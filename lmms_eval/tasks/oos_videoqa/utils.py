import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
import pandas as pd
from loguru import logger as eval_logger

LETTER_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEBUG_EVAL = os.getenv("OOS_DEBUG_EVAL", "0") == "1"
DEBUG_FAIL_ONLY = os.getenv("OOS_DEBUG_FAIL_ONLY", "0") == "1"
DEBUG_REASONING = os.getenv("OOS_DEBUG_REASONING", "0") == "1"
DEBUG_MAX_SAMPLES = int(os.getenv("OOS_DEBUG_MAX_SAMPLES", "0"))

# Controls multi-turn prompt construction.
# gold: previous turns use gold answers when available
# none: each step is asked independently even when mode=multi_turn
OOS_HISTORY_MODE = os.getenv("OOS_HISTORY_MODE", "gold").strip().lower()


OOS_TIME_TOLERANCE_SEC = float(os.getenv("OOS_TIME_TOLERANCE_SEC", "1.0"))
OOS_COORD_TOLERANCE_NORM = float(os.getenv("OOS_COORD_TOLERANCE_NORM", "0.08"))

def _step_id(x: Any) -> str:
    return str(x).strip()


def _normalize_dep_list(dep_list: Optional[List[Any]]) -> List[str]:
    if not dep_list:
        return []
    return [_step_id(x) for x in dep_list]


def _step_sort_key(step: Any):
    s = str(step).strip()
    m = re.match(r"^(\d+)([A-Za-z]*)$", s)
    if m:
        return (int(m.group(1)), m.group(2))
    return (10**9, s)


def _is_time_point_open_task(doc: Dict[str, Any]) -> bool:
    cls = str(doc.get("step_question_class", "")).strip().lower()
    return cls in {
        "oos_step2_last_visible",
        "oos_step3_last_placement",
    }

def _is_multiple_choice(doc: Dict[str, Any]) -> bool:
    return bool(doc.get("choices"))


# def _is_step2_last_visible(doc: Dict[str, Any]) -> bool:
#     return str(doc.get("step_question_class", "")).strip().lower() == "oos_step2_last_visible"
def _time_point_instruction(doc: Dict[str, Any]) -> str:
    example = "Example format: <TIME 00:00:12.3 video 1>; Point=(0.45, 0.62)"

    if DEBUG_REASONING:
        return (
            "Estimate the requested event time and its normalized image location. "
            "Use normalized coordinates where x and y are each between 0 and 1. "
            "First briefly explain your reasoning from the video. "
            "Then on a new line output exactly: "
            "Final Answer: <TIME HH:MM:SS.s video 1>; Point=(<x>, <y>). "
            + example
        )
    return (
        "Give a single structured answer using normalized coordinates. "
        "Output exactly one line in this format: "
        "<TIME HH:MM:SS.s video 1>; Point=(<x>, <y>). "
        "Use x,y normalized to [0,1]. "
        + example
    )


def _format_float(value: float, ndigits: int = 3) -> str:
    text = f"{float(value):.{ndigits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _rewrite_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    base_dir = os.getenv("OOS_VIDEO_BASE_DIR")
    if not base_dir:
        return path
    return str(Path(base_dir) / Path(path).name)


def _stable_cache_dir() -> str:
    cache_dir = os.getenv("OOS_VIDEO_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(tempfile.gettempdir(), "oos_video_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _run_ffmpeg(cmd: List[str], error_prefix: str) -> None:
    # Use custom ffmpeg path if provided
    ffmpeg_path = os.getenv(
        "FFMPEG_PATH"
    )

    # Replace "ffmpeg" with full path
    cmd = [ffmpeg_path if cmd[0] == "ffmpeg" else cmd[0]] + cmd[1:]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ffmpeg not found at {ffmpeg_path}. Please check path."
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore")
        raise RuntimeError(f"{error_prefix}\nSTDERR:\n{stderr}") from e


def _video_preprocess_config() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return (
        os.getenv("OOS_TARGET_FPS"),
        os.getenv("OOS_RESIZE_WIDTH"),
        os.getenv("OOS_RESIZE_HEIGHT"),
    )


def _preprocess_video(video_path: str) -> str:
    """Create one canonical cached version of the full video."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    enable = os.getenv("OOS_PREPROCESS_VIDEO", "0") == "1"
    if not enable:
        return video_path

    target_fps, resize_w, resize_h = _video_preprocess_config()
    if not target_fps and not (resize_w and resize_h):
        return video_path

    cache_dir = _stable_cache_dir()
    config_key = f"preprocess|{video_path}|fps={target_fps}|w={resize_w}|h={resize_h}"
    output_path = os.path.join(cache_dir, hashlib.md5(config_key.encode("utf-8")).hexdigest() + ".mp4")
    if os.path.exists(output_path):
        return output_path

    vf_parts: List[str] = []
    if target_fps:
        vf_parts.append(f"fps={target_fps}")
    if resize_w and resize_h:
        vf_parts.append(f"scale={resize_w}:{resize_h}")

    cmd = ["ffmpeg", "-y", "-i", video_path]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", output_path]
    _run_ffmpeg(cmd, f"Video preprocessing failed for {video_path}")
    return output_path

def _extract_prefix_video(full_video_path: str, end_time: float) -> str:
    """Return the prefix video from 0 to query time."""
    if not os.path.exists(full_video_path):
        raise FileNotFoundError(f"Full video path does not exist: {full_video_path}")
    if end_time <= 0:
        raise ValueError(f"Invalid query end time: {end_time}")

    canonical_video_path = _preprocess_video(full_video_path)
    fast_copy = os.getenv("OOS_FAST_PREFIX_COPY", "0") == "1"

    cache_dir = _stable_cache_dir()
    config_key = f"prefix|src={canonical_video_path}|end={end_time:.3f}|fast_copy={int(fast_copy)}"
    output_path = os.path.join(cache_dir, hashlib.md5(config_key.encode("utf-8")).hexdigest() + ".mp4")
    if os.path.exists(output_path):
        return output_path

    if fast_copy:
        cmd = [
            "ffmpeg", "-y", "-ss", "0", "-i", canonical_video_path,
            "-t", str(float(end_time)),
            "-an", "-c", "copy", "-avoid_negative_ts", "make_zero", output_path,
        ]
        _run_ffmpeg(cmd, f"Fast prefix copy failed for {canonical_video_path}")
        return output_path

    cmd = [
        "ffmpeg", "-y", "-ss", "0", "-i", canonical_video_path,
        "-t", str(float(end_time)),
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", output_path,
    ]
    _run_ffmpeg(cmd, f"Prefix extraction failed for {canonical_video_path}")
    return output_path

def _needs_anchor_marker(doc: Dict[str, Any]) -> bool:
    if os.getenv("OOS_MARK_ANCHOR_OBJECT", "1") != "1":
        return False

    qclass = str(doc.get("step_question_class", "")).strip().lower()
    return qclass in {
        "oos_branch_object_object_relation",
        "oos_branch_object_object_distance",
    }


def _get_anchor_marker_xy_norm(doc: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    meta = doc.get("answer_metadata") or {}

    xy = meta.get("object_y_normalized_projected_pixel")
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return None

    try:
        x_norm, y_norm = float(xy[0]), float(xy[1])
    except Exception:
        return None

    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return None

    return x_norm, y_norm


def _extract_marked_prefix_video(
    full_video_path: str,
    end_time: float,
    marker_xy_norm: Tuple[float, float],
    marker_label: Optional[str] = None,
) -> str:
    """
    Extract video prefix from 0 to end_time and draw a small point marker
    only on the query-time frame.

    marker_xy_norm is normalized coordinate: (x_norm, y_norm), each in [0, 1].
    """
    if not os.path.exists(full_video_path):
        raise FileNotFoundError(f"Full video path does not exist: {full_video_path}")

    if end_time <= 0:
        raise ValueError(f"Invalid query end time: {end_time}")

    canonical_video_path = _preprocess_video(full_video_path)

    # Uses OOS_VIDEO_WIDTH / OOS_VIDEO_HEIGHT in your simplified _ffprobe_size().
    dst_w, dst_h = int(os.environ.get("OOS_VIDEO_WIDTH")), int(os.environ.get("OOS_VIDEO_HEIGHT"))

    x_norm, y_norm = marker_xy_norm

    # Convert normalized coordinate to pixel coordinate in the actual video used by eval.
    x = float(x_norm) * float(dst_w)
    y = float(y_norm) * float(dst_h)

    # Clamp marker center into image bounds.
    x = max(0.0, min(float(dst_w - 1), x))
    y = max(0.0, min(float(dst_h - 1), y))

    # Small square point marker.
    marker_size = int(os.getenv("OOS_MARKER_SIZE_PX", "8"))
    marker_size = max(1, marker_size)

    half = marker_size // 2
    left = int(round(x)) - half
    top = int(round(y)) - half

    # Keep square marker fully inside image.
    left = max(0, min(max(0, int(dst_w) - marker_size), left))
    top = max(0, min(max(0, int(dst_h) - marker_size), top))

    # Mark only the query frame interval.
    # Since your videos are preprocessed to 1 fps, this means the final 1-second interval.
    fps = float(os.getenv("OOS_TARGET_FPS", "1"))
    frame_window = 1.0 / max(fps, 1e-6)

    start_t = max(0.0, float(end_time) - frame_window)
    end_t = float(end_time)

    enable_expr = f"between(t,{start_t:.3f},{end_t:.3f})"

    # Include marker settings in cache key so changing size/fps creates a new cached file.
    config_key = (
        f"marked_prefix|src={canonical_video_path}|end={end_t:.3f}|"
        f"x={x:.2f}|y={y:.2f}|size={marker_size}|"
        f"start={start_t:.3f}|end={end_t:.3f}|point_marker_only=1"
    )

    cache_dir = _stable_cache_dir()
    output_path = os.path.join(
        cache_dir,
        hashlib.md5(config_key.encode("utf-8")).hexdigest() + ".mp4",
    )

    if os.path.exists(output_path):
        return output_path

    # Only drawbox; no drawtext, so it works with your ffmpeg build.
    vf = (
        f"drawbox=x={left}:y={top}:w={marker_size}:h={marker_size}:"
        f"color=red@0.95:t=fill:enable='{enable_expr}'"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        "0",
        "-i",
        canonical_video_path,
        "-t",
        str(end_t),
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        output_path,
    ]

    _run_ffmpeg(cmd, f"Marked prefix extraction failed for {canonical_video_path}")
    return output_path

def _format_choices(choices: List[str]) -> str:
    return "\n".join(f"{LETTER_MAP[i]}. {choice}" for i, choice in enumerate(choices))


def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_relation_text(text: str) -> str:
    text = _normalize_text(text)
    mapping = {
        "right": "to the right",
        "left": "to the left",
        "front": "in front of",
        "in front": "in front of",
        "ahead of": "in front of",
        "back": "behind",
        "at the back": "behind",
    }
    return mapping.get(text, text)


def _clean_prediction(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).strip()

    final_patterns = [
        r"final answer\s*[:\-]\s*([A-Z])\b",
        r"final answer\s*[:\-]\s*(.+)",
        r"answer\s*[:\-]\s*([A-Z])\b",
        r"answer\s*[:\-]\s*(.+)",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            candidate = matches[-1].strip()
            candidate = candidate.splitlines()[0].strip()
            return candidate

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def _extract_letter_index(pred: str, n_choices: int) -> int:
    pred_up = pred.upper().strip()
    patterns = [
        r"^\(?([A-Z])\)?\.?$",
        r"^(?:OPTION|ANSWER)\s*[:\-]?\s*([A-Z])\.?$",
        r"^\s*([A-Z])\s*[\)\.\:\-]\s*",
    ]
    for pattern in patterns:
        m = re.search(pattern, pred_up)
        if m:
            idx = LETTER_MAP.find(m.group(1))
            if 0 <= idx < n_choices:
                return idx
    return -1


def _extract_choice_text_index(pred: str, choices: List[str]) -> int:
    pred_norm = _normalize_relation_text(pred)
    choice_norms = [_normalize_relation_text(c) for c in choices]

    for i, c in enumerate(choice_norms):
        if pred_norm == c:
            return i

    for i, c in enumerate(choice_norms):
        if c and c in pred_norm:
            return i

    for i, c in enumerate(choice_norms):
        if pred_norm and pred_norm in c:
            return i

    return -1


def extract_prediction_index(prediction: str, choices: List[str]) -> int:
    pred = _clean_prediction(prediction)
    idx = _extract_letter_index(pred, len(choices))
    if idx != -1:
        return idx
    idx = _extract_choice_text_index(pred, choices)
    if idx != -1:
        return idx
    return -1


def _step_answer_text(step_doc: Dict[str, Any]) -> Optional[str]:
    if step_doc.get("answer") not in (None, ""):
        return str(step_doc["answer"])
    acceptable_answers = step_doc.get("acceptable_answers") or []
    if acceptable_answers:
        return str(acceptable_answers[0])
    if step_doc.get("choices") and step_doc.get("answer_idx") is not None:
        idx = int(step_doc["answer_idx"])
        if 0 <= idx < len(step_doc["choices"]):
            return str(step_doc["choices"][idx])
    return None

def _history_question_from_step(step: Dict[str, Any]) -> str:
    question_text = str(step.get("question", "")).strip()

    choices = step.get("choices") or []
    if choices:
        choice_lines = "\n".join(
            f"{LETTER_MAP[i]}. {choice}"
            for i, choice in enumerate(choices)
        )
        question_text = f"{question_text}\nOptions:\n{choice_lines}"

    return question_text


def _history_answer_from_step(step: Dict[str, Any], object_name: Optional[str] = None) -> str:
    obj_name = str(object_name or "the object").strip() or "the object"
    qclass = str(step.get("step_question_class", "")).strip().lower()

    answer_text = None

    if step.get("target_text") not in (None, ""):
        answer_text = str(step["target_text"])
    elif step.get("answer") not in (None, ""):
        answer_text = str(step["answer"])
    elif step.get("choices") and step.get("correct_idx") is not None:
        choices = step.get("choices") or []
        idx = int(step["correct_idx"])
        if 0 <= idx < len(choices):
            answer_text = str(choices[idx])
    else:
        acceptable = step.get("acceptable_answers") or []
        if acceptable:
            answer_text = str(acceptable[0])

    if answer_text is None:
        answer_text = ""

    answer_text = str(answer_text).strip()

    m = re.search(
        r"(<TIME\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+video\s+\d+>)"
        r"\s*;\s*Point=\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
        answer_text,
        flags=re.I,
    )

    if m:
        time_token = m.group(1)
        x = m.group(2)
        y = m.group(3)

        if qclass == "oos_step2_last_visible":
            return (
                f"{obj_name} was last visible at {time_token}, "
                f"at normalized image coordinates (x={x}, y={y}), where x and y are in [0, 1]."
            )

        if qclass == "oos_step3_last_placement":
            return (
                f"{obj_name} stopped moving at {time_token}, "
                f"at normalized image coordinates (x={x}, y={y}), where x and y are in [0, 1]."
            )

        return (
            f"The answer is {time_token}, at normalized image coordinates "
            f"(x={x}, y={y}), where x and y are in [0, 1]."
        )

    return answer_text


def _default_open_answer_instruction() -> str:
    if DEBUG_REASONING:
        return (
            "First briefly explain your reasoning from the video. "
            "Then on a new line output: Final Answer: <SHORT TEXT ANSWER>."
        )
    return "Answer briefly and directly."

def _seconds_to_time_token(seconds: float, video_idx: int = 1) -> str:
    seconds = float(seconds)
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    return f"<TIME {hh:02d}:{mm:02d}:{ss:04.1f} video {video_idx}>"


def _parse_time_token_to_seconds(text: str) -> Optional[float]:
    if not text:
        return None

    # Match: <TIME 00:00:55.0 video 1>
    m = re.search(
        r"<TIME\s+(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)\s+video\s+\d+>",
        text,
        flags=re.I,
    )
    if not m:
        return None

    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss

# def _step2_last_visible_instruction(doc: Dict[str, Any]) -> str:
#     example = "Example format: <TIME 00:00:12.3 video 1>; Point=(0.45, 0.62)"

#     if DEBUG_REASONING:
#         return (
#             "Estimate when the target was last visible and its normalized image location. "
#             "Use normalized coordinates where x and y are each between 0 and 1. "
#             "First briefly explain your reasoning from the video. "
#             "Then on a new line output exactly: "
#             "Final Answer: <TIME HH:MM:SS.s video 1>; Point=(<x>, <y>). "
#             + example
#         )
#     return (
#         "Give a single structured answer using normalized coordinates. "
#         "Output exactly one line in this format: "
#         "<TIME HH:MM:SS.s video 1>; Point=(<x>, <y>). "
#         "Use x,y normalized to [0,1]. "
#         + example
#     )



# def _prompt_suffix(doc: Dict[str, Any], include_answer_instruction: bool = True) -> str:
#     lines: List[str] = []
#     if _is_multiple_choice(doc):
#         lines += ["Options:", _format_choices(doc["choices"])]
#         if include_answer_instruction:
#             if DEBUG_REASONING:
#                 lines.append(
#                     "Briefly reason from the video, then on a new line output exactly: Final Answer: <OPTION LETTER>."
#                 )
#             else:
#                 lines.append(
#                     "Select the best option and output only its letter: A, B, C, or D (or the matching option letter if there are more choices)."
#                 )
#     else:
#         if include_answer_instruction:
#             if _is_step2_last_visible(doc):
#                 lines.append(_step2_last_visible_instruction(doc))
#             else:
#                 lines.append(_default_open_answer_instruction())
#     return "\n".join(lines).strip()

def _prompt_suffix(doc: Dict[str, Any], include_answer_instruction: bool = True) -> str:
    lines: List[str] = []
    if _is_multiple_choice(doc):
        lines += ["Options:", _format_choices(doc["choices"])]
        if include_answer_instruction:
            if DEBUG_REASONING:
                lines.append(
                    "Briefly reason from the video, then on a new line output exactly: Final Answer: <OPTION LETTER>."
                )
            else:
                lines.append(
                    "Select the best option and output only its letter."
                )
    else:
        if include_answer_instruction:
            if _is_time_point_open_task(doc):
                lines.append(_time_point_instruction(doc))
            else:
                lines.append(_default_open_answer_instruction())
    return "\n".join(lines).strip()

def _question_block(doc: Dict[str, Any], include_answer_instruction: bool = True) -> str:
    #prefix = f"You are given the video from the beginning until query time {doc['query_time_sec']} seconds."
    suffix = _prompt_suffix(doc, include_answer_instruction=include_answer_instruction)
    blocks = [f"Question: {doc['question']}"]
    if suffix:
        blocks.append(suffix)
    return "\n".join(blocks)


def _normalize_single_turn_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    out["id"] = doc.get("doc_id", doc.get("id", doc.get("trajectory_id")))
    out["doc_id"] = out["id"]
    out["source_video_id"] = doc.get("source_video_id", doc.get("video_id"))
    out["question_class"] = doc.get("step_question_class", doc.get("question_class", "unknown"))
    out["step_question_class"] = doc.get("step_question_class", out["question_class"])
    out["answer_idx"] = None if doc.get("correct_idx") is None else int(doc["correct_idx"])
    out["answer"] = doc.get("target_text", doc.get("answer"))
    out["acceptable_answers"] = doc.get("acceptable_answers") or []
    out["acceptable_answer_idxs"] = doc.get("acceptable_answer_idxs") or []
    out["group_id"] = out["question_class"]
    out["mode"] = doc.get("mode", "single_turn")
    out["video_path"] = _rewrite_path(doc.get("video_path"))
    return out


# def _extract_history_messages_for_step(raw_doc: Dict[str, Any], current_step: int) -> List[Dict[str, Any]]:
#     """
#     Return only prior completed QA turns before the current step.

#     Rules:
#     - keep system messages
#     - for step k, keep only steps < k
#     - never include the current step user question
#     - never include later turns
#     - never include orphan assistant messages
#     """
#     messages = raw_doc.get("gold_history_messages") or []

#     system_msgs: List[Dict[str, Any]] = [
#         msg for msg in messages if msg.get("role") == "system"
#     ]

#     if current_step <= 1:
#         return system_msgs

#     history: List[Dict[str, Any]] = []
#     pending_user: Optional[Dict[str, Any]] = None
#     completed_steps = 0

#     for msg in messages:
#         role = msg.get("role")

#         if role == "system":
#             continue

#         if role == "user":
#             # stop before the current step's user question
#             if completed_steps >= current_step - 1:
#                 break

#             # replace any unfinished pending user
#             pending_user = msg
#             continue

#         if role == "assistant":
#             # only keep complete QA pairs
#             if pending_user is None:
#                 continue

#             history.append(pending_user)
#             history.append(msg)
#             pending_user = None
#             completed_steps += 1

#             if completed_steps >= current_step - 1:
#                 break

#     return system_msgs + history
def _extract_history_messages_for_step(raw_doc: Dict[str, Any], current_step_id: str) -> List[Dict[str, Any]]:
    """
    Return system messages + QA pairs for steps explicitly listed in depends_on_steps.
    Assumes depends_on_steps already contains the full prefix the current step should see.
    """
    messages = raw_doc.get("gold_history_messages") or []

    system_msgs = [msg for msg in messages if msg.get("role") == "system"]

    steps = [s for s in raw_doc.get("steps", []) if not s.get("skipped")]
    step_ids_in_order = [_step_id(s["step"]) for s in steps]

    dep_map = {
        _step_id(s["step"]): _normalize_dep_list(s.get("depends_on_steps"))
        for s in steps
    }
    needed = set(dep_map.get(current_step_id, []))

    if not needed:
        return system_msgs

    history_pairs = {}
    pending_user = None
    pair_index = 0

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            pending_user = msg
            continue
        if role == "assistant" and pending_user is not None:
            if pair_index < len(step_ids_in_order):
                sid = step_ids_in_order[pair_index]
                history_pairs[sid] = [pending_user, msg]
            pending_user = None
            pair_index += 1

    history = []
    step_by_id = {
        _step_id(s["step"]): s
        for s in steps
    }

    object_name = raw_doc.get("object_a_name")

    for sid in step_ids_in_order:
        if sid not in needed:
            continue

        step = step_by_id.get(sid)
        if step is None:
            continue

        history.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": _history_question_from_step(step),
            }],
        })

        history.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": _history_answer_from_step(step, object_name=object_name),
            }],
        })

    return system_msgs + history


# def _expand_multi_turn_doc(raw_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
#     expanded: List[Dict[str, Any]] = []
#     common = {
#         "trajectory_id": raw_doc.get("trajectory_id"),
#         "source_video_id": raw_doc.get("video_id", raw_doc.get("source_video_id")),
#         "video_id": raw_doc.get("video_id"),
#         "video_path": _rewrite_path(raw_doc.get("video_path")),
#         "query_time_sec": raw_doc.get("query_time_sec"),
#         "query_time_in_clip_sec": raw_doc.get("query_time_in_clip_sec"),
#         "clip_start_time_sec": raw_doc.get("clip_start_time_sec"),
#         "clip_end_time_sec": raw_doc.get("clip_end_time_sec"),
#         "clip_duration_sec": raw_doc.get("clip_duration_sec"),
#         "horizon_sec": raw_doc.get("horizon_sec"),
#         "object_a_assoc_id": raw_doc.get("object_a_assoc_id"),
#         "object_a_name": raw_doc.get("object_a_name"),
#         "generation_info": raw_doc.get("generation_info"),
#         "include_gold_history": raw_doc.get("include_gold_history", True),
#         "mode": "multi_turn",
#     }

#     for step in raw_doc.get("steps", []):
#         if step.get("skipped"):
#             continue

#         step_no = int(step.get("step"))
#         item = dict(common)
#         item["id"] = f"{raw_doc.get('doc_id', raw_doc.get('trajectory_id'))}__step_{step_no}"
#         item["doc_id"] = item["id"]
#         item["step"] = step_no
#         item["question_class"] = step.get("step_question_class", raw_doc.get("question_class", "unknown"))
#         item["step_question_class"] = item["question_class"]
#         item["question"] = step.get("question")
#         item["choices"] = step.get("choices") or []
#         item["answer_idx"] = None if step.get("correct_idx") is None else int(step["correct_idx"])
#         item["answer"] = step.get("target_text", step.get("answer"))
#         item["acceptable_answers"] = step.get("acceptable_answers") or []
#         item["acceptable_answer_idxs"] = step.get("acceptable_answer_idxs") or []
#         item["answer_metadata"] = step.get("answer_metadata")
#         item["group_id"] = item["question_class"]

#         if OOS_HISTORY_MODE == "gold" and raw_doc.get("include_gold_history", True):
#             item["history_messages"] = _extract_history_messages_for_step(raw_doc, step_no)
#         else:
#             item["history_messages"] = []

#         expanded.append(item)

#     return expanded
def _expand_multi_turn_doc(raw_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    common = {
        "trajectory_id": raw_doc.get("trajectory_id"),
        "source_video_id": raw_doc.get("video_id", raw_doc.get("source_video_id")),
        "video_id": raw_doc.get("video_id"),
        "video_path": _rewrite_path(raw_doc.get("video_path")),
        "query_time_sec": raw_doc.get("query_time_sec"),
        "query_time_in_clip_sec": raw_doc.get("query_time_in_clip_sec"),
        "clip_start_time_sec": raw_doc.get("clip_start_time_sec"),
        "clip_end_time_sec": raw_doc.get("clip_end_time_sec"),
        "clip_duration_sec": raw_doc.get("clip_duration_sec"),
        "horizon_sec": raw_doc.get("horizon_sec"),
        "object_a_assoc_id": raw_doc.get("object_a_assoc_id"),
        "object_a_name": raw_doc.get("object_a_name"),
        "generation_info": raw_doc.get("generation_info"),
        "include_gold_history": raw_doc.get("include_gold_history", True),
        "mode": "multi_turn",
    }

    for step in raw_doc.get("steps", []):
        if step.get("skipped"):
            continue

        step_id = _step_id(step.get("step"))

        item = dict(common)
        item["id"] = f"{raw_doc.get('doc_id', raw_doc.get('trajectory_id'))}__step_{step_id}"
        item["trajectory_id"] = raw_doc.get("trajectory_id")
        item["doc_id"] = item["id"]
        item["step"] = step_id
        item["branch_group"] = step.get("branch_group")
        item["depends_on_steps"] = _normalize_dep_list(step.get("depends_on_steps"))
        item["question_class"] = step.get("step_question_class", raw_doc.get("question_class", "unknown"))
        item["step_question_class"] = item["question_class"]
        item["question"] = step.get("question")
        item["choices"] = step.get("choices") or []
        item["answer_idx"] = None if step.get("correct_idx") is None else int(step["correct_idx"])
        item["answer"] = step.get("target_text", step.get("answer"))
        item["acceptable_answers"] = step.get("acceptable_answers") or []
        item["acceptable_answer_idxs"] = step.get("acceptable_idxs") or step.get("acceptable_answer_idxs") or []
        item["answer_metadata"] = step.get("answer_metadata")
        item["group_id"] = item["question_class"]

        if OOS_HISTORY_MODE == "gold" and raw_doc.get("include_gold_history", True):
            item["history_messages"] = _extract_history_messages_for_step(raw_doc, step_id)
        else:
            item["history_messages"] = []

        expanded.append(item)

    return expanded

# def warm_video_prefix_cache(dataset: datasets.Dataset) -> None:
#     seen = set()
#     for doc in dataset:
#         video_path = doc.get("video_path")
#         query_time = doc.get("query_time_sec")
#         if not video_path or query_time is None:
#             continue
#         key = (video_path, float(query_time))
#         if key in seen:
#             continue
#         seen.add(key)
#         _extract_prefix_video(video_path, float(query_time))
def warm_video_prefix_cache(dataset: datasets.Dataset) -> None:
    seen = set()

    for doc in dataset:
        video_path = doc.get("video_path")
        query_time = doc.get("query_time_sec")
        if not video_path or query_time is None:
            continue

        query_time = float(query_time)

        if _needs_anchor_marker(doc):
            marker_xy = _get_anchor_marker_xy_norm(doc)
            if marker_xy is not None:
                key = (
                    "marked",
                    video_path,
                    query_time,
                    round(float(marker_xy[0]), 2),
                    round(float(marker_xy[1]), 2),
                )
                if key not in seen:
                    seen.add(key)
                    meta = doc.get("answer_metadata") or {}
                    marker_label = meta.get("object_y_name") or "marked object"
                    _extract_marked_prefix_video(
                        video_path,
                        query_time,
                        marker_xy,
                        marker_label=marker_label,
                    )
                continue

        key = ("prefix", video_path, query_time)
        if key in seen:
            continue
        seen.add(key)
        _extract_prefix_video(video_path, query_time)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    expanded_rows: List[Dict[str, Any]] = []

    for doc in dataset:
        if doc.get("mode") == "multi_turn" and isinstance(doc.get("steps"), list):
            expanded_rows.extend(_expand_multi_turn_doc(doc))
        else:
            expanded_rows.append(_normalize_single_turn_doc(doc))

    debug_step = os.getenv("OOS_DEBUG_STEP", "").strip()

    if debug_step:
        keep_steps = {s.strip() for s in debug_step.split(",")}
        expanded_rows = [
            row for row in expanded_rows
            if str(row.get("step")) in keep_steps
        ]
        eval_logger.info(f"OOS_DEBUG_STEP={debug_step}; kept {len(expanded_rows)} rows")    

    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", "0") == "1":
        eval_logger.info("LMMS_EVAL_SHUFFLE_DOCS detected; shuffling dataset.")
        import random
        rng = random.Random(42)
        rng.shuffle(expanded_rows)

    return datasets.Dataset.from_list(expanded_rows)

def _get_system_prompt(lmms_eval_specific_kwargs=None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    return kwargs["system_prompt"]

# def oos_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
#     if os.getenv("OOS_NO_VIDEO_INPUT", "0") == "1":
#         return []
#     video_path = doc.get("video_path")
#     if not video_path:
#         raise ValueError(f"Missing video_path for doc id={doc.get('id')}")
#     query_time_sec = float(doc.get("query_time_sec", 0.0))
#     prefix_path = _extract_prefix_video(video_path, query_time_sec)
#     return [prefix_path]

def oos_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    if os.getenv("OOS_NO_VIDEO_INPUT", "0") == "1":
        return []

    video_path = doc.get("video_path")
    if not video_path:
        raise ValueError(f"Missing video_path for doc id={doc.get('id')}")

    query_time_sec = float(doc.get("query_time_sec", 0.0))

    if _needs_anchor_marker(doc):
        marker_xy_norm = _get_anchor_marker_xy_norm(doc)
        if marker_xy_norm is not None:
            meta = doc.get("answer_metadata") or {}
            marker_label = meta.get("object_y_name") or "marked object"

            return [
                _extract_marked_prefix_video(
                    video_path,
                    query_time_sec,
                    marker_xy_norm,
                    marker_label=marker_label,
                )
            ]

        eval_logger.warning(
            f"Anchor marker requested but no valid object_y pixel found for doc id={doc.get('id')}"
        )

    prefix_path = _extract_prefix_video(video_path, query_time_sec)
    return [prefix_path]


def oos_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    system_prompt = _get_system_prompt(lmms_eval_specific_kwargs)

    lines: List[str] = [system_prompt]

    if doc.get("history_messages"):
        lines.append("Conversation history:")
        for msg in doc["history_messages"]:
            role = msg.get("role", "user").upper()
            content = msg.get("content") or []
            text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
            if text.strip():
                lines.append(f"{role}: {text}")

    lines.append(_question_block(doc, include_answer_instruction=True))
    return "\n\n".join(lines)


def oos_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> List[Dict[str, Any]]:
    system_prompt = _get_system_prompt(lmms_eval_specific_kwargs)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    if doc.get("history_messages"):
        for msg in doc["history_messages"]:
            role = msg.get("role", "user")
            content = msg.get("content") or []
            if role == "system":
                continue
            messages.append({"role": role, "content": content})

    messages.append(
        {"role": "user", "content": [{"type": "text", "text": _question_block(doc, include_answer_instruction=True)}]}
    )
    return messages


def _safe_jsonable(x: Any):
    if x is None or isinstance(x, (str, int, float, bool, list, dict)):
        return x
    return str(x)




def _parse_first_float(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    return float(match.group(0)) if match else None


def _extract_all_floats(text: str) -> List[float]:
    if not text:
        return []
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)]


def _parse_time_and_point_prediction(prediction: str) -> Dict[str, Optional[float]]:
    clean = _clean_prediction(prediction)
    lowered = clean.lower()

    time_value = _parse_time_token_to_seconds(clean)
    point_x = None
    point_y = None

    # Backward-compatible fallback for old format:
    if time_value is None:
        time_patterns = [
            r"time\s*[:=]\s*([-+]?\d*\.?\d+)\s*s?",
            r"last\s+visible\s*(?:at|time)?\s*[:=]?\s*([-+]?\d*\.?\d+)\s*s?",
            r"([-+]?\d*\.?\d+)\s*s(?:ec(?:ond)?s?)?",
        ]
        for pattern in time_patterns:
            match = re.search(pattern, lowered, flags=re.I)
            if match:
                time_value = float(match.group(1))
                break

    point_patterns = [
        r"point\s*[:=]\s*\(?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)?",
        r"(?:coord(?:inate)?s?|location|position|pixel|xy|x,y)\s*[:=]\s*\(?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)?",
        r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
        r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
        r"x\s*[:=]\s*([-+]?\d*\.?\d+)\s*[,; ]+y\s*[:=]\s*([-+]?\d*\.?\d+)",
    ]
    for pattern in point_patterns:
        match = re.search(pattern, lowered, flags=re.I)
        if match:
            point_x = float(match.group(1))
            point_y = float(match.group(2))
            break

    # Only use generic float fallback if we did NOT already parse a TIME token
    if (time_value is None or point_x is None or point_y is None) and "<time" not in lowered:
        floats = _extract_all_floats(lowered)
        if len(floats) >= 3:
            if time_value is None:
                time_value = floats[0]
            if point_x is None or point_y is None:
                point_x = floats[-2]
                point_y = floats[-1]

    return {
        "time_sec": time_value,
        "x": point_x,
        "y": point_y,
    }


# def _score_step2_last_visible(doc: Dict[str, Any], prediction: str) -> Optional[Dict[str, Any]]:
#     meta = doc.get("answer_metadata") or {}
#     gold_time = meta.get("sampled_last_visible_time_sec")
#     gold_point = meta.get("normalized_projected_pixel") or []

#     if not isinstance(gold_time, (int, float)) or not isinstance(gold_point, list) or len(gold_point) < 2:
#         return None

#     parsed = _parse_time_and_point_prediction(prediction)
#     pred_time = parsed.get("time_sec")
#     pred_x = parsed.get("x")
#     pred_y = parsed.get("y")

#     time_ok = isinstance(pred_time, (int, float)) and abs(float(pred_time) - float(gold_time)) <= OOS_TIME_TOLERANCE_SEC
#     coord_ok = (
#         isinstance(pred_x, (int, float))
#         and isinstance(pred_y, (int, float))
#         and abs(float(pred_x) - float(gold_point[0])) <= OOS_COORD_TOLERANCE_NORM
#         and abs(float(pred_y) - float(gold_point[1])) <= OOS_COORD_TOLERANCE_NORM
#     )
#     correct = float(time_ok and coord_ok)

#     enriched = dict(doc)
#     enriched["prediction"] = prediction
#     enriched["clean_prediction"] = _clean_prediction(prediction)
#     enriched["pred_idx"] = None
#     enriched["pred_choice"] = None
#     enriched["gold_idx"] = None
#     enriched["gold_choice"] = (
#         f"{_seconds_to_time_token(float(gold_time), video_idx=1)}; "
#         f"Point=({_format_float(float(gold_point[0]), 4)}, {_format_float(float(gold_point[1]), 4)})"
#     )
#     enriched["gold_idxs"] = []
#     enriched["gold_choices"] = [enriched["gold_choice"]]
#     enriched["accuracy"] = correct
#     enriched["parsed"] = all(isinstance(v, (int, float)) for v in [pred_time, pred_x, pred_y])
#     enriched["scorable"] = True
#     enriched["pred_time_sec"] = pred_time
#     enriched["pred_point_x"] = pred_x
#     enriched["pred_point_y"] = pred_y
#     enriched["gold_time_sec"] = float(gold_time)
#     enriched["gold_point_x"] = float(gold_point[0])
#     enriched["gold_point_y"] = float(gold_point[1])
#     enriched["time_tolerance_sec"] = OOS_TIME_TOLERANCE_SEC
#     enriched["coord_tolerance_norm"] = OOS_COORD_TOLERANCE_NORM
#     enriched["time_error_sec"] = None if pred_time is None else abs(float(pred_time) - float(gold_time))
#     if pred_x is None or pred_y is None:
#         enriched["coord_error_linf"] = None
#     else:
#         enriched["coord_error_linf"] = max(
#             abs(float(pred_x) - float(gold_point[0])),
#             abs(float(pred_y) - float(gold_point[1])),
#         )
#     enriched["time_within_tolerance"] = bool(time_ok)
#     enriched["coord_within_tolerance"] = bool(coord_ok)
#     _log_sample_result(doc, prediction, None, [], correct)
#     return {"oos_score": enriched}
def _score_time_point_open_task(doc: Dict[str, Any], prediction: str) -> Optional[Dict[str, Any]]:
    meta = doc.get("answer_metadata") or {}
    qclass = str(doc.get("step_question_class", "")).strip().lower()

    if qclass == "oos_step2_last_visible":
        gold_time = meta.get("sampled_last_visible_time_sec")
    elif qclass == "oos_step3_last_placement":
        gold_time = meta.get("last_placement_time_sec")
    else:
        return None

    gold_point = meta.get("normalized_projected_pixel") or []

    if not isinstance(gold_time, (int, float)) or not isinstance(gold_point, list) or len(gold_point) < 2:
        return None

    parsed = _parse_time_and_point_prediction(prediction)
    pred_time = parsed.get("time_sec")
    pred_x = parsed.get("x")
    pred_y = parsed.get("y")

    time_ok = isinstance(pred_time, (int, float)) and abs(float(pred_time) - float(gold_time)) <= OOS_TIME_TOLERANCE_SEC
    coord_ok = (
        isinstance(pred_x, (int, float))
        and isinstance(pred_y, (int, float))
        and abs(float(pred_x) - float(gold_point[0])) <= OOS_COORD_TOLERANCE_NORM
        and abs(float(pred_y) - float(gold_point[1])) <= OOS_COORD_TOLERANCE_NORM
    )
    correct = float(time_ok and coord_ok)

    enriched = dict(doc)
    enriched["prediction"] = prediction
    enriched["clean_prediction"] = _clean_prediction(prediction)
    enriched["pred_idx"] = None
    enriched["pred_choice"] = None
    enriched["gold_idx"] = None
    enriched["gold_choice"] = (
        f"{_seconds_to_time_token(float(gold_time), video_idx=1)}; "
        f"Point=({_format_float(float(gold_point[0]), 4)}, {_format_float(float(gold_point[1]), 4)})"
    )
    enriched["gold_idxs"] = []
    enriched["gold_choices"] = [enriched["gold_choice"]]
    enriched["accuracy"] = correct
    enriched["parsed"] = all(isinstance(v, (int, float)) for v in [pred_time, pred_x, pred_y])
    enriched["scorable"] = True
    enriched["pred_time_sec"] = pred_time
    enriched["pred_point_x"] = pred_x
    enriched["pred_point_y"] = pred_y
    enriched["gold_time_sec"] = float(gold_time)
    enriched["gold_point_x"] = float(gold_point[0])
    enriched["gold_point_y"] = float(gold_point[1])
    enriched["time_tolerance_sec"] = OOS_TIME_TOLERANCE_SEC
    enriched["coord_tolerance_norm"] = OOS_COORD_TOLERANCE_NORM
    enriched["time_error_sec"] = None if pred_time is None else abs(float(pred_time) - float(gold_time))
    enriched["coord_error_linf"] = None if pred_x is None or pred_y is None else max(
        abs(float(pred_x) - float(gold_point[0])),
        abs(float(pred_y) - float(gold_point[1])),
    )
    enriched["time_within_tolerance"] = bool(time_ok)
    enriched["coord_within_tolerance"] = bool(coord_ok)
    _log_sample_result(doc, prediction, None, [], correct)
    return {"oos_score": enriched}


def _log_sample_result(doc: Dict[str, Any], prediction: str, pred_idx: Optional[int], gold_idxs: List[int], correct: float):
    if not DEBUG_EVAL:
        return
    if DEBUG_FAIL_ONLY and correct == 1.0:
        return

    pred_choice = doc["choices"][pred_idx] if pred_idx is not None and 0 <= pred_idx < len(doc.get("choices", [])) else None

    gold_choice = None
    if doc.get("answer_idx") is not None and 0 <= int(doc["answer_idx"]) < len(doc.get("choices", [])):
        gold_choice = doc["choices"][int(doc["answer_idx"])]
    elif doc.get("answer") not in (None, ""):
        gold_choice = doc.get("answer")
    elif doc.get("acceptable_answers"):
        gold_choice = doc["acceptable_answers"][0]

    eval_logger.info("\n===== SAMPLE =====")
    eval_logger.info(f"ID: {doc.get('id')}")
    eval_logger.info(f"Q: {doc.get('question')}")
    eval_logger.info(f"Choices: {doc.get('choices')}")
    eval_logger.info(f"\n--- RAW MODEL OUTPUT ---\n{prediction}")
    eval_logger.info(f"\nPredicted: {pred_choice if pred_choice is not None else _clean_prediction(prediction)}")
    eval_logger.info(f"Gold: {gold_choice}")
    eval_logger.info("========================\n")


# def oos_process_results(doc: Dict[str, Any], results) -> Dict[str, Any]:
#     prediction = results[0] if isinstance(results, (list, tuple)) else results

#     if _is_step2_last_visible(doc):
#         structured_result = _score_step2_last_visible(doc, prediction)
#         if structured_result is not None:
#             return structured_result
def oos_process_results(doc: Dict[str, Any], results) -> Dict[str, Any]:
    prediction = results[0] if isinstance(results, (list, tuple)) else results

    if _is_time_point_open_task(doc):
        structured_result = _score_time_point_open_task(doc, prediction)
        if structured_result is not None:
            return structured_result

    choices = doc.get("choices") or []
    answer_idx = doc.get("answer_idx")
    answer = doc.get("answer")
    acceptable_answers = [str(x) for x in (doc.get("acceptable_answers") or []) if str(x).strip()]

    if choices and answer_idx is not None:
        pred_idx = extract_prediction_index(prediction, choices)
        gold_idx = int(answer_idx)
        gold_idxs = [gold_idx]

        acceptable_answer_idxs = doc.get("acceptable_answer_idxs") or []
        if acceptable_answer_idxs:
            gold_idxs = [int(x) for x in acceptable_answer_idxs]

        correct = float(pred_idx in gold_idxs)
        enriched = dict(doc)
        enriched["prediction"] = prediction
        enriched["clean_prediction"] = _clean_prediction(prediction)
        enriched["pred_idx"] = pred_idx
        enriched["pred_choice"] = choices[pred_idx] if 0 <= pred_idx < len(choices) else None
        enriched["gold_idx"] = gold_idx
        enriched["gold_choice"] = choices[gold_idx] if 0 <= gold_idx < len(choices) else None
        enriched["gold_idxs"] = gold_idxs
        enriched["gold_choices"] = [choices[i] for i in gold_idxs if 0 <= i < len(choices)]
        enriched["accuracy"] = correct
        enriched["parsed"] = pred_idx != -1
        enriched["scorable"] = True
        _log_sample_result(doc, prediction, pred_idx, gold_idxs, correct)
        return {"oos_score": enriched}

    valid_text_answers = []
    if answer not in (None, ""):
        valid_text_answers.append(str(answer))
    valid_text_answers.extend(acceptable_answers)
    valid_text_answers = [x for x in valid_text_answers if _normalize_text(x)]

    if valid_text_answers:
        pred_clean = _normalize_text(_clean_prediction(prediction))
        gold_norms = [_normalize_text(x) for x in valid_text_answers]
        correct = float(pred_clean in gold_norms)
        enriched = dict(doc)
        enriched["prediction"] = prediction
        enriched["clean_prediction"] = _clean_prediction(prediction)
        enriched["pred_idx"] = None
        enriched["pred_choice"] = None
        enriched["gold_idx"] = None
        enriched["gold_choice"] = valid_text_answers[0]
        enriched["gold_idxs"] = []
        enriched["gold_choices"] = valid_text_answers
        enriched["accuracy"] = correct
        enriched["parsed"] = pred_clean != ""
        enriched["scorable"] = True
        _log_sample_result(doc, prediction, None, [], correct)
        return {"oos_score": enriched}

    enriched = dict(doc)
    enriched["prediction"] = prediction
    enriched["clean_prediction"] = _clean_prediction(prediction)
    enriched["pred_idx"] = None
    enriched["pred_choice"] = None
    enriched["gold_idx"] = None
    enriched["gold_choice"] = None
    enriched["gold_idxs"] = []
    enriched["gold_choices"] = []
    enriched["accuracy"] = None
    enriched["parsed"] = _normalize_text(_clean_prediction(prediction)) != ""
    enriched["scorable"] = False
    return {"oos_score": enriched}

# def _aggregate_step_metrics(scored: pd.DataFrame) -> Dict[str, float]:
#     output: Dict[str, float] = {}

#     required_cols = {"mode", "step", "accuracy"}
#     if not required_cols.issubset(scored.columns):
#         return output

#     step_df = scored[scored["mode"] == "multi_turn"].copy()
#     step_df = step_df.dropna(subset=["step"])
#     if len(step_df) == 0:
#         return output

#     step_df["step"] = step_df["step"].astype(int)

#     per_step = step_df.groupby("step")["accuracy"].mean().sort_index()
#     for step, acc in per_step.items():
#         output[f"step_{int(step)}_accuracy"] = float(acc)

#     output["step_macro_avg"] = float(per_step.mean())
#     output["multi_turn_step_count"] = int(len(step_df))
#     return output
def _aggregate_step_metrics(scored: pd.DataFrame) -> Dict[str, float]:
    output: Dict[str, float] = {}

    required_cols = {"mode", "step", "accuracy"}
    if not required_cols.issubset(scored.columns):
        return output

    step_df = scored[scored["mode"] == "multi_turn"].copy()
    step_df = step_df.dropna(subset=["step"])
    if len(step_df) == 0:
        return output

    step_df["step"] = step_df["step"].astype(str)

    step_order = sorted(step_df["step"].unique(), key=_step_sort_key)
    per_step = step_df.groupby("step")["accuracy"].mean()

    for step in step_order:
        output[f"step_{step}_accuracy"] = float(per_step[step])

    output["step_macro_avg"] = float(per_step.mean())
    output["multi_turn_step_count"] = int(len(step_df))
    return output


def _aggregate_trajectory_metrics(scored: pd.DataFrame) -> Dict[str, float]:
    output: Dict[str, float] = {}

    required_cols = {"mode", "trajectory_id", "step", "accuracy"}
    if not required_cols.issubset(scored.columns):
        return output

    traj_df = scored[scored["mode"] == "multi_turn"].copy()
    traj_df = traj_df.dropna(subset=["trajectory_id", "step"])
    if len(traj_df) == 0:
        return output

    traj_df["step"] = traj_df["step"].astype(str)
    traj_df["step_sort_key"] = traj_df["step"].map(_step_sort_key)
    traj_df = traj_df.sort_values(["trajectory_id", "step_sort_key"])

    grouped = traj_df.groupby("trajectory_id", sort=False)

    traj_success = grouped["accuracy"].apply(lambda x: float((x == 1.0).all()))
    output["trajectory_count"] = int(traj_success.shape[0])
    output["trajectory_success_rate"] = float(traj_success.mean())

    first_mistake_steps = []
    completed_without_error = 0

    for _, group in grouped:
        wrong_rows = group[group["accuracy"] != 1.0]
        if len(wrong_rows) == 0:
            completed_without_error += 1
        else:
            first_mistake_steps.append(str(wrong_rows.iloc[0]["step"]))

    output["completed_without_error_rate"] = float(
        completed_without_error / max(len(traj_success), 1)
    )

    if first_mistake_steps:
        numeric_first = []
        for s in first_mistake_steps:
            m = re.match(r"^(\d+)", s)
            if m:
                numeric_first.append(int(m.group(1)))
        output["avg_first_mistake_step"] = float(sum(numeric_first) / len(numeric_first)) if numeric_first else 0.0
        output["median_first_mistake_step"] = float(pd.Series(numeric_first).median()) if numeric_first else 0.0
    else:
        output["avg_first_mistake_step"] = 0.0
        output["median_first_mistake_step"] = 0.0

    propagation_flags = []
    later_error_rates = []

    for _, group in grouped:
        accs = group["accuracy"].tolist()
        first_error_idx = next((i for i, a in enumerate(accs) if a != 1.0), None)
        if first_error_idx is None:
            continue

        later = accs[first_error_idx + 1:]
        if len(later) == 0:
            continue

        has_later_error = any(a != 1.0 for a in later)
        propagation_flags.append(float(has_later_error))
        later_error_rates.append(sum(a != 1.0 for a in later) / len(later))

    output["error_propagation_rate"] = float(sum(propagation_flags) / len(propagation_flags)) if propagation_flags else 0.0
    output["post_error_failure_rate"] = float(sum(later_error_rates) / len(later_error_rates)) if later_error_rates else 0.0

    return output



def oos_aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    df = pd.DataFrame(results)
    if len(df) == 0:
        return {
            "macro_avg": 0.0,
            "micro_avg": 0.0,
            "parse_rate": 0.0,
            "scorable_count": 0,
            "step_macro_avg": 0.0,
            "multi_turn_step_count": 0,
            "trajectory_count": 0,
            "trajectory_success_rate": 0.0,
            "completed_without_error_rate": 0.0,
            "avg_first_mistake_step": 0.0,
            "median_first_mistake_step": 0.0,
            "error_propagation_rate": 0.0,
            "post_error_failure_rate": 0.0,
        }

    output: Dict[str, float] = {}
    output["total_count"] = int(len(df))
    output["scorable_count"] = int(df["scorable"].fillna(False).sum())
    output["unscorable_count"] = int((~df["scorable"].fillna(False)).sum())
    output["parse_rate"] = float(df["parsed"].fillna(False).mean())

    scored = df[df["scorable"] == True].copy()
    if len(scored) == 0:
        output.update(
            {
                "macro_avg": 0.0,
                "micro_avg": 0.0,
                "overall": 0.0,
                "step_macro_avg": 0.0,
                "multi_turn_step_count": 0,
                "trajectory_count": 0,
                "trajectory_success_rate": 0.0,
                "completed_without_error_rate": 0.0,
                "avg_first_mistake_step": 0.0,
                "median_first_mistake_step": 0.0,
                "error_propagation_rate": 0.0,
                "post_error_failure_rate": 0.0,
            }
        )
        return output

    if "question_class" in scored.columns:
        per_class = scored.groupby("question_class")["accuracy"].mean()
        for question_class, acc in per_class.items():
            output[f"{question_class}_accuracy"] = float(acc)
        output["macro_avg"] = float(per_class.mean())
    else:
        output["macro_avg"] = float(scored["accuracy"].mean())

    output["micro_avg"] = float(scored["accuracy"].mean())

    if "source_video_id" in scored.columns:
        output["video_macro_avg"] = float(
            scored.groupby("source_video_id")["accuracy"].mean().mean()
        )

    step_metrics = _aggregate_step_metrics(scored)
    output.update(step_metrics)

    traj_metrics = _aggregate_trajectory_metrics(scored)
    output.update(traj_metrics)

    output["overall"] = output["macro_avg"]

    eval_logger.info(f"Evaluation results: {output}")
    return output