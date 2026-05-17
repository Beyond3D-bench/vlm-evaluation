# """LiteLLM chat backend with doc_to_messages + doc_to_visual video support."""

# from __future__ import annotations

# import os
# import time
# from typing import Any, Dict, List, Optional, Union

# from loguru import logger as eval_logger
# from tqdm import tqdm

# from lmms_eval.api.instance import GenerationResult, TokenCounts
# from lmms_eval.api.registry import register_model
# from lmms_eval.models.chat.openai import OpenAICompatible as OpenAICompatibleChatBase
# from lmms_eval.models.model_utils.usage_metrics import log_usage
# from lmms_eval.models.simple.litellm import _PLACEHOLDER_API_KEY, _LiteLLMClientShim
# from lmms_eval.protocol import ChatMessages


# @register_model("litellm_chat")
# class LiteLLMChatCompatible(OpenAICompatibleChatBase):
#     """LiteLLM-backed chat backend that supports task doc_to_messages + doc_to_visual."""

#     is_simple = False

#     def __init__(
#         self,
#         model_version: str = "openai/gpt-4o-mini",
#         model: Optional[str] = None,
#         base_url: Optional[str] = None,
#         api_key: Optional[str] = None,
#         **kwargs: Any,
#     ) -> None:
#         resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
#         resolved_base_url = base_url or os.getenv("OPENAI_API_BASE")

#         # Force qwen_vl_utils to use decord for local video decoding.
#         os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

#         super().__init__(
#             model_version=model_version,
#             model=model,
#             base_url=resolved_base_url,
#             api_key=resolved_api_key or _PLACEHOLDER_API_KEY,
#             azure_openai=False,
#             **kwargs,
#         )

#         self.client = _LiteLLMClientShim(
#             api_key=resolved_api_key,
#             base_url=resolved_base_url,
#         )

#     def _debug_enabled(self) -> bool:
#         return os.getenv("OOS_CHAT_DEBUG", "0") == "1"

#     def _dbg(self, msg: str) -> None:
#         if self._debug_enabled():
#             print(msg, flush=True)

#     @staticmethod
#     def _ensure_content_list(content: Any) -> List[Dict[str, Any]]:
#         if content is None:
#             return []
#         if isinstance(content, str):
#             return [{"type": "text", "text": content}]
#         if isinstance(content, list):
#             return list(content)
#         raise TypeError(f"Unsupported message content type: {type(content)}")

#     def _normalize_messages(self, raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         if not isinstance(raw_messages, list) or len(raw_messages) == 0:
#             raise ValueError("doc_to_messages returned empty or invalid messages")

#         normalized: List[Dict[str, Any]] = []

#         for msg in raw_messages:
#             if not isinstance(msg, dict):
#                 raise TypeError(f"Message must be dict, got {type(msg)}")

#             role = msg.get("role")
#             if role not in {"system", "user", "assistant"}:
#                 raise ValueError(f"Invalid message role: {role}")

#             content = self._ensure_content_list(msg.get("content"))

#             # Avoid ChatMessages validation errors from text=None.
#             cleaned_content = []
#             for part in content:
#                 if part.get("type") == "text":
#                     text = part.get("text")
#                     if text is None:
#                         text = ""
#                     cleaned_content.append({"type": "text", "text": str(text)})
#                 else:
#                     cleaned_content.append(part)

#             normalized.append({"role": role, "content": cleaned_content})

#         return normalized

#     @staticmethod
#     def _visuals_to_protocol_content(visuals: Any) -> List[Dict[str, Any]]:
#         content: List[Dict[str, Any]] = []

#         if not visuals:
#             return content

#         for visual in visuals:
#             if isinstance(visual, str) and visual.lower().endswith(
#                 (".mp4", ".avi", ".mov", ".mkv", ".webm")
#             ):
#                 content.append({"type": "video", "url": visual})
#             else:
#                 content.append({"type": "image", "url": visual})

#         return content

#     def _attach_visuals_to_first_user(
#         self,
#         messages: List[Dict[str, Any]],
#         visual_content: List[Dict[str, Any]],
#     ) -> List[Dict[str, Any]]:
#         if not visual_content:
#             return messages

#         for msg in messages:
#             if msg.get("role") == "user":
#                 msg["content"] = visual_content + self._ensure_content_list(msg.get("content"))
#                 return messages

#         messages.append({"role": "user", "content": visual_content})
#         return messages

#     def _build_video_kwargs(self) -> Dict[str, Any]:
#         if self.video_fps is not None and self.video_fps > 0:
#             return {
#                 "fps": self.video_fps,
#                 "video_reader_backend": "decord",
#             }

#         return {
#             "nframes": self.max_frames_num,
#             "video_reader_backend": "decord",
#         }

#     def _build_payload_for_request(self, req_args: tuple) -> dict:
#         ctx, doc_to_source, gen_kwargs, doc_id, task, split = req_args
#         doc = self.task_dict[task][split][doc_id]

#         source_name = getattr(doc_to_source, "__name__", "")

#         if source_name == "doc_to_messages" or source_name.endswith("doc_to_messages"):
#             raw_messages = doc_to_source(doc)
#             messages = self._normalize_messages(raw_messages)

#             bound_task = getattr(doc_to_source, "__self__", None)
#             if bound_task is None or not hasattr(bound_task, "doc_to_visual"):
#                 raise ValueError(
#                     f"Could not access doc_to_visual from bound doc_to_messages for task={task}"
#                 )

#             visuals = bound_task.doc_to_visual(doc)
#             visual_content = self._visuals_to_protocol_content(visuals)
#             messages = self._attach_visuals_to_first_user(messages, visual_content)

#             self._dbg("\n" + "=" * 100)
#             self._dbg(
#                 f"[LITELLM CHAT] doc_id={doc_id} "
#                 f"id={doc.get('id')} step={doc.get('step')}"
#             )
#             self._dbg(f"[LITELLM CHAT] visuals={visuals}")

#             for i, msg in enumerate(messages):
#                 types = [
#                     part.get("type")
#                     for part in self._ensure_content_list(msg.get("content"))
#                 ]
#                 self._dbg(f"[MSG] idx={i} role={msg.get('role')} types={types}")

#         else:
#             # Fallback path for non-chat tasks.
#             visuals = doc_to_source(doc)
#             visual_content = self._visuals_to_protocol_content(visuals)
#             messages = [
#                 {
#                     "role": "user",
#                     "content": visual_content + [{"type": "text", "text": str(ctx)}],
#                 }
#             ]

#         request_gen_kwargs = dict(gen_kwargs)
#         max_new_tokens = min(request_gen_kwargs.get("max_new_tokens", 1024), 4096)
#         temperature = request_gen_kwargs.get("temperature", 0)

#         chat_messages = ChatMessages(messages=messages)

#         payload = {
#             "messages": chat_messages.to_openai_messages(
#                 video_kwargs=self._build_video_kwargs()
#             ),
#             "model": self.model_version,
#             "max_tokens": max_new_tokens,
#             "temperature": temperature,
#         }

#         import json
#         payload_size_mb = len(json.dumps(payload).encode("utf-8")) / (1024 * 1024)
#         self._dbg(f"[LITELLM CHAT] payload_size_mb={payload_size_mb:.2f}")
#         # Reasoning/OpenAI-family special handling inherited from chat OpenAI pattern.
#         if (
#             "o1" in self.model_version
#             or "o3" in self.model_version
#             or "o4" in self.model_version
#             or "gpt-5" in self.model_version
#         ):
#             payload.pop("temperature", None)
#             payload.pop("max_tokens", None)
#             payload["response_format"] = {"type": "text"}

#         return payload

#     def generate_until(self, requests) -> List[GenerationResult]:
#         if not requests:
#             return []

#         reordered_requests = list(requests)
#         responses: List[Union[GenerationResult, None]] = [None] * len(reordered_requests)

#         pbar = tqdm(
#             total=len(reordered_requests),
#             disable=(self.rank != 0),
#             desc="Model Responding",
#         )

#         for idx, req in enumerate(reordered_requests):
#             started_at = time.time()

#             try:
#                 payload = self._build_payload_for_request(req.args)
#                 response = self.client.chat.completions.create(**payload)

#                 elapsed = time.time() - started_at
#                 response_text = response.choices[0].message.content or ""

#                 input_tokens = 0
#                 output_tokens = len(response_text.split())
#                 reasoning_tokens = 0

#                 if hasattr(response, "usage") and response.usage:
#                     input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
#                     output_tokens = getattr(response.usage, "completion_tokens", 0) or output_tokens

#                     details = getattr(response.usage, "completion_tokens_details", None)
#                     if details:
#                         reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

#                 log_usage(
#                     model_name=self.model_version,
#                     task_name=None,
#                     input_tokens=input_tokens,
#                     output_tokens=output_tokens,
#                     reasoning_tokens=reasoning_tokens,
#                     source="model",
#                 )

#                 responses[idx] = GenerationResult(
#                     text=response_text,
#                     token_counts=TokenCounts(
#                         input_tokens=input_tokens,
#                         output_tokens=output_tokens,
#                         reasoning_tokens=reasoning_tokens,
#                     ),
#                 )

#             except Exception as exc:
#                 elapsed = time.time() - started_at
#                 error_msg = str(exc)
#                 eval_logger.error(f"LiteLLM chat request failed: {error_msg}")

#                 responses[idx] = GenerationResult(
#                     text=f"[LMMS_EVAL_REQUEST_FAILED] {error_msg[:200]}",
#                     token_counts=TokenCounts(),
#                 )

#             pbar.update(1)

#         pbar.close()
#         return responses

"""LiteLLM chat backend with doc_to_messages + doc_to_visual video support.

Optional OOS chunk-evidence mode:
1. Build the normal prefix video from the task's doc_to_visual.
2. Split that prefix video into short overlapping chunks.
3. Ask the VLM for timestamped evidence for each chunk.
4. Send the final request as text only: previous QA context + chunk evidence + current question.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.openai import OpenAICompatible as OpenAICompatibleChatBase
from lmms_eval.models.model_utils.usage_metrics import log_usage
from lmms_eval.models.simple.litellm import _PLACEHOLDER_API_KEY, _LiteLLMClientShim
from lmms_eval.protocol import ChatMessages


@register_model("litellm_chat")
class LiteLLMChatCompatible(OpenAICompatibleChatBase):
    """LiteLLM-backed chat backend that supports task doc_to_messages + doc_to_visual."""

    is_simple = False

    def __init__(
        self,
        model_version: str = "openai/gpt-4o-mini",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_API_BASE")

        # Force qwen_vl_utils to use decord for local video decoding.
        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

        super().__init__(
            model_version=model_version,
            model=model,
            base_url=resolved_base_url,
            api_key=resolved_api_key or _PLACEHOLDER_API_KEY,
            azure_openai=False,
            **kwargs,
        )

        self.client = _LiteLLMClientShim(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )

    def _debug_enabled(self) -> bool:
        return os.getenv("OOS_CHAT_DEBUG", "0") == "1"

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled():
            print(msg, flush=True)

    def _redact_for_log(self, text: str) -> str:
        """Avoid accidentally logging API keys or huge binary-like strings."""
        if text is None:
            return ""
        text = str(text)
        for key_name in ["OPENAI_API_KEY", "OLLAMA_API_KEY"]:
            key = os.getenv(key_name)
            if key:
                text = text.replace(key, f"<REDACTED_{key_name}>")
        return text


    def _content_to_debug_text(self, content: Any) -> str:
        parts = []
        for part in self._ensure_content_list(content):
            ptype = part.get("type")
            if ptype == "text":
                parts.append(part.get("text", ""))
            elif ptype == "video":
                parts.append(f"<VIDEO url={part.get('url')}>")
            elif ptype == "image":
                parts.append(f"<IMAGE url={part.get('url')}>")
            else:
                parts.append(f"<{ptype}: {part}>")
        return "\n".join(str(x) for x in parts)


    def _log_messages_full(self, title: str, messages: List[Dict[str, Any]]) -> None:
        if not self._debug_enabled():
            return

        self._dbg("\n" + "=" * 100)
        self._dbg(f"[{title}] FULL MESSAGES SENT TO MODEL")
        self._dbg("=" * 100)

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            text = self._content_to_debug_text(msg.get("content"))
            text = self._redact_for_log(text)
            self._dbg(f"\n--- message[{i}] role={role} ---\n{text}")

        self._dbg("\n" + "=" * 100)


    def _clean_for_debug_parse(self, doc: Dict[str, Any], text: str) -> str:
        """Mirror the most important evaluator parsing cases for debug logs only."""
        import re

        text = (text or "").strip()
        qclass = str(doc.get("step_question_class", "")).strip().lower()
        choices = doc.get("choices") or []

        if qclass in {"oos_step2_last_visible", "oos_step3_last_placement"}:
            m = re.search(
                r"(<TIME\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+video\s+\d+>\s*;\s*Point=\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\))",
                text,
                flags=re.I,
            )
            if m:
                return m.group(1)

            # Fallback: show why parser may fail.
            return "[NO STRICT TIME+POINT MATCH] " + text.splitlines()[-1][:500]

        if choices:
            m = re.search(r"(?:final answer|answer)\s*[:\-]?\s*([A-Z])\b", text, flags=re.I)
            if m:
                idx = ord(m.group(1).upper()) - ord("A")
                if 0 <= idx < len(choices):
                    return m.group(1).upper()

            m = re.search(r"\b([A-Z])\b", text.upper())
            if m:
                idx = ord(m.group(1)) - ord("A")
                if 0 <= idx < len(choices):
                    return m.group(1)

            return "[NO OPTION MATCH] " + text.splitlines()[-1][:500]

        return text.splitlines()[-1] if text else ""
    @staticmethod
    def _ensure_content_list(content: Any) -> List[Dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            return list(content)
        raise TypeError(f"Unsupported message content type: {type(content)}")

    def _normalize_messages(self, raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(raw_messages, list) or len(raw_messages) == 0:
            raise ValueError("doc_to_messages returned empty or invalid messages")

        normalized: List[Dict[str, Any]] = []

        for msg in raw_messages:
            if not isinstance(msg, dict):
                raise TypeError(f"Message must be dict, got {type(msg)}")

            role = msg.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Invalid message role: {role}")

            content = self._ensure_content_list(msg.get("content"))

            # Avoid ChatMessages validation errors from text=None.
            cleaned_content = []
            for part in content:
                if part.get("type") == "text":
                    text = part.get("text")
                    if text is None:
                        text = ""
                    cleaned_content.append({"type": "text", "text": str(text)})
                else:
                    cleaned_content.append(part)

            normalized.append({"role": role, "content": cleaned_content})

        return normalized

    @staticmethod
    def _visuals_to_protocol_content(visuals: Any) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []

        if not visuals:
            return content

        for visual in visuals:
            if isinstance(visual, str) and visual.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv", ".webm")
            ):
                content.append({"type": "video", "url": visual})
            else:
                content.append({"type": "image", "url": visual})

        return content

    def _attach_visuals_to_first_user(
        self,
        messages: List[Dict[str, Any]],
        visual_content: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not visual_content:
            return messages

        for msg in messages:
            if msg.get("role") == "user":
                msg["content"] = visual_content + self._ensure_content_list(msg.get("content"))
                return messages

        messages.append({"role": "user", "content": visual_content})
        return messages

    def _build_video_kwargs(self) -> Dict[str, Any]:
        if self.video_fps is not None and self.video_fps > 0:
            return {
                "fps": self.video_fps,
                "video_reader_backend": "decord",
            }

        return {
            "nframes": self.max_frames_num,
            "video_reader_backend": "decord",
        }

    # -------------------------------
    # Chunk-evidence helpers
    # -------------------------------
    @staticmethod
    def _chunk_evidence_enabled() -> bool:
        return os.getenv("OOS_CHUNK_EVIDENCE", "0") == "1"

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    @staticmethod
    def _stable_cache_dir() -> str:
        cache_dir = os.getenv("OOS_VIDEO_CACHE_DIR")
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), "oos_video_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def _ffmpeg_bin() -> str:
        return os.getenv("FFMPEG_PATH") or "ffmpeg"

    @staticmethod
    def _ffprobe_bin() -> str:
        explicit = os.getenv("FFPROBE_PATH")
        if explicit:
            return explicit
        ffmpeg_path = os.getenv("FFMPEG_PATH")
        if ffmpeg_path:
            candidate = str(Path(ffmpeg_path).with_name("ffprobe"))
            if os.path.exists(candidate):
                return candidate
        return "ffprobe"

    @classmethod
    def _ffprobe_duration(cls, video_path: str) -> float:
        cmd = [
            cls._ffprobe_bin(),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    @classmethod
    def _run_ffmpeg(cls, cmd: List[str], error_prefix: str) -> None:
        cmd = [cls._ffmpeg_bin() if cmd[0] == "ffmpeg" else cmd[0]] + cmd[1:]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
            raise RuntimeError(f"{error_prefix}\nSTDERR:\n{stderr}") from exc

    @classmethod
    def _extract_video_chunk(cls, video_path: str, start_s: float, end_s: float) -> str:
        cache_dir = cls._stable_cache_dir()
        duration_s = max(0.05, float(end_s) - float(start_s))
        src_stat = os.stat(video_path)
        key = (
            f"chunk|src={os.path.abspath(video_path)}|mtime={src_stat.st_mtime_ns}|"
            f"start={start_s:.3f}|end={end_s:.3f}|v2"
        )
        output_path = os.path.join(cache_dir, hashlib.md5(key.encode("utf-8")).hexdigest() + ".mp4")
        if os.path.exists(output_path):
            return output_path

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.3f}",
            "-i", video_path,
            "-t", f"{duration_s:.3f}",
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            output_path,
        ]
        cls._run_ffmpeg(cmd, f"Chunk extraction failed for {video_path} [{start_s:.3f}, {end_s:.3f}]")
        return output_path

    @staticmethod
    def _question_class(doc: Dict[str, Any]) -> str:
        return str(doc.get("step_question_class", doc.get("question_class", ""))).strip().lower()

    @staticmethod
    def _history_mode() -> str:
        mode = os.getenv("OOS_HISTORY_MODE", "gold").strip().lower()
        return mode if mode in {"gold", "pred", "none"} else "gold"

    @staticmethod
    def _is_last_visible_class(qclass: str) -> bool:
        return qclass == "oos_step2_last_visible"

    @staticmethod
    def _is_last_placement_class(qclass: str) -> bool:
        return qclass == "oos_step3_last_placement"

    @staticmethod
    def _is_anchor_relation_class(qclass: str) -> bool:
        return qclass in {
            "oos_branch_object_object_relation",
            "oos_branch_object_object_distance",
        }

    @staticmethod
    def _is_event_scan_class(qclass: str) -> bool:
        return qclass in {"oos_step2_last_visible", "oos_step3_last_placement"}

    def _chunk_params_for_doc(self, doc: Dict[str, Any]) -> Tuple[float, float, int]:
        """Choose chunking defaults by question class.

        Step 2/3 need tighter chunks because they require precise time + point.
        Later fixture/spatial questions can tolerate larger chunks. Env vars can override.
        """
        qclass = self._question_class(doc)
        if self._is_event_scan_class(qclass):
            chunk_s = self._float_env("OOS_EVENT_CHUNK_SECONDS", self._float_env("OOS_CHUNK_SECONDS", 10.0))
            overlap_s = self._float_env("OOS_EVENT_CHUNK_OVERLAP_SECONDS", self._float_env("OOS_CHUNK_OVERLAP_SECONDS", 2.0))
        elif self._is_anchor_relation_class(qclass):
            chunk_s = self._float_env("OOS_SPATIAL_CHUNK_SECONDS", self._float_env("OOS_CHUNK_SECONDS", 20.0))
            overlap_s = self._float_env("OOS_SPATIAL_CHUNK_OVERLAP_SECONDS", self._float_env("OOS_CHUNK_OVERLAP_SECONDS", 5.0))
        else:
            chunk_s = self._float_env("OOS_CHUNK_SECONDS", 30.0)
            overlap_s = self._float_env("OOS_CHUNK_OVERLAP_SECONDS", 5.0)

        chunk_s = max(0.5, chunk_s)
        overlap_s = max(0.0, overlap_s)
        if overlap_s >= chunk_s:
            overlap_s = max(0.0, chunk_s * 0.25)
        max_chunks = max(1, self._int_env("OOS_MAX_CHUNKS", 200))
        return chunk_s, overlap_s, max_chunks

    def _split_prefix_video(self, video_path: str, doc: Optional[Dict[str, Any]] = None) -> List[Tuple[float, float, str]]:
        duration = self._ffprobe_duration(video_path)
        doc = doc or {}
        chunk_s, overlap_s, max_chunks = self._chunk_params_for_doc(doc)
        chunks: List[Tuple[float, float, str]] = []
        start = 0.0

        while start < duration and len(chunks) < max_chunks:
            end = min(duration, start + chunk_s)
            if end - start >= 0.05:
                chunks.append((start, end, self._extract_video_chunk(video_path, start, end)))
            if end >= duration:
                break
            next_start = end - overlap_s
            if next_start <= start + 1e-6:
                next_start = end
            start = next_start

        return chunks

    def _payload_from_messages(
        self,
        messages: List[Dict[str, Any]],
        gen_kwargs: Dict[str, Any],
        *,
        max_tokens_override: Optional[int] = None,
        include_video_kwargs: bool = True,
    ) -> Dict[str, Any]:
        request_gen_kwargs = dict(gen_kwargs)
        max_new_tokens = min(request_gen_kwargs.get("max_new_tokens", 1024), 4096)
        if max_tokens_override is not None:
            max_new_tokens = int(max_tokens_override)
        temperature = request_gen_kwargs.get("temperature", 0)

        chat_messages = ChatMessages(messages=messages)
        if include_video_kwargs:
            openai_messages = chat_messages.to_openai_messages(video_kwargs=self._build_video_kwargs())
        else:
            openai_messages = chat_messages.to_openai_messages()

        payload = {
            "messages": openai_messages,
            "model": self.model_version,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        if (
            "o1" in self.model_version
            or "o3" in self.model_version
            or "o4" in self.model_version
            or "gpt-5" in self.model_version
        ):
            payload.pop("temperature", None)
            payload.pop("max_tokens", None)
            payload["response_format"] = {"type": "text"}
            if max_tokens_override is not None:
                payload["max_completion_tokens"] = int(max_tokens_override)

        return payload

    def _message_text(self, msg: Dict[str, Any]) -> str:
        parts = []
        for part in self._ensure_content_list(msg.get("content")):
            if part.get("type") == "text" and str(part.get("text", "")).strip():
                parts.append(str(part.get("text", "")).strip())
        return "\n".join(parts).strip()

    def _messages_to_text(self, messages: List[Dict[str, Any]], *, label_history: bool = True) -> str:
        lines: List[str] = []
        for msg in messages:
            text = self._message_text(msg)
            if not text:
                continue
            role = str(msg.get("role", "user")).upper()
            if label_history:
                lines.append(f"{role}: {text}")
            else:
                lines.append(text)
        return "\n".join(lines).strip()

    def _current_user_message(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for msg in reversed(messages):
            if msg.get("role") == "user" and self._message_text(msg):
                return msg
        return None

    def _system_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [m for m in messages if m.get("role") == "system" and self._message_text(m)]

    def _messages_for_mode(self, messages: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
        """Return the context that is allowed under gold/pred/none.

        gold/pred keep the already-built dependency history from doc_to_messages.
        none strips previous QA and keeps only system + current question.
        """
        current = self._current_user_message(messages)
        if mode == "none":
            out = list(self._system_messages(messages))
            if current is not None:
                out.append(current)
            return out
        return messages

    def _history_label(self, mode: str) -> str:
        if mode == "gold":
            return "Previous gold QA context and current question"
        if mode == "pred":
            return "Previous predicted QA context and current question"
        return "Current question only; no previous QA context"

    def _build_question_type_instructions(self, doc: Dict[str, Any]) -> str:
        qclass = self._question_class(doc)
        object_name = str(doc.get("object_a_name") or "the target object")

        if qclass == "oos_step1_visibility":
            return f"""Report visible/not-visible/uncertain evidence for this chunk."""

        if self._is_last_visible_class(qclass):
            return f"""Also give accurate location with normalized coordinates=[x,y] where x and y are between 0 and 1."""

        if self._is_last_placement_class(qclass):
            return f"""Also give accurate location with normalized coordinates=[x,y] where x and y are between 0 and 1."""

        if qclass == "oos_step4_fixture":
            return f""""""

        if qclass == "oos_branch_object_camera_relative_position":
            return f"""Track the kitchen layout across chunks. Especially keep track of the camera viewpoint and location close to query time as reference."""
        
        if qclass == "oos_branch_object_object_relation":
            return f"""Track the kitchen layout across chunks. Especially keep track of the camera viewpoint and location close to query time as reference."""

        if qclass == "oos_branch_object_object_distance":
            return f"""Track the kitchen layout across chunks."""

        return f"""Evidence schema for GENERAL OOS VIDEO QA.
Extract concise evidence about {object_name}, visibility, movement, placement, nearby fixtures, and spatial relations relevant to the current question."""

    def _build_evidence_prompt(
        self,
        *,
        doc: Dict[str, Any],
        start_s: float,
        end_s: float,
        context_messages: List[Dict[str, Any]],
    ) -> str:
        mode = self._history_mode()
        object_name = str(doc.get("object_a_name") or "the target object")
        question_text = str(doc.get("question") or "").strip()

        choices = doc.get("choices") or []
        if choices:
            choice_lines = "\n".join(
                f"{chr(ord('A') + i)}. {choice}"
                for i, choice in enumerate(choices)
            )
            question_text = f"{question_text}\nOptions:\n{choice_lines}"

        history_block = ""
        if mode != "none":
            current_question = question_text.strip()
            all_context = self._messages_to_text(context_messages)
            history_text = all_context.replace(current_question, "").strip()

            if mode == "gold" and history_text:
                history_block = f"Previous gold QA:\n{history_text}\n\n"
            elif mode == "pred" and history_text:
                history_block = f"Previous predicted QA:\n{history_text}\n\n"

        return f"""You are a helpful assistant trained to answer spatial and visual questions based on egocentric videos. The videos are captured from a first-person perspective and contain various objects and interactions. The video is sampled at 1 frame per second.
    
    You are given a video chunk extracted from the full video. 
    Video chunk global time range: {start_s:.1f}s-{end_s:.1f}s. The global timestamp is displayed in the bottom-right corner of the video. 
    target object: {object_name} 
    {history_block}
    Question: {question_text}
    Extract visual and spatial evidence from this chunk that can later be combined with other chunk evidence to help answer the question. Keep track of the target object through video.
    Describe the target object by color, appearance, human interaction (if any), and location.
    {self._build_question_type_instructions(doc)}
    Do NOT answer the question yet. 
    If you find relevant evidence in this chunk, return concise bullets or JSON-like events.
    If no relevant evidence, return exactly: No relevant evidence.""".strip()

    def _ask_chunk_for_evidence(
        self,
        *,
        chunk_path: str,
        start_s: float,
        end_s: float,
        doc: Dict[str, Any],
        context_messages: List[Dict[str, Any]],
        gen_kwargs: Dict[str, Any],
    ) -> str:
        prompt = self._build_evidence_prompt(
            doc=doc,
            start_s=start_s,
            end_s=end_s,
            context_messages=context_messages,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": chunk_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        self._log_messages_full(
            f"CHUNK_EVIDENCE_PROMPT {start_s:.1f}-{end_s:.1f}",
            messages,
        )
        max_tokens_default = 384 if self._is_event_scan_class(self._question_class(doc)) else 256
        max_tokens = self._int_env("OOS_EVIDENCE_MAX_TOKENS", max_tokens_default)
        payload = self._payload_from_messages(
            messages,
            gen_kwargs,
            max_tokens_override=max_tokens,
            include_video_kwargs=True,
        )
        response = self.client.chat.completions.create(**payload)
        text = response.choices[0].message.content or ""
        if self._debug_enabled():
            self._dbg("\n" + "-" * 100)
            self._dbg(f"[CHUNK_EVIDENCE_RAW_OUTPUT] {start_s:.1f}-{end_s:.1f}")
            self._dbg(self._redact_for_log(text))
            self._dbg("-" * 100)

        input_tokens = 0
        output_tokens = len(text.split())
        reasoning_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or output_tokens
            details = getattr(response.usage, "completion_tokens_details", None)
            if details:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        log_usage(
            model_name=self.model_version,
            task_name=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            source="model",
        )
        return text.strip()

    def _collect_chunk_evidence(
        self,
        visuals: Any,
        base_messages: List[Dict[str, Any]],
        gen_kwargs: Dict[str, Any],
        doc: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        mode = self._history_mode()
        context_messages = self._messages_for_mode(base_messages, mode)
        qclass = self._question_class(doc)
        visual_paths = [
            v for v in (visuals or [])
            if isinstance(v, str) and v.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        ]

        for visual_idx, video_path in enumerate(visual_paths, start=1):
            chunks = self._split_prefix_video(video_path, doc=doc)
            self._dbg(
                f"[CHUNK EVIDENCE] mode={mode} qclass={qclass} "
                f"video={video_path} chunks={len(chunks)}"
            )

            for chunk_idx, (start_s, end_s, chunk_path) in enumerate(chunks, start=1):
                try:
                    text = self._ask_chunk_for_evidence(
                        chunk_path=chunk_path,
                        start_s=start_s,
                        end_s=end_s,
                        doc=doc,
                        context_messages=context_messages,
                        gen_kwargs=gen_kwargs,
                    )
                except Exception as exc:
                    text = f"[chunk evidence failed: {str(exc)[:160]}]"
                    eval_logger.error(
                        f"Chunk evidence failed for {video_path} chunk {chunk_idx} "
                        f"[{start_s:.3f}, {end_s:.3f}]: {exc}"
                    )

                evidence.append(
                    {
                        "video_idx": visual_idx,
                        "chunk_idx": chunk_idx,
                        "start_s": start_s,
                        "end_s": end_s,
                        "text": text,
                    }
                )

        evidence.sort(key=lambda x: (float(x["start_s"]), int(x["video_idx"]), int(x["chunk_idx"])))
        return evidence

    def _final_instruction_for_doc(self, doc: Dict[str, Any]) -> str:
        qclass = self._question_class(doc)

        if qclass == "oos_step2_last_visible":
            return (
                "FINAL ANSWER FORMAT:\n"
                "Return exactly one line and nothing else:\n"
                "<TIME HH:MM:SS.s video 1>; Point=(<x>, <y>)"
            )

        if qclass == "oos_step3_last_placement":
            return (
                "FINAL ANSWER FORMAT:\n"
                "Return exactly one line and nothing else:\n"
                "<TIME HH:MM:SS.s video 1>; Point=(<x>, <y>)"
            )

        choices = doc.get("choices") or []
        if choices:
            letters = ", ".join(chr(ord("A") + i) for i in range(len(choices)))
            return (
                "FINAL ANSWER FORMAT:\n"
                f"Return exactly one letter from: {letters}.\n"
                "Do not explain. Do not include punctuation. Do not include any other words."
            )

        return (
            "FINAL ANSWER FORMAT:\n"
            "Return only the short final answer. Do not explain."
        )
    def _merge_evidence_into_final_messages(
        self,
        messages: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        doc: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Text-only copy using only the context allowed for the selected history mode.
        mode = self._history_mode()
        allowed_messages = self._messages_for_mode(messages, mode)
        final_messages: List[Dict[str, Any]] = []
        for msg in allowed_messages:
            text_parts = []
            for part in self._ensure_content_list(msg.get("content")):
                if part.get("type") == "text":
                    text_parts.append({"type": "text", "text": str(part.get("text", ""))})
            if text_parts:
                final_messages.append({"role": msg.get("role", "user"), "content": text_parts})

        evidence_lines = [
            f"Chunk evidence extracted from the prefix video, sorted by global time. Use this evidence to help answer the question. Do any reasoning internally, but do not write the reasoning.",
        ]
        for item in evidence:
            text = str(item["text"]).strip()
            if not text:
                text = "No relevant evidence."
            evidence_lines.append(
                f"[video {int(item['video_idx'])}, chunk {int(item['chunk_idx'])}, "
                f"{float(item['start_s']):.1f}s-{float(item['end_s']):.1f}s] {text}"
            )
        evidence_block = "\n".join(evidence_lines).strip()
        final_instruction = self._final_instruction_for_doc(doc)

        # Put evidence directly before the current question, i.e. in the last user message.
        for msg in reversed(final_messages):
            if msg.get("role") == "user":
                content = self._ensure_content_list(msg.get("content"))
                if content and content[-1].get("type") == "text":
                    content[-1]["text"] = (
                        f"{evidence_block}\n\n{final_instruction}\n\n"
                        f"Current question:\n{content[-1].get('text', '')}"
                    )
                else:
                    content.append({"type": "text", "text": f"{evidence_block}\n\n{final_instruction}"})
                msg["content"] = content
                return final_messages

        final_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"{evidence_block}\n\n{final_instruction}"}],
        })
        return final_messages

    def _build_payload_for_request(self, req_args: tuple) -> dict:
        ctx, doc_to_source, gen_kwargs, doc_id, task, split = req_args
        doc = self.task_dict[task][split][doc_id]

        source_name = getattr(doc_to_source, "__name__", "")

        if source_name == "doc_to_messages" or source_name.endswith("doc_to_messages"):
            raw_messages = doc_to_source(doc)
            messages = self._normalize_messages(raw_messages)

            bound_task = getattr(doc_to_source, "__self__", None)
            if bound_task is None or not hasattr(bound_task, "doc_to_visual"):
                raise ValueError(
                    f"Could not access doc_to_visual from bound doc_to_messages for task={task}"
                )

            visuals = bound_task.doc_to_visual(doc)

            if self._chunk_evidence_enabled() and visuals:
                evidence = self._collect_chunk_evidence(visuals, messages, gen_kwargs, doc)
                messages = self._merge_evidence_into_final_messages(messages, evidence, doc)
                include_video_kwargs = False
                self._dbg(f"[CHUNK EVIDENCE] collected={len(evidence)} final_request=text_only")
            else:
                visual_content = self._visuals_to_protocol_content(visuals)
                messages = self._attach_visuals_to_first_user(messages, visual_content)
                include_video_kwargs = True

            self._dbg("\n" + "=" * 100)
            self._dbg(
                f"[LITELLM CHAT] doc_id={doc_id} "
                f"id={doc.get('id')} step={doc.get('step')}"
            )
            self._dbg(f"[LITELLM CHAT] visuals={visuals}")

            for i, msg in enumerate(messages):
                types = [
                    part.get("type")
                    for part in self._ensure_content_list(msg.get("content"))
                ]
                self._dbg(f"[MSG] idx={i} role={msg.get('role')} types={types}")

        else:
            # Fallback path for non-chat tasks.
            visuals = doc_to_source(doc)
            visual_content = self._visuals_to_protocol_content(visuals)
            messages = [
                {
                    "role": "user",
                    "content": visual_content + [{"type": "text", "text": str(ctx)}],
                }
            ]
            include_video_kwargs = True

        self._log_messages_full("FINAL_REQUEST", messages)

        payload= self._payload_from_messages(
            messages,
            gen_kwargs,
            include_video_kwargs=include_video_kwargs,
        )
        return payload, doc

    def generate_until(self, requests) -> List[GenerationResult]:
        if not requests:
            return []

        reordered_requests = list(requests)
        responses: List[Union[GenerationResult, None]] = [None] * len(reordered_requests)

        pbar = tqdm(
            total=len(reordered_requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        for idx, req in enumerate(reordered_requests):
            started_at = time.time()

            try:
                payload, doc = self._build_payload_for_request(req.args)
                response = self.client.chat.completions.create(**payload)

                elapsed = time.time() - started_at
                response_text = response.choices[0].message.content or ""

                clean_debug = self._clean_for_debug_parse(doc, response_text)

                if self._debug_enabled():
                    self._dbg("\n" + "=" * 100)
                    self._dbg(
                        f"[FINAL_RAW_OUTPUT] id={doc.get('id')} "
                        f"step={doc.get('step')} "
                        f"qclass={doc.get('step_question_class')}"
                    )
                    self._dbg(self._redact_for_log(response_text))
                    self._dbg("\n[FINAL_CLEAN_DEBUG_PARSE]")
                    self._dbg(self._redact_for_log(clean_debug))
                    self._dbg("=" * 100)

                input_tokens = 0
                output_tokens = len(response_text.split())
                reasoning_tokens = 0

                if hasattr(response, "usage") and response.usage:
                    input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(response.usage, "completion_tokens", 0) or output_tokens

                    details = getattr(response.usage, "completion_tokens_details", None)
                    if details:
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

                log_usage(
                    model_name=self.model_version,
                    task_name=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    source="model",
                )

                responses[idx] = GenerationResult(
                    text=response_text,
                    token_counts=TokenCounts(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                    ),
                )

            except Exception as exc:
                elapsed = time.time() - started_at
                error_msg = str(exc)
                eval_logger.error(f"LiteLLM chat request failed: {error_msg}")

                responses[idx] = GenerationResult(
                    text=f"[LMMS_EVAL_REQUEST_FAILED] {error_msg[:200]}",
                    token_counts=TokenCounts(),
                )

            pbar.update(1)

        pbar.close()
        return responses
