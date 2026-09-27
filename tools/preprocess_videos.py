#!/usr/bin/env python3
"""Prepare HD-EPIC videos with per-video camera masks and timestamp watermarks.

Adapted from benchmark-construction/scripts/preprocessing/preprocess_with_watermark.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


DEFAULT_VIDEO_ROOT = Path(os.getenv("OOS_DATA_ROOT", "data")) / "HD-EPIC" / "Videos"
_INSCRIBED_MASKS: dict[tuple[int, int], np.ndarray] = {}


def inscribed_circle(width: int, height: int) -> tuple[float, float, float]:
    """Match visibility_track_v2's fixed devignetting-circle geometry."""
    return (width - 1) / 2.0, (height - 1) / 2.0, min(width, height) / 2.0


def apply_inscribed_circle_mask(frame: np.ndarray) -> np.ndarray:
    """Black out pixels outside the circle used by visibility_track_v2."""
    import numpy as np

    height, width = frame.shape[:2]
    mask = _INSCRIBED_MASKS.get((height, width))
    if mask is None:
        cx, cy, radius = inscribed_circle(width, height)
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 > radius * radius
        _INSCRIBED_MASKS[(height, width)] = mask
    output = frame.copy()
    output[mask] = 0
    return output


def calibrated_circle(path: Path) -> tuple[int, int, float, float, float]:
    import math
    camera = json.loads(path.read_text())["cameras"]["camera-rgb"]
    width, height = camera["image_size"]
    cx, cy = camera["projection_params"][1:3]
    radius = camera["valid_radius"]
    if radius is None or not all(math.isfinite(float(v)) for v in (width, height, cx, cy, radius)) or min(width, height, radius) <= 0:
        raise ValueError(f"Invalid camera-rgb geometry: {path}")
    return int(width), int(height), float(cx), float(cy), float(radius)


def ffmpeg_calibrated_circle_filter(geometry) -> str:
    width, height, cx, cy, radius = geometry
    x = f"(X*{width}/W-{cx})"
    y = f"(Y*{height}/H-{cy})"
    distance = f"{x}*{x}+{y}*{y}"
    limit = f"{radius * radius}"
    return "geq=" + ":".join(f"{c}='if(lte({distance},{limit}),{c}(X,Y),0)'" for c in "rgb")


def ffmpeg_inscribed_circle_filter() -> str:
    """Return an FFmpeg RGB filter equivalent to apply_inscribed_circle_mask."""
    distance_squared = (
        "(X-(W-1)/2)*(X-(W-1)/2)"
        "+(Y-(H-1)/2)*(Y-(H-1)/2)"
    )
    radius_squared = "min(W,H)*min(W,H)/4"

    def channel_expression(channel: str) -> str:
        return (
            f"if(lte({distance_squared},{radius_squared}),"
            f"{channel}(X,Y),0)"
        )

    return (
        "geq="
        f"r='{channel_expression('r')}':"
        f"g='{channel_expression('g')}':"
        f"b='{channel_expression('b')}'"
    )


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def make_text(seconds: float, watermark_style: str) -> str:
    ts = format_time(seconds)
    if watermark_style == "labeled":
        return f"Time\n{ts}"
    if watermark_style == "token":
        return f"<TIME {ts} video 1>"
    return ts


def draw_watermark(frame: np.ndarray, seconds: float, target_size: tuple[int, int], watermark_style: str) -> np.ndarray:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    text = make_text(seconds, watermark_style)

    font_size = max(4, int(target_size[0] * 0.035))
    margin = 4
    box_pad_x = 2
    box_pad_y = 1
    width, height = target_size
    cx, cy, radius = inscribed_circle(width, height)

    while True:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_left = width - text_w - margin
        text_top = height - text_h - margin
        box_left = text_left - box_pad_x
        box_top = text_top - box_pad_y
        box_right = text_left + text_w + box_pad_x
        box_bottom = text_top + text_h + box_pad_y

        # The nearest point of the watermark box to the circle must remain
        # outside it. Shrink long styles (especially ``token``) until the
        # complete overlay fits in the bottom-right black boundary.
        nearest_x = min(max(cx, box_left), box_right)
        nearest_y = min(max(cy, box_top), box_bottom)
        outside_circle = (
            (nearest_x - cx) ** 2 + (nearest_y - cy) ** 2 > radius * radius
        )
        if outside_circle or font_size <= 4:
            break
        font_size -= 1

    # Account for fonts whose ink bbox does not start at (0, 0).
    draw_x = text_left - bbox[0]
    draw_y = text_top - bbox[1]

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [box_left, box_top, box_right, box_bottom],
        fill=(0, 0, 0, 100),
    )
    overlay_draw.text(
        (draw_x, draw_y), text, fill=(255, 255, 255, 200), font=font
    )

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def load_video_specs(
    jsonl_path: Path | None,
    video_id_file: Path | None,
    participants: list[str] | None,
    video_root: Path,
    trim_to_jsonl_clip_end: bool,
) -> dict[str, float | None]:
    video_specs: dict[str, float | None] = {}

    if jsonl_path is not None:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                video_id = record.get("video_id")
                if not video_id:
                    continue
                end_time = None
                if trim_to_jsonl_clip_end:
                    raw_end_time = record.get("clip_end_time_sec", record.get("query_time_sec"))
                    try:
                        end_time = float(raw_end_time)
                    except (TypeError, ValueError):
                        end_time = None
                current_end_time = video_specs.get(video_id)
                if current_end_time is None:
                    video_specs[video_id] = end_time
                elif end_time is not None:
                    video_specs[video_id] = max(current_end_time, end_time)

    if video_id_file is not None:
        with video_id_file.open("r", encoding="utf-8") as f:
            for line in f:
                video_id = line.strip()
                if video_id and video_id not in video_specs:
                    video_specs[video_id] = None

    if participants:
        for participant_id in participants:
            for video_id in discover_participant_video_ids(video_root, participant_id):
                if video_id not in video_specs:
                    video_specs[video_id] = None

    return video_specs


def participant_source_dirs(video_root: Path, participant_id: str) -> list[Path]:
    return [
        video_root / participant_id,
        video_root / f"{participant_id}_raw",
        video_root / f"{participant_id}selected",
    ]


def discover_participant_video_ids(video_root: Path, participant_id: str) -> list[str]:
    video_ids: list[str] = []
    seen: set[str] = set()
    checked_dirs = participant_source_dirs(video_root, participant_id)
    for source_dir in checked_dirs:
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.glob(f"{participant_id}-*.mp4")):
            video_id = path.stem
            if video_id not in seen:
                seen.add(video_id)
                video_ids.append(video_id)

    if not video_ids:
        checked = ", ".join(str(path) for path in checked_dirs)
        raise FileNotFoundError(
            f"No source videos found for participant {participant_id}. Checked: {checked}"
        )

    return video_ids


def resolve_source_video(video_root: Path, video_id: str) -> Path:
    participant_id = video_id.split("-", 1)[0]
    candidates = [
        source_dir / f"{video_id}.mp4"
        for source_dir in participant_source_dirs(video_root, participant_id)
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find source video for {video_id}")


def output_path_for_video(
    video_root: Path,
    video_id: str,
    output_suffix: str,
    output_root: Path | None = None,
) -> Path:
    if output_root is not None:
        return output_root / f"{video_id}.mp4"
    participant_id = video_id.split("-", 1)[0]
    return video_root / f"{participant_id}{output_suffix}" / f"{video_id}.mp4"


def ffmpeg_has_filter(filter_name: str) -> bool:
    try:
        result = subprocess.run(
            [os.getenv("FFMPEG_PATH", "ffmpeg"), "-hide_banner", "-filters"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return any(filter_name in line.split() for line in result.stdout.splitlines())


def preprocess_one(task: dict[str, Any]) -> tuple[str, str, str]:
    video_id = task["video_id"]
    input_path = Path(task["input_path"])
    output_path = Path(task["output_path"])
    target_fps = float(task["target_fps"])
    target_size = tuple(task["target_size"])
    add_watermark = bool(task["add_watermark"])
    apply_circle_mask = bool(task.get("apply_circle_mask", True))
    watermark_style = str(task["watermark_style"])
    overwrite = bool(task["overwrite"])
    backend = str(task["backend"])
    end_time_sec = task.get("end_time_sec")
    end_time_sec = float(end_time_sec) if end_time_sec is not None else None

    geometry = task.get("calibration_geometry")

    if output_path.exists() and not overwrite:
        return video_id, "skipped_exists", str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    if backend == "ffmpeg":
        filters = [f"fps={target_fps:g}"]
        if apply_circle_mask:
            # Sample first, then match visibility_track_v2 by masking each
            # selected decoded frame before detector-side resizing.
            filters.extend(["format=rgb24", ffmpeg_calibrated_circle_filter(geometry) if geometry else ffmpeg_inscribed_circle_filter()])
        filters.append(f"scale={target_size[0]}:{target_size[1]}")
        if add_watermark:
            timestamp = "%{pts\\:hms}"
            if watermark_style == "labeled":
                text = f"Time\\\\n{timestamp}"
            elif watermark_style == "token":
                text = f"<TIME {timestamp} video 1>"
            else:
                text = timestamp
            filters.append(
                "drawtext="
                f"text='{text}':"
                "x=w-tw-4:y=h-th-4:"
                "fontsize=8:"
                "fontcolor=white@0.8:"
                "box=1:boxcolor=black@0.4:boxborderw=2"
            )
        cmd = [
            os.getenv("FFMPEG_PATH", "ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-i",
            str(input_path),
        ]
        if end_time_sec is not None:
            cmd.extend(["-t", f"{end_time_sec + (1.0 / target_fps):.3f}"])
        cmd.extend(
            [
                "-vf",
                ",".join(filters),
                "-r",
                f"{target_fps:g}",
                "-an",
                "-movflags",
                "+faststart",
                str(tmp_output_path),
            ]
        )
        subprocess.run(cmd, check=True)
        tmp_output_path.replace(output_path)
        return video_id, "processed", str(output_path)

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_output_path), fourcc, target_fps, target_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not write video: {tmp_output_path}")

    # Decode inter-frame dependencies, but retrieve/convert only selected frames.
    # read() also converts discarded frames to BGR, which is costly at source FPS.
    frame_idx = 0
    next_sample_time = 0.0
    calibration_mask = None
    sampled_frames = 0
    print(f"[processing] {video_id}", flush=True)
    while True:
        t = frame_idx / src_fps
        if end_time_sec is not None and t > end_time_sec + 1e-6:
            break
        if not cap.grab():
            break
        if t + 1e-6 < next_sample_time:
            frame_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            cap.release()
            writer.release()
            raise RuntimeError(f"Could not retrieve frame {frame_idx} from {input_path}")

        if apply_circle_mask:
            if geometry:
                if calibration_mask is None:
                    width, height, cx, cy, radius = geometry
                    h, w = frame.shape[:2]
                    yy, xx = np.ogrid[:h, :w]
                    calibration_mask = ((xx * width / w - cx) ** 2 + (yy * height / h - cy) ** 2) > radius ** 2
                frame[calibration_mask] = 0
            else:
                frame = apply_inscribed_circle_mask(frame)
        frame = cv2.resize(frame, target_size)
        if add_watermark:
            frame = draw_watermark(frame, t, target_size, watermark_style)
        writer.write(frame)
        sampled_frames += 1
        if sampled_frames % 300 == 0:
            print(f"[processing] {video_id}: {sampled_frames} output frames", flush=True)
        next_sample_time += 1.0 / target_fps
        frame_idx += 1

    cap.release()
    writer.release()
    tmp_output_path.replace(output_path)
    return video_id, "processed", str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess selected videos with per-video camera calibration "
            "and timestamp watermarks."
        )
    )
    parser.add_argument("--fixed-circle", action="store_true", help="Explicitly use the legacy fixed inscribed circle instead of calibration.")
    parser.add_argument("--intermediate-root", type=Path, help="Use per-video camera-rgb calibration from ROOT/Pxx/video_id/device_calibration.json instead of the fixed circle.")
    parser.add_argument("--jsonl", type=Path, help="JSONL file containing video_id fields")
    parser.add_argument("--video-id-file", type=Path, help="Plain text file with one video_id per line")
    parser.add_argument(
        "--participant",
        action="append",
        default=None,
        help="Participant id, e.g. P01. Discovers videos from P01/, P01_raw/, or P01selected/. Repeatable.",
    )
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Write every processed video directly into this directory. "
            "By default, outputs use participant-specific sibling directories."
        ),
    )
    parser.add_argument("--output-suffix", default="_preprocessed_with_watermark")
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--target-size", type=int, nargs=2, default=(448, 448), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--watermark-style", choices=["plain", "labeled", "token"], default="plain")
    parser.add_argument("--no-watermark", action="store_true")
    parser.add_argument(
        "--no-circle-mask",
        action="store_true",
        help=(
            "Do not black out pixels outside visibility_track_v2's fixed "
            "inscribed circle. The mask is enabled by default."
        ),
    )
    parser.add_argument("--backend", choices=["ffmpeg", "opencv"], default="ffmpeg")
    parser.add_argument(
        "--trim-to-jsonl-clip-end",
        action="store_true",
        help="When --jsonl is used, trim each output to that video's max clip_end_time_sec. By default, full videos are processed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.intermediate_root and not args.fixed_circle and not args.no_circle_mask:
        raise SystemExit("Pass --intermediate-root for per-video calibration (or --fixed-circle for legacy output)")
    if args.intermediate_root and args.fixed_circle:
        raise SystemExit("Choose calibration or --fixed-circle")
    if args.intermediate_root and args.no_circle_mask:
        raise SystemExit("--intermediate-root cannot be combined with --no-circle-mask")
    if args.target_fps <= 0 or min(args.target_size) <= 0 or args.workers < 1:
        raise SystemExit("FPS, dimensions, and workers must be positive")
    backend = args.backend
    if backend == "ffmpeg" and not args.no_watermark and not ffmpeg_has_filter("drawtext"):
        backend = "opencv"
        print("ffmpeg drawtext filter is unavailable; falling back to OpenCV backend for watermarks.")

    video_specs = load_video_specs(
        args.jsonl,
        args.video_id_file,
        args.participant,
        args.video_root,
        trim_to_jsonl_clip_end=args.trim_to_jsonl_clip_end,
    )
    if not video_specs:
        raise SystemExit("No video ids found. Pass --jsonl, --video-id-file, or --participant.")

    tasks: list[dict[str, Any]] = []
    missing: list[str] = []
    for video_id, end_time_sec in video_specs.items():
        try:
            input_path = resolve_source_video(args.video_root, video_id)
        except FileNotFoundError:
            missing.append(video_id)
            continue

        tasks.append(
            {
                "calibration_geometry": calibrated_circle(args.intermediate_root / video_id.split("-", 1)[0] / video_id / "device_calibration.json") if args.intermediate_root else None,
                "video_id": video_id,
                "input_path": str(input_path),
                "output_path": str(
                    output_path_for_video(
                        args.video_root,
                        video_id,
                        args.output_suffix,
                        args.output_root,
                    )
                ),
                "target_fps": args.target_fps,
                "target_size": tuple(args.target_size),
                "add_watermark": not args.no_watermark,
                "apply_circle_mask": not args.no_circle_mask,
                "watermark_style": args.watermark_style,
                "backend": backend,
                "overwrite": args.overwrite,
                "end_time_sec": end_time_sec,
            }
        )

    if missing:
        raise SystemExit("Missing source videos: " + ", ".join(missing))

    trimmed_count = sum(1 for task in tasks if task["end_time_sec"] is not None)
    print(
        f"Selected {len(video_specs)} videos; processing {len(tasks)} found videos "
        f"with {args.workers} workers ({trimmed_count} trimmed to JSONL clip end; "
        f"circle mask {'disabled' if args.no_circle_mask else 'enabled'})."
    )
    status_counts: dict[str, int] = {}
    if args.workers <= 1:
        for task in tasks:
            video_id, status, output_path = preprocess_one(task)
            status_counts[status] = status_counts.get(status, 0) + 1
            print(f"[{status}] {video_id} -> {output_path}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(preprocess_one, task) for task in tasks]
            for future in as_completed(futures):
                video_id, status, output_path = future.result()
                status_counts[status] = status_counts.get(status, 0) + 1
                print(f"[{status}] {video_id} -> {output_path}")

    print("Summary:", status_counts)


if __name__ == "__main__":
    main()
