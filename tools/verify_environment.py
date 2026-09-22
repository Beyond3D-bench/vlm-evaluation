#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from environment_profiles import DEFAULT_MANIFEST, load_environment_manifest


def parse_args() -> argparse.Namespace:
    profiles = load_environment_manifest(DEFAULT_MANIFEST)["profiles"]
    parser = argparse.ArgumentParser(description="Verify a CUDA 12.8 environment profile.")
    parser.add_argument("--profile", required=True, choices=sorted(profiles))
    return parser.parse_args()


def version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def main() -> int:
    args = parse_args()
    if args.profile == "base":
        library_dir = str(Path(sys.prefix) / "ffmpeg-shared" / "lib")
        if library_dir not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = library_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            os.execve(sys.executable, [sys.executable, *sys.argv], env)
    manifest = load_environment_manifest(DEFAULT_MANIFEST)
    expected = {
        "torch": manifest["torch"],
        "torchvision": manifest["torchvision"],
        "imageio-ffmpeg": "0.6.0",
        **manifest["profiles"][args.profile]["verify"],
    }
    failures = []
    if args.profile == "base":
        expected["torchcodec"] = "0.5"
    actual = {}
    for package, wanted in expected.items():
        try:
            found = version(package)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{package} is not installed")
            continue
        actual[package] = found
        if found.removesuffix("+cu128") != wanted:
            failures.append(f"{package}=={found}, expected {wanted}")

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
             "-an", "-c:v", "libx264", "-f", "null", "-"],
            check=True, capture_output=True, timeout=30,
        )
        actual["ffmpeg"] = imageio_ffmpeg.get_ffmpeg_version()
    except (ImportError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        failures.append(f"Bundled FFmpeg H.264 encoding check failed: {exc}")

    import torch

    if args.profile == "base":
        try:
            from torchcodec.decoders import VideoDecoder
            from transformers import InternVLVideoProcessor

            with tempfile.TemporaryDirectory(prefix="oos-video-check-") as temp:
                clip = str(Path(temp) / "clip.mp4")
                subprocess.run(
                    [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5:d=1",
                     "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip],
                    check=True, capture_output=True, timeout=30,
                )
                decoder = VideoDecoder(clip, device="cpu")
                if decoder.get_frames_at(indices=[0, 4]).data.shape != (2, 3, 64, 64):
                    raise RuntimeError("Unexpected decoded frame shape")
                processor = InternVLVideoProcessor()
                frames, metadata = processor.fetch_videos(
                    clip, sample_indices_fn=lambda metadata, **kwargs: [0, metadata.total_num_frames - 1]
                )
                if metadata.video_backend != "torchcodec" or len(frames) != 2:
                    raise RuntimeError("InternVL did not decode using TorchCodec")
                actual["internvl_video_backend"] = metadata.video_backend
        except Exception as exc:
            failures.append(f"InternVL TorchCodec decoding check failed: {exc}")

    actual["python"] = sys.version.split()[0]
    actual["torch_cuda"] = torch.version.cuda
    expected_python = tuple(int(part) for part in manifest["python"].split("."))
    if sys.version_info[:2] != expected_python:
        failures.append(f"Python is {actual['python']}, expected {manifest['python']}.x")
    if torch.version.cuda != manifest["cuda"]:
        failures.append(f"torch CUDA runtime is {torch.version.cuda}, expected {manifest['cuda']}")

    transformers_location = importlib.metadata.distribution("transformers").locate_file("").resolve()
    if not transformers_location.is_relative_to(Path(sys.prefix).resolve()):
        failures.append(f"transformers resolves outside this environment: {transformers_location}")

    print(json.dumps({"profile": args.profile, "versions": actual}, indent=2, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
