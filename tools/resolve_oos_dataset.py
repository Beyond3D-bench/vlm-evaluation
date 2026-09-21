#!/usr/bin/env python3
"""Resolve the local files for the published BEYOND3D dataset."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex


DEFAULT_REPOSITORY = "Ffffangzhu/BEYOND3D"
DEFAULT_JSONL = "vqa_baseline.jsonl"


def resolve_dataset(
    *,
    repository: str,
    revision: str | None,
    cache_dir: str | None,
    offline: bool,
    jsonl_name: str = DEFAULT_JSONL,
    downloader=None,
) -> tuple[Path, Path]:
    """Download or find a dataset snapshot and return its JSONL and video paths."""
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download

    snapshot = Path(
        downloader(
            repo_id=repository,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
    )
    jsonl = snapshot / jsonl_name
    videos = snapshot / "videos"
    if not jsonl.is_file():
        raise FileNotFoundError(f"Dataset snapshot is missing {jsonl_name}: {snapshot}")
    if not videos.is_dir():
        raise FileNotFoundError(f"Dataset snapshot is missing videos/: {snapshot}")
    return jsonl, videos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.getenv("OOS_DATASET_REPO", DEFAULT_REPOSITORY))
    parser.add_argument("--revision", default=os.getenv("OOS_DATASET_REVISION"))
    parser.add_argument("--cache-dir", default=os.getenv("HF_HUB_CACHE"))
    parser.add_argument("--file", default=os.getenv("OOS_DATASET_FILE", DEFAULT_JSONL),
                        help="JSONL filename in the dataset snapshot.")
    parser.add_argument("--offline", action="store_true", default=os.getenv("OOS_OFFLINE", "0") == "1")
    parser.add_argument("--shell", action="store_true", help="Print shell exports for the launcher.")
    args = parser.parse_args()
    if Path(args.file).name != args.file:
        parser.error("--file must be a filename at the dataset repository root")

    try:
        jsonl, videos = resolve_dataset(
            repository=args.repo,
            revision=args.revision,
            cache_dir=args.cache_dir,
            offline=args.offline,
            jsonl_name=args.file,
        )
    except Exception as exc:
        parser.exit(
            1,
            f"Cannot resolve dataset {args.repo}: {exc}\n"
            "Set OOS_DATASET_JSONL to a local file, or run bash setup.sh --model <preset> on a connected machine first.\n",
        )

    if args.shell:
        print(f"export OOS_DATASET_JSONL={shlex.quote(str(jsonl))}")
        if not os.getenv("OOS_VIDEO_BASE_DIR"):
            print(f"export OOS_VIDEO_BASE_DIR={shlex.quote(str(videos))}")
    else:
        print(jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
