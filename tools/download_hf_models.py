#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from model_manifest import iter_huggingface_artifacts, load_manifest, require_pinned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate a standard Hugging Face cache from the pinned model manifest."
    )
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.yaml"))
    parser.add_argument("--model", action="append", dest="models", help="Model preset to download; repeat as needed. Defaults to all models.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Hub cache directory. Defaults to HF_HUB_CACHE or HF_HOME/hub.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-pinned", action="store_true", help="Reject artifacts without a manifest revision.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest)
        artifacts = iter_huggingface_artifacts(manifest, args.models)
        if args.require_pinned:
            require_pinned(artifacts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    cache_dir = args.cache_dir
    if cache_dir is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_dir = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))

    print(f"Hugging Face cache: {cache_dir}")
    if not args.dry_run:
        from huggingface_hub import snapshot_download

    for artifact in artifacts:
        print(f"[{artifact.model}] {artifact.repo_id}@{artifact.revision or 'main'}")
        if args.dry_run:
            continue
        snapshot_path = snapshot_download(
            repo_id=artifact.repo_id,
            revision=artifact.revision or "main",
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"  cached at {snapshot_path} (commit {Path(snapshot_path).name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
