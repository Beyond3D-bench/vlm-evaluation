#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from model_manifest import iter_huggingface_artifacts, load_manifest, require_pinned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that pinned Hugging Face snapshots resolve without network access."
    )
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.yaml"))
    parser.add_argument("--model", action="append", dest="models", help="Model preset to verify; repeat as needed. Defaults to all models.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Hub cache directory. Defaults to HF_HUB_CACHE or HF_HOME/hub.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest)
        artifacts = iter_huggingface_artifacts(manifest, args.models)
        require_pinned(artifacts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir = args.cache_dir or Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))

    # Set offline mode before importing huggingface_hub so its constants observe
    # the same environment that compute jobs use.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download

    failures: list[str] = []
    print(f"Checking offline cache: {cache_dir}")
    for artifact in artifacts:
        label = f"{artifact.repo_id}@{artifact.revision}"
        try:
            snapshot_path = snapshot_download(
                repo_id=artifact.repo_id,
                revision=artifact.revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except Exception as exc:  # report every missing artifact in one run
            failures.append(f"{label}: {exc}")
            print(f"MISSING [{artifact.model}] {label}")
        else:
            print(f"OK      [{artifact.model}] {label} -> {snapshot_path}")

    if failures:
        print("\nOffline verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
