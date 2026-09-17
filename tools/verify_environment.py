#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys

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
    manifest = load_environment_manifest(DEFAULT_MANIFEST)
    expected = {
        "torch": manifest["torch"],
        "torchvision": manifest["torchvision"],
        **manifest["profiles"][args.profile]["verify"],
    }
    failures = []
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

    import torch

    actual["python"] = sys.version.split()[0]
    actual["torch_cuda"] = torch.version.cuda
    expected_python = tuple(int(part) for part in manifest["python"].split("."))
    if sys.version_info[:2] != expected_python:
        failures.append(f"Python is {actual['python']}, expected {manifest['python']}.x")
    if torch.version.cuda != manifest["cuda"]:
        failures.append(f"torch CUDA runtime is {torch.version.cuda}, expected {manifest['cuda']}")

    transformers_location = importlib.metadata.distribution("transformers").locate_file("").resolve()
    if not transformers_location.is_relative_to(sys.prefix):
        failures.append(f"transformers resolves outside this environment: {transformers_location}")

    print(json.dumps({"profile": args.profile, "versions": actual}, indent=2, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
