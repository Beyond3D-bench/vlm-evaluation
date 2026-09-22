#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess

from environment_profiles import get_profile, load_environment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate pinned CUDA 12.8 environment lock files.")
    parser.add_argument("--profile", action="append", dest="profiles", help="Profile to compile; repeat as needed.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--upgrade", action="store_true", help="Allow versions in an existing lock to advance.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_environment_manifest()
    selected = list(manifest["profiles"]) if args.all else args.profiles
    if not selected:
        raise SystemExit("Select at least one --profile or use --all.")
    unknown = set(selected).difference(manifest["profiles"])
    if unknown:
        raise SystemExit(f"Unknown profiles: {', '.join(sorted(unknown))}")
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required to compile environment locks.")

    for name in selected:
        profile = get_profile(name)
        profile.lock.parent.mkdir(parents=True, exist_ok=True)
        command = [
            uv,
            "pip",
            "compile",
            str(profile.requirements),
            "--constraints",
            str(profile.requirements.parent / "cuda128.constraints.txt"),
            "--python-version",
            manifest["python"],
            "--python-platform",
            "x86_64-manylinux_2_28",
            "--torch-backend",
            "cu128",
            "--output-file",
            str(profile.lock),
            "--custom-compile-command",
            f"python tools/compile_environments.py --profile {name}",
        ]
        if args.upgrade:
            command.append("--upgrade")
        print("+", " ".join(command))
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
