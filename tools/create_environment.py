#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from environment_profiles import DEFAULT_MANIFEST, REPO_ROOT, get_profile, load_environment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create locked CUDA 12.8 model environments without overlays.")
    parser.add_argument("--profile", action="append", dest="profiles", help="Profile to create; repeat as needed.")
    parser.add_argument("--all", action="store_true", help="Create every declared profile.")
    parser.add_argument("--venv-root", type=Path, default=None, help="Defaults to OOS_VENV_ROOT or $OOS_STORAGE_ROOT/venvs.")
    parser.add_argument("--python", default=None, help="Python interpreter/version passed to uv venv.")
    parser.add_argument("--recreate", action="store_true", help="Recreate an existing profile environment.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: Sequence[str], *, dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    args = parse_args()
    manifest = load_environment_manifest(DEFAULT_MANIFEST)
    selected = list(manifest["profiles"]) if args.all else args.profiles
    if not selected:
        raise SystemExit("Select at least one --profile or use --all.")
    unknown = set(selected).difference(manifest["profiles"])
    if unknown:
        raise SystemExit(f"Unknown profiles: {', '.join(sorted(unknown))}")

    storage_root = Path(os.environ.get("OOS_STORAGE_ROOT", Path.home() / "scratch"))
    venv_root = args.venv_root or Path(os.environ.get("OOS_VENV_ROOT", storage_root / "venvs"))
    python = args.python or manifest["python"]
    uv = os.environ.get("OOS_UV") or shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required. Install it with: python -m pip install --user uv")

    for name in selected:
        profile = get_profile(name)
        if not profile.lock.is_file():
            raise SystemExit(f"Missing lock file for {name}: {profile.lock}")
        target = venv_root / profile.directory
        if target.exists() and args.recreate:
            print(f"Remove the existing environment before recreating it: {target}")
            if not args.dry_run:
                shutil.rmtree(target)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            run([uv, "venv", "--python", python, str(target)], dry_run=args.dry_run)
        interpreter = target / "bin" / "python"
        run(
            [uv, "pip", "sync", str(profile.lock), "--python", str(interpreter), "--torch-backend", "cu128"],
            dry_run=args.dry_run,
        )
        run([str(interpreter), str(REPO_ROOT / "tools" / "verify_environment.py"), "--profile", name], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
