#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

from model_manifest import SourceRepository, iter_source_repositories, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone, pin, patch, and verify external model source repositories."
    )
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.yaml"))
    parser.add_argument("--model", action="append", dest="models", help="Source model to prepare; repeat as needed. Defaults to all source models.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="External source root. Defaults to OOS_MODEL_SOURCE_DIR or $OOS_STORAGE_ROOT.",
    )
    parser.add_argument("--verify-only", action="store_true", help="Do not clone or patch; only verify the prepared trees.")
    parser.add_argument("--dry-run", action="store_true", help="Print mutating commands without running them.")
    return parser.parse_args()


def run(command: Sequence[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    if not capture:
        print(f"+ {shlex.join(command)}")
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def git_output(target: Path, *arguments: str) -> str:
    result = run(["git", "-C", str(target), *arguments], capture=True)
    return result.stdout.strip()


def git_check(target: Path, *arguments: str) -> bool:
    result = run(["git", "-C", str(target), *arguments], capture=True, check=False)
    return result.returncode == 0


def verify_repository(source: SourceRepository, target: Path) -> None:
    if not (target / ".git").exists():
        raise RuntimeError(f"Missing Git worktree for {source.model}: {target}")

    origin = git_output(target, "remote", "get-url", "origin")
    if origin.rstrip("/").removesuffix(".git") != source.url.rstrip("/").removesuffix(".git"):
        raise RuntimeError(f"Unexpected origin for {source.model}: {origin} (expected {source.url})")

    head = git_output(target, "rev-parse", "HEAD")
    if head != source.revision:
        raise RuntimeError(f"Unexpected revision for {source.model}: {head} (expected {source.revision})")

    for submodule_path, revision in source.submodules.items():
        actual = git_output(target / submodule_path, "rev-parse", "HEAD")
        if actual != revision:
            raise RuntimeError(
                f"Unexpected {source.model} submodule revision for {submodule_path}: "
                f"{actual} (expected {revision})"
            )

    for patch in source.patches:
        patch_target = target / patch.workdir
        if not git_check(patch_target, "apply", "--reverse", "--check", str(patch.file)):
            raise RuntimeError(f"Required patch is not applied for {source.model}: {patch.file}")

    print(f"OK [{source.model}] {target} @ {source.revision}")


def prepare_repository(source: SourceRepository, source_root: Path, *, dry_run: bool) -> None:
    target = source_root / source.directory
    print(f"\n[{source.model}] {target}")

    if dry_run:
        if not target.exists():
            print(f"+ {shlex.join(['git', 'clone', source.url, str(target)])}")
        print(f"+ {shlex.join(['git', '-C', str(target), 'checkout', '--detach', source.revision])}")
        print(f"+ {shlex.join(['git', '-C', str(target), 'submodule', 'update', '--init', '--recursive'])}")
        for patch in source.patches:
            print(
                f"+ {shlex.join(['git', '-C', str(target / patch.workdir), 'apply', str(patch.file)])}"
            )
        return

    cloned = not target.exists()
    if cloned:
        source_root.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", source.url, str(target)])
    elif not (target / ".git").exists():
        raise RuntimeError(f"Refusing to reuse a non-Git directory: {target}")

    origin = git_output(target, "remote", "get-url", "origin")
    if origin.rstrip("/").removesuffix(".git") != source.url.rstrip("/").removesuffix(".git"):
        raise RuntimeError(f"Refusing unexpected origin for {source.model}: {origin}")

    head = git_output(target, "rev-parse", "HEAD")
    if cloned:
        run(["git", "-C", str(target), "checkout", "--detach", source.revision])
    elif head != source.revision:
        status = git_output(target, "status", "--porcelain")
        if status:
            raise RuntimeError(
                f"Refusing to change revision of dirty source tree {target}. "
                "Move it aside or clean it explicitly first."
            )
        run(["git", "-C", str(target), "fetch", "origin", source.revision])
        run(["git", "-C", str(target), "checkout", "--detach", source.revision])

    run(["git", "-C", str(target), "submodule", "update", "--init", "--recursive"])

    for submodule_path, revision in source.submodules.items():
        actual = git_output(target / submodule_path, "rev-parse", "HEAD")
        if actual != revision:
            raise RuntimeError(
                f"Manifest/submodule mismatch for {source.model}/{submodule_path}: "
                f"{actual} (expected {revision})"
            )

    for patch in source.patches:
        patch_target = target / patch.workdir
        if git_check(patch_target, "apply", "--reverse", "--check", str(patch.file)):
            print(f"already applied: {patch.file.name}")
        elif git_check(patch_target, "apply", "--check", str(patch.file)):
            run(["git", "-C", str(patch_target), "apply", str(patch.file)])
        else:
            raise RuntimeError(
                f"Patch neither applies nor cleanly reverses for {source.model}: {patch.file}"
            )

    verify_repository(source, target)


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest)
        sources = iter_source_repositories(manifest, args.manifest, args.models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    source_root = args.source_dir or Path(os.environ.get("OOS_MODEL_SOURCE_DIR", os.environ.get("OOS_STORAGE_ROOT", str(Path.home() / "scratch"))))
    source_root = source_root.expanduser().resolve()
    print(f"External model source root: {source_root}")

    for source in sources:
        target = source_root / source.directory
        if args.verify_only:
            verify_repository(source, target)
        else:
            prepare_repository(source, source_root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
