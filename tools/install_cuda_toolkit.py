#!/usr/bin/env python3
"""Install the minimal pinned CUDA 12.8 toolkit needed to build CuRoPE."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request


BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist"
ARTIFACTS = {
    "cuda_cccl-linux-x86_64-12.8.55-archive.tar.xz": (
        "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.55-archive.tar.xz",
        "dce4f2e7720d4432ab0861ede2243f9cbd46bc675008932bc9dcdb871fc7d60b",
    ),
    "cuda_cudart-linux-x86_64-12.8.57-archive.tar.xz": (
        "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.57-archive.tar.xz",
        "5bd3ac35ea8e8ab880e595d5054ee373abf6d9e53dcb8cef0a5c75358dbc0ae2",
    ),
    "cuda_nvcc-linux-x86_64-12.8.61-archive.tar.xz": (
        "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.61-archive.tar.xz",
        "145f8779bd56bdfa214447e5cb1b3a206ec1b7398da460e257f2898fd8604c54",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
        actual = sha256(partial)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}: expected {expected_sha256}, got {actual}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def extract_component(archive: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="cuda-component-") as tmp_name:
        tmp = Path(tmp_name)
        with tarfile.open(archive, "r:xz") as bundle:
            for member in bundle.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"Unsafe path in {archive.name}: {member.name}")
            extract_kwargs = {"filter": "data"} if "filter" in inspect.signature(bundle.extractall).parameters else {}
            bundle.extractall(tmp, **extract_kwargs)
        roots = [entry for entry in tmp.iterdir() if entry.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Expected one root directory in {archive.name}, found {len(roots)}")
        for entry in roots[0].iterdir():
            target = destination / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True, symlinks=True)
            else:
                shutil.copy2(entry, target, follow_symlinks=False)


def parse_args() -> argparse.Namespace:
    storage = Path(os.environ.get("OOS_STORAGE_ROOT", Path.home() / "scratch"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=storage / "toolchains/cuda-12.8")
    parser.add_argument("--download-dir", type=Path, default=storage / "downloads/cuda-12.8")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nvcc = args.prefix / "bin/nvcc"
    if nvcc.is_file():
        version = subprocess.check_output([str(nvcc), "--version"], text=True)
        if "release 12.8" in version:
            print(f"CUDA 12.8 toolkit: {args.prefix}")
            return 0

    args.prefix.mkdir(parents=True, exist_ok=True)
    for filename, (relative_url, expected_sha256) in ARTIFACTS.items():
        archive = args.download_dir / filename
        print(f"Preparing {filename}", flush=True)
        download(f"{BASE_URL}/{relative_url}", archive, expected_sha256)
        extract_component(archive, args.prefix)

    if not nvcc.is_file():
        raise RuntimeError(f"CUDA installation did not produce {nvcc}")
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "release 12.8" not in version:
        raise RuntimeError(f"Expected CUDA 12.8, found: {version.strip()}")
    print(f"CUDA 12.8 toolkit: {args.prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
