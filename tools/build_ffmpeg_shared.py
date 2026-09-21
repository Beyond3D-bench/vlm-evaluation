#!/usr/bin/env python3
"""Build pinned, CPU-only FFmpeg shared libraries for TorchCodec without root/Conda."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "7.1.1"
SHA256 = "733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1"
URL = f"https://ffmpeg.org/releases/ffmpeg-{VERSION}.tar.xz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded source archive.")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    prefix = args.prefix.resolve()
    marker = prefix / ".oos_ffmpeg_complete"
    if marker.is_file() and marker.read_text().strip() == SHA256:
        if all((prefix / "lib" / name).is_file() for name in
               ("libavutil.so.59", "libavcodec.so.61", "libavformat.so.61",
                "libavdevice.so.61", "libavfilter.so.10", "libswscale.so.8", "libswresample.so.5")):
            print(f"Using FFmpeg shared libraries: {prefix}", flush=True)
            return
    for command in ("cc", "make"):
        if shutil.which(command) is None:
            raise SystemExit(f"Building FFmpeg requires {command} on PATH.")
    prefix.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    log_path = prefix / "build.log"
    with tempfile.TemporaryDirectory(prefix="oos-ffmpeg-") as temp:
        root = Path(temp)
        archive = args.archive or root / "ffmpeg.tar.xz"
        if args.archive is None:
            print(f"Downloading {URL}", flush=True)
            with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise SystemExit("FFmpeg archive SHA256 mismatch")
        with tarfile.open(archive) as source:
            # Only extract regular files/directories from this verified source archive.
            for member in source.getmembers():
                target = (root / member.name).resolve()
                if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                    raise SystemExit(f"Unexpected archive entry: {member.name}")
            if hasattr(tarfile, "data_filter"):
                source.extractall(root, filter="data")
            else:
                source.extractall(root)
        build_dir = root / f"ffmpeg-{VERSION}"
        configure = [
            "./configure", f"--prefix={prefix}", "--enable-shared", "--disable-static",
            "--disable-programs", "--disable-doc", "--disable-debug", "--disable-autodetect",
            "--disable-network", "--disable-x86asm", "--disable-everything",
            "--enable-protocol=file,pipe",
            "--enable-demuxer=mov,matroska,avi,h264,hevc,image2,wav",
            "--enable-parser=h264,hevc,mpeg4video,mjpeg,vp8,vp9,aac",
            "--enable-decoder=h264,hevc,mpeg4,mjpeg,vp8,vp9,rawvideo,aac,pcm_s16le",
            "--enable-filter=buffer,buffersink,abuffer,abuffersink,scale,format,aformat,aresample",
        ]
        print(f"Building FFmpeg {VERSION} shared libraries; log: {log_path}", flush=True)
        with log_path.open("w") as log:
            for command in (configure, ["make", f"-j{args.jobs}"], ["make", "install"]):
                try:
                    subprocess.run(command, cwd=build_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError as exc:
                    raise SystemExit(f"FFmpeg build failed; see {log_path}") from exc
    marker.write_text(SHA256 + "\n")
    print(f"Installed FFmpeg shared libraries: {prefix}", flush=True)


if __name__ == "__main__":
    main()
