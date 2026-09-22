#!/usr/bin/env python3
"""Download the official CUT3R checkpoint to the launcher's default location."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import subprocess

# Official source: https://github.com/CUT3R/CUT3R#download-checkpoints
CHECKPOINT_URL = 'https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view'


def prepare_checkpoint(target: Path, download):
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep failed downloads separate so a rerun never treats them as checkpoints.
    partial = target.with_name(target.name + '.part')
    try:
        download(partial)
        if not partial.is_file() or partial.stat().st_size < 1024 * 1024:
            raise RuntimeError('CUT3R download is missing or too small to be a checkpoint.')
        with partial.open('rb') as handle:
            prefix = handle.read(256).lstrip().lower()
        if prefix.startswith((b'<!doctype html', b'<html')):
            raise RuntimeError('CUT3R download returned an HTML page instead of weights.')
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def main():
    repo = Path(os.environ['VLM3R_REPO'])
    target = Path(os.environ.get('VLM3R_CUT3R_WEIGHTS', repo / 'CUT3R/src/cut3r_512_dpt_4_64.pth'))

    def download(partial):
        subprocess.run([sys.executable, '-m', 'gdown',
                        '--fuzzy', CHECKPOINT_URL, '--output', str(partial)], check=True)

    print(f'CUT3R checkpoint: {prepare_checkpoint(target, download)}')


if __name__ == '__main__':
    main()
