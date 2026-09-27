#!/usr/bin/env python3
"""Fetch benchmark annotations and prepare local videos as part of setup."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .resolve_oos_dataset import DEFAULT_REPOSITORY, resolve_dataset
else:
    from resolve_oos_dataset import DEFAULT_REPOSITORY, resolve_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = Path(os.getenv('OOS_DATA_ROOT', str(Path.home() / 'scratch/data')))
    parser.add_argument('--intermediate-archive', action='append', help='Intermediate-data ZIP URL or local path; repeat for participant archives.')
    parser.add_argument('--intermediate-root', type=Path,
                        default=Path(os.getenv('OOS_INTERMEDIATE_ROOT', str(data_root / 'HD-EPIC/Intermediate_data'))))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    jsonl, _ = resolve_dataset(repository=os.getenv('OOS_DATASET_REPO', DEFAULT_REPOSITORY),
                              revision='main',
                              local_dir=os.getenv('OOS_DATASET_DIR', str(data_root / 'BEYOND3D')),
                              cache_dir=os.getenv('HF_HUB_CACHE'), offline=False, annotations_only=True)
    # Both published variants share one prepared video directory.
    jsonls = [jsonl.parent / name for name in ('vqa_baseline.jsonl', 'vqa_temporal_cues.jsonl')]
    missing = [str(path) for path in jsonls if not path.is_file()]
    if missing:
        raise SystemExit('Missing benchmark annotations: ' + ', '.join(missing))
    output = Path(os.getenv('OOS_VIDEO_BASE_DIR', str(data_root / 'BEYOND3D/videos')))
    command = [sys.executable, str(Path(__file__).with_name('prepare_video_data.py')),
               '--data-root', str(data_root), '--output-root', str(output),
               '--intermediate-root', str(args.intermediate_root), '--download',
               '--backend', 'opencv', '--workers', os.getenv('OOS_VIDEO_PREP_WORKERS', '8'),
               '--download-workers', os.getenv('OOS_VIDEO_DOWNLOAD_WORKERS', '8')]
    for path in jsonls:
        command.extend(['--jsonl', str(path)])
    for archive in args.intermediate_archive or []:
        command.extend(['--intermediate-archive', archive])
    if args.overwrite:
        command.append('--overwrite')
    subprocess.run(command, check=True)
    print(f'Benchmark ready: annotations in {jsonl.parent}; videos in {output}', flush=True)


if __name__ == '__main__':
    main()
