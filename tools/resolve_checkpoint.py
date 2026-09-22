#!/usr/bin/env python3
"""Resolve a local checkpoint or download a manifest-pinned Hub snapshot."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def resolve_checkpoint(value, *, offline=False, revision=None, cache_dir=None, downloader=None):
    path = Path(value).expanduser()
    if path.is_dir():
        if not (path / 'config.json').is_file():
            raise ValueError(f'{path} has no config.json. Use a checkpoint directory or snapshots/<revision>, not a Hub cache root.')
        return str(path.resolve())
    if path.is_absolute() or value.startswith(('.', '~')) or value.count('/') != 1:
        raise ValueError(f'Checkpoint directory does not exist: {value}')
    if revision is None:
        import yaml
        manifest = yaml.safe_load((Path(__file__).resolve().parents[1] / 'models/manifest.yaml').read_text())
        for model in manifest['huggingface_models'].values():
            for artifact in model['artifacts']:
                if artifact['repo_id'] == value and artifact.get('revision'):
                    revision = artifact['revision']
                    break
        if revision is None:
            print(f'Warning: {value} has no pinned revision; resolving main. Record the resolved revision for reproducibility.', file=sys.stderr)
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    return str(downloader(repo_id=value, revision=revision or 'main', local_files_only=offline, cache_dir=cache_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--revision')
    args = parser.parse_args()
    try:
        print(resolve_checkpoint(args.checkpoint, offline=os.getenv('OOS_OFFLINE', '0') == '1', revision=args.revision, cache_dir=os.getenv('HF_HUB_CACHE')))
    except Exception as exc:
        parser.exit(1, f'Cannot resolve checkpoint: {exc}\nFor offline runs, download on a connected node or set OOS_PRETRAINED to a local checkpoint directory.\n')


if __name__ == '__main__':
    main()
