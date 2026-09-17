#!/usr/bin/env python3
"""Prepare model environments, sources, and checkpoints from one command."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def selected_profiles(models, manifest):
    unknown = set(models) - set(manifest['preset_profiles'])
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(sorted(unknown))}")
    return list(dict.fromkeys(manifest['preset_profiles'][model] for model in models))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--model', action='append', help='Model preset; repeat for several models.')
    selection.add_argument('--all', action='store_true', help='Prepare every model (large downloads).')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without installing or downloading.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'environments/profiles.json').read_text())
    models = list(manifest['preset_profiles']) if args.all else args.model
    try:
        profiles = selected_profiles(models, manifest)
    except ValueError as exc:
        parser.error(str(exc))

    def run(command):
        command = [str(arg) for arg in command]
        print('+', shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    storage = Path(os.environ['OOS_STORAGE_ROOT'])
    os.environ.setdefault('UV_CACHE_DIR', str(storage / 'uv-cache'))
    os.environ.setdefault('UV_PYTHON_INSTALL_DIR', str(storage / 'python'))
    os.environ.setdefault('PIP_CACHE_DIR', str(storage / 'pip-cache'))
    # Keep bootstrap packages out of the user's home quota and model environments.
    uv = shutil.which('uv')
    if not uv:
        bootstrap = storage / 'setup-env'
        python = bootstrap / 'bin/python'
        if not python.is_file():
            run([sys.executable, '-m', 'venv', bootstrap])
        uv = str(bootstrap / 'bin/uv')
        run([python, '-m', 'pip', 'install', 'uv==0.11.26'])
    os.environ['OOS_UV'] = uv
    venv_root = Path(os.environ['OOS_VENV_ROOT'])
    # uv is also used to create Python environments on machines without Python 3.10.
    run([sys.executable, ROOT / 'tools/create_environment.py',
         *[arg for profile in profiles for arg in ('--profile', profile)]])
    python = venv_root / manifest['profiles'][profiles[0]]['directory'] / 'bin/python'
    sources = [model for model in models if model in ('cambrian_p', 'spatial_mllm', 'vlm3r')]
    if sources:
        run([python, ROOT / 'tools/bootstrap_model_sources.py',
             *[arg for model in sources for arg in ('--model', model)]])
    run([python, ROOT / 'tools/download_hf_models.py',
         *[arg for model in models for arg in ('--model', model)]])
    if 'vlm3r' in models:
        vlm_python = venv_root / manifest['profiles']['vlm3r']['directory'] / 'bin/python'
        run([vlm_python, ROOT / 'tools/prepare_cut3r.py'])
    if not args.dry_run:
        print('Setup complete. VLM-3R builds its CUDA extension on its first GPU run.'
              if 'vlm3r' in models else 'Setup complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
