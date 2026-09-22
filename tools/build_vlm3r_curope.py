#!/usr/bin/env python3
"""Build VLM-3R's CUDA extension using the active model environment and GPU."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig


def find_cuda(environ, which=shutil.which):
    explicit = environ.get('VLM3R_CUDA_HOME') or environ.get('CUDA_HOME')
    if explicit:
        root = Path(explicit).expanduser()
        if not (root / 'bin/nvcc').is_file():
            raise RuntimeError(f'Configured CUDA toolkit has no bin/nvcc: {root}')
        return root
    nvcc = which('nvcc')
    candidates = ([Path(nvcc).resolve().parent.parent] if nvcc else [])
    candidates += [Path('/usr/local/cuda-12.8'), Path('/usr/local/cuda')]
    for root in candidates:
        if (root / 'bin/nvcc').is_file():
            return root
    raise RuntimeError('CUDA toolkit not found. Load your CUDA 12.8 module or set CUDA_HOME to its installation directory.')


def main():
    import torch

    try:
        import curope
        print(f'Using curope: {curope.__file__}')
        return 0
    except ImportError:
        pass
    root = find_cuda(os.environ)
    version = subprocess.check_output([str(root / 'bin/nvcc'), '--version'], text=True)
    if not re.search(r'release\s+12\.8\b', version):
        raise RuntimeError(f'CUDA 12.8 is required for this environment; found: {version.strip()}')
    arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
    if not arch:
        if not torch.cuda.is_available():
            raise RuntimeError('Build curope on a GPU allocation; the GPU architecture is detected automatically there.')
        arch = ';'.join(sorted({f'{major}.{minor}' for major, minor in
                               (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))}))
    repo = Path(os.environ['VLM3R_REPO'])
    source = repo / 'CUT3R/src/croco/models/curope'
    if not (source / 'setup.py').is_file():
        raise RuntimeError('VLM-3R sources are missing. Run bash setup.sh --model vlm3r first.')
    env = dict(os.environ, CUDA_HOME=str(root), TORCH_CUDA_ARCH_LIST=arch)
    env['PATH'] = str(root / 'bin') + os.pathsep + env.get('PATH', '')
    env.setdefault('MAX_JOBS', env.get('SLURM_CPUS_PER_TASK', '4'))
    build_temp = Path(env.get('TMPDIR', '/tmp')) / f'vlm3r-curope-{os.getpid()}'
    print(f'Building curope with {root} for GPU architecture {arch}', flush=True)
    subprocess.run([sys.executable, 'setup.py', 'build_ext',
                    '--build-lib', sysconfig.get_paths()['purelib'],
                    '--build-temp', str(build_temp)], cwd=source, env=env, check=True)
    subprocess.run([sys.executable, '-c', 'import torch, curope; print(curope.__file__)'], check=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from None
