#!/usr/bin/env bash

#SBATCH --job-name=vlm3r_curope
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=pro_6000:1
#SBATCH --tmp=20G
#SBATCH --output=logs/vlm3r_curope_%j.out
#SBATCH --error=logs/vlm3r_curope_%j.err

set -euo pipefail

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$LAUNCHER_DIR/.." && pwd)"
OOS_STORAGE_ROOT="${OOS_STORAGE_ROOT:-$HOME/scratch}"

export VLM3R_PYTHON="${VLM3R_PYTHON:-$REPO_DIR/.venv-cu128-home/bin/python}"
export VLM3R_REPO="${VLM3R_REPO:-$OOS_STORAGE_ROOT/VLM-3R}"
export VLM3R_OVERLAY="${VLM3R_OVERLAY:-$OOS_STORAGE_ROOT/python-overlays/vlm3r-cu128}"
export VLM3R_CUDA_HOME="${VLM3R_CUDA_HOME:-$OOS_STORAGE_ROOT/toolchains/cuda-12.8}"

export CUDA_HOME="$VLM3R_CUDA_HOME"
export PATH="$CUDA_HOME/bin:$VLM3R_OVERLAY/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$VLM3R_REPO:$VLM3R_REPO/CUT3R:$REPO_DIR:$VLM3R_OVERLAY"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

CUROPE_SOURCE="$VLM3R_REPO/CUT3R/src/croco/models/curope"
CUROPE_BUILD_TEMP="${VLM3R_CUROPE_BUILD_TEMP:-${TMPDIR:-$OOS_STORAGE_ROOT/tmp}/vlm3r-curope-build-${SLURM_JOB_ID:-manual}}"

test -x "$VLM3R_PYTHON"
test -x "$CUDA_HOME/bin/nvcc"
test -f "$CUROPE_SOURCE/setup.py"
test -d "$VLM3R_OVERLAY"

mkdir -p "$CUROPE_BUILD_TEMP"

echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Python: $VLM3R_PYTHON"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUT3R source: $CUROPE_SOURCE"
echo "Overlay output: $VLM3R_OVERLAY"
echo "Temporary build directory: $CUROPE_BUILD_TEMP"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

nvidia-smi
"$CUDA_HOME/bin/nvcc" --version
"$VLM3R_PYTHON" -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))'

cd "$CUROPE_SOURCE"

"$VLM3R_PYTHON" setup.py build_ext \
  --build-lib "$VLM3R_OVERLAY" \
  --build-temp "$CUROPE_BUILD_TEMP"

"$VLM3R_PYTHON" -c 'import torch, curope; print("Loaded curope:", curope.__file__)'

echo "VLM-3R curope build completed successfully."
