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
cd "$REPO_DIR"
source "$LAUNCHER_DIR/oos_env.sh"
VLM3R_PYTHON="${VLM3R_PYTHON:-$OOS_VENV_ROOT/vlm3r-cu128/bin/python}"
exec "$VLM3R_PYTHON" "$REPO_DIR/tools/build_vlm3r_curope.py"
