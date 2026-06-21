#!/usr/bin/env bash
# Submit with, for example:
#   OOS_MODEL=qwen3_6 OOS_VENV=.venv/bin/activate sbatch launchers/slurm_oos_eval.sh
#
# Most clusters require editing the SBATCH account/partition/GPU lines below.

#SBATCH --job-name=oos_videoqa
#SBATCH --account=3dv
#SBATCH --time=23:50:00
#SBATCH --gpus=gb10:1
#SBATCH --output=logs/oos_videoqa_%j.out
#SBATCH --error=logs/oos_videoqa_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_DIR="$SLURM_SUBMIT_DIR"
else
  LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="$(cd "$LAUNCHER_DIR/.." && pwd)"
fi
cd "$REPO_DIR"

VENV_ACTIVATE="${OOS_VENV:-.venv/bin/activate}"
if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Missing virtualenv activate script: $VENV_ACTIVATE" >&2
  echo "Current directory: $(pwd)" >&2
  echo "Set OOS_VENV to an absolute path or a path relative to the sbatch submit directory." >&2
  exit 1
fi
source "$VENV_ACTIVATE"

# Optional debug filter; override at submit time with OOS_DEBUG_STEP=<step>.
export OOS_DEBUG_STEP="${OOS_DEBUG_STEP:-5b}"

if command -v srun >/dev/null 2>&1; then
  srun bash launchers/run_oos_eval.sh
else
  bash launchers/run_oos_eval.sh
fi
