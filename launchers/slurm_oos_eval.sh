#!/usr/bin/env bash
# Submit with, for example:
#   OOS_MODEL=qwen3_6 sbatch launchers/slurm_oos_eval.sh
#   OOS_MODEL=cambrian_p OOS_LIMIT=2 sbatch launchers/slurm_oos_eval.sh
#
# Most clusters require editing the SBATCH account/partition/GPU lines below.
#a100_80gb,pro_6000

#SBATCH --job-name=oos_videoqa
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=25G
#SBATCH --gpus=pro_6000:1 
#SBATCH --tmp=100G
#SBATCH --output=logs/oos_videoqa_%j.out
#SBATCH --error=logs/oos_videoqa_%j.err

set -euo pipefail
export OOS_OFFLINE="${OOS_OFFLINE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_DIR="$SLURM_SUBMIT_DIR"
else
  LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="$(cd "$LAUNCHER_DIR/.." && pwd)"
fi
cd "$REPO_DIR"

export OOS_ENV_FILE="${OOS_ENV_FILE:-launchers/oos_env.sh}"

echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Slurm node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Slurm submit dir: ${SLURM_SUBMIT_DIR:-unset}"
echo "Repo dir: $REPO_DIR"
echo "Working dir: $(pwd)"
echo "Requested CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found on PATH" >&2
fi

if command -v srun >/dev/null 2>&1; then
  srun --unbuffered bash launchers/run_oos_eval.sh
else
  bash launchers/run_oos_eval.sh
fi
