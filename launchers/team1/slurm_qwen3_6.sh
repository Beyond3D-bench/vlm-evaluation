#!/usr/bin/env bash
# Concrete Slurm launcher for Qwen3.6 on the ETH 3dv Team 1 workspace.
# Submit with:
#   sbatch /work/courses/3dv/team1/lmms-eval/launchers/team1/slurm_qwen3_6.sh

#SBATCH --job-name=oos_qwen3_6
#SBATCH --account=3dv
#SBATCH --time=23:50:00
#SBATCH --gpus=gb10:1
#SBATCH --output=/work/courses/3dv/team1/lmms-eval/logs/oos_qwen3_6_%j.out
#SBATCH --error=/work/courses/3dv/team1/lmms-eval/logs/oos_qwen3_6_%j.err

set -euo pipefail

srun bash /work/courses/3dv/team1/lmms-eval/launchers/team1/run_qwen3_6.sh
