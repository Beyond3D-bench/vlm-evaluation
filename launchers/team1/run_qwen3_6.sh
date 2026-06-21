#!/usr/bin/env bash
# Concrete Qwen3.6 launcher for the current ETH 3dv Team 1 workspace.
# Run from anywhere:
#   bash /work/courses/3dv/team1/lmms-eval/launchers/team1/run_qwen3_6.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work/courses/3dv/team1}"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR/lmms-eval}"
cd "$CODE_DIR"

source "${OOS_VENV:-$CODE_DIR/.venv-gb10/bin/activate}"

if [ -f "$CODE_DIR/.env" ]; then
  set -a
  source "$CODE_DIR/.env"
  set +a
fi

export OOS_MODEL="${OOS_MODEL:-qwen3_6}"
export OOS_ENV_FILE="${OOS_ENV_FILE:-launchers/oos_env.sh}"

mkdir -p "$CODE_DIR/logs"

echo "Host: $(hostname)"
echo "Arch: $(uname -m)"
echo "Python: $(which python)"
python --version

bash "$CODE_DIR/launchers/run_oos_eval.sh"
