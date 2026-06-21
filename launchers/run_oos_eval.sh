#!/usr/bin/env bash
set -euo pipefail

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$LAUNCHER_DIR/.." && pwd)"
cd "$REPO_DIR"

source "$LAUNCHER_DIR/common.sh"
load_oos_env
require_oos_env
prepare_oos_dirs
load_oos_model_preset

TASK_NAME="${OOS_TASK:-oos_videoqa}"
BATCH_SIZE="${OOS_BATCH_SIZE:-1}"
LIMIT_ARGS=()
if [ -n "${OOS_LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "$OOS_LIMIT")
fi

OUTPUT_PATH="$OOS_OUTPUT_DIR/$OUTPUT_SUBDIR"
mkdir -p "$OUTPUT_PATH"

print_oos_run_summary

python -m lmms_eval \
  --model "$MODEL" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK_NAME" \
  --batch_size "$BATCH_SIZE" \
  --log_samples \
  --output_path "$OUTPUT_PATH" \
  "${LIMIT_ARGS[@]}"
