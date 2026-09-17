#!/usr/bin/env bash
set -euo pipefail

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$LAUNCHER_DIR/.." && pwd)"
cd "$REPO_DIR"

source "$LAUNCHER_DIR/common.sh"
load_oos_env
resolve_oos_venv

if [ -n "${OOS_VENV:-}" ]; then
  VENV_ACTIVATE="$OOS_VENV"

  if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Missing virtualenv activate script: $OOS_VENV" >&2
    echo "Resolved activate script: $VENV_ACTIVATE" >&2
    echo "Current directory: $(pwd)" >&2
    echo "Create it with tools/create_environment.py, or set OOS_VENV explicitly." >&2
    exit 1
  fi
  source "$VENV_ACTIVATE"
fi

if [ "${OOS_MODEL:-}" = "vlm3r" ]; then
  if [ ! -s "$VLM3R_CUT3R_WEIGHTS" ]; then
    echo "Missing CUT3R weights. Run bash setup.sh --model vlm3r on a connected machine first." >&2
    exit 1
  fi
  python "$REPO_DIR/tools/build_vlm3r_curope.py"
fi

require_oos_env
python "$REPO_DIR/tools/validate_dataset.py" "$OOS_DATASET_JSONL"
prepare_oos_dirs
load_oos_model_preset
resolve_oos_checkpoint
stage_hf_model_to_node_tmp

TASK_NAME="${OOS_TASK:-oos_videoqa}"
BATCH_SIZE="${OOS_BATCH_SIZE:-1}"
LIMIT_ARGS=()
if [ -n "${OOS_LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "$OOS_LIMIT")
fi

OUTPUT_PATH="$OOS_OUTPUT_DIR/$OUTPUT_SUBDIR"
mkdir -p "$OUTPUT_PATH"

if [ "${OOS_STREAM_RESULTS:-0}" = "1" ]; then
  LIVE_RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
  export OOS_LIVE_RESULTS_PATH="${OOS_LIVE_RESULTS_PATH:-$OUTPUT_PATH/${LIVE_RUN_ID}_live_results.jsonl}"
  echo "Live results: $OOS_LIVE_RESULTS_PATH"
fi

print_oos_run_summary
echo "Starting lmms_eval at $(date -Is)"

PYTHON_ARGS=(-u -X faulthandler)
if [ "${OOS_IMPORTTIME:-0}" = "1" ]; then
  PYTHON_ARGS+=(-X importtime)
fi

python "${PYTHON_ARGS[@]}" launchers/run_lmms_eval_with_trace.py \
  --model "$MODEL" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK_NAME" \
  --batch_size "$BATCH_SIZE" \
  --log_samples \
  --output_path "$OUTPUT_PATH" \
  "${LIMIT_ARGS[@]}"
