#!/usr/bin/env bash
set -euo pipefail

# Do not let the submit shell's active Python environment leak into the job.
# The selected model environment is resolved and activated below.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  OOS_INHERITED_VENV_BIN="${VIRTUAL_ENV%/}/bin"
  OOS_CLEAN_PATH=""
  IFS=: read -r -a OOS_PATH_ENTRIES <<< "${PATH:-}"
  for OOS_PATH_ENTRY in "${OOS_PATH_ENTRIES[@]}"; do
    if [ "$OOS_PATH_ENTRY" != "$OOS_INHERITED_VENV_BIN" ]; then
      OOS_CLEAN_PATH="${OOS_CLEAN_PATH:+$OOS_CLEAN_PATH:}$OOS_PATH_ENTRY"
    fi
  done
  export PATH="$OOS_CLEAN_PATH"
  unset OOS_CLEAN_PATH OOS_INHERITED_VENV_BIN OOS_PATH_ENTRIES OOS_PATH_ENTRY
fi
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
hash -r

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

# Resolve after activating the model environment; no system FFmpeg is required.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "$VIRTUAL_ENV/ffmpeg-shared/lib" ]; then
  export LD_LIBRARY_PATH="$VIRTUAL_ENV/ffmpeg-shared/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -z "${FFMPEG_PATH:-}" ]; then
  FFMPEG_PATH="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  export FFMPEG_PATH
fi
if ! "$FFMPEG_PATH" -version >/dev/null 2>&1; then
  echo "Cannot execute FFmpeg: $FFMPEG_PATH. Rerun setup.sh for this model or set FFMPEG_PATH to a working executable." >&2
  exit 1
fi

if [ "${OOS_MODEL:-}" = "vlm3r" ]; then
  if [ ! -s "$VLM3R_CUT3R_WEIGHTS" ]; then
    echo "Missing CUT3R weights. Run bash setup.sh --model vlm3r on a connected machine first." >&2
    exit 1
  fi
  python "$REPO_DIR/tools/build_vlm3r_curope.py"
fi

if [ -z "${OOS_DATASET_JSONL:-}" ]; then
  # Evaluation must not make network requests: compute nodes commonly have no Internet.
  OOS_DATASET_EXPORTS="$(python "$REPO_DIR/tools/resolve_oos_dataset.py" --offline --shell)" || exit 1
  eval "$OOS_DATASET_EXPORTS"
  unset OOS_DATASET_EXPORTS
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
