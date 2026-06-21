#!/usr/bin/env bash
# Shared helpers for OOS launcher scripts. Source this file; do not execute it.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Source launchers/common.sh from another launcher script." >&2
  exit 1
fi

LAUNCHER_DIR="${LAUNCHER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_DIR="${REPO_DIR:-$(cd "$LAUNCHER_DIR/.." && pwd)}"

OOS_MODEL_PRESETS="qwen3_6 qwen3_vl llava internvl phi4 vlm3r custom"

load_oos_env() {
  local env_file="${OOS_ENV_FILE:-launchers/oos_env.sh}"
  if [ ! -f "$env_file" ]; then
    echo "Missing env file: $env_file" >&2
    echo "Use launchers/oos_env.sh as the Team 1 example config, or set OOS_ENV_FILE=/path/to/your_env.sh." >&2
    echo "Edit it for your local cache/output/model settings." >&2
    exit 1
  fi
  source "$env_file"
}

require_oos_env() {
  local required_vars=(
    HF_HOME
    HF_DATASETS_CACHE
    LMMS_EVAL_CACHE
    TMPDIR
    OOS_VIDEO_CACHE_DIR
    OOS_OUTPUT_DIR
    OOS_HISTORY_MODE
    LMMS_EVAL_SHUFFLE_DOCS
    OOS_NO_VIDEO_INPUT
    OOS_CHAT_DEBUG
    OOS_PREPROCESS_VIDEO
    OOS_TARGET_FPS
    OOS_RESIZE_WIDTH
    OOS_RESIZE_HEIGHT
    OOS_VIDEO_WIDTH
    OOS_VIDEO_HEIGHT
    OOS_TIME_TOLERANCE_SEC
    OOS_COORD_TOLERANCE_NORM
  )

  local name
  for name in "${required_vars[@]}"; do
    if [ -z "${!name:-}" ]; then
      echo "Required env var $name is empty. Set it in ${OOS_ENV_FILE:-launchers/oos_env.sh}." >&2
      exit 1
    fi
  done
}

prepare_oos_dirs() {
  mkdir -p \
    logs \
    outputs/oos_videoqa \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$LMMS_EVAL_CACHE" \
    "$TMPDIR" \
    "$OOS_VIDEO_CACHE_DIR" \
    "$OOS_OUTPUT_DIR"
}

load_oos_model_preset() {
  MODEL_PRESET="${OOS_MODEL:-qwen3_6}"

  if [ "$MODEL_PRESET" = "custom" ]; then
    MODEL="${OOS_LMMS_MODEL:?Set OOS_LMMS_MODEL when OOS_MODEL=custom.}"
    MODEL_ARGS="${OOS_MODEL_ARGS:-}"
    OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-custom}"
    return
  fi

  local preset_file="$LAUNCHER_DIR/models/$MODEL_PRESET.sh"
  if [ ! -f "$preset_file" ]; then
    echo "Unknown OOS_MODEL='$MODEL_PRESET'. Use one of: $OOS_MODEL_PRESETS." >&2
    exit 1
  fi

  source "$preset_file"
}

print_oos_run_summary() {
  echo "Repo: $REPO_DIR"
  echo "Model preset: $MODEL_PRESET"
  echo "lmms-eval model: $MODEL"
  echo "Task: ${TASK_NAME:-oos_videoqa}"
  echo "Output: $OUTPUT_PATH"
  python --version
}
