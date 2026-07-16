#!/usr/bin/env bash
# Shared helpers for OOS launcher scripts. Source this file; do not execute it.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Source launchers/common.sh from another launcher script." >&2
  exit 1
fi

LAUNCHER_DIR="${LAUNCHER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_DIR="${REPO_DIR:-$(cd "$LAUNCHER_DIR/.." && pwd)}"

OOS_MODEL_PRESETS="qwen3_6 qwen3_vl llava internvl phi4 vlm3r stream3d_vlm cambrian_p spatial_mllm sensenova_internvl sensenova_qwen custom"

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

stage_hf_model_to_node_tmp() {
  if [ "${OOS_STAGE_MODEL_TO_TMP:-0}" != "1" ]; then
    return
  fi

  local local_model_dir="${OOS_STAGE_MODEL_DIR:-}"
  if [ -n "$local_model_dir" ]; then
    if [ ! -d "$local_model_dir" ]; then
      echo "OOS_STAGE_MODEL_DIR does not exist: $local_model_dir" >&2
      exit 1
    fi

    local model_name
    model_name="$(basename "$local_model_dir")"
    local node_tmp_root="${OOS_NODE_TMPDIR:-$TMPDIR/oos_models/${SLURM_JOB_ID:-manual}}"
    local dst="$node_tmp_root/$model_name"
    mkdir -p "$dst"

    if [ ! -f "$dst/.oos_stage_complete" ]; then
      echo "Staging local model directory to node-local path: $dst"
      rm -f "$dst/.oos_stage_complete"
      cp -aL "$local_model_dir"/. "$dst"/
      touch "$dst/.oos_stage_complete"
      echo "Finished staging local model at $(date -Is)"
    else
      echo "Using already staged local model: $dst"
    fi

    MODEL_ARGS="${MODEL_ARGS/pretrained=$local_model_dir/pretrained=$dst}"
    export OOS_STAGED_MODEL_PATH="$dst"
    return
  fi

  local model_id="${OOS_STAGE_MODEL_ID:-}"
  if [ -z "$model_id" ]; then
    echo "OOS_STAGE_MODEL_TO_TMP=1 requires OOS_STAGE_MODEL_ID or OOS_STAGE_MODEL_DIR." >&2
    exit 1
  fi

  local cache_name="models--${model_id//\//--}"
  local cache_dir="$HF_HOME/$cache_name"
  local ref="${OOS_STAGE_MODEL_REF:-main}"
  local snapshot_id=""
  if [ -f "$cache_dir/refs/$ref" ]; then
    snapshot_id="$(cat "$cache_dir/refs/$ref")"
  elif [ -d "$cache_dir/snapshots/$ref" ]; then
    snapshot_id="$ref"
  fi

  if [ -z "$snapshot_id" ] || [ ! -d "$cache_dir/snapshots/$snapshot_id" ]; then
    echo "Missing cached HF snapshot for $model_id under $cache_dir." >&2
    echo "Expected $cache_dir/refs/$ref or $cache_dir/snapshots/<revision>." >&2
    exit 1
  fi

  local src="$cache_dir/snapshots/$snapshot_id"
  local node_tmp_root="${OOS_NODE_TMPDIR:-/tmp/$USER/oos_hf_models/${SLURM_JOB_ID:-manual}}"
  local dst="$node_tmp_root/$cache_name/$snapshot_id"
  mkdir -p "$dst"

  if [ ! -f "$dst/.oos_stage_complete" ]; then
    echo "Staging $model_id snapshot $snapshot_id to node-local path: $dst"
    rm -f "$dst/.oos_stage_complete"
    cp -aL "$src"/. "$dst"/
    touch "$dst/.oos_stage_complete"
    echo "Finished staging $model_id at $(date -Is)"
  else
    echo "Using already staged model snapshot: $dst"
  fi

  MODEL_ARGS="${MODEL_ARGS/pretrained=$model_id/pretrained=$dst}"
  export OOS_STAGED_MODEL_PATH="$dst"
}

print_oos_run_summary() {
  echo "Repo: $REPO_DIR"
  echo "Model preset: $MODEL_PRESET"
  echo "lmms-eval model: $MODEL"
  echo "Model args: $MODEL_ARGS"
  echo "Task: ${TASK_NAME:-oos_videoqa}"
  echo "Output: $OUTPUT_PATH"
  echo "Env file: ${OOS_ENV_FILE:-launchers/oos_env.sh}"
  echo "Repo .env present: $([ -f "$REPO_DIR/.env" ] && echo yes || echo no)"
  echo "HF_TOKEN set: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"
  echo "HF_HOME: $HF_HOME"
  echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
  echo "LMMS_EVAL_CACHE: $LMMS_EVAL_CACHE"
  echo "TMPDIR: $TMPDIR"
  echo "OOS_STAGED_MODEL_PATH: ${OOS_STAGED_MODEL_PATH:-unset}"
  echo "FFMPEG_PATH: ${FFMPEG_PATH:-unset}"
  python --version
}
