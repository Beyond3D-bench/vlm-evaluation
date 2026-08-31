#!/usr/bin/env bash
# Node-local checkpoint staging for VLM-3R. Source this file; do not execute it.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Source launchers/stage_vlm3r_assets.sh from the VLM-3R model preset." >&2
  exit 1
fi

_vlm3r_stage_directory() {
  local source_dir="$1"
  local target_dir="$2"
  local label="$3"
  local marker="$target_dir/.vlm3r_stage_complete"

  if [ ! -d "$source_dir" ]; then
    echo "Missing VLM-3R $label directory: $source_dir" >&2
    exit 1
  fi

  mkdir -p "$target_dir"
  if [ ! -f "$marker" ]; then
    echo "Staging VLM-3R $label to node-local storage: $target_dir"
    cp -aL "$source_dir"/. "$target_dir"/
    touch "$marker"
    echo "Finished staging VLM-3R $label at $(date -Is)"
  else
    echo "Using already staged VLM-3R $label: $target_dir"
  fi
}

_vlm3r_stage_file() {
  local source_file="$1"
  local target_file="$2"
  local label="$3"
  local marker="${target_file}.vlm3r_stage_complete"

  if [ ! -f "$source_file" ]; then
    echo "Missing VLM-3R $label file: $source_file" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$target_file")"
  if [ ! -f "$marker" ]; then
    echo "Staging VLM-3R $label to node-local storage: $target_file"
    cp -aL "$source_file" "$target_file"
    touch "$marker"
    echo "Finished staging VLM-3R $label at $(date -Is)"
  else
    echo "Using already staged VLM-3R $label: $target_file"
  fi
}

_vlm3r_resolve_siglip_snapshot() {
  if [ -n "${VLM3R_SIGLIP_SOURCE:-}" ]; then
    printf '%s\n' "$VLM3R_SIGLIP_SOURCE"
    return
  fi

  local hub_cache="${HF_HUB_CACHE:-$HF_HOME/hub}"
  local cache_dir="$hub_cache/models--google--siglip-so400m-patch14-384"
  local revision="${VLM3R_SIGLIP_REVISION:-main}"
  local snapshot_id=""

  if [ -f "$cache_dir/refs/$revision" ]; then
    snapshot_id="$(<"$cache_dir/refs/$revision")"
  elif [ -d "$cache_dir/snapshots/$revision" ]; then
    snapshot_id="$revision"
  fi

  if [ -z "$snapshot_id" ] || [ ! -d "$cache_dir/snapshots/$snapshot_id" ]; then
    echo "Missing cached SigLIP snapshot under $cache_dir." >&2
    echo "Download google/siglip-so400m-patch14-384 on a login node first." >&2
    exit 1
  fi

  printf '%s\n' "$cache_dir/snapshots/$snapshot_id"
}

stage_vlm3r_assets_to_node_tmp() {
  if [ "${VLM3R_STAGE_TO_TMP:-1}" != "1" ]; then
    echo "VLM-3R node-local staging disabled (VLM3R_STAGE_TO_TMP=${VLM3R_STAGE_TO_TMP:-unset})."
    return
  fi

  # Avoid duplicating checkpoints into the shared fallback TMPDIR during
  # interactive/login-node runs. Set VLM3R_STAGE_OUTSIDE_SLURM=1 to override.
  if [ -z "${SLURM_JOB_ID:-}" ] && [ "${VLM3R_STAGE_OUTSIDE_SLURM:-0}" != "1" ]; then
    echo "Skipping VLM-3R staging outside Slurm."
    return
  fi

  local stage_root="${VLM3R_NODE_MODEL_ROOT:-$TMPDIR/oos_models/${SLURM_JOB_ID:-manual}/vlm3r}"
  local source_base="$VLM3R_BASE"
  local source_lora="$VLM3R_CKPT"
  local source_cut3r="${VLM3R_CUT3R_WEIGHTS:-$VLM3R_REPO/CUT3R/src/cut3r_512_dpt_4_64.pth}"
  local source_siglip
  source_siglip="$(_vlm3r_resolve_siglip_snapshot)"

  # VLM-3R's loader dispatches on substrings in the checkpoint directory name.
  # Keep llava, qwen, and lora in the staged adapter basename.
  local staged_base="$stage_root/LLaVA-NeXT-Video-7B-Qwen2"
  local staged_lora="$stage_root/vlm-3r-llava-qwen2-lora"
  local staged_siglip="$stage_root/siglip"
  local staged_cut3r="$stage_root/cut3r/cut3r_512_dpt_4_64.pth"

  echo "VLM-3R staging root: $stage_root"
  _vlm3r_stage_directory "$source_base" "$staged_base" "base model"
  _vlm3r_stage_directory "$source_lora" "$staged_lora" "LoRA checkpoint"
  _vlm3r_stage_directory "$source_siglip" "$staged_siglip" "SigLIP vision tower"
  _vlm3r_stage_file "$source_cut3r" "$staged_cut3r" "CUT3R checkpoint"

  python "$LAUNCHER_DIR/prepare_staged_vlm3r_config.py" "$staged_base/config.json" "$staged_siglip"
  python "$LAUNCHER_DIR/prepare_staged_vlm3r_config.py" "$staged_lora/config.json" "$staged_siglip"

  export VLM3R_BASE="$staged_base"
  export VLM3R_CKPT="$staged_lora"
  export VLM3R_SIGLIP="$staged_siglip"
  export VLM3R_CUT3R_WEIGHTS="$staged_cut3r"
  export OOS_STAGED_MODEL_PATH="$stage_root"
  echo "Finished staging all VLM-3R assets at $(date -Is)"
}
