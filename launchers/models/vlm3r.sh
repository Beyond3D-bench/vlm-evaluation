#!/usr/bin/env bash
if [ -z "${VLM3R_REPO:-}" ]; then
  echo "Set VLM3R_REPO=/path/to/VLM-3R before running OOS_MODEL=vlm3r." >&2
  exit 1
fi

# Resolve explicit model arguments before optional local staging.
MODEL_ARGS="${OOS_MODEL_ARGS:-}"
VLM3R_CKPT="${OOS_PRETRAINED:-$(get_model_arg pretrained || printf '%s' "$VLM3R_CKPT")}"
VLM3R_BASE="$(get_model_arg model_base || printf '%s' "$VLM3R_BASE")"
VLM3R_REVISION_ARGS=()
if [ -n "${OOS_MODEL_REVISION:-}" ]; then
  VLM3R_REVISION_ARGS=(--revision "$OOS_MODEL_REVISION")
fi
VLM3R_CKPT="$(python "$REPO_DIR/tools/resolve_checkpoint.py" "$VLM3R_CKPT" "${VLM3R_REVISION_ARGS[@]}")"
VLM3R_BASE="$(python "$REPO_DIR/tools/resolve_checkpoint.py" "$VLM3R_BASE")"
export VLM3R_CKPT VLM3R_BASE
# Resolve the vision tower explicitly so pinned downloads also work offline.
VLM3R_SIGLIP="$(python "$REPO_DIR/tools/resolve_checkpoint.py" "${VLM3R_SIGLIP_SOURCE:-${VLM3R_SIGLIP:-google/siglip-so400m-patch14-384}}")"
export VLM3R_SIGLIP

source "$LAUNCHER_DIR/stage_vlm3r_assets.sh"
stage_vlm3r_assets_to_node_tmp

# Pin the complete cuDNN sublibrary set bundled with the shared PyTorch
# environment. This prevents a system cuDNN from being mixed with wheel-provided
# sublibraries while keeping cuDNN enabled.
VLM3R_CUDNN_LIB="${VLM3R_CUDNN_LIB:-$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib}"
if [ ! -d "$VLM3R_CUDNN_LIB" ]; then
  echo "Missing VLM-3R cuDNN library directory: $VLM3R_CUDNN_LIB" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${VLM3R_CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLM3R_CUDNN_LIB

# CUT3R's pinned checkpoint contains trusted OmegaConf objects and its upstream
# loader relies on the pre-PyTorch-2.6 torch.load default.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Keep the official sources ahead of this repository. All dependencies are
# supplied by the complete VLM-3R CUDA 12.8 environment.
for OOS_PYTHON_PATH in \
  "$REPO_DIR" \
  "$VLM3R_REPO/CUT3R" \
  "$VLM3R_REPO"; do
  case ":${PYTHONPATH:-}:" in
    *":${OOS_PYTHON_PATH}:"*) ;;
    *) PYTHONPATH="${OOS_PYTHON_PATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
done
export PYTHONPATH
unset OOS_PYTHON_PATH

MODEL="${OOS_LMMS_MODEL:-vlm_3r}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${VLM3R_CKPT:-Journey9ni/vlm-3r-llava-qwen2-lora},model_base=${VLM3R_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2},conv_mode=${VLM3R_CONV_MODE:-qwen_1_5},for_get_frames_num=${VLM3R_FRAMES:-150},mm_spatial_pool_stride=${VLM3R_POOL_STRIDE:-2},mm_spatial_pool_mode=${VLM3R_POOL_MODE:-bilinear},mm_newline_position=${VLM3R_NEWLINE_POSITION:-grid},low_cpu_mem_usage=${VLM3R_LOW_CPU_MEM_USAGE:-True},attn_implementation=${VLM3R_ATTN_IMPLEMENTATION:-eager},disable_cudnn=${VLM3R_DISABLE_CUDNN:-False},add_time_instruction=${VLM3R_ADD_TIME_INSTRUCTION:-False}}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-vlm_3r}"

set_model_arg pretrained "$VLM3R_CKPT"
set_model_arg model_base "$VLM3R_BASE"
