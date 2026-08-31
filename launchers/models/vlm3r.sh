#!/usr/bin/env bash
if [ -z "${VLM3R_REPO:-}" ]; then
  echo "Set VLM3R_REPO=/path/to/VLM-3R before running OOS_MODEL=vlm3r." >&2
  exit 1
fi

if [ -z "${VLM3R_OVERLAY:-}" ]; then
  echo "Set VLM3R_OVERLAY=/path/to/vlm3r-cu128 before running OOS_MODEL=vlm3r." >&2
  exit 1
fi

source "$LAUNCHER_DIR/stage_vlm3r_assets.sh"
stage_vlm3r_assets_to_node_tmp

# Pin the complete cuDNN sublibrary set bundled with the shared PyTorch
# environment. This prevents a system cuDNN from being mixed with wheel-provided
# sublibraries while keeping cuDNN enabled.
VLM3R_CUDNN_LIB="${VLM3R_CUDNN_LIB:-$REPO_DIR/.venv-cu128-home/lib/python3.10/site-packages/nvidia/cudnn/lib}"
if [ ! -d "$VLM3R_CUDNN_LIB" ]; then
  echo "Missing VLM-3R cuDNN library directory: $VLM3R_CUDNN_LIB" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${VLM3R_CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLM3R_CUDNN_LIB

# Keep the official sources ahead of the dependency overlay, followed by the
# shared CUDA 12.8 environment. Prepend in reverse priority order.
for OOS_PYTHON_PATH in \
  "$VLM3R_OVERLAY" \
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
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${VLM3R_CKPT:-Journey9ni/vlm-3r-llava-qwen2-lora},model_base=${VLM3R_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2},conv_mode=${VLM3R_CONV_MODE:-qwen_1_5},for_get_frames_num=${VLM3R_FRAMES:-150},mm_spatial_pool_stride=${VLM3R_POOL_STRIDE:-2},mm_spatial_pool_mode=${VLM3R_POOL_MODE:-bilinear},mm_newline_position=${VLM3R_NEWLINE_POSITION:-grid},low_cpu_mem_usage=${VLM3R_LOW_CPU_MEM_USAGE:-True},attn_implementation=${VLM3R_ATTN_IMPLEMENTATION:-eager},disable_cudnn=${VLM3R_DISABLE_CUDNN:-False},add_time_instruction=${VLM3R_ADD_TIME_INSTRUCTION:-False},export_point_cloud=${VLM3R_EXPORT_POINT_CLOUD:-False},point_cloud_output_dir=${VLM3R_POINT_CLOUD_DIR:-/work/courses/3dv/team1/lmms-eval/outputs/oos_videoqa/vlm_3r_point_clouds},point_cloud_export_limit=${VLM3R_POINT_CLOUD_LIMIT:-0}}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-vlm_3r}"
