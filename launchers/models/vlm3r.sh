#!/usr/bin/env bash
if [ -z "${VLM3R_REPO:-}" ]; then
  echo "Set VLM3R_REPO=/path/to/VLM-3R before running OOS_MODEL=vlm3r." >&2
  exit 1
fi

export PYTHONPATH="$VLM3R_REPO:$VLM3R_REPO/CUT3R:$REPO_DIR:${PYTHONPATH:-}"
MODEL="${OOS_LMMS_MODEL:-vlm_3r}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${VLM3R_CKPT:-Journey9ni/vlm-3r-llava-qwen2-lora},model_base=${VLM3R_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2},conv_mode=${VLM3R_CONV_MODE:-qwen_1_5},for_get_frames_num=${VLM3R_FRAMES:-167},mm_spatial_pool_stride=${VLM3R_POOL_STRIDE:-2},mm_spatial_pool_mode=${VLM3R_POOL_MODE:-average},mm_newline_position=${VLM3R_NEWLINE_POSITION:-grid},attn_implementation=${VLM3R_ATTN_IMPLEMENTATION:-eager},disable_cudnn=${VLM3R_DISABLE_CUDNN:-True},add_time_instruction=${VLM3R_ADD_TIME_INSTRUCTION:-False}}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-vlm_3r}"
