#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-qwen3_vl_chat_fixed}"
QWEN3_VL_PRETRAINED="/cluster/home/fangma/scratch/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${QWEN3_VL_PRETRAINED},fps=${OOS_QWEN3_VL_FPS:-${OOS_TARGET_FPS:-1}},max_num_frames=600,min_pixels=200704,max_pixels=200704,local_files_only=True,attn_implementation=flash_attention_2}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-qwen3_vl_chat_fixed}"
export OOS_STAGE_MODEL_TO_TMP="${OOS_STAGE_MODEL_TO_TMP:-1}"
unset OOS_STAGE_MODEL_ID
export OOS_STAGE_MODEL_DIR="$QWEN3_VL_PRETRAINED"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchvision}"
