#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-qwen3_vl_chat_fixed}"
QWEN3_VL_PRETRAINED="${OOS_QWEN3_VL_PRETRAINED:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${QWEN3_VL_PRETRAINED},fps=${OOS_QWEN3_VL_FPS:-${OOS_TARGET_FPS:-1}},max_num_frames=600,min_pixels=200704,max_pixels=200704,local_files_only=${OOS_LOCAL_FILES_ONLY:-False},attn_implementation=flash_attention_2}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-qwen3_vl_chat_fixed}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchvision}"
