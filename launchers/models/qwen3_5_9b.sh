#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-qwen3_5}"
QWEN3_5_9B_MODEL_ID="${OOS_QWEN3_5_9B_MODEL_ID:-Qwen/Qwen3.5-9B}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${QWEN3_5_9B_MODEL_ID},fps=${OOS_QWEN3_5_9B_FPS:-${OOS_TARGET_FPS:-1}},max_num_frames=${OOS_QWEN3_5_9B_MAX_FRAMES:-600},min_pixels=200704,max_pixels=200704,enable_thinking=False,local_files_only=${OOS_LOCAL_FILES_ONLY:-False},device_map=auto,attn_implementation=flash_attention_2}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-qwen3_5_9b}"
export OOS_STAGE_MODEL_REF="${OOS_STAGE_MODEL_REF:-${OOS_QWEN3_5_9B_REVISION:-}}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
