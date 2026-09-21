#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-qwen3_5}"
QWEN3_6_35B_A3B_MODEL_ID="${OOS_QWEN3_6_35B_A3B_MODEL_ID:-${OOS_QWEN3_6_PRETRAINED:-Qwen/Qwen3.6-35B-A3B}}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${QWEN3_6_35B_A3B_MODEL_ID},fps=1,max_num_frames=600,min_pixels=200704,max_pixels=200704,enable_thinking=False,preserve_reasoning=${OOS_PRESERVE_REASONING:-False},local_files_only=${OOS_LOCAL_FILES_ONLY:-False},device_map=auto,attn_implementation=flash_attention_2}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-qwen3_6_35b_a3b}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
