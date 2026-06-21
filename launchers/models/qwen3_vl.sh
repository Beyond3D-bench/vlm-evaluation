#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-qwen3_vl_chat_fixed}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=Qwen/Qwen3-VL-4B-Instruct,max_num_frames=250,min_pixels=200704,max_pixels=200704}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-qwen3_vl_chat_fixed}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
