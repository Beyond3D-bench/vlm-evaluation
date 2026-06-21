#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-qwen3_5}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=Qwen/Qwen3.6-27B,fps=1,max_num_frames=250,min_pixels=200704,max_pixels=200704,enable_thinking=False}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-qwen3_6}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
