#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-qwen3_5}"
#MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=Qwen/Qwen3.6-27B,fps=1,max_num_frames=250,min_pixels=200704,max_pixels=200704,enable_thinking=False}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=/cluster/home/fangma/scratch/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9,fps=1,max_num_frames=250,min_pixels=200704,max_pixels=200704,enable_thinking=False,local_files_only=True,device_map=auto}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-qwen3_5}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
