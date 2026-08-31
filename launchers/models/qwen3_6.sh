#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-qwen3_5}"
#MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=Qwen/Qwen3.6-27B,fps=1,max_num_frames=250,min_pixels=200704,max_pixels=200704,enable_thinking=False}"
QWEN3_6_PRETRAINED="${OOS_QWEN3_6_PRETRAINED:-$HF_HOME/models--Qwen--Qwen3.6-35B-A3B}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${QWEN3_6_PRETRAINED},fps=1,max_num_frames=600,min_pixels=200704,max_pixels=200704,enable_thinking=False,preserve_reasoning=${OOS_PRESERVE_REASONING:-False},local_files_only=True,device_map=auto,attn_implementation=flash_attention_2}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-qwen3_5}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
