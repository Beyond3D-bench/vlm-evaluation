#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-qwen3_vl_chat_fixed}"
SENSENOVA_QWEN_PRETRAINED="${SENSENOVA_QWEN_CKPT:-sensenova/SenseNova-SI-1.3-Qwen3-VL-8B}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${SENSENOVA_QWEN_PRETRAINED},fps=${OOS_SENSENOVA_QWEN_FPS:-${OOS_TARGET_FPS:-1}},max_num_frames=600,min_pixels=200704,max_pixels=200704,enable_thinking=False,local_files_only=${OOS_LOCAL_FILES_ONLY:-False},device_map=auto,attn_implementation=flash_attention_2}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-sensenova_qwen}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchvision}"
