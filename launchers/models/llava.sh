#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-llava_onevision1_5_chat_fixed}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=lmms-lab/LLaVA-OneVision-1.5-8B-Instruct,fps=1,max_num_frames=250,min_pixels=200704,max_pixels=200704}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-llava_onevision1_5_chat_fixed}"
