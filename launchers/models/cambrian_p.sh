#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-cambrian_p}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=nyu-visionx/Cambrian-P-7B,conv_template=qwen_2,video_max_frames=32,local_files_only=True}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-cambrian_p_7b}"
export OOS_STAGE_MODEL_TO_TMP="${OOS_STAGE_MODEL_TO_TMP:-1}"
export OOS_STAGE_MODEL_ID="${OOS_STAGE_MODEL_ID:-nyu-visionx/Cambrian-P-7B}"
