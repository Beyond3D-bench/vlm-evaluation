#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-cambrian_p}"
CAMBRIAN_MODEL_ID="${OOS_CAMBRIAN_MODEL_ID:-nyu-visionx/Cambrian-P-7B}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${CAMBRIAN_MODEL_ID},model_name=llava_qwen,conv_template=qwen_1_5,video_max_frames=150,video_force_sample=True,mm_spatial_pool_stride=2,mm_spatial_pool_mode=bilinear,use_camera_tokens=True,camera_tokens_mode=camera_tokens,camera_tokens_place=append_to_frame,query_mode=query_after_image,local_files_only=${OOS_LOCAL_FILES_ONLY:-False}}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-cambrian_p_7b}"
export OOS_STAGE_MODEL_REF="${OOS_STAGE_MODEL_REF:-${OOS_CAMBRIAN_REVISION:-}}"
