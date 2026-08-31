#!/usr/bin/env bash

SPATIAL_MLLM_CKPT="${SPATIAL_MLLM_CKPT:-/cluster/home/fangma/scratch/checkpoints/Spatial-MLLM-v1.1-Instruct-820K}"
SPATIAL_MLLM_REPO="${SPATIAL_MLLM_REPO:-/cluster/home/fangma/scratch/Spatial-MLLM}"

export OOS_STAGE_MODEL_TO_TMP="${OOS_STAGE_MODEL_TO_TMP:-1}"
export OOS_STAGE_MODEL_DIR="${OOS_STAGE_MODEL_DIR:-$SPATIAL_MLLM_CKPT}"

MODEL="spatial_mllm"
SPATIAL_VIDEO_ARGS="max_num_frames=${OOS_SPATIAL_MAX_NUM_FRAMES:-64}"
if [ -n "${OOS_SPATIAL_VIDEO_FPS:-}" ]; then
  SPATIAL_VIDEO_ARGS="${SPATIAL_VIDEO_ARGS},fps=${OOS_SPATIAL_VIDEO_FPS}"
fi
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${SPATIAL_MLLM_CKPT},${SPATIAL_VIDEO_ARGS},video_resolution=${OOS_SPATIAL_VIDEO_RESOLUTION:-448},spatial_mllm_repo=${SPATIAL_MLLM_REPO},attn_implementation=${SPATIAL_MLLM_ATTN_IMPLEMENTATION:-flash_attention_2},use_fast=${OOS_SPATIAL_USE_FAST:-True}}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-spatial_mllm}"
