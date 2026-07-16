#!/usr/bin/env bash

SPATIAL_MLLM_CKPT="${SPATIAL_MLLM_CKPT:-/cluster/home/fangma/scratch/checkpoints/Spatial-MLLM-v1.1-Instruct-820K}"
SPATIAL_MLLM_REPO="${SPATIAL_MLLM_REPO:-/cluster/home/fangma/scratch/Spatial-MLLM}"

export OOS_STAGE_MODEL_TO_TMP="${OOS_STAGE_MODEL_TO_TMP:-1}"
export OOS_STAGE_MODEL_DIR="${OOS_STAGE_MODEL_DIR:-$SPATIAL_MLLM_CKPT}"

MODEL="spatial_mllm"
MODEL_ARGS="pretrained=${SPATIAL_MLLM_CKPT},fps=${OOS_SPATIAL_VIDEO_FPS:-${OOS_TARGET_FPS:-1}},spatial_mllm_repo=${SPATIAL_MLLM_REPO},attn_implementation=${SPATIAL_MLLM_ATTN_IMPLEMENTATION:-sdpa}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-spatial_mllm}"
