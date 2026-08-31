#!/usr/bin/env bash

export STREAM3D_VLM_REPO="${STREAM3D_VLM_REPO:-/cluster/home/$USER/scratch/Stream3D-VLM}"
export STREAM3D_VLM_SRC="${STREAM3D_VLM_SRC:-${STREAM3D_VLM_REPO}/src}"
export STREAM3D_OVERLAY="${STREAM3D_OVERLAY:-/cluster/home/$USER/scratch/python-overlays/stream3d-cu128}"
export STREAM3D_VLM_CKPT="${STREAM3D_VLM_CKPT:-/cluster/home/$USER/scratch/checkpoints/Stream3D-VLM-4B}"

# Keep the official source ahead of its pinned package overlay, followed by the
# shared CUDA 12.8 environment. Prepend in reverse priority order.
for OOS_PYTHON_PATH in "$STREAM3D_OVERLAY" "$STREAM3D_VLM_SRC"; do
  case ":${PYTHONPATH:-}:" in
    *":${OOS_PYTHON_PATH}:"*) ;;
    *) PYTHONPATH="${OOS_PYTHON_PATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
done
export PYTHONPATH
unset OOS_PYTHON_PATH

export OOS_STAGE_MODEL_TO_TMP="${OOS_STAGE_MODEL_TO_TMP:-1}"
export OOS_STAGE_MODEL_DIR="${OOS_STAGE_MODEL_DIR:-$STREAM3D_VLM_CKPT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${OOS_LMMS_MODEL:-stream3d_vlm}"
STREAM3D_PROMPT_MODE="${STREAM3D_PROMPT_MODE:-query_after_prefix}"
STREAM3D_FRAME_POLICY="${STREAM3D_FRAME_POLICY:-auto}"
STREAM3D_FRAME_TIMESTAMPS="${STREAM3D_FRAME_TIMESTAMPS:-0}"
STREAM3D_MAX_FRAMES="${OOS_STREAM3D_MAX_FRAMES:-600}"
STREAM3D_VIDEO_DECODER="${OOS_STREAM3D_VIDEO_DECODER:-pyav}"
# Preserve the official causal token order while bounding StreamVGGT's
# quadratic per-call geometry attention. Use 0 only for short, full-prefix
# equivalence tests; 4 is a conservative default for long A100 evaluations.
STREAM3D_PREFIX_CHUNK_SIZE="${OOS_STREAM3D_PREFIX_CHUNK_SIZE:-4}"
STREAM3D_PREFIX_CACHE="${OOS_STREAM3D_PREFIX_CACHE:-1}"
STREAM3D_MEMORY_CLEANUP="${OOS_STREAM3D_MEMORY_CLEANUP:-0}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${STREAM3D_VLM_CKPT},stream3d_repo=${STREAM3D_VLM_REPO},require_geometry=true,max_frames=${STREAM3D_MAX_FRAMES},video_decoder=${STREAM3D_VIDEO_DECODER},attn_implementation=${STREAM3D_ATTN_IMPLEMENTATION:-flash_attention_2},stream_prompt_mode=${STREAM3D_PROMPT_MODE},stream_frame_policy=${STREAM3D_FRAME_POLICY},stream_frame_timestamps=${STREAM3D_FRAME_TIMESTAMPS},stream_prefix_chunk_size=${STREAM3D_PREFIX_CHUNK_SIZE},stream_prefix_cache=${STREAM3D_PREFIX_CACHE},stream_memory_cleanup=${STREAM3D_MEMORY_CLEANUP}}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-stream3d_vlm}"
export FORCE_QWENVL_VIDEO_READER="${OOS_STREAM3D_QWENVL_VIDEO_READER:-$STREAM3D_VIDEO_DECODER}"
