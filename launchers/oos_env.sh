#!/usr/bin/env bash
# Team 1 example config. Edit local paths/model choices, then run:
#   bash launchers/run_oos_eval.sh

# Required for private Hugging Face models or gated datasets.
export HF_TOKEN="${SECRET_KEY:-${HF_TOKEN:-}}"

# Local cache/output directories. Defaults are relative to the repository root
# when launchers/run_oos_eval.sh is used.
export HF_HOME="${HF_HOME:-/work/scratch/fangma/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/work/scratch/fangma/hf_cache/datasets}"
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-$HF_DATASETS_CACHE}"
export LMMS_EVAL_CACHE="${LMMS_EVAL_CACHE:-/work/scratch/fangma/lmms_eval_cache}"
export TMPDIR="${TMPDIR:-/work/scratch/fangma/tmp}"
export OOS_VIDEO_CACHE_DIR="${OOS_VIDEO_CACHE_DIR:-/work/scratch/fangma/oos_video_cache}"
export OOS_OUTPUT_DIR="${OOS_OUTPUT_DIR:-/work/courses/3dv/team1/lmms-eval/outputs/oos_videoqa}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# OOS task behavior.
export OOS_HISTORY_MODE="${OOS_HISTORY_MODE:-gold}"          # none, gold, pred
export LMMS_EVAL_SHUFFLE_DOCS="${LMMS_EVAL_SHUFFLE_DOCS:-0}" # keep 0 for pred mode
export OOS_NO_VIDEO_INPUT="${OOS_NO_VIDEO_INPUT:-0}"
export OOS_CHAT_DEBUG="${OOS_CHAT_DEBUG:-1}" # set to 1 to print model prompt/media diagnostics
# Video preprocessing and scoring.
# Currently the input video is already preprocessed to fps=1 & 448x448, so we don't need to preprocess it again.
export OOS_PREPROCESS_VIDEO="${OOS_PREPROCESS_VIDEO:-0}"
export OOS_TARGET_FPS="${OOS_TARGET_FPS:-1}"
export OOS_RESIZE_WIDTH="${OOS_RESIZE_WIDTH:-448}"
export OOS_RESIZE_HEIGHT="${OOS_RESIZE_HEIGHT:-448}"

# Specify the video dimension to draw the marker for the anchored objects at query frame.
# This step is necessary because the coordinates of the anchored object provided in the dataset is normalized to [0,1], so we need to know the actual video dimension to draw the marker at the correct location.
export OOS_VIDEO_WIDTH="${OOS_VIDEO_WIDTH:-448}"
export OOS_VIDEO_HEIGHT="${OOS_VIDEO_HEIGHT:-448}"

# Step 2 and 3 tolerance
export OOS_TIME_TOLERANCE_SEC="${OOS_TIME_TOLERANCE_SEC:-3.0}"
export OOS_COORD_TOLERANCE_NORM="${OOS_COORD_TOLERANCE_NORM:-0.2}"

# Qwen video reader. Use torchcodec on ARM/GB10; decord is often fine on x86_64.
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"

# VLM-3R needs a separate clone on PYTHONPATH.
export VLM3R_REPO="${VLM3R_REPO:-}"
export VLM3R_CKPT="${VLM3R_CKPT:-Journey9ni/vlm-3r-llava-qwen2-lora}"
export VLM3R_BASE="${VLM3R_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"

# Current ETH 3dv GB10 settings used by the existing Qwen3.6 script.
export TORCH_CUDNN_V8_API_DISABLED=1
export FFMPEG_PATH="${FFMPEG_PATH:-/work/courses/3dv/team1/ffmpeg_env/bin/ffmpeg}"
export LD_LIBRARY_PATH="/work/courses/3dv/team1/ffmpeg_env/lib:${LD_LIBRARY_PATH:-}"
export PATH="/work/courses/3dv/team1/ffmpeg_env/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
