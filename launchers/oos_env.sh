#!/usr/bin/env bash
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
# Required for private Hugging Face models or gated datasets.
export HF_TOKEN="${SECRET_KEY:-${HF_TOKEN:-}}"
# export HF_HUB_DISABLE_XET=1

# Persistent storage. Override OOS_STORAGE_ROOT in .env on systems that do not
# use $HOME/scratch. This default preserves the existing Euler layout without
# embedding a cluster hostname or username in the public configuration.
export OOS_STORAGE_ROOT="${OOS_STORAGE_ROOT:-$HOME/scratch}"
export OOS_CHECKPOINT_DIR="${OOS_CHECKPOINT_DIR:-$OOS_STORAGE_ROOT/checkpoints}"
export OOS_OVERLAY_ROOT="${OOS_OVERLAY_ROOT:-$OOS_STORAGE_ROOT/python-overlays}"
export OOS_VENV_ROOT="${OOS_VENV_ROOT:-$OOS_STORAGE_ROOT/venvs}"
export OOS_DATA_ROOT="${OOS_DATA_ROOT:-$OOS_STORAGE_ROOT/data}"
export OOS_MODEL_SOURCE_DIR="${OOS_MODEL_SOURCE_DIR:-$OOS_STORAGE_ROOT}"

# Persistent Hugging Face cache. Downloads made on a login node remain
# available to offline compute jobs through the standard Hub cache layout.
export HF_HOME="${HF_HOME:-$OOS_STORAGE_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-$HF_DATASETS_CACHE}"
export LMMS_EVAL_CACHE="${LMMS_EVAL_CACHE:-$OOS_STORAGE_ROOT/lmms_eval_cache}"
# In Slurm jobs, Euler sets TMPDIR to node-local scratch when #SBATCH --tmp is
# requested. Keep that value; falling back to network scratch makes imports slow.
export TMPDIR="${OOS_TMPDIR:-${TMPDIR:-$OOS_STORAGE_ROOT/tmp}}"
export OOS_JOB_CACHE_ROOT="${OOS_JOB_CACHE_ROOT:-$TMPDIR/oos_vlm_evaluation}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OOS_JOB_CACHE_ROOT/xdg_cache}"
export TORCH_HOME="${TORCH_HOME:-$OOS_JOB_CACHE_ROOT/torch_cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$OOS_JOB_CACHE_ROOT/pycache}"
export OOS_VIDEO_CACHE_DIR="${OOS_VIDEO_CACHE_DIR:-$OOS_STORAGE_ROOT/oos_video_cache3}"
export OOS_OUTPUT_DIR="${OOS_OUTPUT_DIR:-$OOS_STORAGE_ROOT/lmms-eval/outputs/oos_videoqa}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-4}"

if [ "${OOS_MODEL:-}" = "spatial_mllm" ]; then
  OOS_REPO_DIR="${REPO_DIR:-$(pwd)}"
  export OOS_VENV="${OOS_VENV:-${OOS_REPO_DIR}/.venv-cu128-home/bin/activate}"
  export SPATIAL_MLLM_CKPT="${SPATIAL_MLLM_CKPT:-$OOS_CHECKPOINT_DIR/Spatial-MLLM-v1.1-Instruct-820K}"
  export SPATIAL_MLLM_REPO="${SPATIAL_MLLM_REPO:-$OOS_MODEL_SOURCE_DIR/Spatial-MLLM}"
  export SPATIAL_MLLM_EXTERNAL_PATH="${SPATIAL_MLLM_EXTERNAL_PATH:-${SPATIAL_MLLM_REPO}/src/qwenvl/external}"
  export SPATIAL_OVERLAY="${SPATIAL_OVERLAY:-$OOS_OVERLAY_ROOT/spatial-mllm-cu128}"

  # Prepend in reverse priority order so the final search order is
  # Spatial-MLLM, bundled VGGT, the dependency overlay, then the base venv.
  for OOS_PYTHON_PATH in "$SPATIAL_OVERLAY" "$SPATIAL_MLLM_EXTERNAL_PATH" "$SPATIAL_MLLM_REPO"; do
    case ":${PYTHONPATH:-}:" in
      *":${OOS_PYTHON_PATH}:"*) ;;
      *) PYTHONPATH="${OOS_PYTHON_PATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
  done
  export PYTHONPATH
  unset OOS_PYTHON_PATH

  export OOS_STAGE_MODEL_TO_TMP="${OOS_STAGE_MODEL_TO_TMP:-1}"
  export OOS_STAGE_MODEL_DIR="${OOS_STAGE_MODEL_DIR:-$SPATIAL_MLLM_CKPT}"
  export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"
  export DS_BUILD_AIO="${DS_BUILD_AIO:-0}"
  export DS_BUILD_FUSED_ADAM="${DS_BUILD_FUSED_ADAM:-0}"
  export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
  export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
  export OOS_SPATIAL_LOG_SAMPLES="${OOS_SPATIAL_LOG_SAMPLES:-1}"
fi

if [ "${OOS_MODEL:-}" = "sensenova_qwen" ]; then
  OOS_REPO_DIR="${REPO_DIR:-$(pwd)}"
  export OOS_VENV="${OOS_VENV:-${OOS_REPO_DIR}/.venv-cu128-home/bin/activate}"
  export SENSENOVA_QWEN_CKPT="${SENSENOVA_QWEN_CKPT:-$OOS_CHECKPOINT_DIR/SenseNova-SI-1.3-Qwen3-VL-8B}"
  export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchvision}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OOS_JOB_CACHE_ROOT/triton_cache}"
  export OOS_STAGE_MODEL_TO_TMP="${OOS_STAGE_MODEL_TO_TMP:-1}"
  export OOS_STAGE_MODEL_DIR="${OOS_STAGE_MODEL_DIR:-$SENSENOVA_QWEN_CKPT}"
fi

if [ "${OOS_MODEL:-}" = "cambrian_p" ]; then
  OOS_REPO_DIR="${REPO_DIR:-$(pwd)}"
  export OOS_VENV="${OOS_VENV:-${OOS_REPO_DIR}/.venv-cu128-home/bin/activate}"
  export CAMBRIAN_OVERLAY="${CAMBRIAN_OVERLAY:-$OOS_OVERLAY_ROOT/cambrian-cu128}"
  export CAMBRIAN_P_PATH="${CAMBRIAN_P_PATH:-$OOS_MODEL_SOURCE_DIR/cambrian-p}"
  export CAMBRIAN_P_VGGT_PATH="${CAMBRIAN_P_VGGT_PATH:-${CAMBRIAN_P_PATH}/vggt}"

  # Prepend in reverse priority order so the final search order is
  # Cambrian-P, VGGT, then the lightweight dependency overlay.
  for OOS_PYTHON_PATH in "$CAMBRIAN_OVERLAY" "$CAMBRIAN_P_VGGT_PATH" "$CAMBRIAN_P_PATH"; do
    case ":${PYTHONPATH:-}:" in
      *":${OOS_PYTHON_PATH}:"*) ;;
      *) PYTHONPATH="${OOS_PYTHON_PATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
  done
  export PYTHONPATH
  unset OOS_PYTHON_PATH
fi

# OOS task behavior.
export OOS_VENV="${OOS_VENV:-${REPO_DIR:-$(pwd)}/.venv-cu128-home/bin/activate}"
export OOS_DATASET_JSONL="${OOS_DATASET_JSONL:-$OOS_DATA_ROOT/vqa/sample_100_ordered_future_object_reference_next_movement_end.jsonl}"
export OOS_DEBUG_STEP="${OOS_DEBUG_STEP:-}"
export OOS_HISTORY_MODE="${OOS_HISTORY_MODE:-none}"          # none, gold, pred
export LMMS_EVAL_SHUFFLE_DOCS="${LMMS_EVAL_SHUFFLE_DOCS:-0}" # keep 0 for pred mode
export OOS_NO_VIDEO_INPUT="${OOS_NO_VIDEO_INPUT:-0}"
export OOS_CHAT_DEBUG="${OOS_CHAT_DEBUG:-1}" # set to 1 to print model prompt/media diagnostics
export OOS_VIDEO_CONTEXT="${OOS_VIDEO_CONTEXT:-last_frame}" # prefix or last_frame
export OOS_MARK_ANCHOR_OBJECT="${OOS_MARK_ANCHOR_OBJECT:-1}" 
# Position the dataset-provided target reference relative to the video: before or after.
export OOS_TARGET_REFERENCE_POSITION="${OOS_TARGET_REFERENCE_POSITION:-after}"

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
# export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchvision}"

# VLM-3R local source, dependency overlay, LoRA adapter, and base model.
export VLM3R_REPO="${VLM3R_REPO:-$OOS_MODEL_SOURCE_DIR/VLM-3R}"
export VLM3R_OVERLAY="${VLM3R_OVERLAY:-$OOS_OVERLAY_ROOT/vlm3r-cu128}"
export VLM3R_CKPT="${VLM3R_CKPT:-$HF_HOME/vlm-3r-llava-qwen2-lora}"
export VLM3R_BASE="${VLM3R_BASE:-$HF_HOME/LLaVA-NeXT-Video-7B-Qwen2}"
export VLM3R_CUT3R_WEIGHTS="${VLM3R_CUT3R_WEIGHTS:-$VLM3R_REPO/CUT3R/src/cut3r_512_dpt_4_64.pth}"
# Copy all VLM-3R checkpoints to the Slurm job's node-local 100 GB TMPDIR.
export VLM3R_STAGE_TO_TMP="${VLM3R_STAGE_TO_TMP:-1}"

# Keep TORCH_CUDNN_V8_API_DISABLED unset on Euler; enabling it makes Qwen3.5
# inference substantially slower.
export FFMPEG_ROOT="${FFMPEG_ROOT:-$OOS_VENV_ROOT/ffmpeg_env}"
export FFMPEG_PATH="${FFMPEG_PATH:-$FFMPEG_ROOT/bin/ffmpeg}"
case ":${LD_LIBRARY_PATH:-}:" in
  *":${FFMPEG_ROOT}/lib:"*) ;;
  *) export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}" ;;
esac
case ":${PATH:-}:" in
  *":${FFMPEG_ROOT}/bin:"*) ;;
  *) export PATH="${PATH}:${FFMPEG_ROOT}/bin" ;;
esac
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# export OOS_STEP23_EVAL_MODE=time_only
