#!/bin/bash
set -euo pipefail

PROJECT_DIR=/work/courses/3dv/team1
CODE_DIR=$PROJECT_DIR/lmms-eval

cd "$CODE_DIR"

# Activate existing lmms-eval environment
source .venv/bin/activate

# Load .env if it exists
if [ -f "$CODE_DIR/.env" ]; then
  set -a
  source "$CODE_DIR/.env"
  set +a
fi

echo "Python: $(which python)"
python --version

# -------------------------------
# Caches / temp dirs
# -------------------------------
# Per-user tmp dir to avoid permission collisions on the shared project dir.
USER_TMPDIR="$PROJECT_DIR/tmp_${USER}"
mkdir -p "$USER_TMPDIR"
mkdir -p "$CODE_DIR/outputs/oos_ollama_login_sanity"

export TMPDIR="$USER_TMPDIR"
export LMMS_EVAL_DATASETS_CACHE="$USER_TMPDIR/lmms_eval_hf_datasets"

# Encode video frames as JPEG instead of the default PNG
export LMMS_IMAGE_ENCODE_FORMAT=JPEG

export TOKENIZERS_PARALLELISM=false
export FFMPEG_PATH=/work/courses/3dv/team1/ffmpeg-7.0.2-amd64-static/ffmpeg
export FORCE_QWENVL_VIDEO_READER=decord

# -------------------------------
# Ollama Cloud / LiteLLM
# -------------------------------
# Prefer OLLAMA_API_KEY if already set.
# If your .env uses SECRET_KEY for Ollama, this fallback uses it.
export OLLAMA_API_KEY=$OLLAMA_API_KEY
export OLLAMA_API_BASE="https://ollama.com"

if [ -z "${OLLAMA_API_KEY}" ]; then
  echo "ERROR: OLLAMA_API_KEY is not set."
  echo "Set it with: export OLLAMA_API_KEY=your_key"
  exit 1
fi

# Choose your Ollama Cloud model here.
# Replace this with an actual model from your Ollama Cloud account if needed.
# Available VL models on Ollama Cloud (per https://ollama.com/api/tags):
#   - qwen3-vl:235b-instruct  (~470 GB, MoE) -- too big / errors
#   - gemma3:27b              (~55 GB, multimodal)
#   - gemma3:12b              (~24 GB, multimodal) -- 12B param, fits the 10-20B target
#   - gemma3:4b               (~8.6 GB) -- below the range
# Note: qwen3-vl smaller tags (8b/30b-a3b/32b) are NOT hosted on Cloud, only the 235b.
# OLLAMA_MODEL=ollama_chat/qwen3-vl:235b-instruct
OLLAMA_MODEL=ollama_chat/qwen3-vl:235b-instruct # 12B multimodal alternative on Ollama Cloud.

# -------------------------------
# OOS sanity settings
# -------------------------------

# Video ablation switch 
export OOS_NO_VIDEO_INPUT="0"

# Use gold or none for generic LiteLLM sanity check.
# Do NOT use pred unless you have a custom LiteLLM backend with pred-history support.
export OOS_HISTORY_MODE="none"  # gold or none

#Comment this out to Evaluate single step if wanted ()
export OOS_DEBUG_STEP=5b

# Shuffle option (make sure it is set to "0" if using "pred" history mode to ensure alignment between predicted history and current evaluation)
export LMMS_EVAL_SHUFFLE_DOCS="0"

# Set to 1 to log previous-step QA history, model inputs, and outputs in the logs.
# Set to 0 to disable these debug logs.
export OOS_CHAT_DEBUG="1"

# Video dimensions
export OOS_VIDEO_WIDTH=448
export OOS_VIDEO_HEIGHT=448


# Preprocessing options (not needed if you already have preprocessed the videos beforehand)
export OOS_PREPROCESS_VIDEO="0"
export OOS_TARGET_FPS=1
export OOS_RESIZE_WIDTH=224
export OOS_RESIZE_HEIGHT=224

# Extra knobs for the improved chunk-evidence backend.
# Source this file or copy the exports into your existing run script.

export OOS_CHUNK_EVIDENCE="1"

# Global fallback chunking. Used by most non-event questions unless overridden below.
export OOS_CHUNK_SECONDS="60"
export OOS_CHUNK_OVERLAP_SECONDS="5"

# Step 2 / Step 3: precise timestamp + coordinate event scanning.
export OOS_EVENT_CHUNK_SECONDS="60"
export OOS_EVENT_CHUNK_OVERLAP_SECONDS="5"

# Step 5b / Step 5c: anchor relation / distance reasoning.
export OOS_SPATIAL_CHUNK_SECONDS="100"
export OOS_SPATIAL_CHUNK_OVERLAP_SECONDS="5"

export OOS_EVIDENCE_MAX_TOKENS="384"
export OOS_MAX_CHUNKS="200"

# History modes:
#   gold: previous dependency QA are gold/oracle, as provided by utils.py history_messages.
#   none: backend strips previous QA and keeps only system + current question.
#   pred: backend labels previous QA as predicted; requires your custom pipeline to put predicted history into doc_to_messages/history_messages.
export OOS_HISTORY_MODE="none"

# Keep order fixed if you later implement pred-history accumulation.
export LMMS_EVAL_SHUFFLE_DOCS="0"

echo "Running sanity check..."
echo "Model: $OLLAMA_MODEL"
echo "OOS_HISTORY_MODE=$OOS_HISTORY_MODE"
echo "OOS_NO_VIDEO_INPUT=$OOS_NO_VIDEO_INPUT"


python -m lmms_eval \
  --model litellm_chat \
  --model_args model="$OLLAMA_MODEL",api_key="$OLLAMA_API_KEY",base_url="$OLLAMA_API_BASE",video_fps=1,num_concurrent=1,timeout=120,max_retries=1 \
  --tasks oos_videoqa \
  --batch_size 1 \
  --log_samples \
  --output_path "$CODE_DIR/outputs/oos_ollama_login_sanity"

echo "Done. Outputs saved to:"
echo "$CODE_DIR/outputs/oos_ollama_login_sanity"