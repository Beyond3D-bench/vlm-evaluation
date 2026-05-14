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
# OLLAMA_MODEL=ollama_chat/qwen3-vl:235b-instruct
OLLAMA_MODEL=ollama_chat/qwen3-vl:30b-a3b-instruct # Alternative slightly smaller model here.

# -------------------------------
# OOS sanity settings
# -------------------------------

# Video ablation switch 
export OOS_NO_VIDEO_INPUT="0"

# Use gold or none for generic LiteLLM sanity check.
# Do NOT use pred unless you have a custom LiteLLM backend with pred-history support.
export OOS_HISTORY_MODE="gold"  # gold or none

#Comment this out to Evaluate single step if wanted ()
# export OOS_DEBUG_STEP=1

# Shuffle option (make sure it is set to "0" if using "pred" history mode to ensure alignment between predicted history and current evaluation)
export LMMS_EVAL_SHUFFLE_DOCS="0"

# Set to 1 to log previous-step QA history, model inputs, and outputs in the logs.
# Set to 0 to disable these debug logs.
export OOS_CHAT_DEBUG="1"

# Video dimensions
export OOS_VIDEO_WIDTH=224
export OOS_VIDEO_HEIGHT=224

# Debugging options (set to "1" to prompt the VLM to ouput reasoning instead of final answer directly)
export OOS_DEBUG_REASONING="0"
export OOS_DEBUG_EVAL="0"
export OOS_DEBUG_FAIL_ONLY="0"

# Preprocessing options (not needed if you already have preprocessed the videos beforehand)
export OOS_PREPROCESS_VIDEO="0"
export OOS_TARGET_FPS=1
export OOS_RESIZE_WIDTH=224
export OOS_RESIZE_HEIGHT=224

echo "Running sanity check..."
echo "Model: $OLLAMA_MODEL"
echo "OOS_HISTORY_MODE=$OOS_HISTORY_MODE"
echo "OOS_NO_VIDEO_INPUT=$OOS_NO_VIDEO_INPUT"


python -m lmms_eval \
  --model litellm_chat \
  --model_args model="$OLLAMA_MODEL",api_key="$OLLAMA_API_KEY",base_url="$OLLAMA_API_BASE",max_frames_num=768,num_concurrent=1,timeout=120,max_retries=1 \
  --tasks oos_videoqa \
  --batch_size 1 \
  --log_samples \
  --output_path "$CODE_DIR/outputs/oos_ollama_login_sanity"

echo "Done. Outputs saved to:"
echo "$CODE_DIR/outputs/oos_ollama_login_sanity"