#!/bin/bash
# Direct-run version for an already allocated GB10 interactive node.
# Run with:
#   bash run_oos_videoqa_direct_gb10.sh
#
# Do NOT submit this version with sbatch.
#
# Original sbatch options kept for reference only:
# #SBATCH --job-name=oos_videoqa
# #SBATCH --account=3dv
# #SBATCH --time=00:60:00
# #SBATCH --gpus=gb10:1
# #SBATCH --output=/work/courses/3dv/team1/lmms-eval/logs/oos_videoqa_%j.out
# #SBATCH --error=/work/courses/3dv/team1/lmms-eval/logs/oos_videoqa_%j.err

set -euo pipefail

PROJECT_DIR=/work/courses/3dv/team1
CODE_DIR=$PROJECT_DIR/lmms-eval

mkdir -p "$CODE_DIR/logs"
mkdir -p "$CODE_DIR/outputs/oos_videoqa"
mkdir -p "$PROJECT_DIR/hf_cache/datasets"
mkdir -p "$PROJECT_DIR/tmp"

cd "$CODE_DIR"

# Activate GB10/ARM virtual environment created on this node
source "$CODE_DIR/llava15_env_gb10/bin/activate"
unset LD_LIBRARY_PATH


set -a
source "$CODE_DIR/.env"
set +a

echo "Host: $(hostname)"
echo "Arch: $(uname -m)"
echo "Python: $(which python)"
python --version
nvidia-smi || true

# Optional sanity check: GB10 should be ARM/aarch64
python - <<'PY'
import platform, sys
print("Python executable:", sys.executable)
print("Platform machine:", platform.machine())
PY

# ===== HuggingFace cache =====
export HF_HOME=$PROJECT_DIR/hf_cache
export TRANSFORMERS_CACHE=$PROJECT_DIR/hf_cache
export HF_DATASETS_CACHE=$PROJECT_DIR/hf_cache/datasets
export LMMS_EVAL_DATASETS_CACHE=$PROJECT_DIR/hf_cache/datasets
export LMMS_EVAL_CACHE=$PROJECT_DIR/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$SECRET_KEY

# WARNING:
# Your old ffmpeg path contains "amd64", which may not work on GB10 ARM.
# Prefer system ffmpeg if available. Otherwise set this to an ARM-compatible ffmpeg.
if command -v ffmpeg >/dev/null 2>&1; then
  export FFMPEG_PATH="$(command -v ffmpeg)"
else
  echo "Warning: system ffmpeg not found. Your old amd64 ffmpeg may not work on GB10."
  export FFMPEG_PATH=/work/courses/3dv/team1/ffmpeg-7.0.2-amd64-static/ffmpeg
fi

echo "FFMPEG_PATH=$FFMPEG_PATH"

export FORCE_QWENVL_VIDEO_READER=opencv

# Avoid temp writes going to /tmp
export TMPDIR=$PROJECT_DIR/tmp
mkdir -p "$TMPDIR"

export OOS_VIDEO_CACHE_DIR=$PROJECT_DIR/tmp/oos_video_cache_168
mkdir -p "$OOS_VIDEO_CACHE_DIR"

# Step 2 and 3 tolerance
export OOS_TIME_TOLERANCE_SEC=3.0
export OOS_COORD_TOLERANCE_NORM=0.2

# Video ablation switch
export OOS_NO_VIDEO_INPUT="0"

# Video dimensions
export OOS_VIDEO_WIDTH=224
export OOS_VIDEO_HEIGHT=224

# History mode: "gold", "none", "pred"
export OOS_HISTORY_MODE="none"

# Uncomment to evaluate single step
# export OOS_DEBUG_STEP=2

# Keep 0 if using pred history mode
export LMMS_EVAL_SHUFFLE_DOCS="0"

# Debug logs
export OOS_CHAT_DEBUG="1"

# Debugging options
export OOS_DEBUG_REASONING="0"
export OOS_DEBUG_EVAL="0"
export OOS_DEBUG_FAIL_ONLY="0"

# Preprocessing options
export OOS_PREPROCESS_VIDEO="0"
export OOS_TARGET_FPS=1
export OOS_RESIZE_WIDTH=224
export OOS_RESIZE_HEIGHT=224


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GB10: free buffer cache so more shared memory is available for GPU use
/usr/bin/drop-caches || true

# Direct run: no srun here, because you are already inside the GB10 allocation
srun python -m lmms_eval \
  --model=llava_onevision1_5_chat_fixed \
  --model_args=pretrained=lmms-lab/LLaVA-OneVision-1.5-8B-Instruct,fps=1,max_num_frames=768,min_pixels=28824,max_pixels=28824,load_in_4bit=True \
  --tasks=oos_videoqa \
  --batch_size=1 \
  --log_samples \
  --output_path="$PROJECT_DIR/lmms-eval/outputs/oos_videoqa/llava_onevision1_5_chat_fixed"