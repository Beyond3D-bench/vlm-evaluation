#!/bin/bash
#SBATCH --job-name=oos_videoqa
#SBATCH --account=3dv
#SBATCH --time=03:00:00
#SBATCH --output=/work/courses/3dv/team1/lmms-eval/logs/oos_videoqa_%j.out
#SBATCH --error=/work/courses/3dv/team1/lmms-eval/logs/oos_videoqa_%j.err

set -euo pipefail

PROJECT_DIR=/work/courses/3dv/team1
CODE_DIR=$PROJECT_DIR/lmms-eval

# Create shared directories if not exist
mkdir -p $PROJECT_DIR/lmms-eval/logs
mkdir -p $PROJECT_DIR/lmms-eval/outputs/oos_videoqa
mkdir -p $PROJECT_DIR/hf_cache/datasets

cd $CODE_DIR

# Activate virtual environment
source .venv/bin/activate

set -a
source "$CODE_DIR/.env"
set +a

echo "Python: $(which python)"
python --version
nvidia-smi || true

# ===== HuggingFace cache (NOW in course directory) =====
export HF_HOME=$PROJECT_DIR/hf_cache
export TRANSFORMERS_CACHE=$PROJECT_DIR/hf_cache
export HF_DATASETS_CACHE=$PROJECT_DIR/hf_cache/datasets
export LMMS_EVAL_DATASETS_CACHE=$PROJECT_DIR/hf_cache/datasets
export LMMS_EVAL_CACHE=$PROJECT_DIR/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$SECRET_KEY
export FFMPEG_PATH=/work/courses/3dv/team1/ffmpeg-7.0.2-amd64-static/ffmpeg

export FORCE_QWENVL_VIDEO_READER=decord

# Avoid temp writes going to /tmp (optional but safer)
export TMPDIR=$PROJECT_DIR/tmp
mkdir -p $TMPDIR

# If your utils.py supports remapping video paths by basename, keep this:
# export OOS_VIDEO_BASE_DIR=$CODE_DIR/videos

# step 2 (last visible) and 3 (last placement) tolerance
export OOS_TIME_TOLERANCE_SEC=3.0
export OOS_COORD_TOLERANCE_NORM=0.2

# Video ablation switch 
export OOS_NO_VIDEO_INPUT="1"

# History mode: "gold" uses gold history, "none" uses no history, "pred" uses predicted history (if available)
export OOS_HISTORY_MODE="none"  # "gold", "none", "pred"

#Comment this out to Evaluate single step if wanted ()
export OOS_DEBUG_STEP=1

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

export CUDA_LAUNCH_BLOCKING=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# srun python -m lmms_eval \
#   --model qwen2_5_vl \
#   --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
#   --tasks oos_videoqa \
#   --batch_size 1 \
#   --log_samples \
#   --output_path "$PROJECT_DIR/lmms-eval/outputs/oos_videoqa"

srun python -m lmms_eval \
  --model qwen3_vl_chat_fixed \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct,fps=1,max_num_frames=800,min_pixels=38416,max_pixels=38416 \
  --tasks oos_videoqa \
  --batch_size 1 \
  --log_samples \
  --output_path "$PROJECT_DIR/lmms-eval/outputs/oos_videoqa"

# srun python -m lmms_eval \
#   --model qwen3_5 \
#   --model_args pretrained=Qwen/Qwen3.5-2B,fps=1,max_num_frames=800,min_pixels=38416,max_pixels=38416,enable_thinking=False \
#   --tasks oos_videoqa \
#   --batch_size 1 \
#   --log_samples \
#   --output_path "$PROJECT_DIR/lmms-eval/outputs/oos_videoqa"



# srun python -m lmms_eval \
#   --model llava_vid \
#   --model_args pretrained=liuhaotian/llava-v1.5-7bt,max_frames_num=294 \
#   --tasks oos_videoqa \
#   --batch_size 1 \
#   --log_samples \
#   --output_path "$PROJECT_DIR/lmms-eval/outputs/oos_videoqa"