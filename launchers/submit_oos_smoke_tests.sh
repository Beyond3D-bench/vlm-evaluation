#!/usr/bin/env bash
# Submit OOS smoke tests from a Slurm login node.
# examples:
#   launchers/submit_oos_smoke_tests.sh --qwen3-6 --llava --limit 1
#   launchers/submit_oos_smoke_tests.sh --qwen3-6 --dry-run

set -euo pipefail

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$LAUNCHER_DIR/.." && pwd)"
SLURM_LAUNCHER="$LAUNCHER_DIR/slurm_oos_eval.sh"
OOS_STORAGE_ROOT="${OOS_STORAGE_ROOT:-$HOME/scratch}"

DEFAULT_MODELS=(
  qwen3_6
  qwen3_vl
  llava
  internvl
  phi4
  vlm3r
  cambrian_p
  spatial_mllm
  sensenova_internvl
  sensenova_qwen
)

LIMIT="${OOS_LIMIT:-2}"
DRY_RUN=0
SELECTED_MODELS=()

usage() {
  cat <<'EOF'
Usage: launchers/submit_oos_smoke_tests.sh [model flags] [options]

With no model flags, all smoke tests are submitted.

Model flags:
  --qwen3-6
  --qwen3-vl
  --llava
  --internvl
  --phi4
  --vlm3r
  --cambrian-p
  --spatial-mllm
  --sensenova-internvl
  --sensenova-qwen
  --all

Options:
  --limit N    Number of evaluation samples per job (default: 2)
  --dry-run    Print sbatch commands without submitting jobs
  -h, --help   Show this help

Examples:
  launchers/submit_oos_smoke_tests.sh --qwen3-6
  launchers/submit_oos_smoke_tests.sh --qwen3-vl --llava --limit 1
  launchers/submit_oos_smoke_tests.sh --dry-run
EOF
}

add_model() {
  local candidate="$1"
  local existing
  for existing in "${SELECTED_MODELS[@]:-}"; do
    if [ "$existing" = "$candidate" ]; then
      return
    fi
  done
  SELECTED_MODELS+=("$candidate")
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --qwen3-6) add_model qwen3_6 ;;
    --qwen3-vl) add_model qwen3_vl ;;
    --llava) add_model llava ;;
    --internvl) add_model internvl ;;
    --phi4) add_model phi4 ;;
    --vlm3r) add_model vlm3r ;;
    --cambrian-p) add_model cambrian_p ;;
    --spatial-mllm) add_model spatial_mllm ;;
    --sensenova-internvl) add_model sensenova_internvl ;;
    --sensenova-qwen) add_model sensenova_qwen ;;
    --all)
      SELECTED_MODELS=("${DEFAULT_MODELS[@]}")
      ;;
    --limit)
      if [ "$#" -lt 2 ]; then
        echo "--limit requires an integer value." >&2
        exit 2
      fi
      LIMIT="$2"
      shift
      ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "--limit must be a positive integer, got: $LIMIT" >&2
  exit 2
fi

if [ "${#SELECTED_MODELS[@]}" -eq 0 ]; then
  SELECTED_MODELS=("${DEFAULT_MODELS[@]}")
fi

cd "$REPO_DIR"

submit_model() {
  local model="$1"
  local -a command=(env -u OOS_VENV "OOS_MODEL=$model" "OOS_LIMIT=$LIMIT")

  case "$model" in
    vlm3r)
      if [ -z "${VLM3R_REPO:-}" ]; then
        echo "VLM-3R requires VLM3R_REPO=/path/to/VLM-3R before submission." >&2
        return 1
      fi
      command+=("VLM3R_REPO=$VLM3R_REPO")
      ;;
    cambrian_p)
      command=(
        env
        "OOS_MODEL=$model"
        "OOS_LIMIT=$LIMIT"
        "OOS_VENV=${OOS_CAMBRIAN_VENV:-$OOS_STORAGE_ROOT/venvs/oos_vlm_evaluation-cu124-cambrianp/bin/activate}"
      )
      ;;
  esac

  command+=(sbatch "$SLURM_LAUNCHER")

  printf 'Submitting %-22s ' "$model"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '\n  '
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

for model in "${SELECTED_MODELS[@]}"; do
  submit_model "$model"
done
