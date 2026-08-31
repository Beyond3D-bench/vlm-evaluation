#!/usr/bin/env bash
# Reuse the CUDA/PyTorch environment while selecting the dependency versions
# required by microsoft/Phi-4-multimodal-instruct.
PHI4_ENV_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PHI4_ENV_REPO_DIR/.venv-cu128-home/bin/activate"
if [ ! -d "$PHI4_ENV_REPO_DIR/.venv-Phi4-overrides/transformers" ]; then
  echo "Missing Phi-4 compatibility packages. Run: bash launchers/setup_phi4_env.sh" >&2
  return 1
fi
export PYTHONPATH="$PHI4_ENV_REPO_DIR/.venv-Phi4-overrides${PYTHONPATH:+:$PYTHONPATH}"
unset PHI4_ENV_REPO_DIR
