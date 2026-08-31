#!/usr/bin/env bash
set -euo pipefail

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$LAUNCHER_DIR/.." && pwd)"

uv pip install \
  --target "$REPO_DIR/.venv-Phi4-overrides" \
  --no-deps \
  'transformers==4.48.2' \
  'tokenizers==0.21.0' \
  'peft==0.13.2' \
  'huggingface-hub==0.28.1'
