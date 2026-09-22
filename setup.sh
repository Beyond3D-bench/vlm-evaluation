#!/usr/bin/env bash
# Prepare selected models on a machine with Internet access.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source launchers/oos_env.sh
# Preparation is connected even when evaluation jobs are configured offline.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
exec python3 tools/setup_models.py "$@"
