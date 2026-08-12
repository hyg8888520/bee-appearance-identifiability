#!/usr/bin/env bash
set -euo pipefail

# H6 checkpoints include the exact next batch, optimizer, GradScaler, and RNG state.
exec "$(dirname "$0")/run_h6.sh" "${1:-configs/h6.local.yaml}" "${2:-${BEEID_DINO_PYTHON:-}}"
