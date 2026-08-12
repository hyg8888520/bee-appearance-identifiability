#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/h6_smoke.local.yaml}"
PYTHON_BIN="${2:-${BEEID_DINO_PYTHON:-}}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: pass the beeid-dino Python as argument 2" >&2
  exit 2
fi
CPU_THREADS="${BEEID_H6_CPU_THREADS:-16}"
if [[ ! "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BEEID_H6_CPU_THREADS must be a positive integer" >&2
  exit 2
fi
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -m beeid.cli h6-synthetic-smoke
"$PYTHON_BIN" -m beeid.cli h6-run-all \
  --config "$CONFIG" \
  --models resnet50 dinov3 \
  --confirm-full
