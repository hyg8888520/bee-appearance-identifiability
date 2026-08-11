#!/usr/bin/env bash
set -euo pipefail

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=4
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CONFIGS/H5_SMOKE.LOCAL.YAML [BEEID_DINO_PYTHON]" >&2
  exit 2
fi
PYTHON_BIN="${2:-${BEEID_DINO_PYTHON:-}}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Provide the beeid-dino interpreter as argument 2 or BEEID_DINO_PYTHON." >&2
  exit 2
fi
"${PYTHON_BIN}" -m beeid.cli h5-synthetic-smoke
"${PYTHON_BIN}" -m beeid.cli h5-run-all --config "$1" --models resnet50 dinov3
