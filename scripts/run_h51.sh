#!/usr/bin/env bash
set -euo pipefail

CPU_THREAD_DEFAULT="${BEEID_H51_CPU_THREADS:-16}"
if [[ ! "${CPU_THREAD_DEFAULT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BEEID_H51_CPU_THREADS must be a positive integer when set." >&2
  exit 2
fi
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS="${CPU_THREAD_DEFAULT}"
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CONFIGS/H51.LOCAL.YAML [BEEID_DINO_PYTHON]" >&2
  exit 2
fi
PYTHON_BIN="${2:-${BEEID_DINO_PYTHON:-}}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Provide the beeid-dino interpreter as argument 2 or BEEID_DINO_PYTHON." >&2
  exit 2
fi
"${PYTHON_BIN}" -m beeid.cli h51-validate-protocol --protocol configs/h51_protocol.lock.yaml --checksum configs/h51_protocol.lock.sha256
"${PYTHON_BIN}" -m beeid.cli h51-run-all --config "$1" --models resnet50 dinov3 --confirm-full
