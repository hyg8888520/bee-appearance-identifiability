#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CONFIGS/H41.LOCAL.YAML [BEEID_DINO_PYTHON]" >&2
  exit 2
fi

PYTHON_BIN="${2:-${BEEID_DINO_PYTHON:-}}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Provide the beeid-dino interpreter as argument 2 or BEEID_DINO_PYTHON." >&2
  exit 2
fi

"${PYTHON_BIN}" -m beeid.cli h41-validate-protocol \
  --protocol configs/h41_protocol.lock.yaml \
  --checksum configs/h41_protocol.lock.sha256
"${PYTHON_BIN}" -m beeid.cli h41-run-all \
  --config "$1" \
  --models resnet50 dinov3 \
  --confirm-full
