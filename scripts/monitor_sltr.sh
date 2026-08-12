#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CONFIGS/SLTR.LOCAL.YAML BEEID_DINO_PYTHON [INTERVAL_SECONDS]" >&2; exit 2
fi
CONFIG_PATH="$1"; PYTHON_BIN="$2"; INTERVAL_SECONDS="${3:-5}"
if [[ ! -x "${PYTHON_BIN}" || ! "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Require an executable interpreter and positive interval." >&2; exit 2
fi
OUTPUT_ROOT="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
from beeid.config import load_config
print(load_config(sys.argv[1]).paths.output_root)
PY
)"
while true; do
  clear 2>/dev/null || true
  date --iso-8601=seconds
  for file in "${OUTPUT_ROOT}"/logs/sltr_*_progress.json "${OUTPUT_ROOT}/sltr_decision.json"; do
    [[ -f "${file}" ]] && sed -n '1,120p' "${file}"
  done
  sleep "${INTERVAL_SECONDS}"
done
