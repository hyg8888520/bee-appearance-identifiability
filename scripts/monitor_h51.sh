#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CONFIGS/H51.LOCAL.YAML BEEID_DINO_PYTHON [INTERVAL_SECONDS]" >&2
  exit 2
fi
CONFIG_PATH="$1"
PYTHON_BIN="$2"
INTERVAL_SECONDS="${3:-5}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "H5.1 Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
fi
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
OUTPUT_ROOT="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
from beeid.config import load_config
print(load_config(sys.argv[1]).paths.output_root)
PY
)"
while true; do
  clear 2>/dev/null || true
  date --iso-8601=seconds
  echo "H5.1 output: ${OUTPUT_ROOT}"
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
progress = root / "logs" / "h51_progress.json"
decision = root / "h51_decision.json"
if progress.is_file():
    value = json.loads(progress.read_text(encoding="utf-8"))
    keys = ("status", "completed_jobs", "reused_jobs", "total_jobs", "model", "variant", "video_id", "decision", "method_ready", "final_test_read")
    print(json.dumps({key: value.get(key) for key in keys if key in value}, indent=2))
else:
    print("status: not_started")
if decision.is_file():
    value = json.loads(decision.read_text(encoding="utf-8"))
    print(json.dumps({"diagnostic_decision": value.get("status"), "method_ready": value.get("method_ready"), "final_test_unlocked": value.get("final_test_unlocked")}, indent=2))
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
  fi
  sleep "${INTERVAL_SECONDS}"
done
