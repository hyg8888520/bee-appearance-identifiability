#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CONFIGS/H5.LOCAL.YAML BEEID_DINO_PYTHON [INTERVAL_SECONDS]" >&2
  exit 2
fi

CONFIG_PATH="$1"
PYTHON_BIN="$2"
INTERVAL_SECONDS="${3:-5}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "H5 Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=4
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
  echo "H5 output: ${OUTPUT_ROOT}"
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for model in ("resnet50", "dinov3"):
    print(f"\n=== {model} ===")
    training = root / "h5_logs" / f"{model}_progress.json"
    cache = root / "h5_logs" / f"{model}_cache_progress.json"
    selected = training if training.is_file() else cache
    if selected.is_file():
        try:
            print(json.dumps(json.loads(selected.read_text(encoding="utf-8")), indent=2))
        except (OSError, json.JSONDecodeError) as error:
            print(f"progress temporarily unreadable: {error}")
    else:
        print("status: not_started")
    partial = root / "h5_checkpoints" / model / "training-progress.pt"
    final = root / "h5_checkpoints" / model / "final.pt"
    print(f"partial_checkpoint: {partial.is_file()}")
    print(f"final_checkpoint: {final.is_file()}")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
  fi
  sleep "${INTERVAL_SECONDS}"
done
