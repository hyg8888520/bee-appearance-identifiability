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
CPU_THREAD_DEFAULT="${BEEID_H5_CPU_THREADS:-16}"
if [[ ! "${CPU_THREAD_DEFAULT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BEEID_H5_CPU_THREADS must be a positive integer when set." >&2
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
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\n=== tracking / reporting ===")
tracking = root / "h5_logs" / "tracking_progress.json"
run = root / "h5_run_metadata.json"
if tracking.is_file():
    try:
        value = json.loads(tracking.read_text(encoding="utf-8"))
        keys = (
            "status", "completed_jobs", "total_jobs", "completed_frames", "total_frames",
            "model", "video_id", "variant", "frame", "frames_per_second", "eta_seconds",
            "implementation", "final_test_read",
        )
        print(json.dumps({key: value.get(key) for key in keys if key in value}, indent=2))
    except (OSError, json.JSONDecodeError) as error:
        print(f"tracking progress temporarily unreadable: {error}")
elif run.is_file():
    try:
        value = json.loads(run.read_text(encoding="utf-8"))
        print(json.dumps({"status": "reporting_or_completed", "run_status": value.get("status"), "final_test_read": value.get("final_test_read")}, indent=2))
    except (OSError, json.JSONDecodeError) as error:
        print(f"run metadata temporarily unreadable: {error}")
else:
    finals = [root / "h5_checkpoints" / model / "final.pt" for model in ("resnet50", "dinov3")]
    print("status: tracking_initialization_or_training" if not all(path.is_file() for path in finals) else "status: tracking_initialization")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
  fi
  sleep "${INTERVAL_SECONDS}"
done
