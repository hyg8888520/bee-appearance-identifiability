#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-}"
if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "Usage: bash scripts/monitor_h6.sh /path/to/h6-output" >&2
  exit 2
fi
for stage in oracle training tracking; do
  file="$OUTPUT_ROOT/logs/h6_${stage}_progress.json"
  echo "=== $stage ==="
  if [[ -f "$file" ]]; then
    cat "$file"
  else
    echo "status: not_started"
  fi
  echo
done
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits
fi
