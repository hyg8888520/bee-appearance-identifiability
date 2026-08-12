#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIGS/H2_SMOKE.LOCAL.YAML" >&2
  exit 2
fi

python -m beeid.cli h2-synthetic-smoke
python -m beeid.cli h2-orchestrate --config "$1"
