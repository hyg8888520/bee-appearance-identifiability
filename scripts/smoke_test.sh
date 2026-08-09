#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIGS/H1_SMOKE.LOCAL.YAML" >&2
  exit 2
fi
python -m beeid.cli synthetic-smoke
python -m beeid.cli orchestrate --config "$1"
