#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIGS/H1.LOCAL.YAML" >&2
  exit 2
fi
python -m beeid.cli orchestrate --config "$1" --confirm-full
