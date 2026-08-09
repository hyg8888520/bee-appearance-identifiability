#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIG" >&2
  exit 2
fi
python -m beeid.cli server-check --config "$1"
