#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIGS/H2.LOCAL.YAML" >&2
  exit 2
fi

# Every model/variant shard and observation-ID signature is validated before reuse.
python -m beeid.cli h2-orchestrate --config "$1" --confirm-full
