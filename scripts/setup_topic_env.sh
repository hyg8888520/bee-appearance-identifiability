#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 ENV_DIR [PROJECT_ROOT]" >&2
  exit 2
fi
ENV_DIR=$1
PROJECT_ROOT=${2:-$(cd "$(dirname "$0")/.." && pwd)}
python3.11 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$ENV_DIR/bin/python" -m pip install 'numpy>=1.26,<3' 'Pillow>=10,<13' 'PyYAML>=6,<7' ftfy omegaconf regex scikit-learn submitit termcolor torchmetrics yacs tabulate cloudpickle fvcore iopath
"$ENV_DIR/bin/python" -m pip install --no-deps -e "$PROJECT_ROOT"
"$ENV_DIR/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
