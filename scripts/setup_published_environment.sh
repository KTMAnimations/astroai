#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-published
.venv-published/bin/python -m pip install -r environment/published-requirements.lock.txt
git submodule update --init chronoflow
PYTHONPATH=src .venv-published/bin/python - <<'PY'
import torch, zuko
print('PyTorch',torch.__version__,'zuko',zuko.__version__)
PY
