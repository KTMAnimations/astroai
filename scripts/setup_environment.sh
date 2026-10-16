#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements.lock.txt
.venv/bin/python -m pip install -e . --no-deps
mkdir -p .venv/julia_env
cp environment/JuliaProject.toml .venv/julia_env/Project.toml
cp environment/JuliaManifest.toml .venv/julia_env/Manifest.toml
.venv/bin/python - <<'PY'
import juliapkg
juliapkg.require_julia("=1.13.2")
import pysr
from juliacall import Main as jl
print("PySR", pysr.__version__, "Julia", jl.seval("VERSION"))
PY
