#!/usr/bin/env bash
# One-shot setup + verification on a Linux CUDA server (A800/H800).
#   bash setup_a800.sh            # creates .venv, installs deps, runs tools/check_env.py
#   CU=cu128 bash setup_a800.sh   # pick another CUDA wheel tag (torch 2.14.1 exists for cu126)
set -euo pipefail
cd "$(dirname "$0")"
CU="${CU:-cu126}"
PY="${PY:-python3.12}"

if command -v uv >/dev/null 2>&1; then
  uv venv -p 3.12 .venv
  uv pip install -p .venv torch==2.14.1 --index-url "https://download.pytorch.org/whl/${CU}"
  uv pip install -p .venv mujoco==3.17.0 numpy==2.5.4 imageio==2.38.1 imageio-ffmpeg==0.6.0
else
  "$PY" -m venv .venv
  .venv/bin/pip install -q torch==2.14.1 --index-url "https://download.pytorch.org/whl/${CU}"
  .venv/bin/pip install -q mujoco==3.17.0 numpy==2.5.4 imageio==2.38.1 imageio-ffmpeg==0.6.0
fi

# headless rendering needs EGL (libegl1 / libgl1 on Debian/Ubuntu); training does not
.venv/bin/python tools/check_env.py
