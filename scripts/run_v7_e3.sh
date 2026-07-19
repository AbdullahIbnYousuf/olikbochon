#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_BIN:=.venv/Scripts/python.exe}"
PYTHONPATH=src "$PYTHON_BIN" -m olikbochon.v7_e3 \
  --e0-dir artifacts/v7/e0 \
  --e1-dir artifacts/v7/e1 \
  --output-dir artifacts/v7/e3
