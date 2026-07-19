#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_BIN:=.venv/Scripts/python.exe}"
PYTHONPATH=src "$PYTHON_BIN" -m olikbochon.v7_e4 \
  --train-data "data/competition/dataset samples.json"
