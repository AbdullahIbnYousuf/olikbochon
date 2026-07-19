#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_BIN:=.venv/Scripts/python.exe}"
: "${TRAIN_DATA:=data/competition/dataset samples.json}"
PYTHONPATH=src "$PYTHON_BIN" -m olikbochon.v7_e1 \
  --train-data "$TRAIN_DATA" \
  --output-dir artifacts/v7/e1 \
  --folds-path artifacts/v7/folds/common_folds.csv
