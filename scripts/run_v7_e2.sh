#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_BIN:=.venv/Scripts/python.exe}"
PYTHONPATH=src "$PYTHON_BIN" -m olikbochon.v7_e2 \
  --train-data "data/competition/dataset samples.json" \
  --retrieval-cache artifacts/v7/retrieval/official_retrieval.joblib \
  --checkpoint-dir data/models/banglabert-official-9ce791f \
  --checkpoint-revision 9ce791f330578f50da6bc52b54205166fb5d1c8c \
  --checkpoint-weight-sha256 9d33f519f42705d54e65fc1601644a6f4562c3462f96943b32b4184536130f98
