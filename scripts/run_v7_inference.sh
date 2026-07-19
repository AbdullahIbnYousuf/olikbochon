#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_BIN:=.venv/Scripts/python.exe}"
: "${V7_TEST_DATA:?Set V7_TEST_DATA to the local competition test CSV or JSON}"
PYTHONPATH=src "$PYTHON_BIN" inference/run_v7_inference.py \
  --train-data "data/competition/dataset samples.json" \
  --test-data "$V7_TEST_DATA" \
  --wikipedia-root "data/external/wikimedia/bnwiki-20260701-extracted"
