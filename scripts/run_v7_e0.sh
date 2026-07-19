#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_BIN:=.venv/Scripts/python.exe}"
: "${TRAIN_DATA:=data/competition/dataset samples.json}"
: "${WIKIPEDIA_ROOT:?Set WIKIPEDIA_ROOT to the verified bnwiki AA/AB/AC/AD root}"
PYTHONPATH=src "$PYTHON_BIN" -m olikbochon.v7_e0 \
  --train-data "$TRAIN_DATA" \
  --wikipedia-root "$WIKIPEDIA_ROOT" \
  --resource-name "Wikimedia bnwiki pages-articles" \
  --resource-version 20260701 \
  --resource-license "CC-BY-SA-4.0 and GFDL-1.3-or-later" \
  --resource-source-url "https://dumps.wikimedia.org/bnwiki/20260701/bnwiki-20260701-pages-articles.xml.bz2" \
  --resource-sha1 fdcf44a8ec36fc8a82b1b7b9f44878c4cb6dcba9 \
  --output-dir artifacts/v7/e0 \
  --folds-path artifacts/v7/folds/common_folds.csv
