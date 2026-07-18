#!/usr/bin/env python3
"""Generate the canonical self-contained Version 4-A percent notebook."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

from olikbochon.v4_bundle import build_v4_runtime_archive, encode_v4_runtime_archive


ROOT = Path(__file__).resolve().parents[1]
PY_NOTEBOOK = ROOT / "notebooks" / "generated" / "wikipedia_retrieval_v4a.py"
IPYNB_NOTEBOOK = ROOT / "notebooks" / "generated" / "wikipedia_retrieval_v4a.ipynb"


def _wrapped_payload(payload: str) -> str:
    return "\n".join(f'    "{line}"' for line in textwrap.wrap(payload, width=100))


def generate() -> None:
    archive, digest, manifest = build_v4_runtime_archive(ROOT)
    encoded = encode_v4_runtime_archive(archive)
    notebook = f'''# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kaggle:
#     accelerator: none
#     internet: false
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Version 4-A — clean Bengali Wikipedia retrieval reproduction
#
# This notebook independently reproduces the public Version 4 lexical/Wikipedia method using only
# the official competition input and pinned `abyaadrafid/bnwiki` version 1. It reports the original
# same-OOF cutoff-tuning estimate separately from nested duplicate-aware grouped validation.
# No Kaggle submission is made by this notebook.

# %% [markdown]
# ## Input and safety contract
#
# Attach the official competition input and `abyaadrafid/bnwiki`. Keep internet off. The notebook
# authenticates official hashes and the complete Wikipedia path-and-content manifest before parsing.
# It never prints test text, IDs, probabilities, retrieved passages, or individual predictions.

# %%
from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% [markdown]
# ## Materialize the synchronized embedded runtime

# %%
RUNTIME_ARCHIVE_SHA256 = "{digest}"
RUNTIME_FILE_MANIFEST = {json.dumps(manifest, sort_keys=True, indent=4)}
RUNTIME_ARCHIVE_B85 = (
{_wrapped_payload(encoded)}
)

runtime_bytes = base64.b85decode(RUNTIME_ARCHIVE_B85.encode("ascii"))
if hashlib.sha256(runtime_bytes).hexdigest() != RUNTIME_ARCHIVE_SHA256:
    raise RuntimeError("Embedded Version 4-A runtime failed SHA-256 validation")
runtime_root = Path("/kaggle/working/olikbochon_v4a_runtime")
runtime_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(runtime_bytes)) as runtime_zip:
    names = runtime_zip.namelist()
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise RuntimeError("Unsafe embedded runtime path")
    runtime_zip.extractall(runtime_root)
sys.path.insert(0, str(runtime_root))

# %% [markdown]
# ## Run the frozen reproduction
#
# The official labels are evaluated before the authenticated test is loaded. Final outputs are a
# validated candidate submission, honest grouped OOF probabilities, aligned test probabilities,
# and an aggregate-only run summary. These are Kaggle runtime artifacts and must not be committed.

# %%
from olikbochon.v4_kaggle import run_kaggle_v4a  # noqa: E402

RUN_SUMMARY = run_kaggle_v4a(Path("/kaggle/input"), Path("/kaggle/working"))

# %% [markdown]
# ## Expected outputs
#
# - `/kaggle/working/submission.csv` — validated candidate only; do not submit yet.
# - `/kaggle/working/v4a_oof_probabilities.csv` — honest grouped OOF probabilities.
# - `/kaggle/working/v4a_test_probabilities.csv` — ID-aligned label-1 probabilities.
# - `/kaggle/working/v4a_run_summary.json` — aggregate metrics and configuration only.
'''
    PY_NOTEBOOK.write_text(notebook, encoding="utf-8")
    subprocess.run(
        [str(ROOT / ".venv" / "bin" / "jupytext"), "--sync", str(PY_NOTEBOOK)],
        cwd=ROOT,
        env={**os.environ, "JUPYTER_DATA_DIR": "/tmp/olikbochon-jupyter-data"},
        check=True,
    )
    if not IPYNB_NOTEBOOK.is_file():
        raise RuntimeError("Jupytext did not generate the Version 4-A notebook")


if __name__ == "__main__":
    generate()
