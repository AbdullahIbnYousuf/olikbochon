"""Generate the canonical self-contained Version 3 percent notebook."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

from olikbochon.v3_bundle import build_runtime_archive, encode_runtime_archive


ROOT = Path(__file__).resolve().parents[1]
PY_NOTEBOOK = ROOT / "notebooks" / "generated" / "banglabert_v3.py"
IPYNB_NOTEBOOK = ROOT / "notebooks" / "generated" / "banglabert_v3.ipynb"


def _wrapped_payload(payload: str) -> str:
    lines = textwrap.wrap(payload, width=100)
    return "\n".join(f'    "{line}"' for line in lines)


def generate() -> None:
    archive, digest, manifest = build_runtime_archive(ROOT)
    encoded = encode_runtime_archive(archive)
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
#     accelerator: gpu
#     internet: false
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BanglaBERT Version 3 — authenticated offline transformer baseline
#
# This notebook authenticates the private official snapshot, validates a self-contained runtime,
# audits labeled data without displaying rows, compares two same-backbone arms using frozen grouped
# folds, freezes the arm and threshold, trains once on all official labeled rows, and only then reads
# the competition test file for final inference.

# %% [markdown]
# ## 1. Offline and GPU contract
#
# Internet must remain disabled. The notebook performs no installs or downloads and requires a GPU.

# %%
from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# %% [markdown]
# ## 2. Materialize the synchronized embedded runtime
#
# The archive below is generated deterministically from canonical source modules. Its digest and
# per-file manifest are checked by local tests before this clean notebook is committed.

# %%
RUNTIME_ARCHIVE_SHA256 = "{digest}"
RUNTIME_FILE_MANIFEST = {json.dumps(manifest, sort_keys=True, indent=4)}
RUNTIME_ARCHIVE_B85 = (
{_wrapped_payload(encoded)}
)

runtime_bytes = base64.b85decode(RUNTIME_ARCHIVE_B85.encode("ascii"))
observed_runtime_digest = hashlib.sha256(runtime_bytes).hexdigest()
if observed_runtime_digest != RUNTIME_ARCHIVE_SHA256:
    raise RuntimeError("Embedded Version 3 runtime failed SHA-256 validation")
runtime_root = Path("/kaggle/working/olikbochon_v3_runtime")
runtime_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(runtime_bytes)) as runtime_zip:
    names = runtime_zip.namelist()
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise RuntimeError("Unsafe embedded runtime path")
    runtime_zip.extractall(runtime_root)
sys.path.insert(0, str(runtime_root))

# %% [markdown]
# ## 3. Execute the frozen experiment
#
# Startup first checks normalizer reference outputs and authenticates inputs. No real test row is
# loaded until arm selection, threshold selection, and final full-data training are frozen.
# Safe aggregate progress is printed for: data authentication/grouping, Stage A epochs, five fresh
# Arm A folds, five fresh Arm B folds, guarded arm/threshold selection, final training, submission
# validation, runtime, and memory. CUDA OOM restarts are complete phase restarts, never resumes.

# %%
from olikbochon.v3_kaggle import run_kaggle_v3  # noqa: E402

RUN_SUMMARY = run_kaggle_v3(Path("/kaggle/input"))

# %% [markdown]
# ## 4. Outputs
#
# The primary output is `/kaggle/working/submission.csv`. A fixed-0.50 reference is created only
# when the guarded deployed threshold differs from 0.50. The final private model is stored under
# `/kaggle/working/banglabert_v3_model/`, and aggregate metadata is in `v3_run_summary.json`.
'''
    PY_NOTEBOOK.write_text(notebook, encoding="utf-8")
    subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "jupytext"),
            "--sync",
            str(PY_NOTEBOOK),
        ],
        cwd=ROOT,
        env={**os.environ, "JUPYTER_DATA_DIR": "/tmp/olikbochon-jupyter-data"},
        check=True,
    )
    if not IPYNB_NOTEBOOK.is_file():
        raise RuntimeError("Jupytext did not generate the Version 3 notebook")


if __name__ == "__main__":
    generate()
