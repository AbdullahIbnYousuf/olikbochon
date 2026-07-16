from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import nbformat
import numpy as np
import pandas as pd

from olikbochon.metrics import (
    CLASS0_F1_OOF_EXPERIMENTAL,
    DEFAULT_THRESHOLD_STRATEGY,
    MACRO_F1_OOF,
    prediction_collapse_warning,
    select_threshold,
)
from olikbochon.preprocessing import build_marked_text, normalize_context, normalize_text
from olikbochon.submission import build_submission


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCE = ROOT / "notebooks" / "generated" / "tfidf_baseline_v1.py"
NOTEBOOK = ROOT / "notebooks" / "generated" / "tfidf_baseline_v1.ipynb"

FORBIDDEN_IMPORTS = {
    "requests",
    "urllib",
    "transformers",
    "huggingface_hub",
    "openai",
    "anthropic",
    "google.generativeai",
}
FORBIDDEN_CALL_NAMES = {"pipeline", "from_pretrained", "system", "popen"}


def load_notebook_source_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tfidf_baseline_v1_notebook", NOTEBOOK_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def executable_safety_errors(source: str) -> list[str]:
    tree = ast.parse(source)
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    errors.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in FORBIDDEN_IMPORTS:
                errors.append(f"forbidden import: {module}")
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in FORBIDDEN_CALL_NAMES:
                errors.append(f"forbidden call: {name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if "://" in lowered:
                errors.append("executable URL")
            if "data/public-20k" in lowered:
                errors.append("public-data reference")
            if "data/competition/test set.csv" in lowered:
                errors.append("local real-test path")
    return errors


def test_notebook_is_clean_valid_and_offline() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert executable_safety_errors(code) == []
    assert all(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code")
    assert all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code")
    assert notebook.metadata["kaggle"]["internet"] is False
    assert notebook.metadata["kaggle"]["accelerator"] == "none"


def test_percent_source_has_no_forbidden_executable_behavior() -> None:
    assert executable_safety_errors(NOTEBOOK_SOURCE.read_text(encoding="utf-8")) == []


def test_notebook_and_src_behavior_match_on_synthetic_values() -> None:
    notebook_module = load_notebook_source_module()
    values = [None, "[NULL]", " Cafe\u0301  ", " বাংলা\tলেখা "]
    for value in values:
        assert notebook_module.normalize_text(value) == normalize_text(value)
        assert notebook_module.normalize_context(value) == normalize_context(value)

    args = (" প্রশ্ন ", "[NULL]", " উত্তর ")
    assert notebook_module.build_marked_text(*args) == build_marked_text(*args)

    table = pd.DataFrame(
        [
            {"threshold": 0.4, "f1_label0": 0.8, "macro_f1": 0.7},
            {"threshold": 0.6, "f1_label0": 0.8, "macro_f1": 0.7},
        ]
    )
    assert notebook_module.DEFAULT_THRESHOLD_STRATEGY == DEFAULT_THRESHOLD_STRATEGY
    assert notebook_module.select_threshold(table, MACRO_F1_OOF) == select_threshold(
        table, MACRO_F1_OOF
    )
    assert notebook_module.select_threshold(
        table, CLASS0_F1_OOF_EXPERIMENTAL
    ) == select_threshold(table, CLASS0_F1_OOF_EXPERIMENTAL)
    assert notebook_module.prediction_collapse_warning(
        [0] * 19 + [1], name="parity"
    ) == prediction_collapse_warning([0] * 19 + [1], name="parity")

    ids = pd.Series([3, 4])
    labels = np.array([0, 1], dtype=np.int64)
    expected = build_submission(ids, labels)
    observed = notebook_module.build_submission(ids, labels)
    pd.testing.assert_frame_equal(observed, expected)


def test_notebook_labels_experimental_output_and_default() -> None:
    notebook_module = load_notebook_source_module()
    assert notebook_module.DEFAULT_THRESHOLD_STRATEGY == "macro_f1_oof"
    filename = notebook_module.SUBMISSION_FILENAMES["class0_f1_oof_experimental"]
    assert filename == "submission_class0_experimental.csv"
    assert "should not be submitted unless the organizers explicitly confirm" in (
        notebook_module.CLASS0_EXPERIMENTAL_WARNING
    )


def test_original_starter_notebook_hash_is_unchanged() -> None:
    expected_line = (ROOT / "docs" / "original-notebook-sha256.txt").read_text(
        encoding="utf-8"
    )
    expected_hash = expected_line.split()[0]
    original = ROOT / "notebooks" / "original" / "starter-notebook-datathon.ipynb"
    observed_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    assert observed_hash == expected_hash
