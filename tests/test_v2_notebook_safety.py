from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "generated" / "tfidf_baseline_v2.py"
NOTEBOOK = ROOT / "notebooks" / "generated" / "tfidf_baseline_v2.ipynb"
FORBIDDEN_IMPORTS = {
    "requests",
    "urllib",
    "transformers",
    "huggingface_hub",
    "torch",
    "tensorflow",
    "openai",
    "anthropic",
}
FORBIDDEN_CALLS = {"from_pretrained", "pipeline", "system", "popen"}


def load_source_module():
    spec = importlib.util.spec_from_file_location("tfidf_baseline_v2_notebook", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safety_errors(source: str) -> list[str]:
    tree = ast.parse(source)
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            errors.extend(
                f"forbidden import {alias.name}"
                for alias in node.names
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in FORBIDDEN_IMPORTS or module.startswith("olikbochon"):
                errors.append(f"forbidden import {module}")
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in FORBIDDEN_CALLS:
                errors.append(f"forbidden call {name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if "://" in lowered:
                errors.append("executable URL")
    return errors


def test_v2_notebook_is_valid_clean_offline_and_paired() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata["kaggle"] == {"accelerator": "none", "internet": False}
    assert notebook.metadata["jupytext"]["formats"] == "ipynb,py:percent"
    assert all(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code")
    assert all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code")


def test_v2_notebook_has_no_network_api_model_or_repo_dependency() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert safety_errors(source) == []
    assert safety_errors(code) == []


def test_test_csv_loading_is_confined_to_kaggle_inference() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    read_csv_functions: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "read_csv"
            for child in ast.walk(node)
        ):
            read_csv_functions.append(node.name)
    assert read_csv_functions == ["kaggle_inference"]
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "kaggle_inference"
    ]
    assert len(calls) == 1
    assert "if running_on_kaggle:" in ast.unparse(main)


def test_v2_notebook_fingerprints_and_default_threshold_are_deterministic() -> None:
    module = load_source_module()
    assert module.text_fingerprint(" প্রশ্ন ", "[NULL]", " উত্তর ") == module.text_fingerprint(
        "প্রশ্ন", None, "উত্তর"
    )
    assert module.labeled_fingerprint("p", "", "r", 0) != module.labeled_fingerprint(
        "p", "", "r", 1
    )


def test_v2_notebook_removes_exact_public_overlap_from_lower_precedence_role() -> None:
    module = load_source_module()

    def fingerprinted(rows):
        frame = pd.DataFrame(
            rows, columns=["context", "prompt_bn", "response_bn", "label"]
        )
        keys = [
            (
                module.text_fingerprint(row.prompt_bn, row.context, row.response_bn),
                module.labeled_fingerprint(
                    row.prompt_bn, row.context, row.response_bn, row.label
                ),
            )
            for row in frame.itertuples(index=False)
        ]
        frame["text_fingerprint"] = [key[0] for key in keys]
        frame["labeled_row_fingerprint"] = [key[1] for key in keys]
        return frame

    train = fingerprinted([("", "shared", "x", 0), ("", "train", "x", 1)])
    validation = fingerprinted(
        [("", "validation", "x", 0), ("", "validation-1", "x", 1)]
    )
    official = fingerprinted([("", "shared", "x", 0), ("", "official", "x", 1)])
    train_clean, validation_clean, official_clean, removals = (
        module.deduplicate_exact_roles(train, validation, official)
    )
    assert len(train_clean) == 1
    assert len(validation_clean) == 2
    assert len(official_clean) == 2
    assert removals == {"public_train": 1, "public_validation": 0, "official": 0}


def test_original_starter_hash_remains_unchanged_in_v2() -> None:
    expected = (ROOT / "docs" / "original-notebook-sha256.txt").read_text(
        encoding="utf-8"
    ).split()[0]
    original = ROOT / "notebooks" / "original" / "starter-notebook-datathon.ipynb"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
