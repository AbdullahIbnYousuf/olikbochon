from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import nbformat

from olikbochon.v4_bundle import (
    build_v4_runtime_archive,
    canonical_v4_runtime_files,
    encode_v4_runtime_archive,
    materialize_v4_runtime_archive,
)


ROOT = Path(__file__).resolve().parents[1]
PY_NOTEBOOK = ROOT / "notebooks" / "generated" / "wikipedia_retrieval_v4a.py"
IPYNB_NOTEBOOK = ROOT / "notebooks" / "generated" / "wikipedia_retrieval_v4a.ipynb"


def runtime_source() -> str:
    return "\n".join(
        content.decode("utf-8")
        for name, content in canonical_v4_runtime_files(ROOT).items()
        if name.endswith(".py")
    )


def test_v4_runtime_archive_is_deterministic_and_complete() -> None:
    first = build_v4_runtime_archive(ROOT)
    second = build_v4_runtime_archive(ROOT)
    assert first == second
    files = canonical_v4_runtime_files(ROOT)
    assert {
        "olikbochon/v4_kaggle.py",
        "olikbochon/v4_wikipedia.py",
        "olikbochon/modeling.py",
        "olikbochon/submission.py",
    }.issubset(files)


def test_materialized_v4_runtime_imports_without_repository(tmp_path: Path) -> None:
    payload, digest, _ = build_v4_runtime_archive(ROOT)
    target = materialize_v4_runtime_archive(
        encode_v4_runtime_archive(payload), digest, tmp_path / "runtime"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(target)!r}); "
                "from olikbochon.v4_kaggle import run_kaggle_v4a; "
                "assert callable(run_kaggle_v4a)"
            ),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_generated_notebook_runtime_matches_source_archive() -> None:
    tree = ast.parse(PY_NOTEBOOK.read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith("RUNTIME_"):
                constants[target.id] = ast.literal_eval(node.value)
    archive, digest, manifest = build_v4_runtime_archive(ROOT)
    assert archive
    assert constants["RUNTIME_ARCHIVE_SHA256"] == digest
    assert constants["RUNTIME_FILE_MANIFEST"] == manifest


def test_v4_notebook_is_clean_valid_and_offline() -> None:
    notebook = nbformat.read(IPYNB_NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata["kaggle"]["internet"] is False
    assert notebook.metadata["kaggle"]["accelerator"] == "none"
    assert all(
        cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code"
    )
    assert all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code")


def test_v4_runtime_has_no_network_install_or_unsafe_test_display() -> None:
    source = runtime_source().lower()
    forbidden = (
        "requests.",
        "urllib.request",
        "kagglehub",
        "pip install",
        "!pip",
        "wget ",
        "curl ",
        "git clone",
        "openai",
        "print(test)",
        "print(test.head",
        "display(test",
        "print(submission)",
        "display(submission",
    )
    assert all(token not in source for token in forbidden)


def test_v4_outputs_include_aligned_oof_and_test_probabilities() -> None:
    source = runtime_source()
    assert "v4a_oof_probabilities.csv" in source
    assert "v4a_test_probabilities.csv" in source
    assert "validate_test_probability_artifact" in source
    assert "submitted_to_kaggle\": False" in source


def test_v4_summary_records_dynamic_wikipedia_observations() -> None:
    source = runtime_source()
    assert '"discovered_file_count"' in source
    assert '"discovered_aggregate_bytes"' in source
    assert '"deduplicated_path_count"' in source
    assert '"content_manifest_sha256"' not in source
