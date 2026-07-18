from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import nbformat

from olikbochon.v3_bundle import (
    build_runtime_archive,
    canonical_runtime_files,
    encode_runtime_archive,
    materialize_runtime_archive,
)


ROOT = Path(__file__).resolve().parents[1]
PY_NOTEBOOK = ROOT / "notebooks" / "generated" / "banglabert_v3.py"
IPYNB_NOTEBOOK = ROOT / "notebooks" / "generated" / "banglabert_v3.ipynb"


def _notebook_constants() -> dict[str, object]:
    tree = ast.parse(PY_NOTEBOOK.read_text(encoding="utf-8"))
    constants: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith("RUNTIME_"):
                constants[target.id] = ast.literal_eval(node.value)
    return constants


def test_runtime_archive_is_deterministic_and_notebook_is_synchronized() -> None:
    first = build_runtime_archive(ROOT)
    second = build_runtime_archive(ROOT)
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first[0])) as archive:
        entries = archive.infolist()
    assert [entry.filename for entry in entries] == sorted(first[2])
    assert all(entry.create_system == 3 for entry in entries)
    assert all(entry.date_time == (1980, 1, 1, 0, 0, 0) for entry in entries)
    constants = _notebook_constants()
    assert constants["RUNTIME_ARCHIVE_SHA256"] == first[1]
    assert constants["RUNTIME_FILE_MANIFEST"] == first[2]
    assert constants["RUNTIME_ARCHIVE_B85"] == encode_runtime_archive(first[0])


def test_every_required_runtime_module_is_embedded() -> None:
    files = canonical_runtime_files(ROOT)
    required = {
        "olikbochon/v3_kaggle.py",
        "olikbochon/v3_training.py",
        "olikbochon/v3_data.py",
        "olikbochon/v3_preprocessing.py",
        "olikbochon/v3_default_normalizer.py",
        "ftfy/__init__.py",
        "wcwidth/__init__.py",
    }
    assert required.issubset(files)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in files)


def test_materialized_runtime_imports_without_repository_on_sys_path(tmp_path: Path) -> None:
    payload, digest, _ = build_runtime_archive(ROOT)
    target = materialize_runtime_archive(encode_runtime_archive(payload), digest, tmp_path / "rt")
    script = (
        "import sys; "
        "sys.modules['regex']=None; sys.modules['emoji']=None; "
        f"sys.path.insert(0, {str(target)!r}); "
        "from olikbochon.v3_default_normalizer import verify_frozen_reference_corpus; "
        "import ftfy, wcwidth; "
        "assert verify_frozen_reference_corpus()['cases'] > 0; "
        f"assert ftfy.__file__.startswith({str(target)!r}); "
        f"assert wcwidth.__file__.startswith({str(target)!r})"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_vendor_manifest_matches_canonical_files() -> None:
    manifest_path = ROOT / "src" / "olikbochon" / "v3_vendor_notices" / "VENDOR_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded = canonical_runtime_files(ROOT)
    for package in ("ftfy", "wcwidth"):
        for source_name, digest in manifest[package]["files"].items():
            assert source_name in embedded
            import hashlib

            assert hashlib.sha256(embedded[source_name]).hexdigest() == digest


def test_notebook_is_clean_valid_nbformat() -> None:
    notebook = nbformat.read(IPYNB_NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("execution_count") is None for cell in notebook.cells if cell.cell_type == "code")
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
