from __future__ import annotations

import ast
from pathlib import Path

import pytest

from olikbochon.v4_runner import official_training_path
from olikbochon.v4_training import require_local_model_path, resolve_artifact_path


ROOT = Path(__file__).resolve().parents[1]


def test_v4_runner_has_no_competition_test_access() -> None:
    modules = sorted((ROOT / "src" / "olikbochon").glob("v4_*.py"))
    for module in modules:
        source = module.read_text(encoding="utf-8")
        ast.parse(source)
        assert "test set.csv" not in source.lower()
        assert "test_set.csv" not in source.lower()
    assert official_training_path(ROOT).name == "dataset samples.json"


def test_remote_or_relative_model_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Remote"):
        require_local_model_path("https://example.invalid/model")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absolute local"):
        require_local_model_path(Path("relative-model"))
    local = tmp_path / "model"
    local.mkdir()
    assert require_local_model_path(local.resolve()) == local.resolve()


def test_artifact_paths_are_confined_to_explicitly_ignored_v4_root(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v4/\n", encoding="utf-8")
    expected = tmp_path / "artifacts" / "v4" / "smoke" / "summary.json"
    assert resolve_artifact_path(tmp_path, Path("smoke/summary.json")) == expected
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact_path(tmp_path, Path("../../outside.txt"))
