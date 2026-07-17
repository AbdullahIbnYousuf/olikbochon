from __future__ import annotations

import ast
from pathlib import Path

import pytest

from olikbochon.v4_runner import (
    SMOKE_CANDIDATE,
    build_cli_parser,
    official_training_path,
    require_approved_model_path,
    require_smoke_output_path,
    validate_smoke_arguments,
)
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


def test_smoke_cli_help_exposes_only_the_bounded_interface() -> None:
    help_text = build_cli_parser().format_help()
    for option in (
        "--mode",
        "--model-path",
        "--candidate",
        "--seed",
        "--max-steps",
        "--max-length",
        "--output-dir",
    ):
        assert option in help_text
    assert "{smoke}" in help_text
    assert f"{{{SMOKE_CANDIDATE}}}" in help_text


def test_smoke_cli_rejects_unapproved_model_and_artifact_roots(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v4/\n", encoding="utf-8")
    approved_model = tmp_path / "data" / "models" / "snapshot"
    approved_model.mkdir(parents=True)
    outside_model = tmp_path / "other" / "snapshot"
    outside_model.mkdir(parents=True)
    assert require_approved_model_path(tmp_path, approved_model) == approved_model.resolve()
    with pytest.raises(ValueError, match="data/models"):
        require_approved_model_path(tmp_path, outside_model)
    approved_output = tmp_path / "artifacts" / "v4" / "smoke"
    assert require_smoke_output_path(tmp_path, approved_output) == approved_output.resolve()
    with pytest.raises(ValueError, match="artifacts/v4"):
        require_smoke_output_path(tmp_path, (tmp_path / "outside").resolve())


def test_smoke_cli_argument_validation_is_capped_and_local(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v4/\n", encoding="utf-8")
    model = (tmp_path / "data" / "models" / "snapshot").resolve()
    model.mkdir(parents=True)
    output = (tmp_path / "artifacts" / "v4" / "smoke").resolve()
    parser = build_cli_parser()
    args = parser.parse_args(
        [
            "--mode",
            "smoke",
            "--model-path",
            str(model),
            "--candidate",
            SMOKE_CANDIDATE,
            "--seed",
            "17",
            "--max-steps",
            "20",
            "--max-length",
            "256",
            "--output-dir",
            str(output),
        ]
    )
    assert validate_smoke_arguments(args, tmp_path) == (model, output)
    args.max_steps = 21
    with pytest.raises(ValueError, match="between 1 and 20"):
        validate_smoke_arguments(args, tmp_path)
