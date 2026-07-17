from __future__ import annotations

import ast
from pathlib import Path

import pytest

from olikbochon.v4_runner import (
    HISTORICAL_CANDIDATE,
    SMOKE_CANDIDATE,
    build_cli_parser,
    historical_control_config,
    official_training_path,
    require_approved_model_path,
    require_smoke_output_path,
    validate_diagnostic_arguments,
    validate_historical_control_arguments,
    validate_reproduction_arguments,
    validate_smoke_arguments,
)
from olikbochon.v4_training import (
    V4TrainingConfig,
    require_local_model_path,
    resolve_artifact_path,
)


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


def test_v4_cli_help_exposes_only_the_bounded_interfaces() -> None:
    help_text = build_cli_parser().format_help()
    for option in (
        "--mode",
        "--model-path",
        "--candidate",
        "--seed",
        "--seeds",
        "--folds",
        "--max-steps",
        "--max-length",
        "--output-dir",
    ):
        assert option in help_text
    assert "{smoke,reproduce,diagnose-reproduction,historical-control}" in help_text
    assert f"{{{SMOKE_CANDIDATE},{HISTORICAL_CANDIDATE}}}" in help_text


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


def test_reproduce_cli_requires_the_exact_frozen_design(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v4/\n", encoding="utf-8")
    model = (tmp_path / "data" / "models" / "snapshot").resolve()
    model.mkdir(parents=True)
    output = (tmp_path / "artifacts" / "v4" / "v3_reproduction").resolve()
    parser = build_cli_parser()
    args = parser.parse_args(
        [
            "--mode",
            "reproduce",
            "--model-path",
            str(model),
            "--candidate",
            SMOKE_CANDIDATE,
            "--seeds",
            "17",
            "29",
            "43",
            "--folds",
            "5",
            "--max-length",
            "256",
            "--output-dir",
            str(output),
        ]
    )
    assert validate_reproduction_arguments(args, tmp_path) == (model, output)
    args.seeds = [17, 43, 29]
    with pytest.raises(ValueError, match="17 29 43 in that order"):
        validate_reproduction_arguments(args, tmp_path)
    args.seeds = [17, 29, 43]
    args.folds = 4
    with pytest.raises(ValueError, match="folds 5"):
        validate_reproduction_arguments(args, tmp_path)


def test_reproduction_training_configuration_remains_frozen() -> None:
    config = V4TrainingConfig()
    assert config.epochs == 3
    assert config.learning_rate == 2e-5
    assert config.batch_size == 8
    assert config.gradient_accumulation == 2
    assert config.maximum_length == 256
    assert config.mixed_precision == "fp16"
    assert config.checkpoint_policy == "validation_macro_f1_then_loss_then_earlier_epoch"


def test_reproduce_cli_rejects_smoke_controls_and_other_output_names(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v4/\n", encoding="utf-8")
    model = (tmp_path / "data" / "models" / "snapshot").resolve()
    model.mkdir(parents=True)
    parser = build_cli_parser()
    args = parser.parse_args(
        [
            "--mode",
            "reproduce",
            "--model-path",
            str(model),
            "--candidate",
            SMOKE_CANDIDATE,
            "--seeds",
            "17",
            "29",
            "43",
            "--folds",
            "5",
            "--max-length",
            "256",
            "--output-dir",
            str((tmp_path / "artifacts" / "v4" / "other").resolve()),
        ]
    )
    with pytest.raises(ValueError, match="v3_reproduction"):
        validate_reproduction_arguments(args, tmp_path)
    args.seed = 17
    with pytest.raises(ValueError, match="does not accept"):
        validate_reproduction_arguments(args, tmp_path)


def test_diagnostic_cli_requires_existing_completed_reproduction(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v4/\n", encoding="utf-8")
    model = (tmp_path / "data" / "models" / "snapshot").resolve()
    model.mkdir(parents=True)
    output = (tmp_path / "artifacts" / "v4" / "v3_reproduction").resolve()
    output.mkdir(parents=True)
    args = build_cli_parser().parse_args(
        [
            "--mode",
            "diagnose-reproduction",
            "--model-path",
            str(model),
            "--candidate",
            SMOKE_CANDIDATE,
            "--seeds",
            "17",
            "29",
            "43",
            "--folds",
            "5",
            "--max-length",
            "256",
            "--output-dir",
            str(output),
        ]
    )
    assert validate_diagnostic_arguments(args, tmp_path) == (model, output)


def test_historical_control_cli_is_exact_and_cannot_overwrite(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v4/\n", encoding="utf-8")
    model = (tmp_path / "data" / "models" / "snapshot").resolve()
    model.mkdir(parents=True)
    output = (tmp_path / "artifacts" / "v4" / "v3_historical_control").resolve()
    args = build_cli_parser().parse_args(
        [
            "--mode",
            "historical-control",
            "--model-path",
            str(model),
            "--candidate",
            HISTORICAL_CANDIDATE,
            "--seeds",
            "17",
            "29",
            "43",
            "--folds",
            "5",
            "--max-length",
            "512",
            "--output-dir",
            str(output),
        ]
    )
    assert validate_historical_control_arguments(args, tmp_path) == (model, output)
    args.max_length = 256
    with pytest.raises(ValueError, match="length 512"):
        validate_historical_control_arguments(args, tmp_path)


def test_historical_control_training_values_are_frozen() -> None:
    config = historical_control_config()
    assert config.maximum_length == 512
    assert config.field_budget.maximum_length == 512
    assert config.epochs == 4
    assert config.learning_rate == 1e-5
    assert config.batch_size == 8
    assert config.gradient_accumulation == 2
    assert config.mixed_precision == "fp16"
