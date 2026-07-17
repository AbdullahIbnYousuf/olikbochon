from __future__ import annotations

import ast
from pathlib import Path

import pytest

from olikbochon.v3_data import GroupAudit
from olikbochon.v4_runner import (
    CORRECTED_CANDIDATE,
    HISTORICAL_CANDIDATE,
    SMOKE_CANDIDATE,
    build_cli_parser,
    historical_control_config,
    official_training_path,
    require_approved_model_path,
    require_smoke_output_path,
    retain_selected_checkpoint,
    route_coverage_audit,
    split_distribution,
    validate_corrected_baseline_arguments,
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
from olikbochon.v4_validation import make_repeated_grouped_folds


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
    assert (
        "{smoke,reproduce,diagnose-reproduction,historical-control,corrected-baseline}"
        in help_text
    )
    assert (
        f"{{{SMOKE_CANDIDATE},{HISTORICAL_CANDIDATE},{CORRECTED_CANDIDATE}}}"
        in help_text
    )


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


def test_corrected_baseline_cli_is_exact_and_uses_a_new_output(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v4/\n", encoding="utf-8")
    model = (tmp_path / "data" / "models" / "snapshot").resolve()
    model.mkdir(parents=True)
    output = (tmp_path / "artifacts" / "v4" / "schema_corrected_baseline").resolve()
    args = build_cli_parser().parse_args(
        [
            "--mode",
            "corrected-baseline",
            "--model-path",
            str(model),
            "--candidate",
            CORRECTED_CANDIDATE,
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
    assert validate_corrected_baseline_arguments(args, tmp_path) == (model, output)
    args.output_dir = (tmp_path / "artifacts" / "v4" / "v3_reproduction").resolve()
    with pytest.raises(ValueError, match="schema_corrected_baseline"):
        validate_corrected_baseline_arguments(args, tmp_path)
    args.output_dir = output
    args.seeds = [17, 43, 29]
    with pytest.raises(ValueError, match="17 29 43 in that order"):
        validate_corrected_baseline_arguments(args, tmp_path)
    args.seeds = [17, 29, 43]
    args.candidate = SMOKE_CANDIDATE
    with pytest.raises(ValueError, match="v4_schema_corrected_baseline"):
        validate_corrected_baseline_arguments(args, tmp_path)


def test_corrected_baseline_retains_only_the_representative_checkpoint() -> None:
    assert retain_selected_checkpoint("corrected-baseline", 17, 1)
    assert not retain_selected_checkpoint("corrected-baseline", 17, 2)
    assert not retain_selected_checkpoint("corrected-baseline", 29, 1)
    assert retain_selected_checkpoint("reproduce", 29, 1)


def test_route_coverage_audit_requires_complete_repeated_oof() -> None:
    labels = [0, 1] * 30
    contexts = [True, False] * 30
    groups = [f"group-{index:02d}" for index in range(60)]
    group_audit = GroupAudit(tuple(groups), 60, 0, 1, 0, 0, 0, 0)
    folds = make_repeated_grouped_folds(labels, group_audit)
    audit = route_coverage_audit(labels, contexts, folds)
    assert audit["total_rows"] == 60
    assert audit["context_present"] == {"rows": 30, "label_0": 30, "label_1": 0}
    assert audit["context_absent"] == {"rows": 30, "label_0": 0, "label_1": 30}
    assert all(
        row["complete_oof_coverage"]
        for row in audit["per_seed_validation_coverage"].values()
    )


def test_split_distribution_records_only_aggregate_route_and_label_counts() -> None:
    result = split_distribution(
        [0, 1, 0, 1, 1],
        [True, True, False, False, False],
        [0, 1, 2, 4],
    )
    assert result == {
        "rows": 4,
        "label_0": 2,
        "label_1": 2,
        "context_present": {"rows": 2, "label_0": 1, "label_1": 1},
        "context_absent": {"rows": 2, "label_0": 1, "label_1": 1},
    }
