from __future__ import annotations

import ast
from pathlib import Path

import pytest

from olikbochon.v6_null_neural import CANDIDATES, CANDIDATE_SET
from olikbochon.v6_runner import (
    MODE,
    OUTPUT_NAME,
    build_cli_parser,
    frozen_config,
    require_v6_output_path,
    validate_arguments,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v6_modules_have_no_competition_test_or_public_data_access() -> None:
    for name in ("v6_null_neural.py", "v6_runner.py"):
        source = (ROOT / "src" / "olikbochon" / name).read_text(encoding="utf-8")
        ast.parse(source)
        assert "test set.csv" not in source.lower()
        assert "public_5k" not in source.lower()
        assert "public-20k" not in source.lower()
        assert "requests." not in source


def test_v6_cli_exposes_no_data_candidate_or_training_override() -> None:
    parser = build_cli_parser()
    help_text = parser.format_help()
    for forbidden in (
        "--data",
        "--test",
        "--public",
        "--candidate ",
        "--epochs",
        "--batch-size",
        "--learning-rate",
        "--threshold",
    ):
        assert forbidden not in help_text
    config = frozen_config()
    assert config["candidates"] == list(CANDIDATES)
    assert config["fit_route"] == "corrected_context_absent_only"
    assert config["present_route"].startswith("frozen_candidate_a")
    assert config["maximum_length"] == 192


def test_v6_output_is_exact_new_and_ignored(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v6/\n", encoding="utf-8")
    output = (tmp_path / "artifacts" / "v6" / OUTPUT_NAME).resolve()
    assert require_v6_output_path(tmp_path, output) == output
    with pytest.raises(ValueError, match="null_neural_baseline"):
        require_v6_output_path(
            tmp_path, (tmp_path / "artifacts" / "v6" / "other").resolve()
        )
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="must not already exist"):
        require_v6_output_path(tmp_path, output)


def test_v6_arguments_require_local_model_and_exact_frozen_protocol(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v6/\n", encoding="utf-8")
    model = (tmp_path / "data" / "models" / "snapshot").resolve()
    model.mkdir(parents=True)
    output = (tmp_path / "artifacts" / "v6" / OUTPUT_NAME).resolve()
    args = build_cli_parser().parse_args(
        [
            "--mode",
            MODE,
            "--model-path",
            str(model),
            "--candidate-set",
            CANDIDATE_SET,
            "--seeds",
            "17",
            "29",
            "43",
            "--folds",
            "5",
            "--max-length",
            "192",
            "--output-dir",
            str(output),
        ]
    )
    assert validate_arguments(args, tmp_path) == (model, output)
    args.seeds = [17, 43, 29]
    with pytest.raises(ValueError, match="17 29 43"):
        validate_arguments(args, tmp_path)
    args.seeds = [17, 29, 43]
    args.max_length = 256
    with pytest.raises(ValueError, match="192"):
        validate_arguments(args, tmp_path)


def test_v6_artifact_contract_forbids_raw_text_and_row_values() -> None:
    contract = frozen_config()["artifact_contract"]
    assert "no raw text" in contract
    assert "row-level values" in contract
    assert frozen_config()["retention"].endswith("per candidate")


def test_v6_uses_one_global_threshold_and_reuses_v5_groups() -> None:
    source = (ROOT / "src" / "olikbochon" / "v6_null_neural.py").read_text(
        encoding="utf-8"
    )
    assert "build_v5_folds(validated)" in source
    assert "fold_feasibility_audit(validated, folds)" in source
    assert source.count("select_threshold(") == 1
    assert "mean_null_probabilities" in source
    assert "probability_diagnostics" not in source
