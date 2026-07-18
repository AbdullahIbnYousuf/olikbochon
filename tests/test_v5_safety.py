from __future__ import annotations

import ast
from pathlib import Path

import pytest

from olikbochon.v5_runner import (
    CANDIDATE_SET,
    MODE,
    OUTPUT_NAME,
    build_cli_parser,
    frozen_config,
    require_v5_output_path,
    validate_arguments,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v5_modules_have_no_competition_test_or_external_data_access() -> None:
    for module in sorted((ROOT / "src" / "olikbochon").glob("v5_*.py")):
        source = module.read_text(encoding="utf-8")
        ast.parse(source)
        assert "test set.csv" not in source.lower()
        assert "public_5k" not in source.lower()
        assert "public-20k" not in source.lower()


def test_v5_cli_exposes_only_the_frozen_official_experiment() -> None:
    help_text = build_cli_parser().format_help()
    for option in ("--mode", "--candidate-set", "--seeds", "--folds", "--output-dir"):
        assert option in help_text
    assert "--data" not in help_text
    assert "--model" not in help_text
    config = frozen_config()
    assert config["data_role"] == "authenticated_official_labeled_sample_only"
    assert config["candidate_set"] == CANDIDATE_SET


def test_v5_output_is_exact_new_and_explicitly_ignored(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v5/\n", encoding="utf-8")
    output = (tmp_path / "artifacts" / "v5" / OUTPUT_NAME).resolve()
    assert require_v5_output_path(tmp_path, output) == output
    with pytest.raises(ValueError, match="official_lexical_baseline"):
        require_v5_output_path(tmp_path, (tmp_path / "artifacts" / "v5" / "other").resolve())
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="must not already exist"):
        require_v5_output_path(tmp_path, output)


def test_v5_cli_requires_exact_seeds_folds_and_candidate_set(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v5/\n", encoding="utf-8")
    output = (tmp_path / "artifacts" / "v5" / OUTPUT_NAME).resolve()
    args = build_cli_parser().parse_args(
        [
            "--mode",
            MODE,
            "--candidate-set",
            CANDIDATE_SET,
            "--seeds",
            "17",
            "29",
            "43",
            "--folds",
            "5",
            "--output-dir",
            str(output),
        ]
    )
    assert validate_arguments(args, tmp_path) == output
    args.seeds = [17, 43, 29]
    with pytest.raises(ValueError, match="17 29 43"):
        validate_arguments(args, tmp_path)


def test_v5_artifact_contract_forbids_text_and_row_level_tables() -> None:
    contract = frozen_config()["artifact_contract"]
    assert "no text" in contract
    assert "no row-level table" in contract
