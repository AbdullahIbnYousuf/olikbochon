from __future__ import annotations

import ast
from pathlib import Path

import pytest

from olikbochon.v5_sparse import CANDIDATES, CANDIDATE_SET
from olikbochon.v5_sparse_runner import (
    MODE,
    OUTPUT_NAME,
    build_cli_parser,
    frozen_config,
    require_sparse_output_path,
    validate_arguments,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sparse_modules_have_no_competition_test_or_public_data_access() -> None:
    for name in ("v5_sparse.py", "v5_sparse_runner.py"):
        source = (ROOT / "src" / "olikbochon" / name).read_text(encoding="utf-8")
        ast.parse(source)
        assert "test set.csv" not in source.lower()
        assert "public_5k" not in source.lower()
        assert "public-20k" not in source.lower()


def test_sparse_cli_has_no_data_model_or_candidate_override() -> None:
    parser = build_cli_parser()
    help_text = parser.format_help()
    for forbidden in ("--data", "--test", "--public", "--model"):
        assert forbidden not in help_text
    destinations = {action.dest for action in parser._actions}
    assert "candidate" not in destinations
    assert "candidate_set" in destinations
    with pytest.raises(SystemExit):
        parser.parse_args(["--data-path", "anything"])
    config = frozen_config()
    assert config["candidates"] == list(CANDIDATES)
    assert config["data_role"] == "authenticated_official_labeled_sample_only"
    assert config["present_route"].startswith("frozen_candidate_a")


def test_sparse_output_is_exact_new_and_ignored(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v5/\n", encoding="utf-8")
    output = (tmp_path / "artifacts" / "v5" / OUTPUT_NAME).resolve()
    assert require_sparse_output_path(tmp_path, output) == output
    with pytest.raises(ValueError, match="null_sparse_baseline"):
        require_sparse_output_path(
            tmp_path, (tmp_path / "artifacts" / "v5" / "public-data").resolve()
        )
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="must not already exist"):
        require_sparse_output_path(tmp_path, output)


def test_sparse_arguments_require_the_exact_frozen_protocol(tmp_path: Path) -> None:
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


def test_sparse_artifact_contract_forbids_raw_text_vocabularies_and_rows() -> None:
    contract = frozen_config()["artifact_contract"]
    assert "no raw text" in contract
    assert "vocabulary terms" in contract
    assert "row-level values" in contract


def test_sparse_runner_reuses_strengthened_groups_and_substring_rule() -> None:
    source = (ROOT / "src" / "olikbochon" / "v5_sparse.py").read_text(encoding="utf-8")
    assert "build_v5_folds(validated)" in source
    assert "deterministic_substring_prediction" in source
    assert "fold_feasibility_audit(validated, folds)" in source
