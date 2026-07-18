from __future__ import annotations

import ast
from pathlib import Path

import pytest

from olikbochon.v7_runner import (
    CANDIDATES,
    CANDIDATE_SET,
    MODE,
    OUTPUT_NAME,
    build_cli_parser,
    frozen_config,
    require_output_path,
    validate_arguments,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v7_modules_never_reference_competition_test_or_public_labels() -> None:
    for name in ("v7_corpus.py", "v7_retrieval.py", "v7_features.py", "v7_runner.py"):
        source = (ROOT / "src" / "olikbochon" / name).read_text(encoding="utf-8")
        ast.parse(source)
        assert "test set.csv" not in source.lower()
        assert "public_5k" not in source.lower()
        assert "public-20k" not in source.lower()
        assert "requests." not in source
        assert "candidate_o" not in source.lower() or name == "v7_runner.py"


def test_v7_cli_is_exact_and_has_no_nli_or_data_override(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v7/\ndata/\n", encoding="utf-8")
    corpus = (tmp_path / "data" / "retrieval" / "bnwiki").resolve()
    corpus.mkdir(parents=True)
    output = (tmp_path / "artifacts" / "v7" / OUTPUT_NAME).resolve()
    parser = build_cli_parser()
    help_text = parser.format_help()
    for forbidden in ("--test", "--public", "--nli", "--model", "--candidate "):
        assert forbidden not in help_text
    args = parser.parse_args(
        [
            "--mode",
            MODE,
            "--corpus-path",
            str(corpus),
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
    assert validate_arguments(args, tmp_path) == (corpus, output)
    args.seeds = [17, 43, 29]
    with pytest.raises(ValueError, match="seeds 17 29 43"):
        validate_arguments(args, tmp_path)


def test_v7_output_is_exact_new_and_ignored(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v7/\n", encoding="utf-8")
    output = (tmp_path / "artifacts" / "v7" / OUTPUT_NAME).resolve()
    assert require_output_path(tmp_path, output) == output
    with pytest.raises(ValueError, match=OUTPUT_NAME):
        require_output_path(tmp_path, (tmp_path / "artifacts" / "v7" / "other").resolve())
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="must not already exist"):
        require_output_path(tmp_path, output)


def test_v7_frozen_config_excludes_candidate_o_and_raw_artifacts() -> None:
    config = frozen_config()
    assert config["candidates"] == list(CANDIDATES)
    assert all("candidate_o" not in candidate for candidate in config["candidates"])
    assert config["candidate_o_nli"].startswith("unimplemented")
    assert config["retrieval_cutoff_grid"] == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    assert "no raw article/query/evidence" in config["artifact_contract"]


def test_v7_runner_reuses_strengthened_groups_and_frozen_present_rule() -> None:
    source = (ROOT / "src" / "olikbochon" / "v7_runner.py").read_text(encoding="utf-8")
    assert "build_v5_folds(validated)" in source
    assert "fold_feasibility_audit(validated, folds)" in source
    assert "deterministic_substring_prediction" in source
    assert "outer_validation_rows_used_for_selection\": 0" in source
    assert "row_level_evidence_persisted\": False" in source
