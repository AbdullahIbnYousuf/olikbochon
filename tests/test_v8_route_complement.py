from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from olikbochon.v7_retrieval import RetrievedPassage
from olikbochon.v8_route_complement import (
    CLASSIFIER_THRESHOLD,
    EXPECTED_ABSENT_ROWS,
    EXPECTED_PRESENT_ROWS,
    V4A_RETRIEVAL_CUTOFF,
    build_v4a_feature_matrix,
    frozen_v8_config,
    run_authenticated_oof,
)
from olikbochon.v8_runner import (
    MODE,
    OUTPUT_NAME,
    build_cli_parser,
    validate_arguments,
)


def official_shape_frame() -> pd.DataFrame:
    absent = EXPECTED_ABSENT_ROWS
    present = EXPECTED_PRESENT_ROWS
    rows = absent + present
    return pd.DataFrame(
        {
            "context": ["[NULL]"] * absent
            + [f"present evidence statement {index}" for index in range(present)],
            "prompt_bn": [f"unique prompt {index}" for index in range(rows)],
            "response_bn": [
                f"shared null response class {index % 2} topic {index % 7}"
                for index in range(absent)
            ]
            + [f"statement {index}" for index in range(present)],
            "label": [index % 2 for index in range(absent)]
            + [index % 2 for index in range(present)],
        }
    )


def absent_evidence() -> tuple[tuple[RetrievedPassage, ...], ...]:
    return tuple(
        (
            RetrievedPassage(
                f"article-{index}",
                "synthetic title",
                f"shared null response class {index % 2} supported evidence",
                0.30 if index % 3 else 0.20,
            ),
        )
        for index in range(EXPECTED_ABSENT_ROWS)
    )


def test_v8_frozen_configuration_has_no_selection_or_blending() -> None:
    config = frozen_v8_config()
    assert CLASSIFIER_THRESHOLD == 0.50
    assert V4A_RETRIEVAL_CUTOFF == 0.25
    assert config["selection"] == "none_fixed_route_complement"
    assert config["v4a"]["standard_scaler"] is True
    assert config["v4a"]["classifier"] == config["candidate_i"]["classifier"]
    assert config["teammate_artifact_use"].startswith("post_hoc")


def test_v4a_feature_matrix_uses_direct_present_and_retrieved_absent_evidence() -> None:
    matrix = build_v4a_feature_matrix(official_shape_frame(), absent_evidence())
    assert matrix.shape == (299, 8)
    assert np.isfinite(matrix).all()
    assert int(matrix[:, 0].sum()) == EXPECTED_PRESENT_ROWS + 112


@pytest.fixture(scope="module")
def repeated_result():
    frame = official_shape_frame()
    matrix = build_v4a_feature_matrix(frame, absent_evidence())
    first = run_authenticated_oof(
        frame,
        matrix,
        corpus_manifest_sha256="a" * 64,
    )
    second = run_authenticated_oof(
        frame,
        matrix,
        corpus_manifest_sha256="a" * 64,
    )
    return first, second


def test_shared_fold_oof_is_exactly_deterministic(repeated_result) -> None:
    first, second = repeated_result
    assert first == second


def test_v8_results_have_fixed_routes_and_no_row_level_values(repeated_result) -> None:
    (results, provenance), _second = repeated_result
    assert results["route_counts"] == {
        "context_present": 130,
        "context_absent": 169,
    }
    assert results["threshold"] == 0.50
    assert results["genuine_oof_coverage"] is True
    assert results["group_overlap_count"] == 0
    assert results["row_level_probabilities_persisted"] is False
    assert provenance["raw_text_persisted"] is False
    serialized = json.dumps((results, provenance), sort_keys=True)
    assert "unique prompt" not in serialized
    assert "shared null response" not in serialized


def test_v8_cli_is_exact_ignored_and_non_overwriting(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/v8/\n", encoding="utf-8")
    corpus = (tmp_path / "data" / "retrieval" / "bnwiki").resolve()
    corpus.mkdir(parents=True)
    output = (tmp_path / "artifacts" / "v8" / OUTPUT_NAME).resolve()
    args = build_cli_parser().parse_args(
        [
            "--mode",
            MODE,
            "--corpus-path",
            str(corpus),
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
    args.output_dir = (tmp_path / "artifacts" / "v8" / "alternate").resolve()
    with pytest.raises(ValueError, match=OUTPUT_NAME):
        validate_arguments(args, tmp_path)
    args.output_dir = output
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="overwrite is prohibited"):
        validate_arguments(args, tmp_path)
