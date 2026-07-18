from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest
from test_v8_route_complement import absent_evidence, official_shape_frame

from olikbochon.v8_route_complement import (
    build_v4a_feature_matrix,
    run_authenticated_oof,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def provenance_result():
    frame = official_shape_frame()
    matrix = build_v4a_feature_matrix(frame, absent_evidence())
    return run_authenticated_oof(
        frame,
        matrix,
        corpus_manifest_sha256="a" * 64,
    )


def test_every_v8_validation_row_has_provable_exclusion(provenance_result) -> None:
    results, provenance = provenance_result
    assert len(provenance["folds"]) == 15
    coverage: dict[int, Counter[int]] = {}
    for fold in provenance["folds"]:
        seed_counts = coverage.setdefault(fold["seed"], Counter())
        seed_counts.update(fold["validation_row_indices"])
        assert fold["row_overlap_count"] == 0
        assert fold["group_overlap_count"] == 0
        assert not (
            set(fold["training_group_hashes"])
            & set(fold["validation_group_hashes"])
        )
        assert fold["validation_rows_used_for_fitted_preprocessing"] == 0
        assert fold["candidate_i_training_row_count"] < fold["training_row_count"]
        assert fold["candidate_i_preprocessing_fit_rows"] == fold[
            "candidate_i_training_row_count"
        ]
        assert fold["training_index_sha256"] != fold["validation_index_sha256"]
        assert len(fold["v4a_fitted_model_fingerprint"]) == 64
        assert len(fold["candidate_i_fitted_model_fingerprint"]) == 64
    assert set(coverage) == {17, 29, 43}
    for counts in coverage.values():
        assert len(counts) == 299
        assert set(counts.values()) == {1}
    assert results["competition_test_accessed"] is False


def test_v8_sources_never_reference_competition_test_or_public_labels() -> None:
    for name in ("v8_route_complement.py", "v8_runner.py"):
        source = (ROOT / "src" / "olikbochon" / name).read_text(encoding="utf-8")
        ast.parse(source)
        lowered = source.lower()
        assert "test set.csv" not in lowered
        assert "public_5k" not in lowered
        assert "public-20k" not in lowered
        assert "v4a_oof_probabilities" not in lowered


def test_v8_index_fit_scope_is_corpus_only() -> None:
    source = (ROOT / "src" / "olikbochon" / "v8_runner.py").read_text(
        encoding="utf-8"
    )
    index_position = source.index("CharacterTfidfRetriever(articles)")
    official_position = source.index("load_official_training_frame(root)")
    assert index_position < official_position
    assert '"official_queries_used_for_fit": 0' in source
