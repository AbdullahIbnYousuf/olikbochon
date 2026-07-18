from __future__ import annotations

import numpy as np
import pandas as pd

from olikbochon.v7_corpus import Article
from olikbochon.v7_features import TOP5_FEATURES, V4A_FEATURES, top5_features, v4a_features
from olikbochon.v7_retrieval import (
    CharacterTfidfRetriever,
    RetrievedPassage,
    intro_snippet,
    query_centered_snippet,
)
from olikbochon.v7_runner import (
    CANDIDATE_P,
    EvidenceRows,
    RetrievalCandidateModel,
    build_inner_splits,
)


def articles() -> tuple[Article, ...]:
    return (
        Article("a", "https://example.invalid/a", "ঢাকা", "ঢাকা বাংলাদেশের রাজধানী। " * 8),
        Article("b", "https://example.invalid/b", "চট্টগ্রাম", "চট্টগ্রাম একটি বন্দর নগরী। " * 8),
        Article("c", "https://example.invalid/c", "বাংলাদেশ", "বাংলাদেশ দক্ষিণ এশিয়ার দেশ। " * 8),
        Article("d", "https://example.invalid/d", "পদ্মা", "পদ্মা বাংলাদেশের নদী। " * 8),
        Article("e", "https://example.invalid/e", "ভাষা", "বাংলা একটি ভাষা। " * 8),
    )


def test_retrieval_is_deterministic_and_fits_only_unlabeled_corpus() -> None:
    retriever = CharacterTfidfRetriever(articles())
    first = retriever.retrieve("ঢাকা রাজধানী", top_k=5, snippet_method="query_centered")
    second = retriever.retrieve("ঢাকা রাজধানী", top_k=5, snippet_method="query_centered")
    assert first == second
    assert first[0].article_id == "a"
    assert len(first) == 5
    assert retriever.fit_audit["official_query_count_used_for_fit"] == 0
    assert retriever.fit_audit["raw_vocabulary_persisted"] is False


def test_bounded_snippet_methods_are_deterministic() -> None:
    body = "শুরু " * 30 + "বিশেষ প্রশ্নের উত্তর" + " শেষ" * 30
    assert len(intro_snippet(body, characters=40)) <= 40
    centered = query_centered_snippet("বিশেষ প্রশ্ন", body, characters=50)
    assert centered == query_centered_snippet("বিশেষ প্রশ্ন", body, characters=50)
    assert "বিশেষ" in centered
    assert len(centered) <= 50


def test_v4a_and_top5_features_are_finite_aggregate_values() -> None:
    passages = (
        RetrievedPassage("a", "ঢাকা", "ঢাকা বাংলাদেশের রাজধানী ২০২৪", 0.40),
        RetrievedPassage("b", "অন্য", "অন্য প্রমাণ", 0.20),
    )
    m = v4a_features("ঢাকা ২০২৪", passages, cutoff=0.25)
    n = top5_features("ঢাকা ২০২৪", passages, cutoff=0.25)
    assert tuple(m) == V4A_FEATURES
    assert tuple(n) == TOP5_FEATURES
    assert m["ctx_present"] == 1.0
    assert m["numbers_supported"] == 1.0
    assert n["accepted_passage_count"] == 1.0
    assert all(np.isfinite(list(m.values())))
    assert all(np.isfinite(list(n.values())))


def null_frame(size: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "context": ["[NULL]"] * size,
            "prompt_bn": [f"প্রশ্ন {index}" for index in range(size)],
            "response_bn": [f"উত্তর {index % 5}" for index in range(size)],
            "label": [index % 2 for index in range(size)],
        }
    )


def evidence(size: int = 30) -> EvidenceRows:
    rows = tuple(
        (
            RetrievedPassage(
                f"article-{index}",
                "শিরোনাম",
                f"উত্তর {index % 5} সমর্থিত প্রমাণ",
                0.10 + (index % 6) * 0.05,
            ),
        )
        for index in range(size)
    )
    top5 = tuple(row * 5 for row in rows)
    return EvidenceRows(rows, top5)


def test_candidate_p_fits_lexical_state_on_supplied_training_rows_only() -> None:
    frame = null_frame(10)
    model = RetrievalCandidateModel(CANDIDATE_P, 0.25).fit(frame, evidence(10))
    audit = model.fit_audit()
    assert audit["training_rows"] == 10
    assert audit["lexical_fit_scope"] == "current_null_route_training_partition_only"
    probabilities = model.predict_probability(frame.iloc[:3].reset_index(drop=True), evidence(3))
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()


def test_inner_splits_are_deterministic_group_isolated_and_complete() -> None:
    labels = np.asarray([index % 2 for index in range(30)], dtype=np.int64)
    groups = np.asarray([f"group-{index}" for index in range(30)], dtype=object)
    first = build_inner_splits(labels, groups, outer_seed=17, outer_fold=1)
    second = build_inner_splits(labels, groups, outer_seed=17, outer_fold=1)
    assert all(np.array_equal(a, b) for pair_a, pair_b in zip(first, second) for a, b in zip(pair_a, pair_b))
    assignments = np.zeros(30, dtype=np.int64)
    for train, validation in first:
        assert not (set(groups[train]) & set(groups[validation]))
        assignments[validation] += 1
    assert np.all(assignments == 1)
