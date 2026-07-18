from __future__ import annotations

import numpy as np

from olikbochon.v5_features import CONTEXT_ABSENT_FEATURES
from olikbochon.v7_features import TOP5_FEATURES
from olikbochon.v9_features import CLASSIFIER_CONFIG, frozen_feature_fingerprint
from olikbochon.v9_nli import NLI_FEATURES


def test_frozen_v9_feature_dimensions_and_classifier_are_exact() -> None:
    assert len(NLI_FEATURES) == 7
    assert len(TOP5_FEATURES) == 7
    assert len(CONTEXT_ABSENT_FEATURES) == 21
    assert 7 + 7 == 14
    assert 7 + 7 + 7 + 21 == 42
    assert CLASSIFIER_CONFIG == {
        "class_weight": "balanced",
        "C": 1.0,
        "max_iter": 3000,
        "solver": "liblinear",
        "random_state": 42,
    }


def test_feature_configuration_fingerprint_is_deterministic() -> None:
    first = frozen_feature_fingerprint()
    second = frozen_feature_fingerprint()
    assert first == second
    assert len(first) == 64
    assert all(character in "0123456789abcdef" for character in first)


def test_no_sparse_or_oof_probability_feature_is_frozen() -> None:
    names = np.asarray(NLI_FEATURES + TOP5_FEATURES + CONTEXT_ABSENT_FEATURES, dtype=object)
    joined = " ".join(str(value).lower() for value in names)
    assert "tfidf" not in joined
    assert "oof" not in joined
    assert "probability_label1" not in joined
