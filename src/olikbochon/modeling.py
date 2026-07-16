"""Locked sparse CPU model configuration for Version 1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


RANDOM_STATE = 42
WORD_MAX_FEATURES = 20_000
CHAR_MAX_FEATURES = 30_000
LOGISTIC_C = 1.0


def build_model() -> Pipeline:
    """Create the exact untuned word+character TF-IDF logistic pipeline."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=WORD_MAX_FEATURES,
                    min_df=1,
                    max_df=1.0,
                    sublinear_tf=True,
                    lowercase=False,
                    strip_accents=None,
                    dtype=np.float32,
                    token_pattern=r"(?u)\b\w+\b",
                    norm="l2",
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=CHAR_MAX_FEATURES,
                    min_df=1,
                    max_df=1.0,
                    sublinear_tf=True,
                    lowercase=False,
                    strip_accents=None,
                    dtype=np.float32,
                    norm="l2",
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=LOGISTIC_C,
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def label_probability(model: Pipeline, texts: Sequence[str], label: int) -> np.ndarray:
    """Return probabilities for an explicit label without assuming class-column order."""
    classifier = model.named_steps["classifier"]
    classes = np.asarray(classifier.classes_)
    matches = np.flatnonzero(classes == label)
    if len(matches) != 1:
        raise ValueError(f"Model classes {classes.tolist()} do not contain label {label} exactly once")
    return model.predict_proba(texts)[:, int(matches[0])]
