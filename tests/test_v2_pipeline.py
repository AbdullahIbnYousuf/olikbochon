from __future__ import annotations

import numpy as np
import pandas as pd

from olikbochon.modeling import build_model, label_probability
from olikbochon.v2_pipeline import (
    V2_DEFAULT_SUBMISSION,
    V2_FIXED_SUBMISSION,
    build_v2_submission_variants,
)


def test_both_v2_submission_variants_are_independently_valid() -> None:
    ids = pd.Series([9, 7, 7, 3], name="id")
    sample = pd.DataFrame({"id": ids, "label": [1, 1, 1, 1]})
    variants = build_v2_submission_variants(
        ids,
        np.array([0.2, 0.49, 0.51, 0.8]),
        selected_threshold=0.6,
        sample_submission=sample,
    )
    assert set(variants) == {V2_DEFAULT_SUBMISSION, V2_FIXED_SUBMISSION}
    assert variants[V2_DEFAULT_SUBMISSION]["label"].tolist() == [0, 0, 0, 1]
    assert variants[V2_FIXED_SUBMISSION]["label"].tolist() == [0, 0, 1, 1]
    for submission in variants.values():
        assert list(submission.columns) == ["id", "label"]
        assert submission["id"].tolist() == ids.tolist()
        assert pd.api.types.is_integer_dtype(submission["label"])


def test_fixed_model_is_deterministic() -> None:
    texts = np.array(
        [
            "প্রশ্ন এক উত্তর ভুল",
            "প্রশ্ন দুই উত্তর ঠিক",
            "প্রশ্ন তিন তথ্য ভুল",
            "প্রশ্ন চার তথ্য ঠিক",
            "প্রশ্ন পাঁচ দাবি ভুল",
            "প্রশ্ন ছয় দাবি ঠিক",
        ],
        dtype=object,
    )
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    first = build_model().fit(texts, labels)
    second = build_model().fit(texts, labels)
    np.testing.assert_allclose(
        label_probability(first, texts, label=1),
        label_probability(second, texts, label=1),
        rtol=0,
        atol=0,
    )
