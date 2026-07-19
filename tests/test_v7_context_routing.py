from __future__ import annotations

import math

import numpy as np
import pytest

from olikbochon.v7_e2 import NO_EVIDENCE, semantic_pair
from olikbochon.v7_protocol import has_context


@pytest.mark.parametrize(
    "value",
    [None, np.nan, float("nan"), "", "   ", "[NULL]", " null ", "None", "N/A", "NaN", "<NA>"],
)
def test_every_observed_and_supported_null_sentinel_routes_absent(value: object) -> None:
    assert has_context(value) is False


@pytest.mark.parametrize("value", [0, False, "প্রমাণ", "[NULL] কিন্তু তথ্য আছে"])
def test_nonmissing_values_are_not_discarded_by_truthiness(value: object) -> None:
    assert has_context(value) is True


def test_nan_detection_does_not_require_stringification() -> None:
    assert math.isnan(float("nan"))
    assert has_context(float("nan")) is False


def test_semantic_pair_uses_distinct_rejected_marker_and_full_response() -> None:
    sequence_a, sequence_b = semantic_pair(
        "question", NO_EVIDENCE, "complete answer. second sentence."
    )
    assert NO_EVIDENCE in sequence_a
    assert sequence_b == "complete answer. second sentence."
    assert "[CLAIM]" not in sequence_a + sequence_b
