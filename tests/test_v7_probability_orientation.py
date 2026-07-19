from __future__ import annotations

import pandas as pd
import pytest

from olikbochon.data_loading import DataValidationError
from olikbochon.v7_protocol import OUTER_SEEDS, validate_probability_frame


def probability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": row_id,
                "seed": seed,
                "probability_label_0": probability_0,
                "probability_label_1": 1.0 - probability_0,
            }
            for seed in OUTER_SEEDS
            for row_id, probability_0 in (("a", 0.8), ("b", 0.2))
        ]
    )


def test_explicit_probability_orientation_and_per_seed_oof_coverage() -> None:
    validate_probability_frame(probability_frame(), ["a", "b"])


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, float("nan")])
def test_probability_range_is_enforced(bad_value: float) -> None:
    frame = probability_frame()
    frame.loc[0, "probability_label_1"] = bad_value
    with pytest.raises(DataValidationError, match=r"\[0, 1\]"):
        validate_probability_frame(frame, ["a", "b"])


def test_ambiguous_old_probability_column_is_rejected() -> None:
    frame = probability_frame().assign(probability=0.5)
    with pytest.raises(DataValidationError, match="explicit label orientation"):
        validate_probability_frame(frame, ["a", "b"])


def test_binary_probabilities_must_sum_to_one() -> None:
    frame = probability_frame()
    frame.loc[0, "probability_label_0"] = 0.7
    with pytest.raises(DataValidationError, match="sum to one"):
        validate_probability_frame(frame, ["a", "b"])
