from __future__ import annotations

import pandas as pd
import pytest

from olikbochon.near_duplicates import audit_near_duplicates


def frame(prompt: str, response: str, label: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"context": ["প্রসঙ্গ"], "prompt_bn": [prompt], "response_bn": [response], "label": [label]}
    )


def test_identical_normalized_text_is_high_confidence_near_duplicate() -> None:
    audit = audit_near_duplicates(
        frame(" প্রশ্ন ", " উত্তর ", 0),
        frame("প্রশ্ন", "উত্তর", 0),
        left_name="left",
        right_name="right",
    )
    assert audit.high_confidence_pairs == 1
    assert audit.label_agreement_pairs == 1
    assert audit.label_conflict_pairs == 0
    assert audit.maximum_similarity == pytest.approx(1.0, abs=1e-6)


def test_near_duplicate_label_conflict_is_counted_without_text_output() -> None:
    audit = audit_near_duplicates(
        frame("একটি দীর্ঘ প্রশ্ন", "একটি দীর্ঘ উত্তর", 0),
        frame("একটি দীর্ঘ প্রশ্ন", "একটি দীর্ঘ উত্তর", 1),
        left_name="left",
        right_name="right",
    )
    assert audit.label_conflict_pairs == 1
