from __future__ import annotations

import numpy as np
import pandas as pd

from olikbochon.v11_error_audit import lock_discovery_split


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(24):
        rows.append(
            {
                "context": "[NULL]",
                "prompt_bn": f"প্রশ্ন {index}",
                "response_bn": f"উত্তর {index}",
                "label": index % 2,
            }
        )
    for index in range(24, 42):
        rows.append(
            {
                "context": f"প্রসঙ্গ {index}",
                "prompt_bn": f"উপস্থিত প্রশ্ন {index}",
                "response_bn": f"উত্তর {index}",
                "label": index % 2,
            }
        )
    return pd.DataFrame(rows)


def test_locked_split_is_deterministic_group_safe(monkeypatch) -> None:
    frame = _frame()
    monkeypatch.setattr("olikbochon.v11_error_audit.EXPECTED_ABSENT_ROWS", 24)
    monkeypatch.setattr("olikbochon.v11_error_audit.EXPECTED_PRESENT_ROWS", 18)
    first = lock_discovery_split(frame, 0.75)
    second = lock_discovery_split(frame, 0.75)
    assert np.array_equal(first.discovery_indices, second.discovery_indices)
    assert np.array_equal(first.holdout_indices, second.holdout_indices)
    assert first.record == second.record
    assert first.record["group_overlap_count"] == 0
    assert set(first.record["discovery_label_counts"].values()) != {0}
    assert set(first.record["holdout_label_counts"].values()) != {0}
