from __future__ import annotations

import inspect

import pandas as pd

from olikbochon.v11_error_audit import blind_taxonomy, frozen_hypotheses
from olikbochon import v11_runner


def test_blind_taxonomy_accepts_no_label_or_prediction_columns() -> None:
    frame = pd.DataFrame(
        [{"prompt_bn": "বাংলায় ব্যাখ্যা কর", "response_bn": "এটি একটি সাধারণ উত্তর।"}],
        index=[7],
    )
    tagged = blind_taxonomy(frame, workers=4)
    assert tagged.official_index.tolist() == [7]
    assert "label" not in tagged.columns
    assert not any("probability" in column for column in tagged.columns)


def test_hypotheses_are_not_evaluated_and_are_bounded() -> None:
    analysis = {
        "claim_risk_tags_minimum_support_5": [
            {"tag": "numeric claim", "support": 7}
        ],
        "task_families": [
            {"family": "factual entity question", "support": 9, "candidate_u_error_impact": 4.0}
        ],
    }
    hypotheses = frozen_hypotheses(analysis)
    assert len(hypotheses) <= 6
    assert all(record["evaluated_in_v11"] is False for record in hypotheses)


def test_runner_has_no_competition_or_submission_path() -> None:
    source = inspect.getsource(v11_runner)
    assert "test set.csv" not in source
    assert "submission" not in source.casefold()
    assert "holdout_metrics_computed\": False" in source
