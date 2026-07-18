from __future__ import annotations

import numpy as np
import pandas as pd

from olikbochon.v5_lexical import (
    LOGISTIC_C,
    LOGISTIC_MAX_ITERATIONS,
    LOGISTIC_RANDOM_STATE,
    LOGISTIC_SOLVER,
    _probability_diagnostics_with_routes,
    build_logistic_pipeline,
    build_v5_groups,
    deterministic_substring_prediction,
    majority_fallback,
)


def grouped_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "context": ["প্রমাণ এক", "প্রমাণ দুই", "[NULL]", None, "আলাদা"],
            "prompt_bn": ["একই প্রশ্ন", "একই প্রশ্ন", "ভিন্ন", "অন্য", "শেষ"],
            "response_bn": ["উত্তর এক", "উত্তর দুই", "ক", "খ", "গ"],
            "label": [0, 1, 0, 1, 0],
        }
    )


def test_v5_groups_join_prompt_identity_but_not_all_absent_contexts() -> None:
    audit = build_v5_groups(grouped_frame())
    assert audit.group_ids[0] == audit.group_ids[1]
    assert audit.group_ids[2] != audit.group_ids[3]
    assert audit.prompt_identity_links == 1


def test_substring_rule_and_majority_fallback_are_frozen() -> None:
    assert deterministic_substring_prediction("ঢাকা ২০২৬", "ঢাকা 2026") == 1
    assert deterministic_substring_prediction("ঢাকা", "চট্টগ্রাম") == 0
    assert majority_fallback([0, 1]) == 0
    assert majority_fallback([0, 1, 1]) == 1


def test_logistic_configuration_is_exact() -> None:
    pipeline = build_logistic_pipeline()
    model = pipeline.named_steps["logistic"]
    assert pipeline.named_steps["scale"].with_mean
    assert model.class_weight == "balanced"
    assert model.C == LOGISTIC_C == 1.0
    assert model.max_iter == LOGISTIC_MAX_ITERATIONS == 2000
    assert model.random_state == LOGISTIC_RANDOM_STATE == 42
    assert model.solver == LOGISTIC_SOLVER == "liblinear"


def test_probability_reporting_adds_selected_guard_and_route_metrics() -> None:
    diagnostics = _probability_diagnostics_with_routes(
        np.asarray([0, 0, 1, 1, 0, 0, 1, 1]),
        np.asarray([0.10, 0.30, 0.70, 0.90, 0.20, 0.40, 0.60, 0.80]),
        np.asarray([True, True, True, True, False, False, False, False]),
    )
    selected = diagnostics["best_frozen_grid"]
    assert selected["passes_class_collapse_guard"] is True
    assert selected["route_metrics"]["context_present"]["macro_f1"] == 1.0
    assert selected["route_metrics"]["context_absent"]["macro_f1"] == 1.0
