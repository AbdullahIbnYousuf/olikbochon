"""Self-contained Kaggle orchestration for the clean Version 4-A reproduction."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from .metrics import prediction_collapse_warning
from .submission import build_submission, validate_submission
from .v4_wikipedia import (
    FEATURE_COLUMNS,
    ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
    RANDOM_STATE,
    RETRIEVAL_THRESHOLDS,
    SNIPPET_CHARACTERS,
    WIKI_DATASET_ID,
    WIKI_DATASET_LICENSE_METADATA,
    WIKI_DATASET_REF,
    WIKI_DATASET_VERSION,
    WIKI_MAX_FEATURES,
    WikipediaRetriever,
    discover_v4_files,
    evaluate_v4a,
    fit_final_model,
    load_competition_test,
    load_official_labeled,
    load_sample_submission,
    load_wikipedia_corpus,
    predict_probabilities,
    safe_discovery_summary,
    validate_oof_probability_artifact,
    validate_test_probability_artifact,
)


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsafe Version 4-A summary value: {type(value).__name__}")


def _write_safe_summary(path: Path, summary: dict[str, Any]) -> None:
    forbidden = {"id", "ids", "text", "texts", "probabilities", "predictions", "rows"}

    def audit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in forbidden:
                    raise ValueError(f"Unsafe row-level summary key: {key!r}")
                audit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                audit(item)

    audit(summary)
    path.write_text(
        json.dumps(_safe_json_value(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_kaggle_v4a(
    input_root: Path = Path("/kaggle/input"),
    working_root: Path = Path("/kaggle/working"),
) -> dict[str, Any]:
    """Run V4-A without logging test text, identifiers, or row-level predictions."""
    started = perf_counter()
    working_root.mkdir(parents=True, exist_ok=True)

    files = discover_v4_files(input_root)
    discovery = safe_discovery_summary(files)
    print("Authenticated official competition files and validated Bengali Wikipedia layout.")
    print(
        "Safe input summary:",
        {
            "competition_root": discovery["competition_root"],
            "wikipedia_root": discovery["wikipedia_root"],
            "official_file_count": discovery["official_file_count"],
            "wikipedia_chunk_count": discovery["wikipedia_chunk_count"],
            "wikipedia_total_size": discovery["wikipedia_total_size"],
            "wikipedia_duplicate_path_count": discovery[
                "wikipedia_duplicate_path_count"
            ],
        },
    )

    official = load_official_labeled(files.official_train)
    corpus = load_wikipedia_corpus(files)
    retriever = WikipediaRetriever(corpus)
    official_retrieval = retriever.retrieve(official["prompt_bn"])
    validation = evaluate_v4a(official, official_retrieval)
    print(
        "Validation complete:",
        {
            "official_labeled_rows": len(official),
            "original_method_threshold": validation.original_method[
                "selected_retrieval_threshold"
            ],
            "original_method_macro_f1": validation.original_method["selected_metrics"][
                "macro_f1"
            ],
            "honest_grouped_macro_f1": validation.grouped_method["metrics"]["macro_f1"],
            "duplicate_groups": validation.group_audit.group_count,
        },
    )

    scaler, model = fit_final_model(
        official,
        official_retrieval,
        retrieval_threshold=ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
    )

    # Test content is loaded only after every model, feature, CV, and threshold decision is frozen.
    test = load_competition_test(files.test)
    sample_submission = load_sample_submission(files.sample_submission, test["id"])
    test_retrieval = retriever.retrieve(test["prompt_bn"])
    test_probabilities = predict_probabilities(
        test,
        test_retrieval,
        scaler,
        model,
        retrieval_threshold=ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
    )

    oof_artifact = pd.DataFrame(
        {
            "row_index": np.arange(len(official), dtype=np.int64),
            "fold": validation.grouped_fold_by_row,
            "label": official["label"].to_numpy(dtype=np.int64),
            "probability_label1": validation.grouped_oof_probabilities,
        }
    )
    validate_oof_probability_artifact(oof_artifact, len(official))
    oof_path = working_root / "v4a_oof_probabilities.csv"
    oof_artifact.to_csv(oof_path, index=False)

    test_probability_artifact = pd.DataFrame(
        {"id": test["id"].reset_index(drop=True), "probability_label1": test_probabilities}
    )
    validate_test_probability_artifact(test_probability_artifact, test["id"].reset_index(drop=True))
    test_probability_path = working_root / "v4a_test_probabilities.csv"
    test_probability_artifact.to_csv(test_probability_path, index=False)

    predicted_labels = (test_probabilities >= 0.5).astype(np.int64)
    submission = build_submission(test["id"], predicted_labels, sample_submission)
    validate_submission(submission, test["id"], sample_submission)
    submission_path = working_root / "submission.csv"
    submission.to_csv(submission_path, index=False)

    values, counts = np.unique(predicted_labels, return_counts=True)
    prediction_distribution = {int(value): int(count) for value, count in zip(values, counts)}
    for label in (0, 1):
        prediction_distribution.setdefault(label, 0)
    collapse_warning = prediction_collapse_warning(
        predicted_labels, name="V4-A test predictions"
    )
    summary = {
        "version": "4-A",
        "status": "candidate artifacts created; not submitted",
        "source_method_verified_public_score": 0.685,
        "source_method_verified_kaggle_version": 4,
        "official_labeled_count": int(len(official)),
        "test_count": int(len(test)),
        "wikipedia": {
            "dataset_ref": WIKI_DATASET_REF,
            "declared_dataset_id": WIKI_DATASET_ID,
            "declared_dataset_version": WIKI_DATASET_VERSION,
            "kaggle_license_metadata": WIKI_DATASET_LICENSE_METADATA,
            "discovered_file_count": discovery["wikipedia_chunk_count"],
            "discovered_aggregate_bytes": discovery["wikipedia_total_size"],
            "deduplicated_path_count": discovery["wikipedia_duplicate_path_count"],
            "parsed_canonical_chunk_count": corpus.source_chunk_count,
            "decoded_line_count": corpus.decoded_line_count,
            "rejected_line_count": corpus.rejected_line_count,
            "duplicate_url_count": corpus.duplicate_url_count,
            "usable_article_count": int(len(corpus.articles)),
        },
        "configuration": {
            "random_state": RANDOM_STATE,
            "retrieval_threshold_candidates": RETRIEVAL_THRESHOLDS,
            "deployed_retrieval_threshold": ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
            "snippet_characters": SNIPPET_CHARACTERS,
            "wiki_max_features": WIKI_MAX_FEATURES,
            "feature_columns": FEATURE_COLUMNS,
            "classifier_threshold": 0.5,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "original_method_estimate": validation.original_method,
        "honest_grouped_estimate": validation.grouped_method,
        "group_audit": {
            "group_count": validation.group_audit.group_count,
            "nontrivial_groups": validation.group_audit.nontrivial_groups,
            "largest_group": validation.group_audit.largest_group,
            "exact_duplicate_links": validation.group_audit.exact_duplicate_links,
            "prompt_context_links": validation.group_audit.prompt_context_links,
        },
        "test_prediction_distribution": prediction_distribution,
        "test_prediction_collapse_warning": collapse_warning,
        "artifacts": {
            "submission": submission_path.name,
            "oof_probability_file": oof_path.name,
            "test_probability_file": test_probability_path.name,
            "submission_columns": ["id", "label"],
            "oof_probability_columns": list(oof_artifact.columns),
            "test_probability_columns": list(test_probability_artifact.columns),
        },
        "runtime_seconds": perf_counter() - started,
    }
    summary_path = working_root / "v4a_run_summary.json"
    _write_safe_summary(summary_path, summary)
    print(
        "Validated V4-A candidate outputs:",
        {
            "submission_rows": len(submission),
            "oof_probability_rows": len(oof_artifact),
            "test_probability_rows": len(test_probability_artifact),
            "prediction_distribution": prediction_distribution,
            "collapse_warning": collapse_warning,
            "submitted_to_kaggle": False,
        },
    )
    return summary
