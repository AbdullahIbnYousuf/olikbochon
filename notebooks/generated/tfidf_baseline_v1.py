# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kaggle
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kaggle:
#     accelerator: none
#     internet: false
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. Objective
#
# Build a deterministic CPU baseline for Bengali hallucination detection using word and
# character TF-IDF with balanced logistic regression.

# %% [markdown]
# ## 2. Competition labels
#
# Label `0` means hallucinated. Label `1` means faithful. We report class-0 F1 and ordinary
# two-class macro F1 separately because the official metric wording is ambiguous.

# %% [markdown]
# ## 3. Offline and no-API constraints
#
# This notebook is self-contained, performs no package installation or model download, and
# uses no external service. It is designed for Kaggle with internet disabled and a CPU runtime.

# %% [markdown]
# ## 4. Imports

# %%
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

# %% [markdown]
# ## 5. Configuration
#
# The feature and classifier settings are locked for Version 1. `min_df=1` is intentional:
# the official labeled sample is small and rare Bengali terms may be informative.

# %%
RANDOM_STATE = 42
OFFICIAL_SAMPLE_SHA256 = "f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28"
LABELED_COLUMNS = ("context", "prompt_bn", "response_bn", "label")
TEST_COLUMNS = ("id", "context", "prompt_bn", "response_bn")
SUBMISSION_COLUMNS = ("id", "label")
TRAIN_NAMES = ("dataset samples.json",)
TEST_NAMES = ("test set.csv",)
SAMPLE_SUBMISSION_NAMES = ("sample submission.csv", "sample_submission.csv")
THRESHOLDS = np.arange(20, 81, dtype=np.int64) / 100.0

PROMPT_MARKER = "__PROMPT__"
CONTEXT_PRESENT_MARKER = "__CONTEXT_PRESENT__"
CONTEXT_MARKER = "__CONTEXT__"
RESPONSE_MARKER = "__RESPONSE__"
_WHITESPACE_RE = re.compile(r"\s+")
_MISSING_CONTEXT_MARKERS = frozenset({"", "[null]", "nan", "none", "null", "na", "n/a"})


class DataValidationError(ValueError):
    """Raised when labeled data violates the Version 1 contract."""


class DiscoveryError(RuntimeError):
    """Raised when Kaggle inputs cannot be selected safely."""


class SubmissionValidationError(ValueError):
    """Raised when an output violates the submission contract."""


@dataclass(frozen=True)
class DiscoveredFiles:
    train: Path
    test: Path
    sample_submission: Path | None


@dataclass(frozen=True)
class OOFResult:
    probabilities_label1: np.ndarray
    predictions: np.ndarray
    fold_metrics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class NestedCVResult:
    predictions: np.ndarray
    selected_thresholds: tuple[float, ...]
    fold_metrics: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# %% [markdown]
# ## 6. File discovery
#
# Kaggle inputs are found recursively by exact organizer filenames. Ambiguous non-identical
# candidates are fatal. Hashing reads bytes only and never prints file contents.

# %%
def discover_preferred_file(
    root: Path, accepted_names: tuple[str, ...], *, mandatory: bool
) -> Path | None:
    root = Path(root)
    candidates: list[tuple[int, Path]] = []
    for preference, name in enumerate(accepted_names):
        candidates.extend((preference, path) for path in sorted(root.rglob(name)) if path.is_file())
    if not candidates:
        if mandatory:
            raise DiscoveryError(
                f"Missing mandatory file under {root}: accepted names={list(accepted_names)}"
            )
        return None
    hashes = {sha256_file(path) for _, path in candidates}
    if len(hashes) != 1:
        metadata = [f"{path} ({sha256_file(path)})" for _, path in candidates]
        raise DiscoveryError("Non-identical accepted candidates found: " + "; ".join(metadata))
    return min(candidates, key=lambda item: (item[0], str(item[1])))[1]


def discover_kaggle_files(root: Path = Path("/kaggle/input")) -> DiscoveredFiles:
    return DiscoveredFiles(
        train=discover_preferred_file(root, TRAIN_NAMES, mandatory=True),
        test=discover_preferred_file(root, TEST_NAMES, mandatory=True),
        sample_submission=discover_preferred_file(
            root, SAMPLE_SUBMISSION_NAMES, mandatory=False
        ),
    )


# %% [markdown]
# ## 7. Labeled-data validation
#
# The current row count is reported dynamically. Identity comes from the expected filename,
# schema, labels, nonempty content, and known official hash—not from a hard-coded row count.

# %%
def validate_labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != LABELED_COLUMNS:
        raise DataValidationError(
            f"Expected columns {list(LABELED_COLUMNS)}, found {list(frame.columns)}"
        )
    if frame.empty:
        raise DataValidationError("Labeled dataset must be nonempty")
    if frame["label"].isna().any():
        raise DataValidationError("Labels must not be missing")
    labels = frame["label"].tolist()
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) not in {0, 1}
        for value in labels
    ):
        raise DataValidationError("Labels must be integers restricted to 0 and 1")
    if set(map(int, labels)) != {0, 1}:
        raise DataValidationError("Both label classes 0 and 1 must be present")
    if frame["prompt_bn"].isna().any() or frame["response_bn"].isna().any():
        raise DataValidationError("Prompt and response values must not be missing")
    validated = frame.copy(deep=True)
    validated["label"] = validated["label"].astype(np.int64)
    return validated


def load_labeled_json(path: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.name != TRAIN_NAMES[0]:
        raise DataValidationError(f"Unexpected labeled filename: {path.name!r}")
    if expected_sha256 is not None:
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise DataValidationError(
                f"Official labeled file hash mismatch: expected {expected_sha256}, observed {observed}"
            )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise DataValidationError("Labeled JSON must contain a top-level list")
    return validate_labeled_frame(pd.DataFrame(payload))


# %% [markdown]
# ## 8. Text preprocessing
#
# Text is normalized to Unicode NFC, repeated whitespace is collapsed, common missing-context
# markers become empty context, and punctuation/numbers are preserved. No translation, stemming,
# stop-word removal, or online normalizer is used.

# %%
def _is_scalar_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and math.isnan(value)


def normalize_text(value: Any) -> str:
    if _is_scalar_missing(value):
        return ""
    normalized = unicodedata.normalize("NFC", str(value))
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_context(value: Any) -> str:
    normalized = normalize_text(value)
    if normalized.casefold() in _MISSING_CONTEXT_MARKERS:
        return ""
    return normalized


def context_is_present(value: Any) -> bool:
    return bool(normalize_context(value))


def build_marked_text(prompt: Any, context: Any, response: Any) -> str:
    prompt_text = normalize_text(prompt)
    context_text = normalize_context(context)
    response_text = normalize_text(response)
    present = int(bool(context_text))
    return (
        f"{PROMPT_MARKER}\n{prompt_text}\n\n"
        f"{CONTEXT_PRESENT_MARKER}\n{present}\n\n"
        f"{CONTEXT_MARKER}\n{context_text}\n\n"
        f"{RESPONSE_MARKER}\n{response_text}"
    )


def build_text_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        (
            build_marked_text(row.prompt_bn, row.context, row.response_bn)
            for row in frame[["prompt_bn", "context", "response_bn"]].itertuples(index=False)
        ),
        index=frame.index,
        dtype="string",
        name="model_text",
    )


def context_presence_series(frame: pd.DataFrame) -> pd.Series:
    return frame["context"].map(context_is_present).astype(bool)


def build_model() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=20_000,
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
                    max_features=30_000,
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
        C=1.0,
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def label_probability(model: Pipeline, texts: Sequence[str], label: int) -> np.ndarray:
    classes = np.asarray(model.named_steps["classifier"].classes_)
    matches = np.flatnonzero(classes == label)
    if len(matches) != 1:
        raise ValueError(f"Model classes do not contain label {label} exactly once")
    return model.predict_proba(texts)[:, int(matches[0])]


# %% [markdown]
# ## 9. Five-fold validation
#
# Standard five-fold OOF probabilities give the fixed threshold-0.50 baseline. Fold vectorizers
# are fitted only on each training portion.

# %%
def predictions_from_label1(probabilities: Sequence[float], threshold: float) -> np.ndarray:
    return (np.asarray(probabilities, dtype=np.float64) >= float(threshold)).astype(np.int64)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    return {
        "f1_label0": float(f1_score(truth, predicted, pos_label=0, zero_division=0)),
        "f1_label1": float(f1_score(truth, predicted, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).astype(int).tolist(),
    }


def standard_oof_predictions(
    texts: Sequence[str], y_true: Sequence[int], threshold: float = 0.5
) -> OOFResult:
    texts_array = np.asarray(texts, dtype=object)
    truth = np.asarray(y_true, dtype=np.int64)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probabilities = np.empty(len(truth), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (train_index, valid_index) in enumerate(cv.split(texts_array, truth), start=1):
        model = build_model()
        model.fit(texts_array[train_index], truth[train_index])
        fold_probabilities = label_probability(model, texts_array[valid_index], label=1)
        probabilities[valid_index] = fold_probabilities
        fold_predictions = predictions_from_label1(fold_probabilities, threshold)
        fold_rows.append(
            {
                "fold": fold_index,
                "threshold": float(threshold),
                **classification_metrics(truth[valid_index], fold_predictions),
            }
        )
    return OOFResult(
        probabilities,
        predictions_from_label1(probabilities, threshold),
        tuple(fold_rows),
    )


# %% [markdown]
# ## 10. Threshold selection
#
# The honest estimate uses nested 5×3 CV: each outer threshold is selected only from inner OOF
# predictions. The final deployment threshold is separately selected from full five-fold OOF
# probabilities; its same-OOF score is an optimistic tuning estimate.

# %%
def threshold_table(y_true: Sequence[int], probabilities_label1: Sequence[float]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        metrics = classification_metrics(
            y_true, predictions_from_label1(probabilities_label1, float(threshold))
        )
        rows.append(
            {
                "threshold": float(threshold),
                "f1_label0": metrics["f1_label0"],
                "f1_label1": metrics["f1_label1"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> float:
    required = {"threshold", "f1_label0", "macro_f1"}
    if not required.issubset(table.columns) or table.empty:
        raise ValueError(f"Threshold table must be nonempty and contain {sorted(required)}")
    ranked = table.assign(distance=(table["threshold"] - 0.5).abs()).sort_values(
        ["f1_label0", "macro_f1", "distance", "threshold"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["threshold"])


def nested_threshold_predictions(texts: Sequence[str], y_true: Sequence[int]) -> NestedCVResult:
    texts_array = np.asarray(texts, dtype=object)
    truth = np.asarray(y_true, dtype=np.int64)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    outer_predictions = np.empty(len(truth), dtype=np.int64)
    selected_thresholds: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for outer_index, (outer_train, outer_valid) in enumerate(outer_cv.split(texts_array, truth)):
        inner_truth = truth[outer_train]
        inner_texts = texts_array[outer_train]
        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE + outer_index,
        )
        inner_probabilities = np.empty(len(inner_truth), dtype=np.float64)
        for inner_train, inner_valid in inner_cv.split(inner_texts, inner_truth):
            inner_model = build_model()
            inner_model.fit(inner_texts[inner_train], inner_truth[inner_train])
            inner_probabilities[inner_valid] = label_probability(
                inner_model, inner_texts[inner_valid], label=1
            )
        selected = select_threshold(threshold_table(inner_truth, inner_probabilities))
        selected_thresholds.append(selected)
        outer_model = build_model()
        outer_model.fit(texts_array[outer_train], truth[outer_train])
        outer_probabilities = label_probability(outer_model, texts_array[outer_valid], label=1)
        outer_fold_predictions = predictions_from_label1(outer_probabilities, selected)
        outer_predictions[outer_valid] = outer_fold_predictions
        fold_rows.append(
            {
                "fold": outer_index + 1,
                "threshold": selected,
                **classification_metrics(truth[outer_valid], outer_fold_predictions),
            }
        )
    return NestedCVResult(
        outer_predictions,
        tuple(selected_thresholds),
        tuple(fold_rows),
    )


def fold_mean_std(fold_metrics: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    names = ("f1_label0", "f1_label1", "macro_f1", "accuracy")
    return {
        name: {
            "mean": float(np.mean([float(row[name]) for row in fold_metrics])),
            "std": float(np.std([float(row[name]) for row in fold_metrics], ddof=0)),
        }
        for name in names
    }


def subgroup_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], has_context: Sequence[bool]
) -> dict[str, dict[str, Any]]:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    presence = np.asarray(has_context, dtype=bool)
    result: dict[str, dict[str, Any]] = {}
    for name, mask in (("context_present", presence), ("context_absent", ~presence)):
        labels = np.unique(truth[mask])
        row: dict[str, Any] = {"count": int(mask.sum()), "both_classes": len(labels) == 2}
        if len(labels) == 2:
            metrics = classification_metrics(truth[mask], predicted[mask])
            row.update(
                {
                    "f1_label0": metrics["f1_label0"],
                    "macro_f1": metrics["macro_f1"],
                    "confusion_matrix": metrics["confusion_matrix"],
                }
            )
        result[name] = row
    return result


# %% [markdown]
# ## 11. Validation results
#
# Only aggregate metrics, confusion matrices, thresholds, and counts are printed. No examples,
# row-level probabilities, or row-level predictions are displayed or saved.

# %%
def safe_validation_report(frame: pd.DataFrame) -> tuple[float, Pipeline]:
    texts = build_text_series(frame)
    truth = frame["label"].to_numpy(dtype=np.int64)
    context_presence = context_presence_series(frame).to_numpy(dtype=bool)

    fixed = standard_oof_predictions(texts, truth, threshold=0.5)
    fixed_metrics = classification_metrics(truth, fixed.predictions)
    nested = nested_threshold_predictions(texts, truth)
    nested_metrics = classification_metrics(truth, nested.predictions)
    table = threshold_table(truth, fixed.probabilities_label1)
    deployment_threshold = select_threshold(table)
    deployment_predictions = predictions_from_label1(
        fixed.probabilities_label1, deployment_threshold
    )
    deployment_metrics = classification_metrics(truth, deployment_predictions)
    constant_baselines = {
        "always_label0": classification_metrics(truth, np.zeros(len(truth), dtype=np.int64)),
        "always_label1": classification_metrics(truth, np.ones(len(truth), dtype=np.int64)),
    }

    print("Observed labeled rows:", len(frame))
    print("Label counts:", frame["label"].value_counts().sort_index().to_dict())
    print("Context counts:", pd.Series(context_presence).value_counts().sort_index().to_dict())
    print("Package versions:", {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__})
    print("Constant-class baselines:")
    print(json.dumps(constant_baselines, indent=2, sort_keys=True))
    print("\nFixed threshold baseline (0.50):")
    print(json.dumps(fixed_metrics, indent=2, sort_keys=True))
    print("Fixed-threshold fold metrics:")
    print(json.dumps(fixed.fold_metrics, indent=2, sort_keys=True))
    print("Fixed-threshold fold mean/std:")
    print(json.dumps(fold_mean_std(fixed.fold_metrics), indent=2, sort_keys=True))
    print("\nNested-CV threshold-selected estimate:")
    print(json.dumps(nested_metrics, indent=2, sort_keys=True))
    print("Nested outer-fold metrics and independently selected thresholds:")
    print(json.dumps(nested.fold_metrics, indent=2, sort_keys=True))
    print("Nested fold mean/std:")
    print(json.dumps(fold_mean_std(nested.fold_metrics), indent=2, sort_keys=True))
    print("\nFull-OOF threshold-tuning estimate (optimistic):")
    print("Selected deployment threshold:", deployment_threshold)
    print(json.dumps(deployment_metrics, indent=2, sort_keys=True))
    print("Complete threshold table:")
    print(table.to_string(index=False))
    print("\nContext subgroup metrics at threshold 0.50:")
    print(json.dumps(subgroup_metrics(truth, fixed.predictions, context_presence), indent=2, sort_keys=True))
    print("Context subgroup metrics for nested predictions:")
    print(json.dumps(subgroup_metrics(truth, nested.predictions, context_presence), indent=2, sort_keys=True))
    print("Context subgroup metrics at deployment threshold:")
    print(
        json.dumps(
            subgroup_metrics(truth, deployment_predictions, context_presence),
            indent=2,
            sort_keys=True,
        )
    )

    final_model = build_model()
    final_model.fit(texts, truth)
    return deployment_threshold, final_model


# %% [markdown]
# ## 12. Final full-data training
#
# After validation, the locked pipeline is fitted once on every official labeled record. The
# observed row count is dynamic and is never used as a test-data assumption.

# %% [markdown]
# ## 13. Kaggle-only test inference
#
# The real test file is opened only when `/kaggle/input` exists. Local execution completes
# validation and final training, then skips this section deliberately.

# %%
def validate_test_frame(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != TEST_COLUMNS:
        raise SubmissionValidationError(
            f"Expected test columns {list(TEST_COLUMNS)}, found {list(frame.columns)}"
        )
    if frame["id"].isna().any():
        raise SubmissionValidationError("Test IDs must not be missing")


# %% [markdown]
# ## 14. Submission validation
#
# IDs are preserved exactly in input order. Missing IDs, mismatched templates, non-integer labels,
# extra columns, and row-count mismatches are fatal.

# %%
def validate_submission(
    submission: pd.DataFrame,
    test_ids: pd.Series,
    sample_submission: pd.DataFrame | None = None,
) -> None:
    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise SubmissionValidationError("Submission columns must be exactly id,label")
    if len(submission) != len(test_ids):
        raise SubmissionValidationError("Submission row count must match test row count")
    if submission["id"].isna().any() or submission["label"].isna().any():
        raise SubmissionValidationError("Submission IDs and labels must not be missing")
    if submission["id"].tolist() != test_ids.tolist():
        raise SubmissionValidationError("Submission IDs must preserve test ID order")
    if not pd.api.types.is_integer_dtype(submission["label"].dtype):
        raise SubmissionValidationError("Submission labels must have an integer dtype")
    if not submission["label"].isin([0, 1]).all():
        raise SubmissionValidationError("Submission labels must be restricted to 0 and 1")
    if sample_submission is not None:
        if tuple(sample_submission.columns) != SUBMISSION_COLUMNS:
            raise SubmissionValidationError("Sample submission must have exact id,label columns")
        if len(sample_submission) != len(test_ids):
            raise SubmissionValidationError("Sample submission row count must match test data")
        if sample_submission["id"].tolist() != test_ids.tolist():
            raise SubmissionValidationError("Sample submission ID order does not match test IDs")


def build_submission(
    test_ids: pd.Series,
    predicted_labels: Sequence[int],
    sample_submission: pd.DataFrame | None = None,
) -> pd.DataFrame:
    labels = np.asarray(predicted_labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise SubmissionValidationError("Predicted labels must have an integer dtype")
    if len(labels) != len(test_ids):
        raise SubmissionValidationError("Submission row count must match test row count")
    normalized_ids = test_ids.reset_index(drop=True)
    submission = pd.DataFrame(
        {"id": normalized_ids, "label": labels.astype(np.int64)}
    )
    validate_submission(submission, normalized_ids, sample_submission)
    return submission


# %% [markdown]
# ## 15. Output location
#
# Kaggle execution writes exactly `/kaggle/working/submission.csv`. It displays only safe aggregate
# metadata and prediction-label counts, never IDs, individual labels, or test text.

# %%
def run_kaggle_inference(
    files: DiscoveredFiles, model: Pipeline, deployment_threshold: float
) -> None:
    test_frame = pd.read_csv(files.test)
    validate_test_frame(test_frame)
    test_texts = build_text_series(test_frame)
    probabilities_label1 = label_probability(model, test_texts, label=1)
    predicted_labels = predictions_from_label1(probabilities_label1, deployment_threshold)
    sample = pd.read_csv(files.sample_submission) if files.sample_submission else None
    submission = build_submission(test_frame["id"], predicted_labels, sample)
    output_path = Path("/kaggle/working/submission.csv")
    submission.to_csv(output_path, index=False)

    print("Selected test path:", files.test)
    print("Selected sample-submission path:", files.sample_submission)
    print("Test row count:", len(test_frame))
    print("Test columns:", list(test_frame.columns))
    print("Missing-ID count:", int(test_frame["id"].isna().sum()))
    print("Duplicate-ID count:", int(test_frame["id"].duplicated().sum()))
    print("Prediction label counts:", pd.Series(predicted_labels).value_counts().sort_index().to_dict())
    print("Submission shape:", submission.shape)
    print("Submission columns:", list(submission.columns))
    print("Output location:", output_path)


# %% [markdown]
# ## 16. Limitations and next steps
#
# The official labeled sample is small, the exact competition metric remains ambiguous, and the
# full-OOF deployment-threshold score is optimistic. The nested estimate is the honest threshold-
# tuned result. This Version 1 intentionally excludes unresolved external data and all large models.

# %%
def main() -> None:
    started = perf_counter()
    kaggle_input = Path("/kaggle/input")
    running_on_kaggle = kaggle_input.is_dir()
    if running_on_kaggle:
        files = discover_kaggle_files(kaggle_input)
        train_path = files.train
        print("Selected labeled-data path:", train_path)
    else:
        files = None
        train_path = Path.cwd() / "data" / "competition" / TRAIN_NAMES[0]
        print("Local mode: using the explicit official labeled-sample path.")

    frame = load_labeled_json(train_path, expected_sha256=OFFICIAL_SAMPLE_SHA256)
    deployment_threshold, final_model = safe_validation_report(frame)

    if running_on_kaggle:
        assert files is not None
        run_kaggle_inference(files, final_model, deployment_threshold)
    else:
        print("Kaggle-only inference intentionally skipped; no local test file was opened.")
    print("Total runtime seconds:", round(perf_counter() - started, 3))


if __name__ == "__main__":
    main()
