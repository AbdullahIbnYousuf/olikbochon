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
# Train the unchanged Version 1 word/character TF-IDF logistic-regression family on the approved
# public 4k split, select one macro-F1 threshold on the public 1k split, independently evaluate the
# official labeled sample, then refit on every unique allowed labeled row.

# %% [markdown]
# ## 2. Version 1 reference
#
# Version 1 used 299 official rows, threshold 0.53, and scored 0.466 publicly on Kaggle in 53
# seconds. Its official-sample metrics were OOF estimates; Version 2 uses a separately trained
# public-data model, so the comparisons are informative but not perfectly identical.

# %% [markdown]
# ## 3. Version 2 data roles
#
# Public 4k is development training, public 1k selects the threshold, and the official sample is an
# untouched independent evaluation set. The public aggregate is audit-only. Every test CSV is
# excluded from all labeled-data code.

# %% [markdown]
# ## 4. Offline constraints
#
# CPU only, internet disabled, no installs, APIs, model downloads, transformers, or repository
# imports. Local execution never loads the competition test CSV.

# %% [markdown]
# ## 5. Imports

# %%
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion, Pipeline

# %% [markdown]
# ## 6. Configuration

# %%
RANDOM_STATE = 42
PUBLIC_AGGREGATE_FILENAME = "bangla_hallucination_5k_contrastive.json"
PUBLIC_TRAIN_FILENAME = "bangla_hallucination_5k_train.json"
PUBLIC_VALIDATION_FILENAME = "bangla_hallucination_5k_validation.json"
OFFICIAL_FILENAME = "dataset samples.json"
TEST_FILENAME = "test set.csv"
SAMPLE_SUBMISSION_NAMES = ("sample submission.csv", "sample_submission.csv")
LABELED_COLUMNS = ("context", "prompt_bn", "response_bn", "label")
TEST_COLUMNS = ("id", "context", "prompt_bn", "response_bn")
SUBMISSION_COLUMNS = ("id", "label")
THRESHOLDS = np.arange(20, 81, dtype=np.int64) / 100.0
SIMILARITY_THRESHOLD = 0.97
LENGTH_RATIO_THRESHOLD = 0.90
BLOCK_SIZE = 256

PROMPT_MARKER = "__PROMPT__"
CONTEXT_PRESENT_MARKER = "__CONTEXT_PRESENT__"
CONTEXT_MARKER = "__CONTEXT__"
RESPONSE_MARKER = "__RESPONSE__"
_WHITESPACE_RE = re.compile(r"\s+")
_MISSING_CONTEXT_MARKERS = frozenset({"", "[null]", "nan", "none", "null", "na", "n/a"})


class DataSafetyError(RuntimeError):
    pass


class DiscoveryError(RuntimeError):
    pass


class SubmissionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class V2Files:
    public_train: Path
    public_validation: Path
    public_aggregate: Path
    official_train: Path
    test: Path
    sample_submission: Path
    public_roots: tuple[Path, ...]
    competition_root: Path


@dataclass(frozen=True)
class NearAudit:
    left_name: str
    right_name: str
    left_rows: int
    right_rows: int
    pairs: int
    affected_left: tuple[int, ...]
    affected_right: tuple[int, ...]
    agreements: int
    conflicts: int
    maximum: float
    percentiles: dict[str, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# %% [markdown]
# ## 7. File discovery
#
# Directories containing both approved public splits are public roles and all CSVs below them are
# quarantined. A competition root must co-locate the official sample, test, and sample submission.

# %%
def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _identical(paths: Sequence[Path]) -> bool:
    return len({sha256_file(path) for path in paths}) == 1


def discover_files(root: Path) -> V2Files:
    root = Path(root)
    train_parents = {path.parent for path in root.rglob(PUBLIC_TRAIN_FILENAME) if path.is_file()}
    validation_parents = {
        path.parent for path in root.rglob(PUBLIC_VALIDATION_FILENAME) if path.is_file()
    }
    public_roots = sorted(train_parents.intersection(validation_parents), key=str)
    if not public_roots:
        raise DiscoveryError("No public directory contains both approved 4k and 1k files")

    def public_copy(filename: str) -> Path:
        paths = [directory / filename for directory in public_roots]
        if not all(path.is_file() for path in paths) or not _identical(paths):
            raise DiscoveryError(f"Missing or conflicting public copies of {filename!r}")
        return min(paths, key=str)

    public_train = public_copy(PUBLIC_TRAIN_FILENAME)
    public_validation = public_copy(PUBLIC_VALIDATION_FILENAME)
    public_aggregate = public_copy(PUBLIC_AGGREGATE_FILENAME)

    candidate_roots: set[Path] = set()
    for sample_name in SAMPLE_SUBMISSION_NAMES:
        for sample_path in sorted(root.rglob(sample_name)):
            candidate = sample_path.parent
            if any(_within(candidate, public_root) for public_root in public_roots):
                continue
            if (candidate / OFFICIAL_FILENAME).is_file() and (
                candidate / TEST_FILENAME
            ).is_file():
                candidate_roots.add(candidate)
    candidates = [
        (
            next(
                preference
                for preference, sample_name in enumerate(SAMPLE_SUBMISSION_NAMES)
                if (candidate / sample_name).is_file()
            ),
            candidate,
        )
        for candidate in sorted(candidate_roots, key=str)
    ]
    if not candidates:
        raise DiscoveryError("No coherent non-public competition root was found")

    signatures = set()
    for preference, candidate in candidates:
        signatures.add(
            (
                sha256_file(candidate / OFFICIAL_FILENAME),
                sha256_file(candidate / TEST_FILENAME),
                sha256_file(candidate / SAMPLE_SUBMISSION_NAMES[preference]),
            )
        )
    if len(signatures) != 1:
        raise DiscoveryError("Conflicting non-identical competition roots")
    preference, competition_root = min(candidates, key=lambda item: (item[0], str(item[1])))
    test = competition_root / TEST_FILENAME
    if any(_within(test, public_root) for public_root in public_roots):
        raise DiscoveryError("Competition test resolved inside a quarantined public directory")
    return V2Files(
        public_train,
        public_validation,
        public_aggregate,
        competition_root / OFFICIAL_FILENAME,
        test,
        competition_root / SAMPLE_SUBMISSION_NAMES[preference],
        tuple(public_roots),
        competition_root,
    )


# %% [markdown]
# ## 8. Safe duplicate audit
#
# Text fingerprints omit labels to expose conflicts. Labeled-row fingerprints include labels to
# identify complete duplicates. Near-duplicate logs contain only aggregate counts and similarity.

# %%
def _missing(value: Any) -> bool:
    return value is None or value is pd.NA or (
        isinstance(value, float) and math.isnan(value)
    )


def normalize_text(value: Any) -> str:
    if _missing(value):
        return ""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", str(value))).strip()


def normalize_context(value: Any) -> str:
    normalized = normalize_text(value)
    return "" if normalized.casefold() in _MISSING_CONTEXT_MARKERS else normalized


def build_marked_text(prompt: Any, context: Any, response: Any) -> str:
    prompt_text = normalize_text(prompt)
    context_text = normalize_context(context)
    response_text = normalize_text(response)
    return (
        f"{PROMPT_MARKER}\n{prompt_text}\n\n"
        f"{CONTEXT_PRESENT_MARKER}\n{int(bool(context_text))}\n\n"
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
    )


def text_fingerprint(prompt: Any, context: Any, response: Any) -> str:
    return hashlib.sha256(build_marked_text(prompt, context, response).encode()).hexdigest()


def labeled_fingerprint(prompt: Any, context: Any, response: Any, label: int) -> str:
    value = f"{build_marked_text(prompt, context, response)}\n\n__LABEL__\n{int(label)}"
    return hashlib.sha256(value.encode()).hexdigest()


def validate_labeled(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != LABELED_COLUMNS or frame.empty:
        raise DataSafetyError("Invalid labeled schema or empty labeled file")
    if frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
        raise DataSafetyError("Labels must be nonmissing binary integers")
    if not pd.api.types.is_integer_dtype(frame["label"].dtype):
        raise DataSafetyError("Labels must have integer dtype")
    if set(frame["label"]) != {0, 1}:
        raise DataSafetyError("Both labels must be present")
    if frame["prompt_bn"].isna().any() or frame["response_bn"].isna().any():
        raise DataSafetyError("Prompt and response must not be missing")
    result = frame.copy(deep=True)
    result["label"] = result["label"].astype(np.int64)
    return result


def load_labeled(path: Path, expected_name: str) -> pd.DataFrame:
    if Path(path).name != expected_name:
        raise DataSafetyError(f"Unexpected labeled filename: {Path(path).name!r}")
    frame = validate_labeled(pd.read_json(path, orient="records"))
    keys = [
        (
            text_fingerprint(row.prompt_bn, row.context, row.response_bn),
            labeled_fingerprint(row.prompt_bn, row.context, row.response_bn, row.label),
        )
        for row in frame.itertuples(index=False)
    ]
    frame["text_fingerprint"] = [item[0] for item in keys]
    frame["labeled_row_fingerprint"] = [item[1] for item in keys]
    return frame


def exact_audit(train: pd.DataFrame, validation: pd.DataFrame, aggregate: pd.DataFrame, official: pd.DataFrame) -> dict[str, Any]:
    named = {"public_train": train, "public_validation": validation, "public_aggregate": aggregate, "official": official}
    combined = pd.concat(
        [part[["text_fingerprint", "label"]].assign(partition=name) for name, part in named.items()],
        ignore_index=True,
    )
    conflicts = combined.groupby("text_fingerprint")["label"].nunique()
    conflict_count = int((conflicts > 1).sum())
    if conflict_count:
        affected = sorted(combined.loc[combined["text_fingerprint"].isin(conflicts[conflicts > 1].index), "partition"].unique())
        raise DataSafetyError(f"Found {conflict_count} text-label conflicts in {affected}")
    composition = Counter(aggregate["labeled_row_fingerprint"]) == (
        Counter(train["labeled_row_fingerprint"]) + Counter(validation["labeled_row_fingerprint"])
    )
    if not composition:
        raise DataSafetyError("Public 4k plus 1k do not compose the aggregate")
    pairs = {
        "train__validation": (train, validation),
        "public__official": (pd.concat([train, validation]), official),
    }
    return {
        "aggregate_composes_split": composition,
        "conflicting_texts": conflict_count,
        "partition_rows": {name: len(part) for name, part in named.items()},
        "text_duplicate_rows": {name: int(part["text_fingerprint"].duplicated(keep=False).sum()) for name, part in named.items()},
        "labeled_duplicate_rows": {name: int(part["labeled_row_fingerprint"].duplicated(keep=False).sum()) for name, part in named.items()},
        "text_overlaps": {name: len(set(left["text_fingerprint"]) & set(right["text_fingerprint"])) for name, (left, right) in pairs.items()},
    }


def deduplicate_exact_roles(
    train: pd.DataFrame, validation: pd.DataFrame, official: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    official_clean = official.drop_duplicates(
        "labeled_row_fingerprint", keep="first"
    ).reset_index(drop=True)
    validation_clean = validation.loc[
        ~validation["text_fingerprint"].isin(set(official_clean["text_fingerprint"]))
    ].drop_duplicates("labeled_row_fingerprint", keep="first").reset_index(drop=True)
    protected_text = set(official_clean["text_fingerprint"]).union(
        validation_clean["text_fingerprint"]
    )
    train_clean = train.loc[
        ~train["text_fingerprint"].isin(protected_text)
    ].drop_duplicates("labeled_row_fingerprint", keep="first").reset_index(drop=True)
    return train_clean, validation_clean, official_clean, {
        "public_train": len(train) - len(train_clean),
        "public_validation": len(validation) - len(validation_clean),
        "official": len(official) - len(official_clean),
    }


def near_audit(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> NearAudit:
    left_text = build_text_series(left).astype(str).to_numpy(dtype=str)
    right_text = build_text_series(right).astype(str).to_numpy(dtype=str)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=1.0, sublinear_tf=True, lowercase=False, strip_accents=None, dtype=np.float32, norm="l2")
    matrix = vectorizer.fit_transform(np.concatenate([left_text, right_text])).tocsr()
    left_matrix, right_matrix = matrix[: len(left)], matrix[len(left) :]
    left_lengths = np.char.str_len(left_text).astype(np.int64)
    right_lengths = np.char.str_len(right_text).astype(np.int64)
    left_labels = left["label"].to_numpy(dtype=np.int64)
    right_labels = right["label"].to_numpy(dtype=np.int64)
    affected_left: set[int] = set()
    affected_right: set[int] = set()
    pairs = agreements = conflicts = 0
    maxima = np.zeros(len(left), dtype=np.float32)
    for start in range(0, len(left), BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, len(left))
        similarities = cosine_similarity(left_matrix[start:stop], right_matrix, dense_output=True).astype(np.float32, copy=False)
        maxima[start:stop] = similarities.max(axis=1)
        left_block = np.maximum(left_lengths[start:stop], 1)[:, None]
        right_safe = np.maximum(right_lengths, 1)[None, :]
        ratios = np.minimum(left_block, right_safe) / np.maximum(left_block, right_safe)
        for local_left, right_index in np.argwhere((similarities >= SIMILARITY_THRESHOLD) & (ratios >= LENGTH_RATIO_THRESHOLD)):
            left_index = start + int(local_left)
            right_index = int(right_index)
            affected_left.add(left_index)
            affected_right.add(right_index)
            pairs += 1
            if left_labels[left_index] == right_labels[right_index]:
                agreements += 1
            else:
                conflicts += 1
    values = np.percentile(maxima, [50, 90, 95, 99])
    return NearAudit(left_name, right_name, len(left), len(right), pairs, tuple(sorted(affected_left)), tuple(sorted(affected_right)), agreements, conflicts, float(maxima.max(initial=0.0)), {name: float(value) for name, value in zip(("p50", "p90", "p95", "p99"), values)})


def enforce_audits(train_validation: NearAudit, public_official: NearAudit) -> None:
    if train_validation.conflicts + public_official.conflicts:
        raise DataSafetyError("Near-duplicate label conflict detected")
    if len(train_validation.affected_right) / max(train_validation.right_rows, 1) > 0.01:
        raise DataSafetyError("More than 1% of public validation matches public training")
    if len(public_official.affected_right) / max(public_official.right_rows, 1) > 0.01:
        raise DataSafetyError("More than 1% of official evaluation matches public data")


# %% [markdown]
# ## 9. Training-data loading

# %%
def load_and_audit(files: V2Files) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = load_labeled(files.public_train, PUBLIC_TRAIN_FILENAME).reset_index(drop=True)
    validation = load_labeled(files.public_validation, PUBLIC_VALIDATION_FILENAME).reset_index(drop=True)
    aggregate = load_labeled(files.public_aggregate, PUBLIC_AGGREGATE_FILENAME).reset_index(drop=True)
    official = load_labeled(files.official_train, OFFICIAL_FILENAME).reset_index(drop=True)
    exact = exact_audit(train, validation, aggregate, official)
    train, validation, official, exact_removals = deduplicate_exact_roles(
        train, validation, official
    )
    train_validation = near_audit(train, validation, "public_train", "public_validation")
    public = pd.concat([train, validation], ignore_index=True)
    public_official = near_audit(public, official, "public_5k", "official")
    enforce_audits(train_validation, public_official)
    # Passing audits currently contain no flagged rows. These removals protect holdouts if a future
    # byte-identical approved version contains a small number of same-label near matches.
    train_drop = set(train_validation.affected_left)
    for position in public_official.affected_left:
        if position < len(train):
            train_drop.add(position)
    validation_drop = {position - len(train) for position in public_official.affected_left if position >= len(train)}
    train = train.drop(index=sorted(train_drop)).reset_index(drop=True)
    validation = validation.drop(index=sorted(validation_drop)).reset_index(drop=True)
    audit = {
        "exact": exact,
        "near_train_validation": {"pairs": train_validation.pairs, "affected_train": len(train_validation.affected_left), "affected_validation": len(train_validation.affected_right), "agreements": train_validation.agreements, "conflicts": train_validation.conflicts, "maximum": train_validation.maximum, "percentiles": train_validation.percentiles},
        "near_public_official": {"pairs": public_official.pairs, "affected_public": len(public_official.affected_left), "affected_official": len(public_official.affected_right), "agreements": public_official.agreements, "conflicts": public_official.conflicts, "maximum": public_official.maximum, "percentiles": public_official.percentiles},
        "exact_removals": exact_removals,
        "near_removals": {"public_train": len(train_drop), "public_validation": len(validation_drop), "official": 0},
        "total_removals": {
            "public_train": exact_removals["public_train"] + len(train_drop),
            "public_validation": exact_removals["public_validation"] + len(validation_drop),
            "official": exact_removals["official"],
        },
    }
    return train, validation, official, audit


# %% [markdown]
# ## 10. Text preprocessing
#
# NFC, missing-context normalization, whitespace collapse, and fixed field markers are identical
# across audit, development, final refit, and inference.

# %% [markdown]
# ## 11. Development training

# %%
def build_model() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20_000, min_df=1, max_df=1.0, sublinear_tf=True, lowercase=False, strip_accents=None, dtype=np.float32, token_pattern=r"(?u)\b\w+\b", norm="l2")),
            ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30_000, min_df=1, max_df=1.0, sublinear_tf=True, lowercase=False, strip_accents=None, dtype=np.float32, norm="l2")),
        ]
    )
    classifier = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("features", features), ("classifier", classifier)])


def label1_probability(model: Pipeline, texts: Sequence[str]) -> np.ndarray:
    classes = np.asarray(model.named_steps["classifier"].classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("Classifier classes do not contain label 1 exactly once")
    return model.predict_proba(texts)[:, int(matches[0])]


# %% [markdown]
# ## 12. Public-validation threshold selection

# %%
def predictions(probabilities: Sequence[float], threshold: float) -> np.ndarray:
    return (np.asarray(probabilities, dtype=np.float64) >= float(threshold)).astype(np.int64)


def metrics(truth: Sequence[int], predicted: Sequence[int]) -> dict[str, Any]:
    truth_array = np.asarray(truth, dtype=np.int64)
    predicted_array = np.asarray(predicted, dtype=np.int64)
    return {"f1_label0": float(f1_score(truth_array, predicted_array, pos_label=0, zero_division=0)), "f1_label1": float(f1_score(truth_array, predicted_array, pos_label=1, zero_division=0)), "macro_f1": float(f1_score(truth_array, predicted_array, average="macro", zero_division=0)), "accuracy": float(accuracy_score(truth_array, predicted_array)), "confusion_matrix": confusion_matrix(truth_array, predicted_array, labels=[0, 1]).astype(int).tolist()}


def threshold_table(truth: Sequence[int], probabilities: Sequence[float]) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        row = metrics(truth, predictions(probabilities, float(threshold)))
        rows.append({"threshold": float(threshold), "f1_label0": row["f1_label0"], "f1_label1": row["f1_label1"], "macro_f1": row["macro_f1"], "accuracy": row["accuracy"]})
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> float:
    ranked = table.assign(distance=(table["threshold"] - 0.5).abs()).sort_values(["macro_f1", "f1_label0", "distance", "threshold"], ascending=[False, False, True, True], kind="mergesort")
    return float(ranked.iloc[0]["threshold"])


def distribution(values: Sequence[int]) -> dict[int, int]:
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=2)
    return {0: int(counts[0]), 1: int(counts[1])}


def collapse_warning(values: Sequence[int], name: str) -> str | None:
    counts = distribution(values)
    total = counts[0] + counts[1]
    dominant = max(counts, key=counts.get)
    share = counts[dominant] / total
    if share <= 0.90:
        return None
    return f"WARNING: prediction collapse for {name}: label {dominant} is {share:.1%} of predictions. High class-specific F1 may reflect collapse rather than useful discrimination."


# %% [markdown]
# ## 13. Independent official-sample evaluation

# %%
def context_metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, Any]:
    truth = frame["label"].to_numpy(dtype=np.int64)
    presence = frame["context"].map(lambda value: bool(normalize_context(value))).to_numpy()
    result = {}
    for name, mask in (("context_present", presence), ("context_absent", ~presence)):
        result[name] = {"count": int(mask.sum()), **metrics(truth[mask], predicted[mask])}
    return result


def development_evaluation(train: pd.DataFrame, validation: pd.DataFrame, official: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    model = build_model()
    model.fit(build_text_series(train), train["label"].to_numpy(dtype=np.int64))
    validation_truth = validation["label"].to_numpy(dtype=np.int64)
    validation_probabilities = label1_probability(model, build_text_series(validation))
    table = threshold_table(validation_truth, validation_probabilities)
    selected = select_threshold(table)
    validation_fixed = predictions(validation_probabilities, 0.5)
    validation_selected = predictions(validation_probabilities, selected)
    official_truth = official["label"].to_numpy(dtype=np.int64)
    official_probabilities = label1_probability(model, build_text_series(official))
    official_fixed = predictions(official_probabilities, 0.5)
    official_selected = predictions(official_probabilities, selected)
    sets = {"public_validation_fixed_050": validation_fixed, "public_validation_threshold_tuning": validation_selected, "official_fixed_050": official_fixed, "official_selected": official_selected}
    warnings = [warning for name, values in sets.items() if (warning := collapse_warning(values, name))]
    report = {
        "selected_threshold": selected,
        "public_validation_fixed_050": metrics(validation_truth, validation_fixed),
        "Public-validation threshold-tuning estimate": metrics(validation_truth, validation_selected),
        "Independent official-sample evaluation fixed_050": metrics(official_truth, official_fixed),
        "Independent official-sample evaluation selected": metrics(official_truth, official_selected),
        "official_fixed_context": context_metrics(official, official_fixed),
        "official_selected_context": context_metrics(official, official_selected),
        "prediction_distributions": {name: distribution(values) for name, values in sets.items()},
        "collapse_warnings": warnings,
        "threshold_table": table.to_dict(orient="records"),
    }
    return selected, report


# %% [markdown]
# ## 14. Final combined-data training
#
# The threshold is frozen before this refit. Adding former validation and official rows can shift
# probability calibration, so the threshold may be slightly less optimal; Version 2 accepts this
# limitation and never retunes from the official sample or leaderboard.

# %%
def final_refit(train: pd.DataFrame, validation: pd.DataFrame, official: pd.DataFrame) -> tuple[Pipeline, int]:
    combined = pd.concat([train, validation, official], ignore_index=True)
    conflicts = combined.groupby("text_fingerprint")["label"].nunique()
    if bool((conflicts > 1).any()):
        raise DataSafetyError("Conflicting labels remain in final combined data")
    combined = combined.drop_duplicates("labeled_row_fingerprint", keep="first")
    model = build_model()
    model.fit(build_text_series(combined), combined["label"].to_numpy(dtype=np.int64))
    return model, len(combined)


# %% [markdown]
# ## 15. Kaggle-only test inference

# %%
def validate_test(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != TEST_COLUMNS or frame["id"].isna().any():
        raise SubmissionValidationError("Invalid test schema or missing IDs")


# %% [markdown]
# ## 16. Submission validation

# %%
def build_submission(ids: pd.Series, labels: Sequence[int], sample: pd.DataFrame) -> pd.DataFrame:
    label_array = np.asarray(labels)
    if not np.issubdtype(label_array.dtype, np.integer) or len(label_array) != len(ids):
        raise SubmissionValidationError("Labels must be integer and match test row count")
    result = pd.DataFrame({"id": ids.reset_index(drop=True), "label": label_array.astype(np.int64)})
    normalized_ids = ids.reset_index(drop=True)
    if tuple(result.columns) != SUBMISSION_COLUMNS or result["id"].tolist() != normalized_ids.tolist():
        raise SubmissionValidationError("Submission columns or ID order are invalid")
    if result["label"].isna().any() or not result["label"].isin([0, 1]).all():
        raise SubmissionValidationError("Submission labels must be binary integers")
    if tuple(sample.columns) != SUBMISSION_COLUMNS or len(sample) != len(result) or sample["id"].tolist() != normalized_ids.tolist():
        raise SubmissionValidationError("Sample submission does not match test IDs")
    return result


# %% [markdown]
# ## 17. Output files
#
# `submission.csv` uses the frozen macro-F1-selected threshold. `submission_fixed_050.csv` is the
# clean reference. Both are independently validated; no experimental class-0 file is created.

# %%
def kaggle_inference(files: V2Files, model: Pipeline, selected_threshold: float) -> None:
    test = pd.read_csv(files.test)
    validate_test(test)
    probabilities = label1_probability(model, build_text_series(test))
    sample = pd.read_csv(files.sample_submission)
    outputs = {
        "submission.csv": build_submission(test["id"], predictions(probabilities, selected_threshold), sample),
        "submission_fixed_050.csv": build_submission(test["id"], predictions(probabilities, 0.5), sample),
    }
    for filename, submission in outputs.items():
        output = Path("/kaggle/working") / filename
        submission.to_csv(output, index=False)
        print(filename, {"shape": submission.shape, "columns": list(submission.columns), "label_counts": submission["label"].value_counts().sort_index().to_dict(), "output": str(output)})


# %% [markdown]
# ## 18. Limitations and next steps
#
# Public validation is a threshold-tuning estimate, not independent. Only the official evaluation
# is independent. The final-refit calibration shift is accepted. One Kaggle score must not drive
# repeated threshold probing.

# %%
def main() -> None:
    started = perf_counter()
    kaggle_root = Path("/kaggle/input")
    running_on_kaggle = kaggle_root.is_dir()
    root = kaggle_root if running_on_kaggle else Path.cwd() / "data"
    files = discover_files(root)
    train, validation, official, audit = load_and_audit(files)
    selected_threshold, evaluation = development_evaluation(train, validation, official)
    final_model, final_count = final_refit(train, validation, official)
    print("Safe file roles:", {"public_train": str(files.public_train), "public_validation": str(files.public_validation), "public_aggregate_audit_only": str(files.public_aggregate), "official_train": str(files.official_train), "competition_root": str(files.competition_root)})
    print("Audit:", json.dumps(audit, indent=2, sort_keys=True))
    print("Evaluation:", json.dumps(evaluation, indent=2, sort_keys=True))
    print("Final combined unique labeled count:", final_count)
    print("Package versions:", {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__})
    if running_on_kaggle:
        kaggle_inference(files, final_model, selected_threshold)
    else:
        print("Kaggle-only inference skipped; no local competition-test text was loaded.")
    print("Runtime seconds:", round(perf_counter() - started, 3))


if __name__ == "__main__":
    main()
