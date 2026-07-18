"""Discovery-only V11 label-definition and frozen-model error audit."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from .metrics import classification_metrics, predictions_from_label1
from .v10_ensemble import CANDIDATE_I, _label1_probability, _pipeline
from .v4_preprocessing import official_context_is_present
from .v5_lexical import build_v5_folds, build_v5_groups
from .v5_normalization import normalize_v5
from .v5_sparse import SparseNullModel


SPLIT_SEEDS = (20260719, 20260720, 20260721, 20260722, 20260723)
EXPECTED_PRESENT_ROWS = 130
EXPECTED_ABSENT_ROWS = 169
THRESHOLD = 0.50
NEAR_DUPLICATE_THRESHOLD = 0.90
PRIMARY_FAMILIES = (
    "factual entity question",
    "definition or explanation",
    "numeric, date, quantity, or ranking",
    "list or enumeration",
    "procedural instruction or advice",
    "comparison or recommendation",
    "translation or text transformation",
    "summarization",
    "creative, opinion, or open-ended generation",
    "current or time-sensitive information",
    "ambiguous or underspecified request",
    "other",
)


@dataclass(frozen=True)
class LockedSplit:
    discovery_indices: np.ndarray
    holdout_indices: np.ndarray
    record: dict[str, Any]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_hash(indices: Sequence[int]) -> str:
    values = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def lock_discovery_split(frame: pd.DataFrame, discovery_fraction: float) -> LockedSplit:
    """Lock the first feasible group-only split without consulting any model score."""
    if discovery_fraction != 0.75:
        raise ValueError("V11 discovery fraction is frozen at 0.75")
    presence = np.asarray(
        [official_context_is_present(value) for value in frame["context"]], dtype=bool
    )
    if int(presence.sum()) != EXPECTED_PRESENT_ROWS or int((~presence).sum()) != EXPECTED_ABSENT_ROWS:
        raise RuntimeError("V11 corrected route totals differ from 130/169")
    null_indices = np.flatnonzero(~presence)
    labels = frame["label"].to_numpy(dtype=np.int64)[null_indices]
    audit = build_v5_groups(frame)
    groups = np.asarray(audit.group_ids, dtype=object)[null_indices]
    for seed in SPLIT_SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, train_size=discovery_fraction, random_state=seed)
        discovery_local, holdout_local = next(splitter.split(null_indices, labels, groups))
        if set(groups[discovery_local]) & set(groups[holdout_local]):
            continue
        if set(labels[discovery_local]) != {0, 1} or set(labels[holdout_local]) != {0, 1}:
            continue
        discovery = np.sort(null_indices[discovery_local])
        holdout = np.sort(null_indices[holdout_local])
        discovery_groups = sorted(set(groups[discovery_local]))
        holdout_groups = sorted(set(groups[holdout_local]))
        record = {
            "selected_seed": seed,
            "discovery_fraction_target": discovery_fraction,
            "discovery_row_count": int(discovery.size),
            "holdout_row_count": int(holdout.size),
            "discovery_label_counts": {
                str(key): int(value)
                for key, value in enumerate(np.bincount(frame.iloc[discovery]["label"], minlength=2))
            },
            "holdout_label_counts": {
                str(key): int(value)
                for key, value in enumerate(np.bincount(frame.iloc[holdout]["label"], minlength=2))
            },
            "discovery_row_index_sha256": _index_hash(discovery),
            "holdout_row_index_sha256": _index_hash(holdout),
            "discovery_group_hashes": [_sha256_json(group) for group in discovery_groups],
            "holdout_group_hashes": [_sha256_json(group) for group in holdout_groups],
            "discovery_group_count": len(discovery_groups),
            "holdout_group_count": len(holdout_groups),
            "group_overlap_count": 0,
            "model_scores_used": False,
        }
        return LockedSplit(discovery, holdout, record)
    raise RuntimeError("No approved deterministic seed produced a feasible V11 split")


def _contains(text: str, terms: Iterable[str]) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def _primary_family(prompt: str) -> str:
    normalized = normalize_v5(prompt)
    if _contains(normalized, ("বর্তমান", "সর্বশেষ", "আজ", "এখন", "current", "latest", "today")):
        return PRIMARY_FAMILIES[9]
    if _contains(normalized, ("অনুবাদ", "translate", "rewrite", "রূপান্তর")):
        return PRIMARY_FAMILIES[6]
    if _contains(normalized, ("সারাংশ", "সংক্ষেপ", "summar")):
        return PRIMARY_FAMILIES[7]
    if _contains(normalized, ("তালিকা", "list", "কয়টি", "গুলো কী", "enumerat")):
        return PRIMARY_FAMILIES[3]
    if _contains(normalized, ("কীভাবে", "কিভাবে", "উপায়", "পরামর্শ", "how to", "advice")):
        return PRIMARY_FAMILIES[4]
    if _contains(normalized, ("তুলনা", "পার্থক্য", "সেরা", "ভালো", "compare", "recommend")):
        return PRIMARY_FAMILIES[5]
    if re.search(r"\d", normalized) or _contains(
        normalized, ("কত", "কবে", "তারিখ", "শতাংশ", "ranking", "rank")
    ):
        return PRIMARY_FAMILIES[2]
    if _contains(normalized, ("সংজ্ঞা", "ব্যাখ্যা", "কাকে বলে", "মানে কী", "explain", "define")):
        return PRIMARY_FAMILIES[1]
    if _contains(normalized, ("লিখ", "রচনা", "গল্প", "কবিতা", "মতামত", "imagine", "creative")):
        return PRIMARY_FAMILIES[8]
    if len(normalized) < 12 or _contains(normalized, ("এটি কী", "এটা কী", "কি বলো")):
        return PRIMARY_FAMILIES[10]
    if "?" in normalized or _contains(normalized, ("কে", "কোথায়", "কোন", "কি", "কী")):
        return PRIMARY_FAMILIES[0]
    return PRIMARY_FAMILIES[11]


def _risk_tags(prompt: str, response: str) -> list[str]:
    combined = f"{prompt} {response}"
    response_norm = normalize_v5(response)
    tags: list[str] = []
    patterns: tuple[tuple[str, bool], ...] = (
        ("numeric claim", bool(re.search(r"\d", response_norm))),
        ("date or temporal claim", bool(re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", response_norm)) or _contains(response_norm, ("সাল", "তারিখ", "বছর"))),
        ("geographic claim", _contains(combined, ("দেশ", "শহর", "জেলা", "নদী", "রাজধানী", "geograph"))),
        ("superlative or ranking", _contains(combined, ("সর্ব", "প্রথম", "শ্রেষ্ঠ", "বৃহত্তম", "ranking"))),
        ("causal claim", _contains(response_norm, ("কারণ", "ফলে", "তাই", "because", "caused"))),
        ("quotation or attribution", bool(re.search(r"[\"“”‘’]", response_norm)) or _contains(response_norm, ("বলেছেন", "মতে"))),
        ("citation, source, or URL claim", bool(re.search(r"https?://|www\.|\[[0-9]+\]", response_norm))),
        ("legal claim", _contains(combined, ("আইন", "আদালত", "legal", "সংবিধান"))),
        ("medical claim", _contains(combined, ("রোগ", "চিকিৎসা", "ওষুধ", "স্বাস্থ্য", "medical"))),
        ("financial claim", _contains(combined, ("টাকা", "ব্যাংক", "বিনিয়োগ", "অর্থ", "financial"))),
        ("technical specification", _contains(combined, ("প্রযুক্তি", "সফটওয়্যার", "হার্ডওয়্যার", "মডেল", "version", "specification"))),
        ("unsupported certainty", _contains(response_norm, ("নিশ্চিত", "অবশ্যই", "নিঃসন্দেহে", "definitely"))),
        ("refusal or uncertainty", _contains(response_norm, ("জানি না", "বলতে পারি না", "দুঃখিত", "নিশ্চিত নই", "cannot", "don't know"))),
        ("hedging language", _contains(response_norm, ("সম্ভবত", "হতে পারে", "মনে হয়", "perhaps", "likely"))),
        ("prompt echo", normalize_v5(prompt) == response_norm or (len(response_norm) > 20 and response_norm in normalize_v5(prompt))),
        ("generic low-information answer", len(response_norm) < 25),
        ("self-contradiction", _contains(response_norm, ("কিন্তু একই", "আবার নয়", "however, not"))),
        ("instruction noncompliance", len(response_norm) == 0),
        ("malformed or incomplete response", len(response_norm) < 5 or response_norm.endswith(("…", "...", ":"))),
    )
    tags.extend(name for name, present in patterns if present)
    bengali_entity = re.findall(r"(?:^|[।.!?]\s+)([\u0980-\u09ff]{3,})", response_norm)
    if bengali_entity or re.search(r"\b[A-Z][a-z]{2,}\b", response):
        tags.append("named entity")
    if len(response_norm) > 160 and ("named entity" in tags or "numeric claim" in tags):
        tags.append("unverifiable specificity")
    return sorted(set(tags))


def _structural(prompt: str, response: str) -> dict[str, Any]:
    prompt_tokens = set(normalize_v5(prompt).split())
    response_tokens = normalize_v5(response).split()
    response_set = set(response_tokens)
    overlap = len(prompt_tokens & response_set) / max(1, len(prompt_tokens))
    repetition = 1.0 - len(response_set) / max(1, len(response_tokens))
    return {
        "response_character_length": len(response),
        "response_token_estimate": len(response_tokens),
        "number_count": len(re.findall(r"\d+(?:[.,]\d+)?", response)),
        "date_like_pattern_count": len(re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", response)),
        "named_entity_like_token_count": len(re.findall(r"\b[A-Z][a-z]{2,}\b", response)),
        "sentence_count": max(1, len(re.findall(r"[।.!?]+", response))),
        "question_response_lexical_overlap": float(overlap),
        "response_repetition_ratio": float(repetition),
        "url_or_citation_presence": bool(re.search(r"https?://|www\.|\[[0-9]+\]", response)),
        "refusal_flag": _contains(response, ("জানি না", "বলতে পারি না", "দুঃখিত", "cannot")),
        "hedging_flag": _contains(response, ("সম্ভবত", "হতে পারে", "মনে হয়", "perhaps")),
    }


def blind_taxonomy(discovery_frame: pd.DataFrame, *, workers: int) -> pd.DataFrame:
    """Tag discovery text without accepting labels or predictions as inputs."""
    if workers != 4:
        raise ValueError("V11 automatic tagging requires four workers")

    def tag(item: tuple[int, str, str]) -> dict[str, Any]:
        official_index, prompt, response = item
        return {
            "official_index": official_index,
            "primary_task_family": _primary_family(prompt),
            "claim_risk_tags": _risk_tags(prompt, response),
            **_structural(prompt, response),
        }

    items = [
        (int(index), str(row.prompt_bn), str(row.response_bn))
        for index, row in discovery_frame.iterrows()
    ]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v11-tag") as executor:
        records = list(executor.map(tag, items))
    tagged = pd.DataFrame(records).sort_values("official_index").reset_index(drop=True)
    tagged.attrs["taxonomy_sha256"] = _sha256_json(tagged.to_dict(orient="records"))
    return tagged


def generate_discovery_oof(
    frame: pd.DataFrame,
    candidate_r_matrix: np.ndarray,
    discovery_indices: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Regenerate frozen I/R OOF values, predicting discovery validation rows only."""
    folds = build_v5_folds(frame)
    presence = np.asarray(
        [official_context_is_present(value) for value in frame["context"]], dtype=bool
    )
    absent_indices = np.flatnonzero(~presence)
    if candidate_r_matrix.shape != (len(absent_indices), 14):
        raise ValueError("V11 Candidate R feature matrix is not the frozen 169 by 14 matrix")
    absent_position = {int(index): position for position, index in enumerate(absent_indices)}
    discovery = set(map(int, discovery_indices))
    groups = np.asarray(folds.audit.group_ids, dtype=object)
    blocks: dict[int, dict[str, list[float]]] = {
        index: {"I": [], "R": []} for index in sorted(discovery)
    }
    provenance: list[dict[str, Any]] = []
    for split in folds.folds:
        train = np.asarray(split.train_indices, dtype=np.int64)
        validation = np.asarray(split.validation_indices, dtype=np.int64)
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("V11 source OOF fold has strengthened-group overlap")
        train_absent = train[~presence[train]]
        validation_discovery = np.asarray(
            [index for index in validation if int(index) in discovery], dtype=np.int64
        )
        if validation_discovery.size == 0:
            continue
        train_positions = np.asarray([absent_position[int(index)] for index in train_absent])
        validation_positions = np.asarray(
            [absent_position[int(index)] for index in validation_discovery]
        )
        train_frame = frame.iloc[train_absent].reset_index(drop=True)
        validation_frame = frame.iloc[validation_discovery].reset_index(drop=True)
        model_i = SparseNullModel(CANDIDATE_I).fit(train_frame)
        probability_i = model_i.predict_label1_probability(validation_frame)
        model_r = _pipeline().fit(
            candidate_r_matrix[train_positions], train_frame["label"].to_numpy(dtype=np.int64)
        )
        probability_r = _label1_probability(model_r, candidate_r_matrix[validation_positions])
        for index, value_i, value_r in zip(
            validation_discovery, probability_i, probability_r, strict=True
        ):
            blocks[int(index)]["I"].append(float(value_i))
            blocks[int(index)]["R"].append(float(value_r))
        provenance.append(
            {
                "seed": split.seed,
                "fold": split.fold,
                "train_index_sha256": _index_hash(train_absent),
                "predicted_discovery_index_sha256": _index_hash(validation_discovery),
                "train_rows": int(train_absent.size),
                "predicted_discovery_rows": int(validation_discovery.size),
                "group_overlap_count": 0,
                "holdout_predictions_generated": 0,
            }
        )
    records = []
    for index, values in blocks.items():
        if len(values["I"]) != 3 or len(values["R"]) != 3:
            raise RuntimeError("V11 discovery OOF coverage is not exactly once per seed")
        probability_i = float(np.mean(values["I"]))
        probability_r = float(np.mean(values["R"]))
        records.append(
            {
                "official_index": index,
                "candidate_i_probability": probability_i,
                "candidate_r_probability": probability_r,
                "candidate_u_probability": 0.5 * probability_i + 0.5 * probability_r,
            }
        )
    return pd.DataFrame(records), {
        "source_seeds": [17, 29, 43],
        "source_folds_per_seed": 5,
        "probability_aggregation": "mean_of_three_genuine_strengthened_group_oof_values",
        "label1_orientation": True,
        "discovery_rows_with_exact_three_seed_coverage": len(records),
        "holdout_predictions_generated": 0,
        "folds": provenance,
    }


def _metric_record(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = predictions_from_label1(probability, THRESHOLD)
    counts = np.bincount(prediction, minlength=2)
    calibration = []
    for lower in np.arange(0.0, 1.0, 0.2):
        upper = lower + 0.2
        mask = (probability >= lower) & (probability < upper if upper < 1.0 else probability <= 1.0)
        calibration.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()) if mask.any() else None,
                "label1_rate": float(truth[mask].mean()) if mask.any() else None,
            }
        )
    return {
        **classification_metrics(truth, prediction),
        "prediction_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "brier_score": float(brier_score_loss(truth, probability)),
        "false_positives": int(np.sum((truth == 0) & (prediction == 1))),
        "false_negatives": int(np.sum((truth == 1) & (prediction == 0))),
        "calibration_bins": calibration,
    }


def _f1_or_none(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    if truth.size < 2:
        return None
    return float(classification_metrics(truth, prediction)["macro_f1"])


def _error_reason(row: Any) -> str:
    tags = set(row.claim_risk_tags)
    wrong_i = row.candidate_i_prediction != row.gold_label
    wrong_r = row.candidate_r_prediction != row.gold_label
    if "numeric claim" in tags or "date or temporal claim" in tags:
        return "numeric/date inconsistency"
    if "refusal or uncertainty" in tags or "hedging language" in tags:
        return "refusal/hedging misinterpreted"
    if "generic low-information answer" in tags:
        return "generic but acceptable response"
    if row.primary_task_family == "creative, opinion, or open-ended generation":
        return "creative/non-factual task misclassified"
    if row.primary_task_family == "ambiguous or underspecified request":
        return "ambiguous annotation"
    if wrong_i and not wrong_r:
        return "model lexical shortcut"
    if wrong_r and not wrong_i:
        return "model semantic shortcut"
    if wrong_i and wrong_r and abs(row.candidate_i_probability - row.candidate_r_probability) < 0.05:
        return "suspected label inconsistency"
    if "named entity" in tags:
        return "incorrect entity or relation"
    if "unverifiable specificity" in tags or "unsupported certainty" in tags:
        return "unsupported factual specificity"
    return "answer plausibility mistaken for factuality"


def _duplicate_clusters(values: Sequence[str]) -> list[list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        buckets[value].append(index)
    return [members for members in buckets.values() if len(members) > 1]


def duplicate_audit(frame: pd.DataFrame, *, workers: int) -> tuple[dict[str, Any], set[int]]:
    if workers != 4:
        raise ValueError("V11 similarity audit requires four workers")
    prompts = [str(value) for value in frame["prompt_bn"]]
    responses = [str(value) for value in frame["response_bn"]]
    combined = [f"{prompt}\n{response}" for prompt, response in zip(prompts, responses, strict=True)]
    normalized = [normalize_v5(value) for value in combined]
    cluster_sets = {
        "exact_prompt": _duplicate_clusters(prompts),
        "exact_response": _duplicate_clusters(responses),
        "exact_prompt_response": _duplicate_clusters(combined),
        "normalized_prompt_response": _duplicate_clusters(normalized),
    }
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1).fit_transform(normalized)
    chunks = np.array_split(np.arange(len(frame)), workers)

    def compare(indices: np.ndarray) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for left in indices:
            similarities = (matrix[left] @ matrix[left + 1 :].T).toarray().ravel()
            pairs.extend(
                (int(left), int(left + 1 + offset))
                for offset in np.flatnonzero(similarities >= NEAR_DUPLICATE_THRESHOLD)
            )
        return pairs

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v11-sim") as executor:
        near_pairs = sorted(pair for block in executor.map(compare, chunks) for pair in block)
    parent = list(range(len(frame)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in near_pairs:
        first, second = find(left), find(right)
        if first != second:
            parent[max(first, second)] = min(first, second)
    near_buckets: dict[int, list[int]] = defaultdict(list)
    for index in range(len(frame)):
        near_buckets[find(index)].append(index)
    near_clusters = [members for members in near_buckets.values() if len(members) > 1]
    labels = frame["label"].to_numpy(dtype=np.int64)
    all_cluster_rows = {
        index
        for clusters in (*cluster_sets.values(), near_clusters)
        for members in clusters
        for index in members
    }

    def summarize(clusters: list[list[int]]) -> dict[str, int]:
        return {
            "cluster_count": len(clusters),
            "rows_in_clusters": len({index for members in clusters for index in members}),
            "mixed_label_clusters": sum(len(set(labels[members])) > 1 for members in clusters),
        }

    return {
        **{name: summarize(clusters) for name, clusters in cluster_sets.items()},
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_pair_count": len(near_pairs),
        "near_duplicate": summarize(near_clusters),
    }, all_cluster_rows


def analyze_discovery(
    frame: pd.DataFrame,
    discovery_indices: Sequence[int],
    taxonomy: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    similarity_workers: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    discovery = frame.iloc[list(discovery_indices)].copy()
    discovery["official_index"] = list(discovery_indices)
    joined = taxonomy.merge(oof, on="official_index", validate="one_to_one").merge(
        discovery[["official_index", "label"]], on="official_index", validate="one_to_one"
    )
    if len(joined) != len(discovery) or joined.isna().any().any():
        raise RuntimeError("V11 discovery taxonomy/OOF alignment is incomplete")
    joined = joined.rename(columns={"label": "gold_label"})
    for short in ("i", "r", "u"):
        joined[f"candidate_{short}_prediction"] = predictions_from_label1(
            joined[f"candidate_{short}_probability"], THRESHOLD
        )
        joined[f"candidate_{short}_confidence"] = np.maximum(
            joined[f"candidate_{short}_probability"],
            1.0 - joined[f"candidate_{short}_probability"],
        )
    review = (
        (joined.candidate_i_prediction != joined.candidate_r_prediction)
        | (
            (joined.candidate_i_prediction != joined.gold_label)
            & (joined.candidate_r_prediction != joined.gold_label)
        )
        | (
            ((joined.candidate_i_probability < 0.1) | (joined.candidate_i_probability > 0.9))
            & (joined.candidate_i_prediction != joined.gold_label)
        )
        | (
            ((joined.candidate_r_probability < 0.1) | (joined.candidate_r_probability > 0.9))
            & (joined.candidate_r_prediction != joined.gold_label)
        )
    )
    joined["review_required"] = review
    joined["error_reason"] = [
        _error_reason(row) if row.review_required else None for row in joined.itertuples()
    ]
    truth = joined.gold_label.to_numpy(dtype=np.int64)
    metrics = {
        name: _metric_record(truth, joined[f"candidate_{name.lower()}_probability"].to_numpy())
        for name in ("I", "R", "U")
    }
    family_records: list[dict[str, Any]] = []
    for family, block in joined.groupby("primary_task_family", sort=True):
        block_truth = block.gold_label.to_numpy(dtype=np.int64)
        record: dict[str, Any] = {
            "family": family,
            "support": len(block),
            "label_counts": {
                str(key): int(value)
                for key, value in enumerate(np.bincount(block_truth, minlength=2))
            },
            "disagreement_rate": float(
                np.mean(block.candidate_i_prediction != block.candidate_r_prediction)
            ),
        }
        confidences = []
        for name in ("i", "r", "u"):
            prediction = block[f"candidate_{name}_prediction"].to_numpy(dtype=np.int64)
            record[f"candidate_{name}_macro_f1"] = _f1_or_none(block_truth, prediction)
            record[f"candidate_{name}_error_count"] = int(np.sum(prediction != block_truth))
            confidences.extend(block[f"candidate_{name}_confidence"].tolist())
        record["mean_confidence"] = float(np.mean(confidences))
        record["candidate_u_error_impact"] = float(
            len(block) * record["candidate_u_error_count"] / len(block)
        )
        family_records.append(record)
    tag_records: list[dict[str, Any]] = []
    tags = sorted({tag for values in joined.claim_risk_tags for tag in values})
    for tag in tags:
        mask = joined.claim_risk_tags.map(lambda values, needle=tag: needle in values)
        if int(mask.sum()) < 5:
            continue
        block = joined.loc[mask]
        block_truth = block.gold_label.to_numpy(dtype=np.int64)
        i_correct = block.candidate_i_prediction.to_numpy() == block_truth
        r_correct = block.candidate_r_prediction.to_numpy() == block_truth
        pred_u = block.candidate_u_prediction.to_numpy(dtype=np.int64)
        negatives = block_truth == 0
        positives = block_truth == 1
        tag_records.append(
            {
                "tag": tag,
                "support": len(block),
                "label_counts": {
                    str(key): int(value)
                    for key, value in enumerate(np.bincount(block_truth, minlength=2))
                },
                "candidate_u_false_positive_rate": float(np.mean(pred_u[negatives] == 1)) if negatives.any() else None,
                "candidate_u_false_negative_rate": float(np.mean(pred_u[positives] == 0)) if positives.any() else None,
                "i_only_correct": int(np.sum(i_correct & ~r_correct)),
                "r_only_correct": int(np.sum(~i_correct & r_correct)),
                "both_wrong": int(np.sum(~i_correct & ~r_correct)),
            }
        )
    i_correct = joined.candidate_i_prediction.to_numpy() == truth
    r_correct = joined.candidate_r_prediction.to_numpy() == truth
    disagreement = joined.candidate_i_prediction != joined.candidate_r_prediction
    disagreement_record = {
        "hard_label_disagreement_rate": float(disagreement.mean()),
        "i_only_correct": int(np.sum(i_correct & ~r_correct)),
        "r_only_correct": int(np.sum(~i_correct & r_correct)),
        "both_correct": int(np.sum(i_correct & r_correct)),
        "both_wrong": int(np.sum(~i_correct & ~r_correct)),
        "probability_correlation": float(
            np.corrcoef(joined.candidate_i_probability, joined.candidate_r_probability)[0, 1]
        ),
        "reviewed_row_count": int(review.sum()),
        "error_reason_counts": dict(Counter(joined.loc[review, "error_reason"])),
        "confidence_i_only_correct_mean": float(joined.loc[i_correct & ~r_correct, "candidate_i_confidence"].mean()) if np.any(i_correct & ~r_correct) else None,
        "confidence_r_only_correct_mean": float(joined.loc[~i_correct & r_correct, "candidate_r_confidence"].mean()) if np.any(~i_correct & r_correct) else None,
    }
    discovery_text = discovery.set_index("official_index").loc[joined.official_index]
    duplicate_record, clustered_positions = duplicate_audit(
        discovery_text.reset_index(drop=True), workers=similarity_workers
    )
    for name in ("i", "r", "u"):
        errors = joined[f"candidate_{name}_prediction"].to_numpy() != truth
        inside = np.asarray([index in clustered_positions for index in range(len(joined))])
        duplicate_record[f"candidate_{name}_error_rate_inside"] = float(errors[inside].mean()) if inside.any() else None
        duplicate_record[f"candidate_{name}_error_rate_outside"] = float(errors[~inside].mean()) if (~inside).any() else None
    safe_columns = [column for column in joined.columns if column not in {"prompt_bn", "response_bn", "context"}]
    joined = joined[safe_columns]
    return {
        "candidate_metrics": metrics,
        "task_families": family_records,
        "claim_risk_tags_minimum_support_5": tag_records,
        "highest_impact_families": sorted(
            family_records,
            key=lambda value: (-value["candidate_u_error_impact"], -value["support"], value["family"]),
        )[:6],
        "disagreement": disagreement_record,
        "duplicates": duplicate_record,
        "suspected_ambiguity_count": int(
            joined.error_reason.isin(("ambiguous annotation", "suspected label inconsistency")).sum()
        ),
    }, joined


def frozen_hypotheses(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate, but never evaluate, at most six discovery-supported V12 hypotheses."""
    supported_tags = {
        record["tag"]: record for record in analysis["claim_risk_tags_minimum_support_5"]
    }
    hypotheses = []
    templates = (
        ("numeric claim", "numeric/date consistency", "Count normalized number/date mentions in prompt and response; flag response-only values."),
        ("refusal or uncertainty", "refusal/hedging handling", "Detect the frozen refusal and hedging lexicons and expose a binary audit feature."),
        ("generic low-information answer", "generic-answer detection", "Flag normalized responses shorter than 25 characters with low prompt lexical overlap."),
        ("named entity", "factual-specificity risk", "Count deterministic named-entity-like and numeric tokens in the response."),
        ("unverifiable specificity", "external-knowledge requirement", "Flag responses over 160 normalized characters containing named-entity or numeric claims."),
    )
    for tag, name, rule in templates:
        record = supported_tags.get(tag)
        if record is None:
            continue
        hypotheses.append(
            {
                "name": name,
                "targeted_error_family": tag,
                "deterministic_rule": rule,
                "expected_direction": "reduce route-specific false classifications",
                "why_current_models_miss_it": "Candidate I has no explicit rule and Candidate R compresses the signal into semantic aggregates.",
                "leakage_risk": "low when fitted or counted on each training partition only",
                "overfitting_risk": "moderate on the 169-row route; confirm once on locked holdout",
                "requires_external_evidence": tag in {"named entity", "unverifiable specificity"},
                "automatic_on_test_rows": True,
                "minimum_discovery_support_required": 5,
                "observed_discovery_support": record["support"],
                "evaluated_in_v11": False,
            }
        )
    family_support = {record["family"]: record for record in analysis["task_families"]}
    eligible = [record for record in family_support.values() if record["support"] >= 5]
    if eligible and len(hypotheses) < 6:
        hypotheses.append(
            {
                "name": "prompt-task-family routing",
                "targeted_error_family": max(eligible, key=lambda value: value["candidate_u_error_impact"])["family"],
                "deterministic_rule": "Apply the frozen twelve-family prompt classifier before any outcome is joined.",
                "expected_direction": "allow a predeclared route-specific correction for the largest supported family",
                "why_current_models_miss_it": "Neither frozen base candidate exposes an explicit task-family decision.",
                "leakage_risk": "low if taxonomy rules remain frozen",
                "overfitting_risk": "high; one locked-holdout confirmation only",
                "requires_external_evidence": False,
                "automatic_on_test_rows": True,
                "minimum_discovery_support_required": 5,
                "observed_discovery_support": max(eligible, key=lambda value: value["candidate_u_error_impact"])["support"],
                "evaluated_in_v11": False,
            }
        )
    return hypotheses[:6]
