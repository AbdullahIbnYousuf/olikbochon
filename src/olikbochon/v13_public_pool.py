"""Authenticated V13 public-pool normalization, provenance, and contamination controls."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loading import sha256_file


PUBLIC_AUTH = {
    "bangla_hallucination_5k_train.json": (4000, 3_959_445, "0f988c5b09ba1b5214a93adc04e994a3c915fb0aff7eca9e2b1aca36aa62c07f"),
    "bangla_hallucination_5k_validation.json": (1000, 988_551, "308c2e55bacc7552d6421101060a9b0e8c56136dd90b5cdc55f66f6375d3fa66"),
    "bangla_hallucination_5k_contrastive.json": (5000, 4_947_994, "cbba66057253545d39914c147843bdca88a299baa0ea6ae6932a09d64e2e8d13"),
}
REQUIRED_COLUMNS = ("context", "prompt_bn", "response_bn", "label")
SEPARATOR = " [SEP] "
_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.translate(_BENGALI_DIGITS).casefold()
    text = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", text).strip()


def skeleton(value: object) -> str:
    text = normalize(value)
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b", "<DATE>", text)
    text = re.sub(r"\b(?:19|20)\d{2}\b", "<DATE>", text)
    text = re.sub(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", "<NUM>", text)
    text = re.sub(r"(?:^|\s)(?:\d+|[a-z])[.)]\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def add_keys(frame: pd.DataFrame, *, workers: int) -> pd.DataFrame:
    if workers != 6:
        raise ValueError("V13 normalization requires six workers")

    def keys(row: Any) -> tuple[str, str, str, str, str]:
        prompt = normalize(row.prompt_bn)
        response = normalize(row.response_bn)
        context = normalize(row.context)
        pair = f"{prompt}{SEPARATOR}{response}"
        triple = f"{context}{SEPARATOR}{pair}"
        return prompt, response, pair, triple, f"{skeleton(prompt)}{SEPARATOR}{skeleton(response)}"

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v13-normalize") as executor:
        values = list(executor.map(keys, frame.itertuples(index=False)))
    result = frame.copy(deep=True).reset_index(drop=True)
    result[["prompt_key", "response_key", "pair_key", "triple_key", "skeleton_key"]] = values
    return result


def authenticate_public(path: Path, *, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected = PUBLIC_AUTH.get(path.name)
    if expected is None or not path.is_file():
        raise ValueError("V13 public path is not one of the three authenticated files")
    rows, size, digest = expected
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise RuntimeError(f"V13 public authentication failed for {path.name}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(raw)
    if len(frame) != rows or tuple(frame.columns) != REQUIRED_COLUMNS:
        raise RuntimeError(f"V13 public schema/count changed for {path.name}")
    if set(frame.label.unique()) != {0, 1} or frame.isna().any().any():
        raise RuntimeError(f"V13 public labels/missing values invalid for {path.name}")
    keyed = add_keys(frame, workers=workers)
    counts = keyed.label.value_counts().sort_index()
    stat = path.stat()
    audit = {
        "path": str(path.resolve()),
        "filename": path.name,
        "bytes": size,
        "sha256": digest,
        "rows": rows,
        "schema": list(REQUIRED_COLUMNS),
        "label_orientation": {"0": "hallucinated_or_unacceptable", "1": "faithful_or_acceptable"},
        "label_counts": {str(label): int(counts.get(label, 0)) for label in (0, 1)},
        "duplicate_ids": None,
        "id_column_present": False,
        "missing_values": int(frame.isna().sum().sum()),
        "creation_time_utc": stat.st_ctime,
        "last_write_time_utc": stat.st_mtime,
        "provenance": "Kaggle public dataset abidur14004/new-dataset; user-approved for this university event; upstream license metadata unknown",
        "explicit_test_label_mapping_columns": False,
    }
    return keyed, audit


def conflict_audit(frame: pd.DataFrame, key: str) -> dict[str, int]:
    groups = frame.groupby(key, sort=False).label.agg(["size", "nunique"])
    duplicate = groups[groups["size"] > 1]
    return {
        "clusters": int(len(duplicate)),
        "rows": int(duplicate["size"].sum()),
        "mixed_label_clusters": int((duplicate["nunique"] > 1).sum()),
    }


def deduplicate_pool(train: pd.DataFrame, contrastive: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    combined = pd.concat(
        [train.assign(source="train", source_index=range(len(train))), contrastive.assign(source="contrastive", source_index=range(len(contrastive)))],
        ignore_index=True,
    )
    validation_pairs = set(validation.pair_key)
    leaked_validation = combined.pair_key.isin(validation_pairs)
    clean = combined.loc[~leaked_validation].copy()
    conflicts = clean.groupby("pair_key").label.nunique()
    conflicting_keys = set(conflicts[conflicts > 1].index)
    conflict_rows = clean.pair_key.isin(conflicting_keys)
    clean = clean.loc[~conflict_rows]
    before = len(clean)
    clean = clean.drop_duplicates("pair_key", keep="first").reset_index(drop=True)
    manifest_rows = [
        f"{row.source}:{row.source_index}:{row.label}:{hashlib.sha256(row.pair_key.encode()).hexdigest()}"
        for row in clean.itertuples(index=False)
    ]
    return clean, {
        "input_rows": len(combined),
        "validation_exact_pair_rows_excluded": int(leaked_validation.sum()),
        "mixed_label_pair_rows_excluded": int(conflict_rows.sum()),
        "same_label_duplicate_rows_removed": before - len(clean),
        "final_rows": len(clean),
        "manifest_sha256": hashlib.sha256("\n".join(manifest_rows).encode()).hexdigest(),
        "label_counts": {str(label): int((clean.label == label).sum()) for label in (0, 1)},
    }


def official_safe_pool(pool: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    pair_keys = set(official.pair_key)
    skeleton_keys = set(official.skeleton_key)
    pair_match = pool.pair_key.isin(pair_keys)
    skeleton_match = pool.skeleton_key.isin(skeleton_keys)
    safe = pool.loc[~pair_match & ~skeleton_match].reset_index(drop=True)
    return safe, {
        "official_exact_pair_rows_excluded": int(pair_match.sum()),
        "official_skeleton_rows_excluded": int((~pair_match & skeleton_match).sum()),
        "safe_rows": len(safe),
    }


def overlap_audit(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    return {
        "exact_prompt_count": int(right.prompt_key.isin(set(left.prompt_key)).sum()),
        "exact_response_count": int(right.response_key.isin(set(left.response_key)).sum()),
        "exact_pair_count": int(right.pair_key.isin(set(left.pair_key)).sum()),
        "exact_triple_count": int(right.triple_key.isin(set(left.triple_key)).sum()),
        "exact_skeleton_count": int(right.skeleton_key.isin(set(left.skeleton_key)).sum()),
    }


def pool_manifest(pool: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(pool),
        "label_counts": {str(label): int((pool.label == label).sum()) for label in (0, 1)},
        "pair_key_sha256": hashlib.sha256("\n".join(sorted(pool.pair_key)).encode()).hexdigest(),
    }
