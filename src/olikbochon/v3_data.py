"""Authenticated Version 3 input discovery, loading, auditing, and grouped folds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data_loading import DataValidationError, sha256_file, validate_labeled_frame
from .file_discovery import DiscoveryError
from .v3_default_normalizer import normalize_default
from .v3_preprocessing import build_transformer_pair, raw_context_is_present


PUBLIC_TRAIN_FILENAME = "bangla_hallucination_5k_train.json"
PUBLIC_VALIDATION_FILENAME = "bangla_hallucination_5k_validation.json"
PUBLIC_AGGREGATE_FILENAME = "bangla_hallucination_5k_contrastive.json"
OFFICIAL_FILENAME = "dataset samples.json"
TEST_FILENAME = "test set.csv"
SAMPLE_NAMES = ("sample submission.csv", "sample_submission.csv")
LABELED_COLUMNS = ("context", "prompt_bn", "response_bn", "label")
TEST_COLUMNS = ("id", "context", "prompt_bn", "response_bn")
MODEL_REQUIRED_FILES = (
    ".gitattributes",
    "README.md",
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)
MODEL_OPTIONAL_FILES = ("SNAPSHOT_INFO.md",)
MODEL_REVISION = "9ce791f330578f50da6bc52b54205166fb5d1c8c"
MODEL_WEIGHT_SHA256 = "9d33f519f42705d54e65fc1601644a6f4562c3462f96943b32b4184536130f98"
KNOWN_HASHES = {
    PUBLIC_TRAIN_FILENAME: "0f988c5b09ba1b5214a93adc04e994a3c915fb0aff7eca9e2b1aca36aa62c07f",
    PUBLIC_VALIDATION_FILENAME: "308c2e55bacc7552d6421101060a9b0e8c56136dd90b5cdc55f66f6375d3fa66",
    OFFICIAL_FILENAME: "f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28",
    TEST_FILENAME: "db75049956c6fa00e4d9c476716ee34bc4cc17a737f52ada06d2c0f80d567b81",
    "sample submission.csv": "c02eaf0f12504c79cc2c8874cd012f4829bc1d2e00eea1ed0d50be1867541e55",
}
NEAR_SIMILARITY = 0.97
NEAR_LENGTH_RATIO = 0.90


@dataclass(frozen=True)
class V3Files:
    """Resolved paths for the exact three-input Version 3 contract."""

    public_train: Path
    public_validation: Path
    public_aggregate: Path | None
    official_train: Path
    test: Path
    sample_submission: Path
    model_directory: Path
    public_roots: tuple[Path, ...]
    competition_root: Path


@dataclass(frozen=True)
class GroupAudit:
    """Aggregate-only official grouping facts."""

    group_ids: tuple[str, ...]
    group_count: int
    nontrivial_groups: int
    largest_group: int
    exact_duplicate_pairs: int
    prompt_context_pairs: int
    near_duplicate_pairs: int
    conflicting_exact_texts: int


@dataclass(frozen=True)
class FoldAssignment:
    """Deterministic validated fold indices shared by both arms."""

    strategy: str
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    validation_fold_by_row: tuple[int, ...]
    fold_class_counts: tuple[dict[int, int], ...]


def audit_partition_safety(partitions: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Fail on exact conflicts/overlap and report aggregate duplicate counts."""
    expected = {"public_train", "public_validation", "official"}
    if set(partitions) != expected:
        raise ValueError(f"Expected partitions {sorted(expected)}")
    keyed: dict[str, pd.DataFrame] = {}
    for name, frame in partitions.items():
        keys: list[str] = []
        for row in frame.itertuples(index=False):
            prompt = normalize_default(str(row.prompt_bn))
            context = (
                normalize_default(str(row.context)) if raw_context_is_present(row.context) else ""
            )
            response = normalize_default(str(row.response_bn))
            keys.append(_fingerprint(prompt, context, response))
        part = pd.DataFrame({"key": keys, "label": frame["label"].astype(int)})
        keyed[name] = part
    combined = pd.concat(
        [part.assign(partition=name) for name, part in keyed.items()], ignore_index=True
    )
    conflicts = combined.groupby("key")["label"].nunique()
    conflict_count = int((conflicts > 1).sum())
    if conflict_count:
        raise DataValidationError(f"Found {conflict_count} exact texts with conflicting labels")
    overlap: dict[str, int] = {}
    names = sorted(keyed)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            count = len(set(keyed[left]["key"]) & set(keyed[right]["key"]))
            overlap[f"{left}__{right}"] = count
            if count:
                raise DataValidationError(
                    f"Exact labeled-text overlap between {left} and {right}: {count}"
                )
    return {
        "partition_row_counts": {name: int(len(frame)) for name, frame in partitions.items()},
        "internal_duplicate_rows": {
            name: int(part["key"].duplicated(keep=False).sum()) for name, part in keyed.items()
        },
        "cross_partition_overlap": overlap,
        "conflicting_exact_texts": conflict_count,
    }


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _identical(paths: list[Path]) -> bool:
    return bool(paths) and len({sha256_file(path) for path in paths}) == 1


def _select_identical(roots: list[Path], filename: str) -> Path:
    paths = [root / filename for root in roots]
    if not all(path.is_file() for path in paths):
        raise DiscoveryError(f"Every public root must contain {filename!r}")
    if not _identical(paths):
        raise DiscoveryError(f"Public copies of {filename!r} are not byte-identical")
    return min(paths, key=str)


def _discover_public(root: Path) -> tuple[list[Path], Path, Path, Path | None]:
    train_roots = {path.parent for path in root.rglob(PUBLIC_TRAIN_FILENAME) if path.is_file()}
    valid_roots = {
        path.parent for path in root.rglob(PUBLIC_VALIDATION_FILENAME) if path.is_file()
    }
    roots = sorted(train_roots & valid_roots, key=str)
    if not roots:
        raise DiscoveryError("No coherent public-data root contains both labeled splits")
    train = _select_identical(roots, PUBLIC_TRAIN_FILENAME)
    validation = _select_identical(roots, PUBLIC_VALIDATION_FILENAME)
    aggregates = [directory / PUBLIC_AGGREGATE_FILENAME for directory in roots]
    existing = [path for path in aggregates if path.is_file()]
    aggregate: Path | None = None
    if existing:
        if len(existing) != len(roots) or not _identical(existing):
            raise DiscoveryError("Optional public aggregate copies are incomplete or conflicting")
        aggregate = min(existing, key=str)
    return roots, train, validation, aggregate


def _discover_competition(root: Path, public_roots: list[Path]) -> tuple[Path, Path]:
    candidates: list[tuple[int, Path]] = []
    for preference, name in enumerate(SAMPLE_NAMES):
        for sample in root.rglob(name):
            candidate = sample.parent
            if any(_inside(candidate, public) for public in public_roots):
                continue
            if (candidate / OFFICIAL_FILENAME).is_file() and (candidate / TEST_FILENAME).is_file():
                candidates.append((preference, candidate))
    if not candidates:
        raise DiscoveryError("No coherent non-public competition root was found")
    signatures = {
        (
            sha256_file(directory / OFFICIAL_FILENAME),
            sha256_file(directory / TEST_FILENAME),
            sha256_file(directory / SAMPLE_NAMES[preference]),
        )
        for preference, directory in candidates
    }
    if len(signatures) != 1:
        raise DiscoveryError("Conflicting coherent competition roots were found")
    preference, directory = min(candidates, key=lambda item: (item[0], str(item[1])))
    return directory, directory / SAMPLE_NAMES[preference]


def authenticate_model_directory(path: Path) -> dict[str, Any]:
    """Authenticate the exact official snapshot before executable loading."""
    path = Path(path)
    missing = [name for name in MODEL_REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise DiscoveryError(f"Model directory is missing required files: {missing}")
    allowed = set(MODEL_REQUIRED_FILES) | set(MODEL_OPTIONAL_FILES)
    extras = sorted(item.name for item in path.iterdir() if item.is_file() and item.name not in allowed)
    directories = sorted(item.name for item in path.iterdir() if item.is_dir())
    if extras or directories:
        raise DiscoveryError(
            f"Authenticated model directory contains unexpected entries: {extras + directories}"
        )
    weight_digest = sha256_file(path / "pytorch_model.bin")
    if weight_digest != MODEL_WEIGHT_SHA256:
        raise DiscoveryError(
            f"BanglaBERT weight digest mismatch: expected {MODEL_WEIGHT_SHA256}, "
            f"observed {weight_digest}"
        )
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "electra" or config.get("vocab_size") != 32_000:
        raise DiscoveryError("Authenticated config must identify ELECTRA with vocabulary 32000")
    return {
        "directory": str(path),
        "revision": MODEL_REVISION,
        "weight_sha256": weight_digest,
        "required_file_count": len(MODEL_REQUIRED_FILES),
    }


def _discover_model(root: Path) -> Path:
    candidates = sorted({path.parent for path in root.rglob("pytorch_model.bin")}, key=str)
    valid: list[Path] = []
    for candidate in candidates:
        try:
            authenticate_model_directory(candidate)
        except DiscoveryError:
            continue
        valid.append(candidate)
    if not valid:
        raise DiscoveryError("No authenticated official BanglaBERT snapshot was found")
    if len(valid) > 1:
        signatures = {
            tuple(sha256_file(path / name) for name in MODEL_REQUIRED_FILES) for path in valid
        }
        if len(signatures) != 1:
            raise DiscoveryError("Conflicting authenticated model directories were found")
    return min(valid, key=str)


def discover_v3_files(root: Path = Path("/kaggle/input")) -> V3Files:
    """Discover the exact three coherent input roles and quarantine public CSVs."""
    root = Path(root)
    public_roots, train, validation, aggregate = _discover_public(root)
    competition_root, sample = _discover_competition(root, public_roots)
    test = competition_root / TEST_FILENAME
    if any(_inside(test, public) for public in public_roots):
        raise DiscoveryError("Competition test resolved inside a quarantined public root")
    files = V3Files(
        public_train=train,
        public_validation=validation,
        public_aggregate=aggregate,
        official_train=competition_root / OFFICIAL_FILENAME,
        test=test,
        sample_submission=sample,
        model_directory=_discover_model(root),
        public_roots=tuple(public_roots),
        competition_root=competition_root,
    )
    def input_section(path: Path) -> Path:
        relative = path.resolve().relative_to(root.resolve())
        if not relative.parts:
            raise DiscoveryError("A required file resolved directly under /kaggle/input")
        return root / relative.parts[0]

    required_sections = {
        input_section(files.public_train),
        input_section(files.official_train),
        input_section(files.model_directory),
    }
    actual_sections = {path for path in root.iterdir() if path.is_dir()}
    if len(required_sections) != 3 or actual_sections != required_sections:
        raise DiscoveryError(
            "Version 3 requires exactly three Kaggle input sections: competition data, "
            "abidur14004/new-dataset, and the authenticated private BanglaBERT snapshot"
        )
    return files


def authenticate_known_file(path: Path) -> str:
    """Require a recorded digest when one exists and return the observed digest."""
    digest = sha256_file(path)
    expected = KNOWN_HASHES.get(path.name)
    if expected is not None and digest != expected:
        raise DataValidationError(
            f"Authenticated digest mismatch for {path.name!r}: expected {expected}, observed {digest}"
        )
    return digest


def load_labeled_json(path: Path) -> pd.DataFrame:
    """Authenticate and validate one labeled JSON with dynamic row counts."""
    authenticate_known_file(path)
    frame = pd.read_json(path, orient="records")
    if tuple(frame.columns) != LABELED_COLUMNS:
        raise DataValidationError(
            f"Expected columns {list(LABELED_COLUMNS)}, found {list(frame.columns)}"
        )
    return validate_labeled_frame(frame).reset_index(drop=True)


def data_manifest(files: V3Files, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Build aggregate-only authenticated file metadata."""
    paths = {
        "public_train": files.public_train,
        "public_validation": files.public_validation,
        "official": files.official_train,
        "test": files.test,
        "sample_submission": files.sample_submission,
    }
    return {
        name: {
            "filename": path.name,
            "sha256": authenticate_known_file(path),
            **({"row_count": int(len(frames[name]))} if name in frames else {}),
        }
        for name, path in paths.items()
    }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        first, second = self.find(left), self.find(right)
        if first != second:
            low, high = sorted((first, second))
            self.parent[high] = low


def _fingerprint(*values: str) -> str:
    joined = "\n\x1f\n".join(values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_official_groups(frame: pd.DataFrame) -> GroupAudit:
    """Group exact, prompt/context-related, and high-confidence near-duplicate rows."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    normalized: list[tuple[str, str, str]] = []
    model_text: list[str] = []
    for row in validated.itertuples(index=False):
        prompt = normalize_default(str(row.prompt_bn))
        context = normalize_default(str(row.context)) if raw_context_is_present(row.context) else ""
        response = normalize_default(str(row.response_bn))
        normalized.append((prompt, context, response))
        pair = build_transformer_pair(row.prompt_bn, row.context, row.response_bn)
        model_text.append(f"{pair.sequence_a}\n\x1e\n{pair.sequence_b}")

    union = _UnionFind(len(validated))
    exact_groups: dict[str, list[int]] = {}
    prompt_context_groups: dict[str, list[int]] = {}
    for index, (prompt, context, response) in enumerate(normalized):
        exact_groups.setdefault(_fingerprint(prompt, context, response), []).append(index)
        prompt_context_groups.setdefault(_fingerprint(prompt, context), []).append(index)

    conflicts = 0
    exact_pairs = 0
    prompt_context_pairs = 0
    labels = validated["label"].to_numpy(dtype=np.int64)
    for members in exact_groups.values():
        if len({int(labels[index]) for index in members}) > 1:
            conflicts += 1
        for right in members[1:]:
            union.union(members[0], right)
            exact_pairs += 1
    if conflicts:
        raise DataValidationError(f"Found {conflicts} exact official texts with conflicting labels")
    for members in prompt_context_groups.values():
        for right in members[1:]:
            union.union(members[0], right)
            prompt_context_pairs += 1

    near_pairs = 0
    if len(model_text) > 1:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            lowercase=False,
            dtype=np.float32,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(model_text)
        similarities = cosine_similarity(matrix, dense_output=True)
        lengths = np.maximum(np.asarray([len(value) for value in model_text]), 1)
        for left in range(len(model_text)):
            for right in range(left + 1, len(model_text)):
                ratio = min(lengths[left], lengths[right]) / max(lengths[left], lengths[right])
                if similarities[left, right] >= NEAR_SIMILARITY and ratio >= NEAR_LENGTH_RATIO:
                    union.union(left, right)
                    near_pairs += 1

    roots = [union.find(index) for index in range(len(validated))]
    root_order = {root: position for position, root in enumerate(sorted(set(roots)))}
    group_ids = tuple(f"group-{root_order[root]:04d}" for root in roots)
    counts = pd.Series(group_ids).value_counts()
    return GroupAudit(
        group_ids=group_ids,
        group_count=int(counts.size),
        nontrivial_groups=int((counts > 1).sum()),
        largest_group=int(counts.max()),
        exact_duplicate_pairs=exact_pairs,
        prompt_context_pairs=prompt_context_pairs,
        near_duplicate_pairs=near_pairs,
        conflicting_exact_texts=conflicts,
    )


def make_official_folds(labels: np.ndarray, audit: GroupAudit) -> FoldAssignment:
    """Create and validate deterministic five-fold assignments without group leakage."""
    truth = np.asarray(labels, dtype=np.int64)
    if len(truth) != len(audit.group_ids):
        raise ValueError("Labels and group IDs must have equal length")
    groups = np.asarray(audit.group_ids, dtype=object)
    rows = np.arange(len(truth))
    if audit.nontrivial_groups:
        splitter: Any = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        split_iterator = splitter.split(rows, truth, groups)
        strategy = "stratified_group_5fold"
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iterator = splitter.split(rows, truth)
        strategy = "stratified_5fold_after_no_group_audit"

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    assignments = np.full(len(truth), -1, dtype=np.int64)
    class_counts: list[dict[int, int]] = []
    for fold, (train, validation) in enumerate(split_iterator):
        train_groups = set(groups[train])
        validation_groups = set(groups[validation])
        if train_groups & validation_groups:
            raise DataValidationError(f"Group leakage detected in fold {fold + 1}")
        values, counts = np.unique(truth[validation], return_counts=True)
        if set(values) != {0, 1}:
            raise DataValidationError(f"Fold {fold + 1} does not contain both classes")
        if np.any(assignments[validation] != -1):
            raise DataValidationError("A row appears in more than one validation fold")
        assignments[validation] = fold
        folds.append((train.astype(np.int64), validation.astype(np.int64)))
        class_counts.append({int(value): int(count) for value, count in zip(values, counts)})
    if len(folds) != 5 or np.any(assignments < 0):
        raise DataValidationError("Five valid folds did not cover every official row exactly once")
    return FoldAssignment(strategy, tuple(folds), tuple(assignments.tolist()), tuple(class_counts))
