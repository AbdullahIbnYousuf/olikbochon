"""Safe, deterministic Wikipedia-retrieval baseline utilities for Version 4-A."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .data_loading import DataValidationError, sha256_file, validate_labeled_frame
from .metrics import classification_metrics, prediction_collapse_warning
from .submission import validate_test_frame


WIKI_DATASET_REF = "abyaadrafid/bnwiki"
WIKI_DATASET_ID = 228152
WIKI_DATASET_VERSION = 1
WIKI_DATASET_LICENSE_METADATA = "CC0-1.0"
MIN_WIKI_CHUNK_COUNT = 250
WIKI_SECTION_NAMES = ("AA", "AB", "AC", "AD")
WIKI_CHUNK_PATTERN = re.compile(r"^wiki_\d+$")
OFFICIAL_FILENAME = "dataset samples.json"
TEST_FILENAME = "test set.csv"
SAMPLE_SUBMISSION_NAMES = ("sample submission.csv", "sample_submission.csv")
KNOWN_FILE_HASHES = {
    OFFICIAL_FILENAME: "f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28",
    TEST_FILENAME: "db75049956c6fa00e4d9c476716ee34bc4cc17a737f52ada06d2c0f80d567b81",
    "sample submission.csv": "c02eaf0f12504c79cc2c8874cd012f4829bc1d2e00eea1ed0d50be1867541e55",
    "sample_submission.csv": "c02eaf0f12504c79cc2c8874cd012f4829bc1d2e00eea1ed0d50be1867541e55",
}
FEATURE_COLUMNS = (
    "ctx_present",
    "word_overlap_ratio",
    "numbers_supported",
    "number_overlap_ratio",
    "has_numbers_in_response",
    "is_substring",
    "response_len_chars",
    "context_len_chars",
)
RETRIEVAL_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 0.41, 0.05))
ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD = 0.25
RANDOM_STATE = 42
SNIPPET_CHARACTERS = 800
WIKI_MAX_FEATURES = 50_000
BANGLA_DIGIT_TRANSLATION = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
NULL_CONTEXT_VALUES = {"", "nan", "[null]"}


class V4DiscoveryError(RuntimeError):
    """Raised when official competition or discovered Wikipedia inputs are unsafe."""


@dataclass(frozen=True)
class V4Files:
    """Resolved official competition files and dynamically validated Wikipedia chunks."""

    official_train: Path
    test: Path
    sample_submission: Path
    competition_root: Path
    wikipedia_root: Path
    wikipedia_chunks: tuple[Path, ...]
    wikipedia_duplicate_path_count: int


@dataclass(frozen=True)
class WikipediaCorpus:
    """Parsed non-row-level corpus state used to construct the retrieval index."""

    articles: pd.DataFrame
    source_chunk_count: int
    decoded_line_count: int
    rejected_line_count: int
    duplicate_url_count: int


@dataclass(frozen=True)
class GroupAudit:
    """Aggregate duplicate-family facts; group identifiers contain no text."""

    group_ids: tuple[str, ...]
    group_count: int
    nontrivial_groups: int
    largest_group: int
    exact_duplicate_links: int
    prompt_context_links: int


@dataclass(frozen=True)
class FoldAssignment:
    """Validated duplicate-aware folds and one fold assignment per row."""

    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    validation_fold_by_row: tuple[int, ...]
    class_counts: tuple[dict[int, int], ...]
    strategy: str


@dataclass(frozen=True)
class RetrievalResult:
    """Top-one retrieval output retained in memory without logging text."""

    snippets: tuple[str, ...]
    similarities: np.ndarray


@dataclass(frozen=True)
class ValidationResults:
    """Original-method and honest grouped estimates plus honest OOF probabilities."""

    original_method: dict[str, Any]
    grouped_method: dict[str, Any]
    grouped_oof_probabilities: np.ndarray
    grouped_fold_by_row: np.ndarray
    group_audit: GroupAudit


def _wiki_chunks(root: Path) -> tuple[Path, ...]:
    """Return direct WikiExtractor chunks beneath a coherent AA/AB/AC/AD root."""
    root = Path(root)
    section_directories = tuple(root / name for name in WIKI_SECTION_NAMES)
    if not all(path.is_dir() for path in section_directories):
        return ()
    paths = tuple(
        sorted(
            path
            for directory in section_directories
            for path in directory.iterdir()
            if path.is_file() and WIKI_CHUNK_PATTERN.fullmatch(path.name)
        )
    )
    if len(paths) < MIN_WIKI_CHUNK_COUNT:
        return ()
    return paths


def _wiki_root_signature(root: Path, chunks: tuple[Path, ...]) -> tuple[tuple[str, int, str], ...]:
    """Hash candidate roots only when needed to prove duplicate layouts are identical."""
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            sha256_file(path),
        )
        for path in chunks
    )


def _discover_wikipedia(root: Path) -> tuple[Path, tuple[Path, ...], int]:
    """Find one logical corpus, accepting byte-identical single or duplicate layouts."""
    candidates: set[Path] = set()
    for path in root.rglob("wiki_*"):
        if (
            path.is_file()
            and WIKI_CHUNK_PATTERN.fullmatch(path.name)
            and path.parent.name in WIKI_SECTION_NAMES
        ):
            candidates.add(path.parent.parent)
    structural = [(candidate, _wiki_chunks(candidate)) for candidate in sorted(candidates)]
    structural = [(candidate, chunks) for candidate, chunks in structural if chunks]
    if not structural:
        raise V4DiscoveryError(
            f"No mounted Wikipedia root has AA/AB/AC/AD and at least "
            f"{MIN_WIKI_CHUNK_COUNT} valid wiki_* files"
        )

    structural.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    selected_root, selected_chunks = structural[0]
    duplicate_path_count = 0
    if len(structural) > 1:
        try:
            selected_signature = _wiki_root_signature(selected_root, selected_chunks)
            for candidate, chunks in structural[1:]:
                if _wiki_root_signature(candidate, chunks) != selected_signature:
                    raise V4DiscoveryError(
                        "More than one distinct Wikipedia corpus layout was found: "
                        + "; ".join(str(path.resolve()) for path, _ in structural)
                    )
                duplicate_path_count += len(chunks)
        except OSError as exc:
            raise V4DiscoveryError(
                f"Wikipedia duplicate-layout comparison failed: {type(exc).__name__}"
            ) from exc

    if len(selected_chunks) < MIN_WIKI_CHUNK_COUNT:
        raise V4DiscoveryError(
            f"Wikipedia discovery retained fewer than {MIN_WIKI_CHUNK_COUNT} chunks"
        )
    return selected_root, selected_chunks, duplicate_path_count


def _discover_competition(root: Path) -> tuple[Path, Path]:
    candidates: dict[Path, tuple[int, Path]] = {}
    for preference, filename in enumerate(SAMPLE_SUBMISSION_NAMES):
        for sample in root.rglob(filename):
            parent = sample.parent
            if (parent / OFFICIAL_FILENAME).is_file() and (parent / TEST_FILENAME).is_file():
                current = candidates.get(parent)
                if current is None or preference < current[0]:
                    candidates[parent] = (preference, sample)
    if not candidates:
        raise V4DiscoveryError("No coherent official competition root was found")
    if len(candidates) > 1:
        raise V4DiscoveryError(
            "More than one coherent official competition root was found: "
            + "; ".join(str(path.resolve()) for path in sorted(candidates))
        )
    competition_root, (_, sample) = next(iter(candidates.items()))
    return competition_root, sample


def authenticate_official_file(path: Path) -> str:
    """Authenticate a known competition file using metadata only before it is loaded."""
    expected = KNOWN_FILE_HASHES.get(path.name)
    if expected is None:
        raise V4DiscoveryError(f"No recorded digest for official file {path.name!r}")
    observed = sha256_file(path)
    if observed != expected:
        raise V4DiscoveryError(
            f"Official file digest mismatch for {path.name!r}: expected {expected}, "
            f"observed {observed}"
        )
    return observed


def discover_v4_files(root: Path = Path("/kaggle/input")) -> V4Files:
    """Recursively and independently resolve official competition and Wikipedia roots."""
    root = Path(root)
    competition_root, sample = _discover_competition(root)
    wikipedia_root, chunks, duplicate_path_count = _discover_wikipedia(root)
    files = V4Files(
        official_train=competition_root / OFFICIAL_FILENAME,
        test=competition_root / TEST_FILENAME,
        sample_submission=sample,
        competition_root=competition_root,
        wikipedia_root=wikipedia_root,
        wikipedia_chunks=chunks,
        wikipedia_duplicate_path_count=duplicate_path_count,
    )
    for path in (files.official_train, files.test, files.sample_submission):
        authenticate_official_file(path)
    return files


def safe_discovery_summary(files: V4Files) -> dict[str, Any]:
    """Return safe paths, counts, and sizes only; never inspect or expose competition rows."""
    return {
        "competition_root": str(files.competition_root.resolve()),
        "wikipedia_root": str(files.wikipedia_root.resolve()),
        "official_file_count": 3,
        "wikipedia_chunk_count": len(files.wikipedia_chunks),
        "wikipedia_total_size": sum(path.stat().st_size for path in files.wikipedia_chunks),
        "wikipedia_duplicate_path_count": files.wikipedia_duplicate_path_count,
        "wikipedia_minimum_chunk_count": MIN_WIKI_CHUNK_COUNT,
    }


def load_official_labeled(path: Path) -> pd.DataFrame:
    """Load the authenticated official labels without printing individual values."""
    authenticate_official_file(path)
    frame = pd.read_json(path, orient="records")
    return validate_labeled_frame(frame).reset_index(drop=True)


def load_competition_test(path: Path) -> pd.DataFrame:
    """Load the authenticated test only after validation decisions are frozen."""
    authenticate_official_file(path)
    frame = pd.read_csv(path)
    validate_test_frame(frame)
    return frame.reset_index(drop=True)


def load_sample_submission(path: Path, test_ids: pd.Series) -> pd.DataFrame:
    """Validate the official output template without printing IDs."""
    authenticate_official_file(path)
    frame = pd.read_csv(path)
    if tuple(frame.columns) != ("id", "label"):
        raise DataValidationError("Sample submission must contain exactly id,label")
    if len(frame) != len(test_ids) or frame["id"].tolist() != test_ids.tolist():
        raise DataValidationError("Sample submission IDs must exactly match test IDs in order")
    return frame


def clean_context(value: Any) -> str:
    """Match the source baseline's null-context semantics without unsafe string coercion."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    stripped = str(value).strip()
    return "" if stripped.lower() in NULL_CONTEXT_VALUES else stripped


def normalize_numbers(text: str) -> str:
    return text.translate(BANGLA_DIGIT_TRANSLATION)


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", normalize_numbers(text)))


def tokenize(text: str) -> set[str]:
    normalized = re.sub(r"[^\w\s]", " ", normalize_numbers(text))
    return {token for token in normalized.split() if len(token) > 1}


def make_lexical_features(context: str, response: str) -> dict[str, float | int]:
    """Reproduce the eight published lexical features exactly and deterministically."""
    context = clean_context(context)
    response = str(response)
    context_present = bool(context)
    context_tokens = tokenize(context)
    response_tokens = tokenize(response)
    overlap = (
        len(response_tokens & context_tokens) / max(len(response_tokens), 1)
        if context_present
        else 0.0
    )
    context_numbers = extract_numbers(context)
    response_numbers = extract_numbers(response)
    if context_present and response_numbers:
        numbers_supported = int(response_numbers.issubset(context_numbers))
        number_overlap = len(response_numbers & context_numbers) / len(response_numbers)
    else:
        numbers_supported = 1
        number_overlap = 1.0 if not response_numbers else 0.0
    stripped_response = response.strip()
    return {
        "ctx_present": int(context_present),
        "word_overlap_ratio": float(overlap),
        "numbers_supported": numbers_supported,
        "number_overlap_ratio": float(number_overlap),
        "has_numbers_in_response": int(bool(response_numbers)),
        "is_substring": int(
            context_present and bool(stripped_response) and stripped_response in context
        ),
        "response_len_chars": len(response),
        "context_len_chars": len(context),
    }


def build_lexical_frame(contexts: list[str], responses: pd.Series) -> pd.DataFrame:
    """Build an ordered numeric frame without retaining prompt or response text."""
    if len(contexts) != len(responses):
        raise ValueError("Contexts and responses must have equal length")
    frame = pd.DataFrame(
        [make_lexical_features(context, response) for context, response in zip(contexts, responses)]
    )
    if tuple(frame.columns) != FEATURE_COLUMNS:
        raise RuntimeError("Unexpected Version 4-A feature order")
    return frame


def load_wikipedia_corpus(files: V4Files) -> WikipediaCorpus:
    """Strictly parse every dynamically discovered WikiExtractor chunk."""
    canonical = files.wikipedia_chunks
    if len(canonical) < MIN_WIKI_CHUNK_COUNT:
        raise DataValidationError(
            f"Wikipedia corpus requires at least {MIN_WIKI_CHUNK_COUNT} valid chunks"
        )
    records: list[dict[str, str]] = []
    decoded = 0
    for path in canonical:
        decoded_in_file = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise DataValidationError(
                            f"Wikipedia chunk {path.name!r} has invalid JSON at line {line_number}"
                        ) from exc
                    if not isinstance(payload, dict):
                        raise DataValidationError(
                            f"Wikipedia chunk {path.name!r} has a non-object JSON record"
                        )
                    if not isinstance(payload.get("text", ""), str):
                        raise DataValidationError(
                            f"Wikipedia chunk {path.name!r} has a non-string text field"
                        )
                    decoded_in_file += 1
                    decoded += 1
                    text = str(payload.get("text", ""))
                    if "\n\n" in text:
                        title, body = text.split("\n\n", 1)
                    else:
                        title, body = text[:80], text
                    records.append(
                        {
                            "url": str(payload.get("url", "")),
                            "title": title.strip(),
                            "snippet": body.strip()[:SNIPPET_CHARACTERS],
                        }
                    )
        except UnicodeDecodeError as exc:
            raise DataValidationError(
                f"Wikipedia chunk {path.name!r} is not valid UTF-8"
            ) from exc
        except OSError as exc:
            raise DataValidationError(
                f"Wikipedia chunk {path.name!r} could not be read: {type(exc).__name__}"
            ) from exc
        if decoded_in_file == 0:
            raise DataValidationError(f"Wikipedia chunk {path.name!r} contains no JSON records")
    articles = pd.DataFrame(records, columns=["url", "title", "snippet"])
    before = len(articles)
    articles = articles.drop_duplicates(subset="url", keep="first")
    duplicate_url_count = before - len(articles)
    articles = articles[articles["snippet"].str.len() > 50].reset_index(drop=True)
    if articles.empty:
        raise DataValidationError("Discovered Wikipedia corpus produced no usable articles")
    articles["search_blob"] = articles["title"] + " " + articles["snippet"]
    return WikipediaCorpus(articles, len(canonical), decoded, 0, duplicate_url_count)


class WikipediaRetriever:
    """Top-one character-TF-IDF retriever matching the verified public method."""

    def __init__(self, corpus: WikipediaCorpus) -> None:
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=WIKI_MAX_FEATURES
        )
        self.matrix = self.vectorizer.fit_transform(corpus.articles["search_blob"])

    def retrieve(self, prompts: pd.Series) -> RetrievalResult:
        snippets: list[str] = []
        similarities = np.empty(len(prompts), dtype=np.float64)
        for index, prompt in enumerate(prompts.astype(str)):
            query = self.vectorizer.transform([prompt])
            scores = cosine_similarity(query, self.matrix, dense_output=True).ravel()
            best = int(np.argmax(scores))
            snippets.append(str(self.corpus.articles.iloc[best]["snippet"]))
            similarities[index] = float(scores[best])
        if not np.isfinite(similarities).all() or np.any((similarities < 0) | (similarities > 1)):
            raise DataValidationError("Wikipedia retrieval produced invalid similarities")
        return RetrievalResult(tuple(snippets), similarities)


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


def _group_text(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def _fingerprint(*values: str) -> str:
    return hashlib.sha256("\n\x1f\n".join(values).encode()).hexdigest()


def build_duplicate_groups(frame: pd.DataFrame) -> GroupAudit:
    """Join exact rows and all response variants sharing normalized prompt/context."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    union = _UnionFind(len(validated))
    exact: dict[str, list[int]] = {}
    prompt_context: dict[str, list[int]] = {}
    labels = validated["label"].to_numpy(dtype=np.int64)
    for index, row in enumerate(validated.itertuples(index=False)):
        prompt = _group_text(row.prompt_bn)
        context = _group_text(clean_context(row.context))
        response = _group_text(row.response_bn)
        exact.setdefault(_fingerprint(prompt, context, response), []).append(index)
        prompt_context.setdefault(_fingerprint(prompt, context), []).append(index)
    exact_links = 0
    for members in exact.values():
        if len({int(labels[index]) for index in members}) > 1:
            raise DataValidationError("Conflicting labels occur in an exact duplicate family")
        for member in members[1:]:
            union.union(members[0], member)
            exact_links += 1
    prompt_links = 0
    for members in prompt_context.values():
        for member in members[1:]:
            union.union(members[0], member)
            prompt_links += 1
    roots = [union.find(index) for index in range(len(validated))]
    order = {root: index for index, root in enumerate(sorted(set(roots)))}
    groups = tuple(f"v4-group-{order[root]:04d}" for root in roots)
    counts = pd.Series(groups).value_counts()
    return GroupAudit(
        groups,
        int(len(counts)),
        int((counts > 1).sum()),
        int(counts.max()),
        exact_links,
        prompt_links,
    )


def make_grouped_folds(
    labels: np.ndarray,
    group_ids: tuple[str, ...] | np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> FoldAssignment:
    """Create validated stratified-group folds and prohibit family overlap."""
    truth = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=object)
    if len(truth) != len(groups):
        raise ValueError("Labels and group IDs must have equal length")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments = np.full(len(truth), -1, dtype=np.int64)
    rows = np.arange(len(truth))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    class_counts: list[dict[int, int]] = []
    for fold, (train, validation) in enumerate(splitter.split(rows, truth, groups)):
        if set(groups[train]) & set(groups[validation]):
            raise DataValidationError(f"Duplicate-family leakage in fold {fold + 1}")
        values, counts = np.unique(truth[validation], return_counts=True)
        if set(values) != {0, 1}:
            raise DataValidationError(f"Grouped fold {fold + 1} does not contain both classes")
        if np.any(assignments[validation] >= 0):
            raise DataValidationError("A row appears in multiple validation folds")
        assignments[validation] = fold
        folds.append((train.astype(np.int64), validation.astype(np.int64)))
        class_counts.append({int(value): int(count) for value, count in zip(values, counts)})
    if len(folds) != n_splits or np.any(assignments < 0):
        raise DataValidationError("Grouped folds did not cover every row exactly once")
    return FoldAssignment(tuple(folds), tuple(assignments), tuple(class_counts), "stratified_group")


def _effective_contexts(
    frame: pd.DataFrame, retrieval: RetrievalResult, retrieval_threshold: float
) -> list[str]:
    contexts: list[str] = []
    for index, raw_context in enumerate(frame["context"]):
        cleaned = clean_context(raw_context)
        if cleaned:
            contexts.append(cleaned)
        elif retrieval.similarities[index] >= retrieval_threshold:
            contexts.append(retrieval.snippets[index])
        else:
            contexts.append("")
    return contexts


def _fit_predict_probabilities(
    features: pd.DataFrame,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features.iloc[train_indices])
    validation_features = scaler.transform(features.iloc[validation_indices])
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(train_features, labels[train_indices])
    matches = np.flatnonzero(model.classes_ == 1)
    if len(matches) != 1:
        raise DataValidationError("Logistic regression exposes no unique label-1 probability")
    probabilities = model.predict_proba(validation_features)[:, int(matches[0])]
    if not np.isfinite(probabilities).all():
        raise DataValidationError("Logistic regression produced non-finite probabilities")
    return probabilities.astype(np.float64)


def _oof_for_folds(
    features: pd.DataFrame,
    labels: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    for train, validation in folds:
        if np.isfinite(oof[validation]).any():
            raise DataValidationError("OOF validation positions overlap")
        oof[validation] = _fit_predict_probabilities(features, labels, train, validation)
    if not np.isfinite(oof).all():
        raise DataValidationError("OOF probabilities do not cover every row")
    return oof


def _threshold_rank(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple:
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = classification_metrics(labels, predictions)
    return (
        float(metrics["macro_f1"]),
        float(metrics["f1_label0"]),
        -abs(float(threshold) - ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD),
        -float(threshold),
    )


def original_method_estimate(
    frame: pd.DataFrame, retrieval: RetrievalResult
) -> dict[str, Any]:
    """Reproduce the published non-grouped, same-OOF retrieval-cutoff estimate."""
    no_context = np.asarray([not bool(clean_context(value)) for value in frame["context"]])
    positions = np.flatnonzero(no_context)
    labels = frame.iloc[positions]["label"].to_numpy(dtype=np.int64)
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = tuple(
        (train.astype(np.int64), validation.astype(np.int64))
        for train, validation in split.split(np.arange(len(labels)), labels)
    )
    table: list[dict[str, Any]] = []
    for threshold in RETRIEVAL_THRESHOLDS:
        contexts = [
            retrieval.snippets[index] if retrieval.similarities[index] >= threshold else ""
            for index in positions
        ]
        features = build_lexical_frame(contexts, frame.iloc[positions]["response_bn"])
        probabilities = _oof_for_folds(features, labels, folds)
        predictions = (probabilities >= 0.5).astype(np.int64)
        metrics = classification_metrics(labels, predictions)
        table.append(
            {
                "retrieval_threshold": threshold,
                "matched_fraction": float(np.mean([bool(context) for context in contexts])),
                "metrics": metrics,
                "rank": _threshold_rank(labels, probabilities, threshold),
            }
        )
    selected = max(table, key=lambda row: row["rank"])
    return {
        "name": "original-method full-OOF retrieval-threshold tuning estimate",
        "scope": "official no-context subset only",
        "row_count": int(len(positions)),
        "selected_retrieval_threshold": float(selected["retrieval_threshold"]),
        "published_deployment_threshold": ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
        "selected_metrics": selected["metrics"],
        "threshold_table": [
            {key: value for key, value in row.items() if key != "rank"} for row in table
        ],
        "optimistic": True,
    }


def honest_grouped_estimate(
    frame: pd.DataFrame, retrieval: RetrievalResult, audit: GroupAudit
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Nested 5x3 grouped estimate with retrieval cutoff selected inside each outer fold."""
    labels = frame["label"].to_numpy(dtype=np.int64)
    outer = make_grouped_folds(labels, audit.group_ids, n_splits=5, seed=RANDOM_STATE)
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    selected_thresholds: list[float] = []
    fold_metrics: list[dict[str, Any]] = []
    all_features = {
        threshold: build_lexical_frame(
            _effective_contexts(frame, retrieval, threshold), frame["response_bn"]
        )
        for threshold in RETRIEVAL_THRESHOLDS
    }
    groups = np.asarray(audit.group_ids, dtype=object)
    for fold_index, (outer_train, outer_validation) in enumerate(outer.folds):
        inner_assignment = make_grouped_folds(
            labels[outer_train],
            groups[outer_train],
            n_splits=3,
            seed=RANDOM_STATE + fold_index + 1,
        )
        ranked: list[tuple[tuple, float]] = []
        for threshold, features in all_features.items():
            inner_oof = np.full(len(outer_train), np.nan, dtype=np.float64)
            for inner_train, inner_validation in inner_assignment.folds:
                global_train = outer_train[inner_train]
                global_validation = outer_train[inner_validation]
                inner_oof[inner_validation] = _fit_predict_probabilities(
                    features, labels, global_train, global_validation
                )
            if not np.isfinite(inner_oof).all():
                raise DataValidationError("Nested grouped OOF did not cover outer-training rows")
            ranked.append((_threshold_rank(labels[outer_train], inner_oof, threshold), threshold))
        _, selected_threshold = max(ranked)
        selected_thresholds.append(float(selected_threshold))
        probabilities = _fit_predict_probabilities(
            all_features[selected_threshold], labels, outer_train, outer_validation
        )
        oof[outer_validation] = probabilities
        predictions = (probabilities >= 0.5).astype(np.int64)
        fold_metrics.append(classification_metrics(labels[outer_validation], predictions))
    if not np.isfinite(oof).all():
        raise DataValidationError("Honest grouped OOF did not cover every official row")
    predictions = (oof >= 0.5).astype(np.int64)
    metrics = classification_metrics(labels, predictions)
    warning = prediction_collapse_warning(predictions, name="V4-A honest grouped OOF")
    return (
        {
            "name": "nested 5x3 duplicate-aware grouped estimate",
            "scope": "all official labeled rows",
            "metrics": metrics,
            "outer_fold_metrics": fold_metrics,
            "outer_selected_retrieval_thresholds": selected_thresholds,
            "collapse_warning": warning,
            "optimistic": False,
        },
        oof,
        np.asarray(outer.validation_fold_by_row, dtype=np.int64),
    )


def evaluate_v4a(frame: pd.DataFrame, retrieval: RetrievalResult) -> ValidationResults:
    """Calculate both explicitly labeled estimates without using competition test data."""
    audit = build_duplicate_groups(frame)
    original = original_method_estimate(frame, retrieval)
    grouped, oof, folds = honest_grouped_estimate(frame, retrieval, audit)
    return ValidationResults(original, grouped, oof, folds, audit)


def fit_final_model(
    frame: pd.DataFrame,
    retrieval: RetrievalResult,
    *,
    retrieval_threshold: float = ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
) -> tuple[StandardScaler, LogisticRegression]:
    """Fit the published lexical classifier on all official labels after decisions freeze."""
    contexts = _effective_contexts(frame, retrieval, retrieval_threshold)
    features = build_lexical_frame(contexts, frame["response_bn"])
    labels = frame["label"].to_numpy(dtype=np.int64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(scaled, labels)
    return scaler, model


def predict_probabilities(
    frame: pd.DataFrame,
    retrieval: RetrievalResult,
    scaler: StandardScaler,
    model: LogisticRegression,
    *,
    retrieval_threshold: float = ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
) -> np.ndarray:
    """Produce label-1 probabilities without making assumptions about class-column order."""
    contexts = _effective_contexts(frame, retrieval, retrieval_threshold)
    features = build_lexical_frame(contexts, frame["response_bn"])
    matches = np.flatnonzero(model.classes_ == 1)
    if len(matches) != 1:
        raise DataValidationError("Final model exposes no unique label-1 probability")
    probabilities = model.predict_proba(scaler.transform(features))[:, int(matches[0])]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise DataValidationError("Final model produced invalid probabilities")
    return probabilities.astype(np.float64)


def validate_oof_probability_artifact(frame: pd.DataFrame, expected_rows: int) -> None:
    """Validate the row-indexed honest OOF artifact without retaining source text."""
    expected = ("row_index", "fold", "label", "probability_label1")
    if tuple(frame.columns) != expected or len(frame) != expected_rows:
        raise DataValidationError("OOF probability artifact schema or row count is invalid")
    if frame["row_index"].tolist() != list(range(expected_rows)):
        raise DataValidationError("OOF row indices must be complete and ordered")
    if not frame["label"].isin([0, 1]).all() or frame["fold"].min() < 0:
        raise DataValidationError("OOF labels or fold assignments are invalid")
    probabilities = frame["probability_label1"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise DataValidationError("OOF probabilities must be finite values in [0,1]")


def validate_test_probability_artifact(frame: pd.DataFrame, test_ids: pd.Series) -> None:
    """Validate ID alignment and probability range without printing any identifier."""
    if tuple(frame.columns) != ("id", "probability_label1") or len(frame) != len(test_ids):
        raise DataValidationError("Test probability artifact schema or row count is invalid")
    if frame["id"].tolist() != test_ids.tolist():
        raise DataValidationError("Test probability IDs must exactly preserve test order")
    probabilities = frame["probability_label1"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise DataValidationError("Test probabilities must be finite values in [0,1]")
