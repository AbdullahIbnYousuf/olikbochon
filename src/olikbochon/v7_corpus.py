"""Deterministic, unlabeled Bengali Wikipedia ingestion for V7 retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .v5_normalization import normalize_v5


CHUNK_NAME = re.compile(r"^wiki_[0-9a-f]+$", re.IGNORECASE)
MINIMUM_ARTICLE_CHARACTERS = 70
FORBIDDEN_LABEL_KEYS = frozenset(
    {"label", "labels", "target", "class", "hallucination", "is_hallucinated"}
)


@dataclass(frozen=True)
class Article:
    """One normalized unlabeled retrieval document retained only in memory."""

    article_id: str
    url: str
    title: str
    body: str


@dataclass(frozen=True)
class CorpusManifest:
    """Aggregate authentication data without article content or source paths."""

    recursive_file_count: int
    candidate_chunk_file_count: int
    unique_chunk_count: int
    duplicate_chunk_file_count: int
    total_physical_bytes: int
    total_unique_bytes: int
    physical_manifest_sha256: str
    logical_content_manifest_sha256: str
    parsed_record_count: int
    usable_article_count: int
    duplicate_url_count: int
    malformed_record_count: int
    empty_article_count: int
    too_short_article_count: int
    minimum_article_characters: int
    chunk_name_pattern: str = "wiki_<hex> without extension"
    corpus_is_unlabeled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_corpus_path(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_dir():
        raise ValueError("V7 corpus path must be an existing absolute directory")
    return candidate.resolve()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _article_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def load_corpus(path: Path) -> tuple[tuple[Article, ...], CorpusManifest]:
    """Parse strict UTF-8 JSON lines after exact chunk-content deduplication."""
    root = require_corpus_path(path)
    all_files = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    chunks = [candidate for candidate in all_files if CHUNK_NAME.fullmatch(candidate.name)]
    if not chunks:
        raise ValueError("No WikiExtractor wiki_<hex> chunks were found")

    physical_manifest = hashlib.sha256()
    content_groups: dict[tuple[int, str], list[Path]] = {}
    for chunk in chunks:
        raw = chunk.read_bytes()
        size = len(raw)
        digest = _sha256_bytes(raw)
        relative = chunk.relative_to(root).as_posix()
        physical_manifest.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))
        content_groups.setdefault((size, digest), []).append(chunk)

    logical_manifest = hashlib.sha256()
    canonical_chunks: list[Path] = []
    for (size, digest), copies in sorted(content_groups.items()):
        logical_manifest.update(f"{size}\0{digest}\n".encode("ascii"))
        canonical_chunks.append(
            min(
                copies,
                key=lambda candidate: (
                    len(candidate.relative_to(root).parts),
                    candidate.relative_to(root).as_posix(),
                ),
            )
        )

    parsed = malformed = empty = too_short = duplicate_urls = 0
    articles: list[Article] = []
    seen_urls: set[str] = set()
    for chunk in canonical_chunks:
        try:
            handle = chunk.open("r", encoding="utf-8", errors="strict")
            with handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    if not isinstance(record, dict):
                        malformed += 1
                        continue
                    if FORBIDDEN_LABEL_KEYS & {str(key).lower() for key in record}:
                        raise ValueError("Corpus record contains a prohibited label field")
                    parsed += 1
                    title_value = record.get("title")
                    body_value = record.get("text")
                    url_value = record.get("url")
                    if not all(
                        isinstance(value, str)
                        for value in (title_value, body_value, url_value)
                    ):
                        malformed += 1
                        continue
                    title = normalize_v5(title_value)
                    body = normalize_v5(body_value)
                    url = url_value.strip()
                    if not title or not body or not url:
                        empty += 1
                        continue
                    if len(body) < MINIMUM_ARTICLE_CHARACTERS:
                        too_short += 1
                        continue
                    if url in seen_urls:
                        duplicate_urls += 1
                        continue
                    seen_urls.add(url)
                    articles.append(Article(_article_id(url), url, title, body))
        except UnicodeDecodeError as exc:
            raise ValueError(f"Corpus chunk is not strict UTF-8: {chunk.name}") from exc

    manifest = CorpusManifest(
        recursive_file_count=len(all_files),
        candidate_chunk_file_count=len(chunks),
        unique_chunk_count=len(canonical_chunks),
        duplicate_chunk_file_count=len(chunks) - len(canonical_chunks),
        total_physical_bytes=sum(candidate.stat().st_size for candidate in chunks),
        total_unique_bytes=sum(candidate.stat().st_size for candidate in canonical_chunks),
        physical_manifest_sha256=physical_manifest.hexdigest(),
        logical_content_manifest_sha256=logical_manifest.hexdigest(),
        parsed_record_count=parsed,
        usable_article_count=len(articles),
        duplicate_url_count=duplicate_urls,
        malformed_record_count=malformed,
        empty_article_count=empty,
        too_short_article_count=too_short,
        minimum_article_characters=MINIMUM_ARTICLE_CHARACTERS,
    )
    return tuple(articles), manifest


def manifest_only(path: Path) -> CorpusManifest:
    """Return aggregate authentication while allowing article memory to be released."""
    _articles, manifest = load_corpus(path)
    return manifest
