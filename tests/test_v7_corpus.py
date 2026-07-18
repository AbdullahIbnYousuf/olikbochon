from __future__ import annotations

import json
from pathlib import Path

import pytest

from olikbochon.v7_corpus import MINIMUM_ARTICLE_CHARACTERS, load_corpus


def record(index: int, *, body: str | None = None) -> dict[str, object]:
    return {
        "id": str(index),
        "url": f"https://example.invalid/{index}",
        "title": f"শিরোনাম {index}",
        "text": body if body is not None else ("বাংলা নিবন্ধ " * 12),
    }


def write_chunk(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


def test_corpus_deduplicates_exact_chunk_copies_and_is_deterministic(tmp_path: Path) -> None:
    rows = [record(1), record(2), record(3, body="ছোট")]
    first = tmp_path / "AA" / "wiki_00"
    duplicate = tmp_path / "copy" / "AA" / "wiki_00"
    write_chunk(first, rows)
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(first.read_bytes())
    articles, manifest = load_corpus(tmp_path)
    repeated_articles, repeated_manifest = load_corpus(tmp_path)
    assert articles == repeated_articles
    assert manifest == repeated_manifest
    assert manifest.recursive_file_count == 2
    assert manifest.candidate_chunk_file_count == 2
    assert manifest.unique_chunk_count == 1
    assert manifest.duplicate_chunk_file_count == 1
    assert manifest.parsed_record_count == 3
    assert manifest.usable_article_count == 2
    assert manifest.too_short_article_count == 1
    assert manifest.duplicate_url_count == 0
    assert manifest.malformed_record_count == 0
    assert manifest.corpus_is_unlabeled
    assert MINIMUM_ARTICLE_CHARACTERS == 70


def test_corpus_manifest_detects_content_change(tmp_path: Path) -> None:
    chunk = tmp_path / "AA" / "wiki_00"
    write_chunk(chunk, [record(1)])
    _articles, before = load_corpus(tmp_path)
    write_chunk(chunk, [record(1, body="পরিবর্তিত বাংলা নিবন্ধ " * 12)])
    _articles, after = load_corpus(tmp_path)
    assert before.logical_content_manifest_sha256 != after.logical_content_manifest_sha256
    assert before.physical_manifest_sha256 != after.physical_manifest_sha256


def test_corpus_rejects_any_label_field(tmp_path: Path) -> None:
    labeled = record(1)
    labeled["label"] = 1
    write_chunk(tmp_path / "AA" / "wiki_00", [labeled])
    with pytest.raises(ValueError, match="prohibited label"):
        load_corpus(tmp_path)


def test_corpus_counts_malformed_empty_and_duplicate_urls(tmp_path: Path) -> None:
    chunk = tmp_path / "AA" / "wiki_00"
    chunk.parent.mkdir(parents=True)
    duplicate = record(2)
    duplicate["url"] = record(1)["url"]
    lines = [
        json.dumps(record(1), ensure_ascii=False),
        json.dumps(duplicate, ensure_ascii=False),
        json.dumps(record(3, body=""), ensure_ascii=False),
        "{not-json}",
    ]
    chunk.write_text("\n".join(lines) + "\n", encoding="utf-8")
    articles, manifest = load_corpus(tmp_path)
    assert len(articles) == 1
    assert manifest.parsed_record_count == 3
    assert manifest.duplicate_url_count == 1
    assert manifest.empty_article_count == 1
    assert manifest.malformed_record_count == 1
