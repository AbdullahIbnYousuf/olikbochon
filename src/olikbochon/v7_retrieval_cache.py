"""Build an ignored official-training retrieval cache without competition test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file
from .v4_wikipedia import (
    V4Files,
    WikipediaRetriever,
    _wiki_chunks,
    load_wikipedia_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--wikipedia-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-name", required=True)
    parser.add_argument("--resource-version", required=True)
    parser.add_argument("--resource-license", required=True)
    parser.add_argument("--resource-source-url", required=True)
    parser.add_argument("--resource-sha1", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = load_labeled_json(args.train_data, expected_sha256=OFFICIAL_SAMPLE_SHA256)
    chunks = _wiki_chunks(args.wikipedia_root)
    if not chunks:
        raise FileNotFoundError("A valid WikiExtractor AA/AB/AC/AD root is required")
    placeholder = args.train_data
    corpus = load_wikipedia_corpus(
        V4Files(
            args.train_data,
            placeholder,
            placeholder,
            args.train_data.parent,
            args.wikipedia_root,
            chunks,
            0,
        )
    )
    retrieval = WikipediaRetriever(corpus).retrieve(frame["prompt_bn"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snippets": retrieval.snippets,
        "similarities": retrieval.similarities,
        "official_rows": len(frame),
        "test_rows_accessed": 0,
        "resource": {
            "name": args.resource_name,
            "version": args.resource_version,
            "license": args.resource_license,
            "source_url": args.resource_source_url,
            "source_sha1": args.resource_sha1,
            "chunk_count": len(chunks),
            "aggregate_bytes": sum(path.stat().st_size for path in chunks),
            "usable_article_count": len(corpus.articles),
        },
    }
    joblib.dump(payload, args.output, compress=3)
    manifest = {
        "status": "complete",
        "cache": args.output.name,
        "cache_sha256": sha256_file(args.output),
        "official_rows": len(frame),
        "test_rows_accessed": 0,
        "resource": payload["resource"],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: manifest[k] for k in ("status", "official_rows", "test_rows_accessed")}))


if __name__ == "__main__":
    main()
