"""Frozen corpus-only character TF-IDF retrieval for V7."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .v5_normalization import normalize_v5, tokenize_v5
from .v7_corpus import Article


ANALYZER = "char_wb"
NGRAM_RANGE = (2, 4)
MAX_FEATURES = 50_000
TOP_K = 5
SNIPPET_CHARACTERS = 800
QUERY_WINDOW_CHARACTERS = 800


@dataclass(frozen=True)
class RetrievedPassage:
    """Bounded in-memory evidence; never serialized to tracked output."""

    article_id: str
    title: str
    snippet: str
    score: float


def intro_snippet(body: str, *, characters: int = SNIPPET_CHARACTERS) -> str:
    normalized = normalize_v5(body)
    return normalized[:characters]


def query_centered_snippet(
    query: str,
    body: str,
    *,
    characters: int = QUERY_WINDOW_CHARACTERS,
) -> str:
    """Choose the highest distinct-query-token-coverage window, then earliest."""
    normalized_query = normalize_v5(query)
    normalized_body = normalize_v5(body)
    if len(normalized_body) <= characters:
        return normalized_body
    tokens = tuple(sorted(set(tokenize_v5(normalized_query)), key=lambda value: (-len(value), value)))
    candidate_starts = {0}
    for token in tokens:
        start = 0
        while token:
            position = normalized_body.find(token, start)
            if position < 0:
                break
            candidate_starts.add(max(0, min(position - characters // 2, len(normalized_body) - characters)))
            start = position + max(1, len(token))
    ranked: list[tuple[int, int]] = []
    for start in candidate_starts:
        window = normalized_body[start : start + characters]
        coverage = sum(token in window for token in tokens)
        ranked.append((coverage, -start))
    _coverage, negative_start = max(ranked)
    selected_start = -negative_start
    return normalized_body[selected_start : selected_start + characters]


class CharacterTfidfRetriever:
    """An immutable article-only index that never fits on official queries."""

    def __init__(self, articles: Sequence[Article]) -> None:
        if not articles:
            raise ValueError("V7 retrieval requires a nonempty unlabeled corpus")
        started = perf_counter()
        self.articles = tuple(articles)
        self.vectorizer = TfidfVectorizer(
            analyzer=ANALYZER,
            ngram_range=NGRAM_RANGE,
            max_features=MAX_FEATURES,
            lowercase=False,
            dtype=np.float32,
            norm="l2",
        )
        documents = [f"{article.title}\n{article.body}" for article in self.articles]
        self.matrix = self.vectorizer.fit_transform(documents).tocsr()
        if not np.isfinite(self.matrix.data).all():
            raise ValueError("V7 retrieval index contains NaN or Inf")
        self.fit_audit = {
            "corpus_article_count": len(self.articles),
            "official_query_count_used_for_fit": 0,
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "raw_vocabulary_persisted": False,
            "index_rows": int(self.matrix.shape[0]),
            "index_columns": int(self.matrix.shape[1]),
            "index_nonzero_values": int(self.matrix.nnz),
            "index_sparse_bytes": int(
                self.matrix.data.nbytes
                + self.matrix.indices.nbytes
                + self.matrix.indptr.nbytes
            ),
            "index_build_seconds": perf_counter() - started,
        }

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        snippet_method: str,
    ) -> tuple[RetrievedPassage, ...]:
        if not 1 <= top_k <= TOP_K:
            raise ValueError("V7 top-k must be between 1 and the frozen value 5")
        if snippet_method not in {"intro", "query_centered"}:
            raise ValueError("Unknown V7 snippet method")
        normalized_query = normalize_v5(query)
        vector = self.vectorizer.transform([normalized_query])
        scores = (self.matrix @ vector.T).toarray().ravel().astype(np.float64)
        order = sorted(
            range(len(self.articles)),
            key=lambda index: (-float(scores[index]), self.articles[index].article_id),
        )[:top_k]
        passages: list[RetrievedPassage] = []
        for index in order:
            article = self.articles[index]
            snippet = (
                intro_snippet(article.body)
                if snippet_method == "intro"
                else query_centered_snippet(normalized_query, article.body)
            )
            passages.append(
                RetrievedPassage(
                    article.article_id,
                    article.title,
                    snippet,
                    float(scores[index]),
                )
            )
        return tuple(passages)

    def retrieve_queries(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        snippet_method: str,
    ) -> tuple[tuple[RetrievedPassage, ...], ...]:
        return tuple(
            self.retrieve(query, top_k=top_k, snippet_method=snippet_method)
            for query in queries
        )
