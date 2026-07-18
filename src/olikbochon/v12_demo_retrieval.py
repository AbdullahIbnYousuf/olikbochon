"""Frozen label-balanced demonstration retrieval for V12."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .v5_normalization import normalize_v5


SEPARATOR = " [SEP] "
CHARACTER_WEIGHT = 0.65
WORD_WEIGHT = 0.35
DEMONSTRATIONS_PER_LABEL = 3


@dataclass(frozen=True)
class Demonstration:
    source: str
    source_index: int
    group_id: str | None
    prompt: str
    response: str
    label: int

    @property
    def normalized_pair(self) -> str:
        return normalized_pair(self.prompt, self.response)


@dataclass(frozen=True)
class RetrievedSet:
    demonstrations: tuple[Demonstration, ...]
    scores: tuple[float, ...]
    exact_override: int | None


def normalized_pair(prompt: object, response: object) -> str:
    return f"{normalize_v5(prompt)}{SEPARATOR}{normalize_v5(response)}"


def _vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    return (
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
            min_df=1,
            norm="l2",
            max_features=100_000,
        ),
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            norm="l2",
            max_features=50_000,
        ),
    )


class FrozenDemonstrationIndex:
    """Train-label-only sparse index with deterministic balanced retrieval."""

    def __init__(self, demonstrations: Sequence[Demonstration]) -> None:
        self.demonstrations = tuple(demonstrations)
        if not self.demonstrations or {item.label for item in self.demonstrations} != {0, 1}:
            raise ValueError("V12 demonstration pool must contain both labels")
        texts = [item.normalized_pair for item in self.demonstrations]
        self.character, self.word = _vectorizers()
        self.character_matrix = self.character.fit_transform(texts).tocsr()
        self.word_matrix = self.word.fit_transform(texts).tocsr()
        self.labels = np.asarray([item.label for item in self.demonstrations], dtype=np.int64)
        exact: dict[str, set[int]] = {}
        for item in self.demonstrations:
            exact.setdefault(item.normalized_pair, set()).add(item.label)
        self.exact_labels = exact
        self.manifest_sha256 = hashlib.sha256(
            "\n".join(
                f"{item.source}:{item.source_index}:{item.group_id}:{item.label}:"
                f"{hashlib.sha256(item.normalized_pair.encode('utf-8')).hexdigest()}"
                for item in self.demonstrations
            ).encode("utf-8")
        ).hexdigest()

    def retrieve(
        self,
        prompt: object,
        response: object,
        *,
        target_source: str | None = None,
        target_index: int | None = None,
        excluded_group: str | None = None,
    ) -> RetrievedSet:
        text = normalized_pair(prompt, response)
        char_query = self.character.transform([text])
        word_query = self.word.transform([text])
        scores = (
            CHARACTER_WEIGHT * (char_query @ self.character_matrix.T).toarray().ravel()
            + WORD_WEIGHT * (word_query @ self.word_matrix.T).toarray().ravel()
        )
        allowed = np.ones(len(self.demonstrations), dtype=bool)
        for position, item in enumerate(self.demonstrations):
            if target_source == item.source and target_index == item.source_index:
                allowed[position] = False
            if excluded_group is not None and item.group_id == excluded_group:
                allowed[position] = False
        selected: list[int] = []
        for label in (0, 1):
            positions = np.flatnonzero(allowed & (self.labels == label))
            order = sorted(positions, key=lambda index: (-float(scores[index]), int(index)))
            if len(order) < DEMONSTRATIONS_PER_LABEL:
                raise RuntimeError("V12 cannot retrieve three demonstrations for each label")
            selected.extend(order[:DEMONSTRATIONS_PER_LABEL])
        selected.sort(key=lambda index: (int(self.labels[index]), -float(scores[index]), int(index)))
        exact_labels = {
            self.demonstrations[position].label
            for position in np.flatnonzero(allowed)
            if self.demonstrations[position].normalized_pair == text
        }
        override = next(iter(exact_labels)) if len(exact_labels) == 1 else None
        return RetrievedSet(
            tuple(self.demonstrations[index] for index in selected),
            tuple(float(scores[index]) for index in selected),
            override,
        )

    def aggregate_overlap(self, prompts: Sequence[object], responses: Sequence[object]) -> dict:
        pairs = [normalized_pair(p, r) for p, r in zip(prompts, responses, strict=True)]
        char_queries = self.character.transform(pairs)
        word_queries = self.word.transform(pairs)
        similarities = (
            CHARACTER_WEIGHT * (char_queries @ self.character_matrix.T).toarray()
            + WORD_WEIGHT * (word_queries @ self.word_matrix.T).toarray()
        )
        matched_labels = [self.exact_labels.get(pair, set()) for pair in pairs]
        exact_pair = np.asarray([bool(labels) for labels in matched_labels], dtype=bool)
        prompt_labels: dict[str, set[int]] = {}
        for item in self.demonstrations:
            prompt_labels.setdefault(normalize_v5(item.prompt), set()).add(item.label)
        exact_prompt = np.asarray(
            [normalize_v5(prompt) in prompt_labels for prompt in prompts], dtype=bool
        )
        top5 = np.argpartition(-similarities, kth=4, axis=1)[:, :5]
        unanimous = np.asarray([len(set(self.labels[row])) == 1 for row in top5])
        nearest = similarities.max(axis=1)
        return {
            "rows": len(pairs),
            "exact_prompt_response_count": int(exact_pair.sum()),
            "exact_prompt_response_fraction": float(exact_pair.mean()),
            "exact_prompt_count": int(exact_prompt.sum()),
            "exact_prompt_fraction": float(exact_prompt.mean()),
            "nearest_similarity_quantiles": {
                str(q): float(np.quantile(nearest, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
            "unanimous_top5_label_count": int(unanimous.sum()),
            "unanimous_top5_label_fraction": float(unanimous.mean()),
            "exact_matches_label_consistent": all(
                len(labels) == 1 for labels in matched_labels if labels
            ),
        }
