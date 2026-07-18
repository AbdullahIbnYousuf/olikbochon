"""Frozen, interpretable lexical features and aggregate-only V5 audits."""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .v4_preprocessing import official_context_is_present
from .v5_normalization import normalize_v5, strip_v5_punctuation, tokenize_v5


RARE_TOKEN_MAX_FREQUENCY = 2
NEGATION_MARKERS = frozenset({"না", "নয়", "নয়", "নেই", "not", "no", "never", "without"})
UNCERTAINTY_MARKERS = frozenset(
    {
        "সম্ভবত",
        "হয়তো",
        "হয়তো",
        "সম্ভাব্য",
        "অনিশ্চিত",
        "maybe",
        "perhaps",
        "possibly",
        "likely",
        "approximately",
    }
)
INTERROGATIVE_MARKERS = frozenset(
    {"কি", "কী", "কে", "কেন", "কখন", "কোথায়", "কোথায়", "কিভাবে", "how", "why", "when", "where", "who", "what"}
)
INSTRUCTION_MARKERS = frozenset(
    {"বলুন", "লিখুন", "ব্যাখ্যা", "বর্ণনা", "তালিকা", "give", "write", "explain", "describe", "list"}
)

CONTEXT_PRESENT_FEATURES = (
    "response_in_context_exact",
    "response_in_context_no_punctuation",
    "context_in_response_exact",
    "response_token_coverage",
    "response_character_coverage",
    "token_jaccard",
    "char3_jaccard",
    "char4_jaccard",
    "longest_common_subsequence_ratio",
    "longest_common_substring_ratio",
    "response_context_length_ratio",
    "response_char_count",
    "response_token_count",
    "prompt_char_count",
    "prompt_token_count",
    "context_char_count",
    "context_token_count",
    "numeric_consistency_ratio",
    "numeral_mismatch_count",
    "response_numeral_count",
    "response_bengali_numeral_count",
    "response_arabic_numeral_count",
    "context_bengali_numeral_count",
    "context_arabic_numeral_count",
    "rare_token_coverage",
    "prompt_response_token_jaccard",
    "prompt_context_token_jaccard",
)

CONTEXT_ABSENT_FEATURES = (
    "response_char_count",
    "response_token_count",
    "prompt_char_count",
    "prompt_token_count",
    "prompt_response_token_jaccard",
    "prompt_response_char3_jaccard",
    "response_numeral_count",
    "response_bengali_numeral_count",
    "response_arabic_numeral_count",
    "response_year_like_count",
    "response_punctuation_count",
    "response_negation_count",
    "response_uncertainty_count",
    "response_bengali_script_ratio",
    "response_latin_script_ratio",
    "prompt_has_question_mark",
    "prompt_interrogative_count",
    "prompt_instruction_count",
    "response_rare_token_count",
    "response_named_entity_like_count",
    "response_specificity_score",
)


def _compact(value: Any) -> str:
    return "".join(strip_v5_punctuation(value).split())


def _jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def _ngrams(text: str, width: int) -> set[str]:
    if not text:
        return set()
    if len(text) < width:
        return {text}
    return {text[index : index + width] for index in range(len(text) - width + 1)}


def _multiset_coverage(values: Sequence[str], reference: Sequence[str]) -> float:
    if not values:
        return 1.0
    overlap = Counter(values) & Counter(reference)
    return float(sum(overlap.values()) / len(values))


def _longest_common_subsequence_length(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_character in right:
        current = [0]
        for index, left_character in enumerate(left, start=1):
            if left_character == right_character:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _longest_common_substring_length(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    best = 0
    for right_character in right:
        current = [0]
        for index, left_character in enumerate(left, start=1):
            value = previous[index - 1] + 1 if left_character == right_character else 0
            current.append(value)
            best = max(best, value)
        previous = current
    return best


def _numeral_counts(value: Any) -> tuple[int, int, int]:
    text = unicodedata.normalize("NFC", "" if value is None else str(value))
    bengali = sum("০" <= character <= "৯" for character in text)
    arabic = sum(
        character.isdecimal() and not ("০" <= character <= "৯") for character in text
    )
    return bengali + arabic, bengali, arabic


def _normalized_numerals(value: Any) -> tuple[str, ...]:
    return tuple(token for token in tokenize_v5(value) if token.isdecimal())


def _punctuation_count(value: Any) -> int:
    text = unicodedata.normalize("NFC", "" if value is None else str(value))
    return sum(unicodedata.category(character).startswith("P") for character in text)


def _script_ratios(value: Any) -> tuple[float, float]:
    text = unicodedata.normalize("NFC", "" if value is None else str(value))
    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    if not letters:
        return 0.0, 0.0
    bengali = sum("BENGALI" in unicodedata.name(character, "") for character in letters)
    latin = sum("LATIN" in unicodedata.name(character, "") for character in letters)
    return float(bengali / len(letters)), float(latin / len(letters))


def _source_tokens(value: Any) -> tuple[str, ...]:
    text = unicodedata.normalize("NFC", "" if value is None else str(value))
    tokens: list[str] = []
    current: list[str] = []
    for character in text:
        if unicodedata.category(character)[0] in {"L", "M", "N"}:
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _named_entity_like_count(value: Any) -> int:
    count = 0
    for token in _source_tokens(value):
        has_letter = any(character.isalpha() for character in token)
        has_digit = any(character.isdecimal() for character in token)
        latin_letters = [
            character
            for character in token
            if character.isalpha() and "LATIN" in unicodedata.name(character, "")
        ]
        capitalized = bool(latin_letters) and (
            latin_letters[0].isupper()
            or (len(latin_letters) > 1 and all(character.isupper() for character in latin_letters))
        )
        if capitalized or (has_letter and has_digit):
            count += 1
    return count


@dataclass(frozen=True)
class V5FeatureExtractor:
    """Feature extractor whose only fitted state is train-fold response-token frequency."""

    response_token_frequency: Mapping[str, int]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> V5FeatureExtractor:
        frequencies: Counter[str] = Counter()
        for response in frame["response_bn"]:
            frequencies.update(tokenize_v5(response))
        return cls(dict(frequencies))

    def _rare_response_tokens(self, value: Any) -> tuple[str, ...]:
        return tuple(
            token
            for token in tokenize_v5(value)
            if self.response_token_frequency.get(token, 0) <= RARE_TOKEN_MAX_FREQUENCY
        )

    def context_present_features(self, prompt: Any, context: Any, response: Any) -> dict[str, float]:
        normalized_prompt = normalize_v5(prompt)
        normalized_context = normalize_v5(context)
        normalized_response = normalize_v5(response)
        prompt_tokens = tokenize_v5(normalized_prompt)
        context_tokens = tokenize_v5(normalized_context)
        response_tokens = tokenize_v5(normalized_response)
        compact_context = _compact(normalized_context)
        compact_response = _compact(normalized_response)
        response_numerals = _normalized_numerals(response)
        context_numerals = _normalized_numerals(context)
        response_total, response_bengali, response_arabic = _numeral_counts(response)
        _, context_bengali, context_arabic = _numeral_counts(context)
        rare_tokens = self._rare_response_tokens(response)
        response_length = len(compact_response)
        numeral_overlap = Counter(response_numerals) & Counter(context_numerals)
        return {
            "response_in_context_exact": float(
                bool(normalized_response) and normalized_response in normalized_context
            ),
            "response_in_context_no_punctuation": float(
                bool(compact_response) and compact_response in compact_context
            ),
            "context_in_response_exact": float(
                bool(normalized_context) and normalized_context in normalized_response
            ),
            "response_token_coverage": _multiset_coverage(response_tokens, context_tokens),
            "response_character_coverage": _multiset_coverage(
                tuple(compact_response), tuple(compact_context)
            ),
            "token_jaccard": _jaccard(response_tokens, context_tokens),
            "char3_jaccard": _jaccard(_ngrams(compact_response, 3), _ngrams(compact_context, 3)),
            "char4_jaccard": _jaccard(_ngrams(compact_response, 4), _ngrams(compact_context, 4)),
            "longest_common_subsequence_ratio": (
                _longest_common_subsequence_length(compact_response, compact_context)
                / response_length
                if response_length
                else 1.0
            ),
            "longest_common_substring_ratio": (
                _longest_common_substring_length(compact_response, compact_context)
                / response_length
                if response_length
                else 1.0
            ),
            "response_context_length_ratio": len(normalized_response)
            / max(len(normalized_context), 1),
            "response_char_count": float(len(normalized_response)),
            "response_token_count": float(len(response_tokens)),
            "prompt_char_count": float(len(normalized_prompt)),
            "prompt_token_count": float(len(prompt_tokens)),
            "context_char_count": float(len(normalized_context)),
            "context_token_count": float(len(context_tokens)),
            "numeric_consistency_ratio": (
                sum(numeral_overlap.values()) / len(response_numerals)
                if response_numerals
                else 1.0
            ),
            "numeral_mismatch_count": float(
                len(response_numerals) - sum(numeral_overlap.values())
            ),
            "response_numeral_count": float(response_total),
            "response_bengali_numeral_count": float(response_bengali),
            "response_arabic_numeral_count": float(response_arabic),
            "context_bengali_numeral_count": float(context_bengali),
            "context_arabic_numeral_count": float(context_arabic),
            "rare_token_coverage": float(len(rare_tokens) / len(response_tokens))
            if response_tokens
            else 0.0,
            "prompt_response_token_jaccard": _jaccard(prompt_tokens, response_tokens),
            "prompt_context_token_jaccard": _jaccard(prompt_tokens, context_tokens),
        }

    def context_absent_features(self, prompt: Any, response: Any) -> dict[str, float]:
        normalized_prompt = normalize_v5(prompt)
        normalized_response = normalize_v5(response)
        prompt_tokens = tokenize_v5(normalized_prompt)
        response_tokens = tokenize_v5(normalized_response)
        compact_prompt = _compact(normalized_prompt)
        compact_response = _compact(normalized_response)
        numeral_total, numeral_bengali, numeral_arabic = _numeral_counts(response)
        normalized_numerals = _normalized_numerals(response)
        rare_count = len(self._rare_response_tokens(response))
        entity_count = _named_entity_like_count(response)
        bengali_ratio, latin_ratio = _script_ratios(response)
        return {
            "response_char_count": float(len(normalized_response)),
            "response_token_count": float(len(response_tokens)),
            "prompt_char_count": float(len(normalized_prompt)),
            "prompt_token_count": float(len(prompt_tokens)),
            "prompt_response_token_jaccard": _jaccard(prompt_tokens, response_tokens),
            "prompt_response_char3_jaccard": _jaccard(
                _ngrams(compact_prompt, 3), _ngrams(compact_response, 3)
            ),
            "response_numeral_count": float(numeral_total),
            "response_bengali_numeral_count": float(numeral_bengali),
            "response_arabic_numeral_count": float(numeral_arabic),
            "response_year_like_count": float(
                sum(token.isdecimal() and len(token) == 4 for token in normalized_numerals)
            ),
            "response_punctuation_count": float(_punctuation_count(response)),
            "response_negation_count": float(
                sum(token in NEGATION_MARKERS for token in response_tokens)
            ),
            "response_uncertainty_count": float(
                sum(token in UNCERTAINTY_MARKERS for token in response_tokens)
            ),
            "response_bengali_script_ratio": bengali_ratio,
            "response_latin_script_ratio": latin_ratio,
            "prompt_has_question_mark": float("?" in normalized_prompt),
            "prompt_interrogative_count": float(
                sum(token in INTERROGATIVE_MARKERS for token in prompt_tokens)
            ),
            "prompt_instruction_count": float(
                sum(token in INSTRUCTION_MARKERS for token in prompt_tokens)
            ),
            "response_rare_token_count": float(rare_count),
            "response_named_entity_like_count": float(entity_count),
            "response_specificity_score": float(numeral_total + rare_count + entity_count),
        }

    def transform_row(self, row: Any, *, context_present: bool) -> dict[str, float]:
        if context_present:
            return self.context_present_features(row.prompt_bn, row.context, row.response_bn)
        return self.context_absent_features(row.prompt_bn, row.response_bn)

    def matrix(
        self,
        frame: pd.DataFrame,
        indices: Sequence[int],
        *,
        context_present: bool,
    ) -> np.ndarray:
        names = CONTEXT_PRESENT_FEATURES if context_present else CONTEXT_ABSENT_FEATURES
        rows = [
            self.transform_row(frame.iloc[int(index)], context_present=context_present)
            for index in indices
        ]
        matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("V5 lexical feature matrix contains NaN or Inf")
        return matrix


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Aggregate feature audit groups must be nonempty")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate_feature_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Return route/label numeric summaries without retaining row text or row-level values."""
    extractor = V5FeatureExtractor.fit(frame)
    presence = np.asarray(
        [official_context_is_present(value) for value in frame["context"]], dtype=bool
    )
    labels = frame["label"].to_numpy(dtype=np.int64)
    result: dict[str, Any] = {
        "row_count": int(len(frame)),
        "context_present_count": int(presence.sum()),
        "context_absent_count": int((~presence).sum()),
        "rare_token_max_training_frequency": RARE_TOKEN_MAX_FREQUENCY,
        "row_level_values_persisted": False,
        "routes": {},
    }
    for route_name, route_present, names in (
        ("context_present", True, CONTEXT_PRESENT_FEATURES),
        ("context_absent", False, CONTEXT_ABSENT_FEATURES),
    ):
        route_mask = presence if route_present else ~presence
        route_result: dict[str, Any] = {}
        for label in (0, 1):
            indices = np.flatnonzero(route_mask & (labels == label))
            feature_rows = [
                extractor.transform_row(frame.iloc[int(index)], context_present=route_present)
                for index in indices
            ]
            route_result[f"label_{label}"] = {
                "row_count": int(len(indices)),
                "features": {
                    name: _summary([row[name] for row in feature_rows]) for name in names
                },
            }
        result["routes"][route_name] = route_result
    return result
