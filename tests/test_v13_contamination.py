from __future__ import annotations

import pandas as pd

from olikbochon.v13_public_pool import add_keys, deduplicate_pool, normalize, official_safe_pool, skeleton


def _frame(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"context": "[NULL]", "prompt_bn": p, "response_bn": r, "label": y} for p, r, y in rows]
    )


def test_normalization_canonicalizes_digits_without_removing_entities_or_negation() -> None:
    assert normalize("ঢাকা ২০২৬ নয়") == "ঢাকা 2026 নয়"
    assert skeleton("ঢাকা ২০২৬ নয়") == "ঢাকা <DATE> নয়"


def test_validation_and_official_pairs_are_excluded() -> None:
    train = add_keys(_frame([("ক", "খ", 1), ("গ", "ঘ", 0)]), workers=6)
    contrastive = add_keys(_frame([("ক", "খ", 1), ("চ", "ছ", 0)]), workers=6)
    validation = add_keys(_frame([("ক", "খ", 1)]), workers=6)
    pool, audit = deduplicate_pool(train, contrastive, validation)
    assert audit["validation_exact_pair_rows_excluded"] == 2
    official = add_keys(_frame([("চ", "ছ", 0)]), workers=6)
    safe, safe_audit = official_safe_pool(pool, official)
    assert not safe.pair_key.isin(set(official.pair_key)).any()
    assert safe_audit["official_exact_pair_rows_excluded"] == 1
