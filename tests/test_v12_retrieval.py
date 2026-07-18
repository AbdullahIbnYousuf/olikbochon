from __future__ import annotations

from olikbochon.v12_demo_retrieval import Demonstration, FrozenDemonstrationIndex


def _pool() -> list[Demonstration]:
    return [
        Demonstration("public", index, None, f"প্রশ্ন {index}", f"উত্তর {index}", index % 2)
        for index in range(12)
    ]


def test_retrieval_is_balanced_deterministic_and_exact_pair_only() -> None:
    index = FrozenDemonstrationIndex(_pool())
    first = index.retrieve("প্রশ্ন 4", "উত্তর 4")
    second = index.retrieve("প্রশ্ন 4", "উত্তর 4")
    assert first == second
    assert [item.label for item in first.demonstrations].count(0) == 3
    assert [item.label for item in first.demonstrations].count(1) == 3
    assert first.exact_override == 0
    assert index.retrieve("প্রশ্ন 4", "ভিন্ন উত্তর").exact_override is None


def test_same_source_row_and_group_are_excluded() -> None:
    pool = _pool() + [
        Demonstration("official", 99, "locked-group", "একই", "একই", 0),
        Demonstration("official", 100, "locked-group", "অন্য", "অন্য", 1),
    ]
    index = FrozenDemonstrationIndex(pool)
    result = index.retrieve(
        "একই", "একই", target_source="official", target_index=99, excluded_group="locked-group"
    )
    assert all(item.group_id != "locked-group" for item in result.demonstrations)
    assert result.exact_override is None
