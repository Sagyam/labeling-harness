"""Tests for frozen split assignment."""

from __future__ import annotations

import pytest

from app.services.splits import assign_split, split_hash_unit

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def test_assignment_is_deterministic() -> None:
    first = assign_split("show-a_ep012", seed=20260101, ratios=RATIOS)
    second = assign_split("show-a_ep012", seed=20260101, ratios=RATIOS)
    assert first == second


def test_assignment_changes_with_the_seed() -> None:
    """A different seed is a different corpus division, so it must be able to differ."""
    assignments = {assign_split("show-a_ep012", seed=s, ratios=RATIOS) for s in range(200)}
    assert len(assignments) > 1


def test_assignment_is_one_of_the_three_splits() -> None:
    for i in range(50):
        assert assign_split(f"ep{i}", seed=1, ratios=RATIOS) in {"train", "val", "test"}


def test_hash_is_uniform_enough_over_many_episodes() -> None:
    counts = {"train": 0, "val": 0, "test": 0}
    n = 4000
    for i in range(n):
        counts[assign_split(f"show-a_ep{i:05d}", seed=20260101, ratios=RATIOS)] += 1
    assert counts["train"] / n == pytest.approx(0.8, abs=0.03)
    assert counts["val"] / n == pytest.approx(0.1, abs=0.02)
    assert counts["test"] / n == pytest.approx(0.1, abs=0.02)


def test_hash_unit_is_in_range() -> None:
    for i in range(100):
        value = split_hash_unit(f"ep{i}", seed=7)
        assert 0.0 <= value < 1.0


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum"):
        assign_split("ep", seed=1, ratios={"train": 0.5, "val": 0.1, "test": 0.1})


def test_ratios_must_name_the_three_splits() -> None:
    with pytest.raises(ValueError, match="train"):
        assign_split("ep", seed=1, ratios={"train": 0.5, "holdout": 0.5})


def test_known_vector_does_not_drift() -> None:
    """Pinning one value catches an accidental change to the hash, which would move episodes
    between train and test and silently invalidate every earlier export."""
    assert round(split_hash_unit("show-a_ep012", seed=20260101), 12) == 0.7146923149
