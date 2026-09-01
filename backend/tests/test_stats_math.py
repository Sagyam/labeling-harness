"""Tests for the shared median, which backs every throughput projection."""

from __future__ import annotations

from app.utils.stats_math import median


def test_no_values_has_no_median() -> None:
    assert median([]) is None


def test_odd_length_takes_the_middle_value() -> None:
    assert median([5.0, 1.0, 3.0]) == 3.0


def test_even_length_averages_the_two_middle_values() -> None:
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_a_single_value_is_its_own_median() -> None:
    assert median([7.5]) == 7.5


def test_the_input_order_does_not_matter() -> None:
    assert median([9.0, 2.0, 4.0, 1.0]) == median([1.0, 2.0, 4.0, 9.0])
