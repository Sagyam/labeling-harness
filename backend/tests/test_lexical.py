"""Tests for the two token-set agreement signals (D67)."""

from __future__ import annotations

import pytest

from app.services.lexical import roman_gap, seed_orphan_rate


def test_a_seed_every_other_system_repeats_has_no_orphans() -> None:
    assert seed_orphan_rate("हामी जान्छौँ", ["हामी जान्छौँ", "हामी जान्छौँ"]) == pytest.approx(0.0)


def test_a_token_no_other_system_has_is_an_orphan() -> None:
    # "wire" appears in neither reference: 1 of 3 seed tokens.
    rate = seed_orphan_rate("मेरो wire हो", ["मेरो उहाँ हो", "मेरो उहाँहरू हो"])
    assert rate == pytest.approx(1 / 3)


def test_one_system_agreeing_is_enough_to_settle_a_token() -> None:
    """A token only counts as orphaned when *no* other system has it (D67)."""
    rate = seed_orphan_rate("मेरो wire हो", ["मेरो wire हो", "मेरो उहाँ हो"])
    assert rate == pytest.approx(0.0)


def test_orphan_rate_is_a_share_not_a_count() -> None:
    """Two errors in five words must outrank two errors in fifty (the D54 failure)."""
    short = seed_orphan_rate("a b c d e", ["a b x y z", "a b p q r"])
    long = seed_orphan_rate(
        " ".join(["a", "b", "c", "d", "e"] + [f"w{i}" for i in range(45)]),
        [" ".join(["a", "b", "x", "y", "z"] + [f"w{i}" for i in range(45)])] * 2,
    )
    assert short > long


def test_an_empty_seed_is_unmeasurable_not_perfect() -> None:
    assert seed_orphan_rate("", ["something"]) == 0.0
    assert seed_orphan_rate("something", []) == 0.0


def test_a_latin_word_both_others_have_and_the_seed_lacks_is_a_gap() -> None:
    """The Devanagari-loanword signal: others wrote it in Latin, the seed did not."""
    gap = roman_gap("यो ट्राफिक पुलिस हो", ["यो traffic police हो", "यो traffic police हो"])
    assert gap > 0.0


def test_one_system_alone_is_not_enough_for_a_roman_gap() -> None:
    """MAI alone romanizing proves nothing -- it must be unanimous among the others."""
    gap = roman_gap("यो ट्राफिक पुलिस हो", ["यो traffic police हो", "यो ट्राफिक पुलिस हो"])
    assert gap == pytest.approx(0.0)


def test_a_seed_already_in_latin_has_no_gap() -> None:
    gap = roman_gap("यो traffic police हो", ["यो traffic police हो", "यो traffic police हो"])
    assert gap == pytest.approx(0.0)


def test_the_roman_gap_is_case_insensitive() -> None:
    gap = roman_gap("यो Traffic हो", ["यो traffic हो", "यो TRAFFIC हो"])
    assert gap == pytest.approx(0.0)


def test_the_roman_gap_saturates() -> None:
    """Normalized to 0-1 so it cannot dominate the sum on one long clip."""
    many = " ".join(f"word{i}" for i in range(40))
    assert roman_gap("नेपाली मात्र", [many, many]) == pytest.approx(1.0)


def test_both_signals_are_bounded() -> None:
    for seed, others in [
        ("", []),
        ("a", ["b", "c"]),
        ("क ख ग", ["a b c", "d e f"]),
    ]:
        assert 0.0 <= seed_orphan_rate(seed, others) <= 1.0
        assert 0.0 <= roman_gap(seed, others) <= 1.0
