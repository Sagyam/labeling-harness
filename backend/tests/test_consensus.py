"""Tests for time-aligned cross-system consensus.

The unit under test puts every system's word spans on one clock and asks where they disagree.
That is a different question from the string comparison in ``analysis.py``: two systems can emit
the same tokens in the same order and still be placed differently, and a system can omit a word
without any of its neighbours moving.
"""

from __future__ import annotations

import pytest

from app.services.consensus import (
    ConsensusHypothesis,
    ConsensusWord,
    build_slots,
    seed_disputes,
    seed_outvoted_fraction,
)


def hyp(system_id: str, *words: tuple[str, float, float]) -> ConsensusHypothesis:
    """A hypothesis from ``(word, start, end)`` triples, positions assigned in order."""
    return ConsensusHypothesis(
        system_id=system_id,
        words=[
            ConsensusWord(position=i, word=w, start=s, end=e) for i, (w, s, e) in enumerate(words)
        ],
    )


def test_no_hypotheses_produce_no_slots() -> None:
    assert build_slots([]) == []


def test_identical_systems_agree_on_every_slot() -> None:
    a = hyp("a", ("hello", 0.0, 0.5), ("world", 0.6, 1.0))
    b = hyp("b", ("hello", 0.0, 0.5), ("world", 0.6, 1.0))
    slots = build_slots([a, b])
    assert len(slots) == 2
    assert all(len(s.by_system) == 2 for s in slots)
    assert all(s.is_unanimous for s in slots)


def test_words_at_the_same_time_with_different_text_are_contested() -> None:
    a = hyp("a", ("Hisense", 0.0, 0.6))
    b = hyp("b", ("Hi", 0.0, 0.6))
    (slot,) = build_slots([a, b])
    assert len(slot.by_system) == 2
    assert not slot.is_unanimous


def test_words_that_do_not_overlap_land_in_separate_slots() -> None:
    a = hyp("a", ("one", 0.0, 0.4))
    b = hyp("b", ("two", 1.0, 1.4))
    slots = build_slots([a, b])
    assert len(slots) == 2
    assert all(len(s.by_system) == 1 for s in slots)


def test_overlap_below_the_threshold_does_not_merge() -> None:
    # 0.1 s of overlap across a 1.0 s union is an IoU of 0.1.
    a = hyp("a", ("x", 0.0, 0.6))
    b = hyp("b", ("x", 0.5, 1.0))
    assert len(build_slots([a, b], iou_threshold=0.5)) == 2
    assert len(build_slots([a, b], iou_threshold=0.05)) == 1


def test_one_system_may_contribute_only_one_word_per_slot() -> None:
    """A system that split a word in two must not occupy the same slot twice."""
    a = hyp("a", ("football", 0.0, 0.8))
    b = hyp("b", ("foot", 0.0, 0.4), ("ball", 0.4, 0.8))
    slots = build_slots([a, b], iou_threshold=0.3)
    for slot in slots:
        assert len(slot.by_system) == len({w.system_id for w in slot.by_system.values()})


def test_words_without_timings_are_ignored() -> None:
    a = ConsensusHypothesis(
        system_id="a",
        words=[ConsensusWord(position=0, word="x", start=None, end=None)],
    )
    b = hyp("b", ("x", 0.0, 0.5))
    (slot,) = build_slots([a, b])
    assert set(slot.by_system) == {"b"}


def test_a_seed_word_every_other_system_contradicts_is_disputed() -> None:
    seed = hyp("seed", ("चैँ", 0.0, 0.5))
    other = hyp("other", ("चाहिँ", 0.0, 0.5))
    third = hyp("third", ("चाहिँ", 0.0, 0.5))
    disputes = seed_disputes(build_slots([seed, other, third]), seed_system_id="seed")
    assert len(disputes) == 1
    assert disputes[0].seed_position == 0
    assert disputes[0].seed_word == "चैँ"
    assert {(a.system_id, a.word) for a in disputes[0].alternatives} == {
        ("other", "चाहिँ"),
        ("third", "चाहिँ"),
    }


def test_a_seed_word_one_system_agrees_with_is_not_disputed() -> None:
    """Outvoted means every other system disagrees, not merely one of them."""
    seed = hyp("seed", ("yes", 0.0, 0.5))
    agrees = hyp("agrees", ("yes", 0.0, 0.5))
    differs = hyp("differs", ("no", 0.0, 0.5))
    assert seed_disputes(build_slots([seed, agrees, differs]), seed_system_id="seed") == []


def test_disagreement_is_judged_after_normalization() -> None:
    """Punctuation and case are not evidence that anything was misheard."""
    seed = hyp("seed", ("Hello,", 0.0, 0.5))
    other = hyp("other", ("hello", 0.0, 0.5))
    assert seed_disputes(build_slots([seed, other]), seed_system_id="seed") == []


def test_a_slot_the_seed_is_absent_from_is_not_a_dispute() -> None:
    """It is a hole in the seed, not a word of the seed to swap; the UI has nothing to anchor."""
    other = hyp("other", ("extra", 0.0, 0.5))
    third = hyp("third", ("extra", 0.0, 0.5))
    seed = hyp("seed", ("later", 2.0, 2.5))
    disputes = seed_disputes(build_slots([seed, other, third]), seed_system_id="seed")
    assert disputes == []


def test_a_seed_alone_in_its_slot_is_not_disputed() -> None:
    """Nothing contradicted it, so there is no alternative to offer."""
    seed = hyp("seed", ("solo", 0.0, 0.5))
    other = hyp("other", ("elsewhere", 3.0, 3.5))
    assert seed_disputes(build_slots([seed, other]), seed_system_id="seed") == []


def test_alternatives_are_deduplicated_but_keep_every_system() -> None:
    seed = hyp("seed", ("a", 0.0, 0.5))
    b = hyp("b", ("bb", 0.0, 0.5))
    c = hyp("c", ("bb", 0.0, 0.5))
    (dispute,) = seed_disputes(build_slots([seed, b, c]), seed_system_id="seed")
    assert [a.system_id for a in dispute.alternatives] == ["b", "c"]
    assert dispute.start == 0.0 and dispute.end == 0.5


def test_slots_are_returned_in_time_order() -> None:
    a = hyp("a", ("three", 2.0, 2.5), ("one", 0.0, 0.5), ("two", 1.0, 1.5))
    slots = build_slots([a])
    assert [s.start for s in slots] == sorted(s.start for s in slots)


def test_outvoted_fraction_is_zero_when_everyone_agrees() -> None:
    a = hyp("seed", ("x", 0.0, 1.0))
    b = hyp("b", ("x", 0.0, 1.0))
    assert seed_outvoted_fraction(build_slots([a, b]), seed_system_id="seed") == 0.0


def test_outvoted_fraction_is_measured_in_time_not_words() -> None:
    """One long disputed word outweighs several short agreed ones."""
    seed = hyp("seed", ("a", 0.0, 0.1), ("b", 0.2, 0.3), ("long", 1.0, 2.0))
    other = hyp("other", ("a", 0.0, 0.1), ("b", 0.2, 0.3), ("different", 1.0, 2.0))
    fraction = seed_outvoted_fraction(build_slots([seed, other]), seed_system_id="seed")
    # 1.0 s disputed out of 1.2 s of slot time.
    assert fraction == pytest.approx(1.0 / 1.2, rel=1e-3)


def test_outvoted_fraction_of_a_segment_with_no_slots_is_zero() -> None:
    assert seed_outvoted_fraction([], seed_system_id="seed") == 0.0


def test_outvoted_fraction_ignores_slots_the_seed_never_entered() -> None:
    """A hole in the seed is not evidence the seed is wrong; it is a different measurement."""
    seed = hyp("seed", ("kept", 0.0, 1.0))
    other = hyp("other", ("kept", 0.0, 1.0), ("extra", 2.0, 3.0))
    assert seed_outvoted_fraction(build_slots([seed, other]), seed_system_id="seed") == 0.0
