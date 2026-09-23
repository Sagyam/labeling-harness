"""Tests for where a voice is heard alone in a clip (D99)."""

from __future__ import annotations

import pytest

from app.services.speaker_lanes import Turn
from app.services.voice_clips import alone_stretches, best_stretch


def test_a_voice_with_nobody_else_is_alone_throughout() -> None:
    assert alone_stretches([Turn(1, 0.0, 4.0)], [], 1) == [(0.0, 4.0)]


def test_another_voice_cuts_a_hole() -> None:
    turns = [Turn(1, 0.0, 6.0), Turn(2, 2.0, 2.5)]
    assert alone_stretches(turns, [], 1) == [(0.0, 2.0), (2.5, 6.0)]


def test_detected_crosstalk_cuts_a_hole_too() -> None:
    assert alone_stretches([Turn(1, 0.0, 6.0)], [[1.0, 1.5]], 1) == [(0.0, 1.0), (1.5, 6.0)]


def test_touching_turns_of_one_voice_are_one_stretch() -> None:
    assert alone_stretches([Turn(1, 0.0, 2.0), Turn(1, 2.0, 3.0)], [], 1) == [(0.0, 3.0)]


def test_a_voice_that_is_never_alone_has_no_stretch() -> None:
    turns = [Turn(1, 1.0, 2.0), Turn(2, 0.0, 3.0)]
    assert alone_stretches(turns, [], 1) == []
    assert best_stretch(turns, [], 1) is None


def test_the_longest_stretch_is_offered() -> None:
    turns = [Turn(1, 0.0, 8.0), Turn(2, 1.0, 1.2), Turn(2, 5.0, 5.3)]
    assert best_stretch(turns, [], 1) == pytest.approx((1.2, 5.0))


def test_a_stretch_too_short_to_recognise_is_not_offered() -> None:
    turns = [Turn(1, 0.0, 1.2)]
    assert best_stretch(turns, [], 1) is None
