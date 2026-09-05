"""Tests for rule-based segment flags."""

from __future__ import annotations

import pytest

from app.config import load_settings
from app.services.flags import ALL_FLAGS, FlagHypothesis, compute_flags, rule_flag_score

SETTINGS = load_settings()


def hyp(text: str = "hello world", no_speech_prob: float | None = 0.01) -> FlagHypothesis:
    return FlagHypothesis(text=text, no_speech_prob=no_speech_prob)


def flags(duration: float = 5.0, hypotheses: list[FlagHypothesis] | None = None) -> list[str]:
    return compute_flags(
        duration_seconds=duration,
        hypotheses=hypotheses if hypotheses is not None else [hyp()],
        settings=SETTINGS,
    )


def test_a_clean_segment_has_no_flags() -> None:
    assert flags(duration=5.0, hypotheses=[hyp("यो एउटा राम्रो वाक्य हो साथी")]) == []


def test_empty_transcript_is_flagged() -> None:
    assert "empty_transcript" in flags(hypotheses=[hyp(""), hyp("something")])


def test_whitespace_only_transcript_is_flagged() -> None:
    assert "empty_transcript" in flags(hypotheses=[hyp("   \n ")])


def test_no_hypotheses_at_all_is_flagged() -> None:
    assert "empty_transcript" in flags(hypotheses=[])


def test_repeated_ngram_is_flagged() -> None:
    text = "म आज यो कुरा " * 5
    assert "repeated_ngram" in flags(duration=20.0, hypotheses=[hyp(text)])


def test_ordinary_repetition_is_not_flagged() -> None:
    text = "यो राम्रो छ र त्यो पनि राम्रो छ तर अर्को कुरा फरक छ"
    assert "repeated_ngram" not in flags(duration=10.0, hypotheses=[hyp(text)])


def test_high_no_speech_probability_is_flagged() -> None:
    assert "high_no_speech_prob" in flags(hypotheses=[hyp("hi", no_speech_prob=0.95)])


def test_missing_no_speech_probability_is_not_flagged() -> None:
    assert "high_no_speech_prob" not in flags(hypotheses=[hyp("hi", no_speech_prob=None)])


def test_short_segment_is_flagged() -> None:
    assert "too_short" in flags(duration=0.4)


def test_long_segment_is_flagged() -> None:
    assert "too_long" in flags(duration=45.0)


def test_implausible_speaking_rate_is_flagged() -> None:
    fast = " ".join(["word"] * 60)
    assert "implausible_speaking_rate" in flags(duration=5.0, hypotheses=[hyp(fast)])


def test_slow_speaking_rate_is_flagged() -> None:
    assert "implausible_speaking_rate" in flags(duration=25.0, hypotheses=[hyp("एउटा")])


def test_plausible_speaking_rate_is_not_flagged() -> None:
    text = " ".join(["शब्द"] * 12)
    assert "implausible_speaking_rate" not in flags(duration=5.0, hypotheses=[hyp(text)])


def test_script_conflict_between_systems_is_flagged() -> None:
    devanagari = hyp("यो पूरै देवनागरीमा लेखिएको वाक्य हो")
    latin = hyp("yo purai latin ma lekhieko wakya ho")
    assert "script_conflict" in flags(duration=5.0, hypotheses=[devanagari, latin])


def test_agreeing_scripts_are_not_flagged() -> None:
    a = hyp("So today म Python मा loops बारे कुरा गर्छु")
    b = hyp("So today म Python मा loops बारे कुरा गर्दछु")
    assert "script_conflict" not in flags(duration=6.0, hypotheses=[a, b])


def test_a_single_hypothesis_cannot_conflict_with_itself() -> None:
    assert "script_conflict" not in flags(hypotheses=[hyp("yo latin ma cha")])


def test_flags_are_sorted_and_unique() -> None:
    result = flags(duration=0.2, hypotheses=[hyp(""), hyp("", no_speech_prob=0.99)])
    assert result == sorted(set(result))


def test_every_flag_is_declared() -> None:
    result = flags(duration=0.2, hypotheses=[hyp("")])
    assert set(result) <= set(ALL_FLAGS)


def test_rule_flag_score_is_the_fraction_of_flags_raised() -> None:
    assert rule_flag_score([]) == 0.0
    assert rule_flag_score(list(ALL_FLAGS)) == 1.0
    assert rule_flag_score(["too_short"]) == pytest.approx(1 / len(ALL_FLAGS))


def test_rule_flag_score_ignores_unknown_flags() -> None:
    """Imported flags from upstream must not inflate the score past 1.0."""
    assert rule_flag_score(["something_upstream_invented"]) == 0.0
    assert rule_flag_score([*ALL_FLAGS, "extra"]) == 1.0


# --- speech no system transcribed ----------------------------------------------------------


SETTINGS = load_settings()


def hyp_with_words(*spans: tuple[float, float]) -> FlagHypothesis:
    return FlagHypothesis(text="x " * len(spans), word_spans=list(spans))


def test_speech_every_system_covered_raises_nothing() -> None:
    flags = compute_flags(
        duration_seconds=5.0,
        hypotheses=[hyp_with_words((0.0, 2.0), (2.0, 4.0))],
        vad_spans=[(0.0, 4.0)],
        settings=SETTINGS,
    )
    assert "missed_speech" not in flags


def test_speech_no_system_transcribed_is_flagged() -> None:
    """The truncation failure D50 found by hand: a system stops before the audio does."""
    flags = compute_flags(
        duration_seconds=20.0,
        hypotheses=[hyp_with_words((0.0, 5.0))],
        vad_spans=[(0.0, 18.0)],
        settings=SETTINGS,
    )
    assert "missed_speech" in flags


def test_a_word_from_any_system_covers_the_speech() -> None:
    """One system hearing it means the audio was transcribable; that is not a missed region."""
    flags = compute_flags(
        duration_seconds=20.0,
        hypotheses=[hyp_with_words((0.0, 5.0)), hyp_with_words((0.0, 17.9))],
        vad_spans=[(0.0, 18.0)],
        settings=SETTINGS,
    )
    assert "missed_speech" not in flags


def test_without_vad_spans_the_flag_cannot_fire() -> None:
    """Segments imported before the spans existed must not all look defective."""
    flags = compute_flags(
        duration_seconds=20.0,
        hypotheses=[hyp_with_words((0.0, 1.0))],
        vad_spans=None,
        settings=SETTINGS,
    )
    assert "missed_speech" not in flags


def test_without_word_spans_the_flag_cannot_fire() -> None:
    """No timings is not evidence of missed speech; it is absence of evidence."""
    flags = compute_flags(
        duration_seconds=20.0,
        hypotheses=[FlagHypothesis(text="something was said")],
        vad_spans=[(0.0, 18.0)],
        settings=SETTINGS,
    )
    assert "missed_speech" not in flags


def test_missed_speech_is_part_of_the_flag_vocabulary() -> None:
    assert "missed_speech" in ALL_FLAGS
