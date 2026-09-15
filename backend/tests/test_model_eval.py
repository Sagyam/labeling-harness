"""Scoring a fine-tuned model's transcripts against the labels (D83)."""

from __future__ import annotations

import random

import pytest

from app.services.fold import _levenshtein
from app.services.model_eval import (
    ScoredClip,
    is_loop,
    levenshtein,
    overlap_bucket,
    overlap_share,
    score_clip,
    summarize,
)

# --- edit distance, loops ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("", "", 0), ("abc", "", 3), ("", "abc", 3), ("kitten", "sitting", 3), ("नमस्ते", "नमस्त", 1)],
)
def test_levenshtein_known_pairs(a: str, b: str, expected: int) -> None:
    assert levenshtein(a, b) == expected


def test_levenshtein_matches_the_plain_dynamic_programme() -> None:
    """The bit-parallel version must agree with the textbook one, at every length."""
    rng = random.Random(7)
    for _ in range(500):
        a = "".join(rng.choice("abcक ख") for _ in range(rng.randint(0, 90)))
        b = "".join(rng.choice("abcक ख") for _ in range(rng.randint(0, 90)))
        assert levenshtein(a, b) == _levenshtein(a, b), (a, b)


def test_a_three_word_run_repeated_five_times_is_a_loop() -> None:
    assert is_loop("त्यो " + "एक दुई तीन " * 5)
    assert not is_loop("एक दुई तीन " * 4)
    assert not is_loop("")


# --- one clip ---------------------------------------------------------------------------------


def test_an_identical_transcript_has_no_errors() -> None:
    score = score_clip("म phone किन्छु", "म phone किन्छु")
    assert (score.errors, score.raw_errors, score.char_errors) == (0, 0, 0)
    assert score.ref_words == 3


def test_a_word_in_the_other_script_is_folded_but_counted_raw() -> None:
    """``टिम``/``team`` is one word written two ways: no folded error, one raw error."""
    score = score_clip("हाम्रो टिम राम्रो छ", "हाम्रो team राम्रो छ")
    assert score.errors == 0
    assert score.raw_errors == 1
    assert score.raw_ref_words == 4


def test_substitutions_deletions_and_insertions_are_counted_apart() -> None:
    score = score_clip("एक दुई तीन चार", "एक पाँच तीन चार छ")
    assert (score.substitutions, score.deletions, score.insertions) == (1, 0, 1)
    score = score_clip("एक दुई तीन चार", "एक चार")
    assert score.deletions == 2
    assert score.errors == 2


def test_characters_are_compared_case_blind_for_latin_only() -> None:
    score = score_clip("Phone राम्रो", "phone राम्रो")
    assert score.char_errors == 0
    assert score.ref_chars == len("phone राम्रो")


def test_a_looping_hypothesis_is_marked() -> None:
    assert score_clip("अँ", "अँ " + "यो हो नि " * 6).is_loop


# --- overlap ----------------------------------------------------------------------------------


def test_overlap_share_is_the_fraction_of_the_clip_in_crosstalk() -> None:
    assert overlap_share([[0.0, 1.0], [3.0, 4.0]], 10.0) == pytest.approx(0.2)
    assert overlap_share([], 10.0) == 0.0
    assert overlap_share(None, 10.0) is None  # never measured is not clean


@pytest.mark.parametrize(
    ("share", "bucket"),
    [(None, "unmeasured"), (0.0, "none"), (0.03, "0-5%"), (0.05, "5-15%"), (0.2, ">15%")],
)
def test_overlap_buckets_follow_the_crosstalk_findings(share: float | None, bucket: str) -> None:
    assert overlap_bucket(share) == bucket


# --- summary ----------------------------------------------------------------------------------


def _clip(episode: str, ref: str, hyp: str, genre: str = "podcast", share: float = 0.0):
    return ScoredClip(
        episode=episode, genre=genre, overlap=overlap_bucket(share), score=score_clip(ref, hyp)
    )


def test_wer_pools_errors_over_words_not_clip_rates() -> None:
    clips = [
        _clip("a", "एक दुई तीन चार", "एक दुई तीन चार"),  # 0 / 4
        _clip("b", "एक", "दुई"),  # 1 / 1
    ]
    summary = summarize(clips)
    assert summary["wer"] == pytest.approx(100 * 1 / 5)
    assert summary["clips"] == 2
    assert summary["ref_words"] == 5


def test_the_interval_brackets_the_estimate_and_is_reproducible() -> None:
    clips = [
        _clip(f"ep{e}", "एक दुई तीन चार", "एक दुई तीन पाँच" if e % 2 else "एक दुई तीन चार")
        for e in range(10)
        for _ in range(3)
    ]
    first, second = summarize(clips), summarize(clips)
    low, high = first["wer_ci"]
    assert low <= first["wer"] <= high
    assert low < high
    assert first["wer_ci"] == second["wer_ci"]


def test_one_episode_has_no_interval() -> None:
    summary = summarize([_clip("a", "एक दुई", "एक"), _clip("a", "एक", "एक")])
    assert summary["wer_ci"] is None


def test_breakdowns_by_genre_and_overlap_carry_their_share_of_errors() -> None:
    clips = [
        _clip("a", "एक दुई", "एक", genre="podcast", share=0.2),
        _clip("b", "एक दुई", "एक दुई", genre="tech_review", share=0.0),
        _clip("c", "एक दुई तीन", "चार", genre="podcast", share=0.0),
    ]
    summary = summarize(clips)
    podcast = summary["by_genre"]["podcast"]
    assert podcast["clips"] == 2
    assert podcast["wer"] == pytest.approx(100 * 4 / 5)
    assert podcast["share_of_errors"] == pytest.approx(1.0)
    assert summary["by_genre"]["tech_review"]["wer"] == 0.0
    assert summary["by_overlap"][">15%"]["clips"] == 1
    assert summary["by_overlap"]["none"]["share_of_errors"] == pytest.approx(3 / 4)


def test_an_empty_run_summarizes_to_zeroes() -> None:
    summary = summarize([])
    assert summary["clips"] == 0
    assert summary["wer"] == 0.0
