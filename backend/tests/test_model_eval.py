"""Scoring a fine-tuned model's transcripts against the labels (D83)."""

from __future__ import annotations

import random

import pytest

from app.services.clip_classes import overlap_bucket
from app.services.fold import _levenshtein, word_errors
from app.services.model_eval import (
    ScoredClip,
    is_loop,
    levenshtein,
    score_clip,
    summarize,
    word_class_counts,
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


# --- summary ----------------------------------------------------------------------------------


def _clip(
    episode: str,
    ref: str,
    hyp: str,
    genre: str = "podcast",
    share: float = 0.0,
    **classes: str,
):
    return ScoredClip(
        episode=episode,
        genre=genre,
        classes={"overlap": overlap_bucket(share)} | classes,
        score=score_clip(ref, hyp),
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
    overlap = summary["by_class"]["overlap"]
    assert overlap[">15%"]["clips"] == 1
    assert overlap["none"]["share_of_errors"] == pytest.approx(3 / 4)


def test_an_empty_run_summarizes_to_zeroes() -> None:
    summary = summarize([])
    assert summary["clips"] == 0
    assert summary["wer"] == 0.0


# --- classes: within-episode rate ratios (D87) ---------------------------------------------------


def test_the_rate_ratio_compares_buckets_inside_each_episode() -> None:
    """Episode a is clean and easy, episode b is overlapped and hard. Pooled, overlap looks
    terrible; inside each episode it doubles the error rate, and the ratio says 2."""
    clips = []
    for episode, base_errors in (("a", 1), ("b", 4)):
        for _ in range(3):
            ref = " ".join(["एक"] * 10)
            clean = " ".join(["दुई"] * base_errors + ["एक"] * (10 - base_errors))
            dirty = " ".join(["दुई"] * (2 * base_errors) + ["एक"] * (10 - 2 * base_errors))
            clips.append(_clip(episode, ref, clean, share=0.0))
            clips.append(_clip(episode, ref, dirty, share=0.1))
    by_overlap = summarize(clips)["by_class"]["overlap"]
    assert by_overlap["5-15%"]["rate_ratio"] == pytest.approx(2.0)
    assert "rate_ratio" not in by_overlap["none"]  # the baseline compares against nothing


def test_a_bucket_never_seen_beside_the_baseline_in_one_episode_has_no_ratio() -> None:
    clips = [_clip("a", "एक दुई", "एक", share=0.0), _clip("b", "एक दुई", "एक", share=0.3)]
    by_overlap = summarize(clips)["by_class"]["overlap"]
    assert by_overlap[">15%"]["rate_ratio"] is None
    assert by_overlap[">15%"]["rate_ratio_episodes"] == 0


def test_the_ratio_interval_is_reproducible_and_brackets_the_estimate() -> None:
    clips = [
        _clip(f"ep{e}", "एक दुई तीन चार", hyp, share=share)
        for e in range(8)
        for share, hyp in ((0.0, "एक दुई तीन चार"), (0.0, "एक दुई तीन पाँच"), (0.1, "एक दुई पाँच पाँच"))
    ]
    first = summarize(clips)["by_class"]["overlap"]["5-15%"]
    second = summarize(clips)["by_class"]["overlap"]["5-15%"]
    low, high = first["rate_ratio_ci"]
    assert low <= first["rate_ratio"] <= high
    assert first["rate_ratio_ci"] == second["rate_ratio_ci"]
    assert first["rate_ratio_episodes"] == 8


def test_unmeasured_buckets_and_axes_without_a_baseline_get_no_ratio() -> None:
    clips = [
        _clip("a", "एक दुई", "एक", gender="male"),
        _clip("a", "एक दुई", "एक", gender="female"),
    ]
    clips.append(ScoredClip("a", "podcast", {"overlap": "unmeasured"}, score_clip("एक", "एक")))
    by_class = summarize(clips)["by_class"]
    assert "rate_ratio" not in by_class["overlap"]["unmeasured"]
    assert all("rate_ratio" not in b for b in by_class["gender"].values())


def test_buckets_come_in_the_axis_order_and_carry_cer() -> None:
    clips = [
        _clip("a", "एक", "एक", share=0.3),
        _clip("a", "एक", "एक", share=0.0),
        _clip("a", "एक दुई", "एक", share=0.01),
    ]
    by_overlap = summarize(clips)["by_class"]["overlap"]
    assert list(by_overlap) == ["none", "0-5%", ">15%"]
    assert by_overlap["0-5%"]["cer"] > 0


# --- word classes (D87) --------------------------------------------------------------------------


def test_every_reference_word_is_counted_in_its_classes() -> None:
    counts = word_class_counts(word_errors("मेरो phone मा 5 वटा app छ", "मेरो phone मा 5 वटा app छ"))
    assert counts["devanagari"].words == 4  # मेरो मा वटा छ
    assert counts["latin"].words == 2
    assert counts["number"].words == 1
    assert counts["edge"].words == 2  # the first and the last word
    # phone and app, and the Devanagari word on each side of them; the digit has no script.
    assert counts["switch"].words == 6
    assert all(c.errors == 0 and c.char_errors == 0 for c in counts.values())


def test_errors_land_on_the_class_of_the_reference_word() -> None:
    counts = word_class_counts(word_errors("मेरो phone राम्रो छ", "मेरो फुल राम्रो"))
    # phone -> फुल is a substitution of a Latin word; छ is deleted at the clip's edge.
    assert counts["latin"].errors == 1
    assert counts["devanagari"].errors == 1
    assert counts["edge"].errors == 1
    assert counts["latin"].char_errors > 0
    assert counts["devanagari"].char_errors == pytest.approx(1)  # छ, one character


def test_insertions_belong_to_no_class() -> None:
    counts = word_class_counts(word_errors("एक दुई", "एक दुई तीन"))
    assert sum(c.errors for c in counts.values()) == 0


def test_a_run_reports_wer_and_cer_per_word_class() -> None:
    clip = ScoredClip(
        "a",
        "podcast",
        {},
        score_clip("मेरो phone राम्रो छ", "मेरो फुल राम्रो"),
        words=word_class_counts(word_errors("मेरो phone राम्रो छ", "मेरो फुल राम्रो")),
    )
    by_word = summarize([clip])["by_word_class"]
    assert by_word["latin"]["words"] == 1
    assert by_word["latin"]["wer"] == pytest.approx(100.0)
    assert by_word["devanagari"]["wer"] == pytest.approx(100 * 1 / 3)
    assert by_word["latin"]["share_of_words"] == pytest.approx(1 / 4)
    assert "cer" in by_word["latin"]


def test_without_word_counts_a_run_has_no_word_breakdown() -> None:
    assert "by_word_class" not in summarize([_clip("a", "एक", "एक")])
