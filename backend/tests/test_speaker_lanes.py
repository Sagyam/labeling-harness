"""Tests for the multitrack editor's lanes: proposal, candidates and what may be saved (D98)."""

from __future__ import annotations

import math

import pytest

from app.services.consensus import ConsensusHypothesis, ConsensusWord
from app.services.speaker_lanes import (
    LaneWord,
    TimedWord,
    Turn,
    check_attribution,
    dominant_speaker,
    flatten,
    is_unchanged,
    place_text_on_clock,
    propose_lanes,
    recogniser_candidates,
)

SEED = [
    TimedWord("हो", 0.10, 0.30),
    TimedWord("team", 0.40, 0.80),
    TimedWord("ले", 0.85, 1.00),
    TimedWord("गर्यो", 1.10, 1.50),
]


def hyp(system: str, words: list[tuple[str, float, float]]) -> ConsensusHypothesis:
    return ConsensusHypothesis(
        system_id=system,
        words=[ConsensusWord(i, w, a, b) for i, (w, a, b) in enumerate(words)],
    )


# --- the verified text on the seed's clock ----------------------------------------------------


def test_an_unchanged_text_keeps_every_seed_span() -> None:
    assert place_text_on_clock("हो team ले गर्यो", SEED) == SEED


def test_a_substituted_word_takes_the_span_it_replaced() -> None:
    placed = place_text_on_clock("हो टिम ले गर्यो", SEED)
    assert placed[1] == TimedWord("टिम", 0.40, 0.80)


def test_a_word_the_seed_lacks_gets_no_span() -> None:
    placed = place_text_on_clock("हो team ले अनि गर्यो", SEED)
    assert [w.word for w in placed] == ["हो", "team", "ले", "अनि", "गर्यो"]
    assert placed[3] == TimedWord("अनि", None, None)
    assert placed[4] == TimedWord("गर्यो", 1.10, 1.50)


def test_a_word_dropped_from_the_seed_is_dropped() -> None:
    placed = place_text_on_clock("हो team गर्यो", SEED)
    assert [w.word for w in placed] == ["हो", "team", "गर्यो"]
    assert placed[2].start == 1.10


def test_no_seed_words_leaves_every_word_untimed() -> None:
    untimed = [TimedWord("a", None, None), TimedWord("b", None, None)]
    assert place_text_on_clock("a b", []) == untimed


def test_empty_text_places_nothing() -> None:
    assert place_text_on_clock("", SEED) == []
    assert place_text_on_clock(None, SEED) == []


# --- which lane a word starts on --------------------------------------------------------------


def test_a_word_goes_to_the_speaker_covering_most_of_it() -> None:
    turns = [Turn(1, 0.0, 0.5), Turn(2, 0.45, 2.0)]
    assert dominant_speaker(0.40, 0.80, turns) == 2
    assert dominant_speaker(0.10, 0.30, turns) == 1


def test_a_tie_goes_to_the_speaker_with_more_talk_time() -> None:
    turns = [Turn(2, 0.0, 1.0), Turn(1, 0.0, 1.0)]
    assert dominant_speaker(0.2, 0.4, turns) == 1


def test_a_word_outside_every_turn_is_on_no_lane() -> None:
    assert dominant_speaker(3.0, 3.2, [Turn(1, 0.0, 1.0)]) is None


def test_the_proposal_records_where_each_word_started() -> None:
    turns = [Turn(1, 0.0, 0.35), Turn(2, 0.35, 2.0)]
    lanes = propose_lanes(place_text_on_clock("हो team अनि ले गर्यो", SEED), turns)
    assert [(w.word, w.speaker) for w in lanes] == [
        ("हो", 1),
        ("team", 2),
        ("अनि", None),
        ("ले", 2),
        ("गर्यो", 2),
    ]
    assert all(w.speaker == w.proposed_speaker and w.source == "label" for w in lanes)


# --- words the recognisers heard that the text lacks ------------------------------------------


def test_a_word_every_recogniser_heard_in_a_hole_is_offered() -> None:
    placed = place_text_on_clock("हो team ले गर्यो", SEED)
    recognisers = [
        hyp("a", [("हो", 0.1, 0.3), ("हजुर", 1.6, 1.9)]),
        hyp("b", [("हजुर", 1.62, 1.88)]),
        hyp("c", [("हजूर", 1.6, 1.9)]),
    ]
    [candidate] = recogniser_candidates(placed, recognisers)
    assert candidate.word == "हजुर"
    assert (candidate.start, candidate.end) == (1.6, 1.9)
    assert candidate.systems == ("a", "b")


def test_a_word_only_one_recogniser_heard_is_not_offered() -> None:
    placed = place_text_on_clock("हो team ले गर्यो", SEED)
    recognisers = [hyp("a", [("हजुर", 1.6, 1.9)]), hyp("b", [("अँ", 1.6, 1.9)])]
    assert recogniser_candidates(placed, recognisers) == []


def test_a_text_word_heard_on_a_different_clock_is_not_offered() -> None:
    """The recognisers' spans and the aligner's differ; the same word nearby is the same word."""
    placed = place_text_on_clock("हो team ले गर्यो", SEED)
    recognisers = [hyp("a", [("गर्यो", 1.35, 1.8)]), hyp("b", [("गर्‍यो", 1.4, 1.85)])]
    assert recogniser_candidates(placed, recognisers) == []


def test_a_word_the_text_already_has_is_not_offered() -> None:
    placed = place_text_on_clock("हो team ले गर्यो", SEED)
    recognisers = [hyp("a", [("team", 0.42, 0.78)]), hyp("b", [("टिम", 0.40, 0.80)])]
    assert recogniser_candidates(placed, recognisers) == []


# --- what may be saved ------------------------------------------------------------------------


def word(**kwargs) -> LaneWord:
    base = {
        "word": "हो",
        "start": 0.1,
        "end": 0.3,
        "speaker": 1,
        "proposed_speaker": 1,
        "source": "label",
    }
    return LaneWord(**{**base, **kwargs})


def test_a_clean_attribution_passes() -> None:
    assert not check_attribution([word(), word(speaker=2)], clip_seconds=2.0, speakers=[1, 2])


@pytest.mark.parametrize(
    ("bad", "fragment"),
    [
        ({"speaker": None}, "no lane"),
        ({"speaker": 3}, "not one of the episode's"),
        ({"start": None, "end": None}, "no span"),
        ({"start": -0.1}, "outside the clip"),
        ({"end": 2.5}, "outside the clip"),
        ({"end": 0.11}, "shorter than"),
        ({"start": math.nan}, "not a number"),
        ({"word": "  "}, "one token"),
        ({"word": "दुई शब्द"}, "one token"),
        ({"source": "invented"}, "unknown source"),
    ],
)
def test_a_bad_word_is_refused(bad: dict, fragment: str) -> None:
    problem = check_attribution([word(**bad)], clip_seconds=2.0, speakers=[1, 2])
    assert problem and fragment in problem.errors[0]


def test_no_words_at_all_is_refused() -> None:
    assert check_attribution([], clip_seconds=2.0, speakers=[1])


def test_the_lanes_flatten_in_time_order() -> None:
    words = [word(word="b", start=0.5, end=0.7, speaker=2), word(word="a", start=0.1, end=0.3)]
    assert flatten(words) == "a b"


def test_a_word_said_by_both_voices_appears_twice_in_the_flat_text() -> None:
    words = [word(), word(speaker=2, source="copy")]
    assert flatten(words) == "हो हो"


def test_saving_the_proposal_as_served_is_unchanged() -> None:
    proposal = [word(), word(word="b", start=0.4, end=0.6, speaker=2, proposed_speaker=2)]
    assert is_unchanged(proposal, list(reversed(proposal)))
    assert is_unchanged(proposal, [word(start=0.102), proposal[1]])


@pytest.mark.parametrize(
    "change",
    [{"speaker": 2}, {"start": 0.15}, {"word": "हजुर"}, {"source": "typed"}],
)
def test_any_move_retime_or_retype_is_an_edit(change: dict) -> None:
    assert not is_unchanged([word()], [word(**change)])


def test_an_added_word_is_an_edit() -> None:
    assert not is_unchanged([word()], [word(), word(word="x", start=1.0, end=1.2, source="typed")])
