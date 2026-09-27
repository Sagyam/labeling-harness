"""The episode metadata editor's rules: closed genres and topics, and speaker rows (D102)."""

from __future__ import annotations

import pytest

from app.llm.topic import TOPIC_LABELS
from app.services.episode_meta import (
    AGE_BRACKETS,
    GENRE_LABELS,
    GENRES,
    MetadataEditError,
    edited_metadata,
    speaker_rows,
)
from app.services.speaker_meta import ALLOWED_VALUES, MAX_SPEAKERS


def test_every_genre_has_a_test_a_person_can_apply() -> None:
    assert len(GENRE_LABELS) == len(set(GENRE_LABELS))
    assert all(g.description.strip() for g in GENRES)
    assert {"advert", "explainer", "commentary", "podcast", "interview"} <= set(GENRE_LABELS)


def test_the_ordered_age_brackets_are_exactly_the_allowed_ones() -> None:
    assert set(AGE_BRACKETS) == ALLOWED_VALUES["age_bracket"]
    assert len(AGE_BRACKETS) == len(ALLOWED_VALUES["age_bracket"])


def test_the_retired_genres_are_not_in_the_list() -> None:
    for retired in ("reels", "tech_review", "cooking", "finance", "education", "sports"):
        assert retired not in GENRE_LABELS


def test_rows_come_back_in_speaker_order_with_blank_rows_for_the_uncounted() -> None:
    metadata = {
        "speaker_count": 3,
        "speakers": {
            "spk2": {"role": "guest", "gender": "female"},
            "spk0": {"role": "host", "gender": "male", "age_bracket": "20_39"},
        },
    }

    assert speaker_rows(metadata) == [
        {"role": "host", "gender": "male", "age_bracket": "20_39"},
        {"role": None, "gender": None, "age_bracket": None},
        {"role": "guest", "gender": "female", "age_bracket": None},
    ]


def test_an_episode_without_speakers_has_no_rows() -> None:
    assert speaker_rows({}) == []
    assert speaker_rows(None) == []


def test_speakers_keyed_by_something_else_are_kept_in_sorted_order() -> None:
    metadata = {"speakers": {"B": {"gender": "female"}, "A": {"gender": "male"}}}

    assert [r["gender"] for r in speaker_rows(metadata)] == ["male", "female"]


def test_an_edit_writes_the_form_shape_and_keeps_everything_else() -> None:
    current = {"genre": "podcast", "source": "upload", "topic": "technology"}

    out = edited_metadata(
        current,
        genre="interview",
        topic="technology",
        speakers=[
            {"role": "host", "gender": "male", "age_bracket": "20_39"},
            {"role": "guest", "gender": None, "age_bracket": None},
        ],
    )

    assert out == {
        "genre": "interview",
        "source": "upload",
        "topic": "technology",
        "speaker_count": 2,
        "speakers": {"spk0": {"role": "host", "gender": "male", "age_bracket": "20_39"}},
    }
    assert current == {"genre": "podcast", "source": "upload", "topic": "technology"}


def test_a_changed_topic_is_marked_manual_and_loses_the_model() -> None:
    current = {"topic": "technology", "topic_source": "llm", "topic_model": "gemini"}

    out = edited_metadata(current, genre=None, topic="sports", speakers=[])

    assert out == {"topic": "sports", "topic_source": "manual"}


def test_an_unchanged_topic_keeps_its_provenance() -> None:
    current = {"topic": "technology", "topic_source": "llm", "topic_model": "gemini"}

    assert edited_metadata(current, genre=None, topic="technology", speakers=[]) == current


def test_clearing_genre_topic_and_speakers_removes_their_keys() -> None:
    current = {
        "genre": "vlog",
        "topic": "travel_food",
        "topic_source": "llm",
        "topic_model": "gemini",
        "speaker_count": 1,
        "speakers": {"spk0": {"gender": "male"}},
    }

    assert edited_metadata(current, genre="", topic=None, speakers=[]) == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [("genre", "reels"), ("genre", "Podcast"), ("topic", "cooking")],
)
def test_a_value_off_the_closed_list_is_refused(field: str, value: str) -> None:
    kwargs = {"genre": "podcast", "topic": TOPIC_LABELS[0], "speakers": []} | {field: value}

    with pytest.raises(MetadataEditError, match=field):
        edited_metadata({}, **kwargs)


@pytest.mark.parametrize(
    "row",
    [
        {"gender": "other"},
        {"age_bracket": "30s"},
        {"name": "Sushant"},
    ],
)
def test_a_speaker_row_off_the_allowlist_is_refused(row: dict[str, str]) -> None:
    with pytest.raises(MetadataEditError):
        edited_metadata({}, genre=None, topic=None, speakers=[row])


def test_more_rows_than_the_diarizer_takes_are_refused() -> None:
    rows = [{"gender": "male"}] * (MAX_SPEAKERS + 1)

    with pytest.raises(MetadataEditError, match="speakers"):
        edited_metadata({}, genre=None, topic=None, speakers=rows)


def test_rows_round_trip_through_the_editor() -> None:
    metadata = {
        "speaker_count": 2,
        "speakers": {"spk1": {"role": "guest", "gender": "female", "age_bracket": "40_59"}},
    }

    out = edited_metadata(metadata, genre=None, topic=None, speakers=speaker_rows(metadata))

    assert out == metadata
