"""Tests for automatic topic classification (D57).

Two things are being protected. The transcript sample must be drawn across the whole episode, not
off the front of it, because a Nepali podcast opens with sponsor reads and greetings that say
nothing about what the episode is about. And the answer must be a label from the taxonomy or
nothing at all -- a free-text topic is not a variable anything can be stratified on, so a model
that invents one is treated as having failed to answer.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import LlmRequestFailed, LlmRouteNotConfigured
from app.llm.topic import TOPIC_LABELS, classify_topic, sample_transcript

ROUTE = "classify_topic"


def routes(**kwargs: Any) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 1,
        "retry_backoff_seconds": 0.0,
        "routes": {
            ROUTE: LlmRoute(provider="openrouter", api="chat", model="google/gemini-3.8-flash")
        },
    }
    return LlmRoutes(**{**base, **kwargs})


def completion(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "model": "google/gemini-3.8-flash",
        "usage": {"prompt_tokens": 200, "completion_tokens": 3},
    }


def client_returning(body: dict[str, Any]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))


# --- the transcript sample, no database -----------------------------------------------------


def test_the_sample_is_drawn_across_the_episode_not_off_the_front() -> None:
    """A budget that fits three of ten segments takes them from the start, middle and end."""
    texts = [f"segment {i}" for i in range(10)]
    sample = sample_transcript(texts, max_chars=len("segment 0") * 3 + 2)
    assert "segment 0" in sample
    assert "segment 9" in sample
    assert "segment 1" not in sample


def test_the_sample_stays_within_its_budget() -> None:
    texts = ["एउटा लामो वाक्य " * 20] * 50
    assert len(sample_transcript(texts, max_chars=500)) <= 500


def test_a_short_episode_is_sampled_whole_and_in_order() -> None:
    texts = ["first", "second", "third"]
    assert sample_transcript(texts, max_chars=1000) == "first second third"


def test_blank_hypotheses_are_skipped() -> None:
    assert sample_transcript(["", "  ", "real text"], max_chars=1000) == "real text"


def test_an_empty_episode_samples_to_nothing() -> None:
    assert sample_transcript([], max_chars=1000) == ""


# --- the classification, against a mocked provider -------------------------------------------


@pytest.mark.db
def test_a_taxonomy_label_is_returned(db_session: Session) -> None:
    topic, meta = classify_topic(
        db_session,
        title="Kathmandu मा startup कसरी खोल्ने",
        transcript="हामीले यो startup को funding round को कुरा गर्यौं",
        route=ROUTE,
        config=routes(),
        client=client_returning(completion("business_finance")),
    )
    assert topic == "business_finance"
    assert meta["topic_source"] == "llm"
    assert meta["topic_model"] == "google/gemini-3.8-flash"


@pytest.mark.db
def test_a_label_wrapped_in_prose_or_a_fence_is_still_read(db_session: Session) -> None:
    for content in ("```\ntechnology\n```", 'The topic is "technology".', "  TECHNOLOGY  "):
        topic, _ = classify_topic(
            db_session,
            title="t",
            transcript="x",
            route=ROUTE,
            config=routes(),
            client=client_returning(completion(content)),
        )
        assert topic == "technology", content


@pytest.mark.db
def test_a_label_outside_the_taxonomy_is_refused(db_session: Session) -> None:
    """An invented label would be a one-episode category nothing can be stratified on."""
    topic, meta = classify_topic(
        db_session,
        title="t",
        transcript="x",
        route=ROUTE,
        config=routes(),
        client=client_returning(completion("nepali_startup_scene")),
    )
    assert topic is None
    assert meta["topic_error"] == "off_taxonomy"


@pytest.mark.db
def test_an_episode_with_no_transcript_is_not_sent_anywhere(db_session: Session) -> None:
    """No text means no evidence; guessing from a title alone is not worth a billed call."""
    topic, meta = classify_topic(
        db_session,
        title="t",
        transcript="   ",
        route=ROUTE,
        config=routes(),
        client=client_returning(completion("technology")),
    )
    assert topic is None
    assert meta["topic_error"] == "no_transcript"


@pytest.mark.db
def test_a_dry_run_costs_nothing_and_claims_nothing(db_session: Session) -> None:
    topic, meta = classify_topic(
        db_session,
        title="t",
        transcript="कुरा",
        route=ROUTE,
        config=routes(dry_run=True),
        client=client_returning(completion("technology")),
    )
    assert topic is None
    assert meta["topic_source"] == "dry_run"


@pytest.mark.db
def test_an_unconfigured_route_is_named_in_the_error(db_session: Session) -> None:
    with pytest.raises(LlmRouteNotConfigured, match="classify_topic"):
        classify_topic(
            db_session,
            title="t",
            transcript="कुरा",
            route=ROUTE,
            config=routes(routes={}),
            client=client_returning(completion("technology")),
        )


@pytest.mark.db
def test_a_provider_failure_propagates(db_session: Session) -> None:
    """The caller decides whether a missing topic is worth failing an ingest over; this does not."""
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500, json={})))
    with pytest.raises(LlmRequestFailed):
        classify_topic(
            db_session,
            title="t",
            transcript="कुरा",
            route=ROUTE,
            config=routes(),
            client=client,
        )


def test_the_taxonomy_has_an_escape_hatch() -> None:
    """Without `other`, an off-taxonomy episode forces the model to pick a wrong label."""
    assert "other" in TOPIC_LABELS
    assert len(set(TOPIC_LABELS)) == len(TOPIC_LABELS)
