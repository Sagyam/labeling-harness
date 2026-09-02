"""Tests for the ElevenLabs Scribe client.

Scribe is the one provider called directly rather than through OpenRouter, so the guarantees that
made that acceptable -- prepaid billing, a row in ``llm_requests`` for every attempt, and no
fabricated transcript when the key is missing -- are the ones under test here. Everything runs
against a mocked HTTP layer; the suite never makes a paid call.
"""

from __future__ import annotations

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import LlmDisabledError, LlmRequestFailed, LlmRouteNotConfigured
from app.llm.elevenlabs import ElevenLabsClient
from app.models import LlmRequest

pytestmark = pytest.mark.db


def routes(**kwargs) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "routes": {
            "asr_scribe_v2": LlmRoute(
                provider="elevenlabs",
                api="transcription",
                model="scribe_v2",
                language="ne",
            )
        },
    }
    return LlmRoutes(**{**base, **kwargs})


def make_client(
    session: Session, handler, *, config: LlmRoutes | None = None, api_key: str = "test-key"
) -> ElevenLabsClient:
    return ElevenLabsClient(
        session,
        config=config or routes(),
        api_key=api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def transcription(text: str = "आजको meeting मा data हेर्यौं", **extra) -> dict:
    """A Scribe response, in the shape the live API returns."""
    return {
        "language_code": "nep",
        "language_probability": 1.0,
        "text": text,
        "words": [],
        "transcription_id": "abc123",
        "audio_duration_secs": 2.0,
        **extra,
    }


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "seg.flac"
    path.write_bytes(b"not really audio, but the client only reads its size")
    return path


# --- never invent a transcript -----------------------------------------------------------


def test_a_missing_api_key_never_yields_a_fabricated_transcript(db_session: Session, clip) -> None:
    """The failure mode that matters: a deploy loses its key and the corpus fills with mock text."""
    client = make_client(db_session, lambda r: httpx.Response(200), api_key="")
    with pytest.raises(LlmRequestFailed, match="ELEVEN_LABS_API_KEY"):
        client.transcribe(clip, route="asr_scribe_v2")

    assert [row.status for row in db_session.scalars(sa.select(LlmRequest))] == ["failed"]


def test_a_disabled_configuration_refuses_to_transcribe(db_session: Session, clip) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200), config=routes(enabled=False))
    with pytest.raises(LlmDisabledError):
        client.transcribe(clip, route="asr_scribe_v2")


def test_an_unknown_route_is_refused(db_session: Session, clip) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200))
    with pytest.raises(LlmRouteNotConfigured, match="nope"):
        client.transcribe(clip, route="nope")


# --- the request ---------------------------------------------------------------------------


def test_the_request_targets_scribe_with_the_key_and_the_language(
    db_session: Session, clip
) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("xi-api-key")
        seen["body"] = bytes(request.content)
        return httpx.Response(200, json=transcription())

    make_client(db_session, handler).transcribe(clip, route="asr_scribe_v2")

    assert seen["url"] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert seen["key"] == "test-key"
    body = seen["body"]
    assert b"scribe_v2" in body
    assert b"language_code" in body and b"ne" in body
    assert b"word" in body  # timestamps_granularity


def test_keyterms_are_sent_because_scribe_takes_no_prompt(db_session: Session, clip) -> None:
    """Scribe has no free-text prompt. Key terms are the only lexical steering it accepts."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = bytes(request.content)
        return httpx.Response(200, json=transcription())

    make_client(db_session, handler).transcribe(
        clip, route="asr_scribe_v2", keyterms=["meeting", "podcast"]
    )
    assert b"keyterms[0]" in seen["body"]
    assert b"meeting" in seen["body"]
    assert b"podcast" in seen["body"]


# --- the response --------------------------------------------------------------------------


def test_a_successful_transcription_returns_the_text(db_session: Session, clip) -> None:
    result = make_client(
        db_session, lambda r: httpx.Response(200, json=transcription())
    ).transcribe(clip, route="asr_scribe_v2")

    assert result.text == "आजको meeting मा data हेर्यौं"
    assert result.model == "scribe_v2"
    assert result.dry_run is False


def test_spacing_and_audio_event_entries_are_not_counted_as_words(
    db_session: Session, clip
) -> None:
    """Scribe interleaves non-word entries; keeping them would inflate every disagreement rate."""
    body = transcription(
        words=[
            {"text": "आजको", "type": "word", "start": 0.0, "end": 0.4},
            {"text": " ", "type": "spacing", "start": 0.4, "end": 0.45},
            {"text": "meeting", "type": "word", "start": 0.45, "end": 0.9},
            {"text": "(laughter)", "type": "audio_event", "start": 0.9, "end": 1.2},
        ]
    )
    result = make_client(db_session, lambda r: httpx.Response(200, json=body)).transcribe(
        clip, route="asr_scribe_v2"
    )

    assert result.words == [
        {"word": "आजको", "start": 0.0, "end": 0.4},
        {"word": "meeting", "start": 0.45, "end": 0.9},
    ]


def test_word_logprobs_are_averaged_into_a_confidence_signal(db_session: Session, clip) -> None:
    """Scribe is the only configured transcriber reporting confidence; it drives low_confidence."""
    body = transcription(
        words=[
            {"text": "आजको", "type": "word", "logprob": -0.2},
            {"text": " ", "type": "spacing", "logprob": -9.0},
            {"text": "meeting", "type": "word", "logprob": -0.4},
        ]
    )
    result = make_client(db_session, lambda r: httpx.Response(200, json=body)).transcribe(
        clip, route="asr_scribe_v2"
    )
    # The spacing entry is excluded, so the mean is over the two real words.
    assert result.avg_logprob == pytest.approx(-0.3)


def test_absent_confidence_stays_none_rather_than_being_invented(db_session: Session, clip) -> None:
    """A default would silently drive the low_confidence term of the priority score."""
    result = make_client(
        db_session, lambda r: httpx.Response(200, json=transcription())
    ).transcribe(clip, route="asr_scribe_v2")

    assert result.avg_logprob is None
    assert result.no_speech_prob is None
    assert result.words is None


# --- logging -------------------------------------------------------------------------------


def test_every_transcription_is_logged(db_session: Session, clip) -> None:
    """The direct call must leave the same audit trail an OpenRouter call would."""
    make_client(db_session, lambda r: httpx.Response(200, json=transcription())).transcribe(
        clip, route="asr_scribe_v2"
    )

    logged = db_session.scalars(sa.select(LlmRequest)).one()
    assert logged.route == "asr_scribe_v2"
    assert logged.model == "scribe_v2"
    assert logged.status == "succeeded"
    assert logged.latency_ms is not None
    assert logged.request_hash


def test_a_failure_is_logged_with_the_error(db_session: Session, clip) -> None:
    client = make_client(
        db_session, lambda r: httpx.Response(400, text="bad audio"), config=routes(max_retries=1)
    )
    with pytest.raises(LlmRequestFailed):
        client.transcribe(clip, route="asr_scribe_v2")

    logged = db_session.scalars(sa.select(LlmRequest)).one()
    assert logged.status == "failed"
    assert "400" in logged.error_message


# --- dry run -------------------------------------------------------------------------------


def test_a_dry_run_is_marked_and_makes_no_http_call(db_session: Session, clip) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a dry run must not reach the network")

    result = make_client(db_session, handler, config=routes(dry_run=True)).transcribe(
        clip, route="asr_scribe_v2"
    )
    assert result.dry_run is True
    assert result.text
    # A mock carries no confidence signal, so it must not claim one.
    assert result.avg_logprob is None
    assert db_session.scalars(sa.select(LlmRequest.status)).all() == ["dry_run"]


# --- retries -------------------------------------------------------------------------------


def test_a_rate_limited_transcription_is_retried_then_succeeds(db_session: Session, clip) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=transcription())

    result = make_client(db_session, handler).transcribe(clip, route="asr_scribe_v2")
    assert calls["n"] == 2
    assert result.text


def test_a_client_error_is_not_retried(db_session: Session, clip) -> None:
    """A 400 will fail identically every time; retrying it only spends time."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad audio")

    with pytest.raises(LlmRequestFailed):
        make_client(db_session, handler).transcribe(clip, route="asr_scribe_v2")
    assert calls["n"] == 1


def test_retries_are_bounded(db_session: Session, clip) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    with pytest.raises(LlmRequestFailed):
        make_client(db_session, handler, config=routes(max_retries=2)).transcribe(
            clip, route="asr_scribe_v2"
        )
    assert calls["n"] == 2
