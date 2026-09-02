"""Tests for transcriber dispatch and the transcript policy.

The ingestion pipeline names a route and gets a transcript; which vendor answered is this
module's business. What is under test is that the right vendor is reached, that the policy prompt
goes to every provider able to read one, and that a provider which cannot is steered by the means
it does have.
"""

from __future__ import annotations

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import LlmRouteNotConfigured
from app.llm.transcription import (
    ASR_PROMPT,
    asr_route_names,
    system_id_for,
    transcribe,
)
from app.models import LlmRequest

pytestmark = pytest.mark.db


def routes(**kwargs) -> LlmRoutes:
    """A routing table with one route of each shape the harness supports."""
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 1,
        "retry_backoff_seconds": 0.0,
        "routes": {
            "asr_scribe_v2": LlmRoute(
                provider="elevenlabs",
                api="transcription",
                system_id="elevenlabs-scribe-v2",
                model="scribe_v2",
                language="ne",
            ),
            "asr_gemini_flash_lite": LlmRoute(
                provider="openrouter",
                api="audio_chat",
                system_id="gemini-3.5-flash-lite",
                model="google/gemini-3.5-flash-lite",
                temperature=0.0,
            ),
            "asr_whisper_large_v3": LlmRoute(
                provider="openrouter",
                api="transcription",
                system_id="whisper-large-v3",
                model="openai/whisper-large-v3",
                language="ne",
            ),
            "check": LlmRoute(model="anthropic/claude-sonnet-5"),
        },
    }
    return LlmRoutes(**{**base, **kwargs})


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "seg.flac"
    path.write_bytes(b"not really audio")
    return path


@pytest.fixture
def recorder(monkeypatch):
    """Capture every outbound request, answering each provider in its own response shape."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "elevenlabs" in str(request.url):
            return httpx.Response(200, json={"text": "स्क्राइब भन्छ", "words": []})
        if "chat/completions" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "model": "google/gemini-3.5-flash-lite",
                    "choices": [{"message": {"content": "  जेमिनाइ भन्छ  "}}],
                    "usage": {"prompt_tokens": 88, "completion_tokens": 4, "cost": 3.3e-05},
                },
            )
        return httpx.Response(
            200,
            json={"text": "विस्पर भन्छ", "usage": {"seconds": 2, "cost": 0.00005}},
        )

    mock = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("app.llm.base.ProviderClient._get_client", lambda self: mock)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ELEVEN_LABS_API_KEY", "test-key")
    return seen


# --- route selection and naming ------------------------------------------------------------


def test_only_asr_routes_become_transcribers() -> None:
    """A `check` route is text inference, not an ASR system; it must not transcribe anything."""
    assert asr_route_names(routes()) == [
        "asr_scribe_v2",
        "asr_gemini_flash_lite",
        "asr_whisper_large_v3",
    ]


def test_the_configured_system_id_wins_over_the_route_name() -> None:
    """system_id is how a hypothesis is attributed for the life of the corpus."""
    table = routes().routes
    assert system_id_for("asr_scribe_v2", table["asr_scribe_v2"]) == "elevenlabs-scribe-v2"
    assert system_id_for("asr_gemini_flash_lite", table["asr_gemini_flash_lite"]) == (
        "gemini-3.5-flash-lite"
    )


def test_a_route_without_a_system_id_falls_back_to_its_name() -> None:
    assert system_id_for("asr_something_new", LlmRoute(model="x")) == "something-new"


def test_an_unknown_route_is_refused(db_session: Session, clip) -> None:
    with pytest.raises(LlmRouteNotConfigured, match="nope"):
        transcribe(db_session, clip, route="nope", config=routes())


# --- dispatch ------------------------------------------------------------------------------


def test_an_elevenlabs_route_reaches_elevenlabs(db_session: Session, clip, recorder) -> None:
    result = transcribe(db_session, clip, route="asr_scribe_v2", config=routes())
    assert result.text == "स्क्राइब भन्छ"
    assert str(recorder[-1].url) == "https://api.elevenlabs.io/v1/speech-to-text"


def test_an_audio_chat_route_reaches_chat_completions(db_session: Session, clip, recorder) -> None:
    result = transcribe(db_session, clip, route="asr_gemini_flash_lite", config=routes())
    assert str(recorder[-1].url) == "https://openrouter.ai/api/v1/chat/completions"
    # A chat model pads its answer; the transcript is what gets stored.
    assert result.text == "जेमिनाइ भन्छ"


def test_a_transcription_route_reaches_the_transcription_endpoint(
    db_session: Session, clip, recorder
) -> None:
    result = transcribe(db_session, clip, route="asr_whisper_large_v3", config=routes())
    assert str(recorder[-1].url) == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert result.text == "विस्पर भन्छ"


# --- the transcript policy reaches every provider that can read it -------------------------


def test_the_policy_prompt_states_both_scripts() -> None:
    """The corpus rule: each word in the script of its own language."""
    assert "Devanagari" in ASR_PROMPT
    assert "Latin" in ASR_PROMPT
    assert "code-switched" in ASR_PROMPT


def test_a_chat_model_is_told_to_answer_with_the_transcript_alone() -> None:
    """A general LLM will otherwise reply in prose, or narrate what it heard."""
    assert "nothing else" in ASR_PROMPT


def test_the_policy_prompt_is_sent_to_the_audio_chat_model(
    db_session: Session, clip, recorder
) -> None:
    transcribe(db_session, clip, route="asr_gemini_flash_lite", config=routes())
    body = bytes(recorder[-1].content)
    assert b"Devanagari" in body
    assert b"input_audio" in body


def test_the_policy_prompt_is_sent_to_the_transcription_endpoint(
    db_session: Session, clip, recorder
) -> None:
    transcribe(db_session, clip, route="asr_whisper_large_v3", config=routes())
    body = bytes(recorder[-1].content)
    assert b"Devanagari" in body
    assert b"ne" in body  # the route's language hint


def test_scribe_is_steered_by_language_and_keyterms_instead_of_a_prompt(
    db_session: Session, clip, recorder
) -> None:
    """Scribe has no prompt parameter, so the policy has to be expressed the ways it accepts."""
    transcribe(db_session, clip, route="asr_scribe_v2", config=routes())
    body = bytes(recorder[-1].content)
    assert b"language_code" in body
    assert b"keyterms[0]" in body
    assert b"Devanagari" not in body, "Scribe has no prompt field to send the policy to"


# --- billing audit trail -------------------------------------------------------------------


def test_every_provider_writes_a_request_log_row(db_session: Session, clip, recorder) -> None:
    """The direct ElevenLabs call must leave the same trail an OpenRouter call does."""
    for route in asr_route_names(routes()):
        transcribe(db_session, clip, route=route, config=routes())

    logged = {row.route: row for row in db_session.scalars(sa.select(LlmRequest))}
    assert set(logged) == {
        "asr_scribe_v2",
        "asr_gemini_flash_lite",
        "asr_whisper_large_v3",
    }
    assert all(row.status == "succeeded" for row in logged.values())


def test_the_charged_cost_is_recorded_from_the_usage_block(
    db_session: Session, clip, recorder
) -> None:
    """OpenRouter reports `cost`; reading only the older `total_cost` logs every call as free."""
    transcribe(db_session, clip, route="asr_whisper_large_v3", config=routes())
    row = db_session.scalars(sa.select(LlmRequest)).one()
    assert row.estimated_cost_usd is not None
    assert float(row.estimated_cost_usd) == pytest.approx(0.00005)


def test_a_dry_run_reaches_no_provider(db_session: Session, clip, recorder) -> None:
    for route in asr_route_names(routes()):
        result = transcribe(db_session, clip, route=route, config=routes(dry_run=True))
        assert result.dry_run is True
        assert result.text
    assert recorder == []


def test_transcribe_accepts_and_uses_custom_http_client(
    db_session: Session, clip, monkeypatch
) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"text": "सफलता", "words": []})

    custom_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("ELEVEN_LABS_API_KEY", "test-key")

    result = transcribe(
        db_session,
        clip,
        route="asr_scribe_v2",
        config=routes(),
        client=custom_client,
    )
    assert result.text == "सफलता"
    assert len(calls) == 1
