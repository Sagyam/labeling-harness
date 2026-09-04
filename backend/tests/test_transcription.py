"""Tests for transcriber dispatch and the transcript policy.

The ingestion pipeline names a route and gets a transcript; which vendor answered is this
module's business. What is under test is that the right vendor is reached, that the policy prompt
goes to every provider able to read one, and that a provider which cannot is steered by the means
it does have.
"""

from __future__ import annotations

import json

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import LlmRouteNotConfigured
from app.llm.transcription import (
    ASR_PROMPT,
    DEFAULT_KEYTERMS,
    SCRIPT_POLICY,
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
            "asr_gemini_flash": LlmRoute(
                provider="openrouter",
                api="audio_chat",
                system_id="gemini-3.8-flash",
                model="google/gemini-3.8-flash",
                temperature=0.0,
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
                    "model": "google/gemini-3.8-flash",
                    "choices": [{"message": {"content": "  जेमिनाइ भन्छ  "}}],
                    "usage": {"prompt_tokens": 88, "completion_tokens": 4, "cost": 3.3e-05},
                },
            )
        if "interactions:create" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "steps": [
                        {
                            "modelOutput": {
                                "content": [
                                    {
                                        "text": {
                                            "text": "भर्टेक्स भन्छ",
                                            "annotations": [
                                                {
                                                    "wordInfo": {
                                                        "text": "भर्टेक्स",
                                                        "startOffset": "0s",
                                                        "endOffset": "0.5s",
                                                        "speaker": "spk_1",
                                                    }
                                                }
                                            ],
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        if "aiplatform.googleapis.com" in str(request.url):
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "फ्ल्यास भन्छ"}]}}]},
            )
        return httpx.Response(
            200,
            json={"text": "विस्पर भन्छ", "usage": {"seconds": 2, "cost": 0.00005}},
        )

    mock = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("app.llm.base.ProviderClient._get_client", lambda self: mock)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ELEVEN_LABS_API_KEY", "test-key")
    monkeypatch.setattr("app.llm.vertex.VertexClient._bearer_token", lambda self: "test-token")
    return seen


# --- route selection and naming ------------------------------------------------------------


def test_only_asr_routes_become_transcribers() -> None:
    """A `check` route is text inference, not an ASR system; it must not transcribe anything."""
    assert asr_route_names(routes()) == [
        "asr_scribe_v2",
        "asr_gemini_flash",
    ]


def test_the_configured_system_id_wins_over_the_route_name() -> None:
    """system_id is how a hypothesis is attributed for the life of the corpus."""
    table = routes().routes
    assert system_id_for("asr_scribe_v2", table["asr_scribe_v2"]) == "elevenlabs-scribe-v2"
    assert system_id_for("asr_gemini_flash", table["asr_gemini_flash"]) == ("gemini-3.8-flash")


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
    result = transcribe(db_session, clip, route="asr_gemini_flash", config=routes())
    assert str(recorder[-1].url) == "https://openrouter.ai/api/v1/chat/completions"
    # A chat model pads its answer; the transcript is what gets stored.
    assert result.text == "जेमिनाइ भन्छ"


def test_a_transcription_route_reaches_the_transcription_endpoint(
    db_session: Session, clip, recorder
) -> None:
    test_routes = routes(
        routes={
            **routes().routes,
            "asr_custom_endpoint": LlmRoute(
                provider="openrouter",
                api="transcription",
                system_id="custom-endpoint",
                model="custom/asr-model",
            ),
        }
    )
    result = transcribe(db_session, clip, route="asr_custom_endpoint", config=test_routes)
    assert str(recorder[-1].url) == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert result.text == "विस्पर भन्छ"


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
    transcribe(db_session, clip, route="asr_gemini_flash", config=routes())
    body = bytes(recorder[-1].content)
    assert b"Devanagari" in body
    assert b"input_audio" in body


def test_the_policy_prompt_is_sent_to_the_transcription_endpoint(
    db_session: Session, clip, recorder
) -> None:
    test_routes = routes(
        routes={
            **routes().routes,
            "asr_custom_endpoint": LlmRoute(
                provider="openrouter",
                api="transcription",
                system_id="custom-endpoint",
                model="custom/asr-model",
                language="ne",
            ),
        }
    )
    transcribe(db_session, clip, route="asr_custom_endpoint", config=test_routes)
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
        "asr_gemini_flash",
    }
    assert all(row.status == "succeeded" for row in logged.values())


def test_the_charged_cost_is_recorded_from_the_usage_block(
    db_session: Session, clip, recorder
) -> None:
    """OpenRouter reports `cost`; reading only the older `total_cost` logs every call as free."""
    transcribe(db_session, clip, route="asr_gemini_flash", config=routes())
    row = db_session.scalars(sa.select(LlmRequest)).one()
    assert row.estimated_cost_usd is not None
    assert float(row.estimated_cost_usd) == pytest.approx(3.3e-05)


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


# --- Vertex AI dispatch ---------------------------------------------------------------------


def _vertex_routes(**route_kwargs) -> LlmRoutes:
    """The base table plus one Vertex route of the requested shape."""
    return routes(
        vertex_project="test-project",
        vertex_location="global",
        routes={**routes().routes, "asr_vertex": LlmRoute(provider="vertex", **route_kwargs)},
    )


def test_a_vertex_transcription_route_reaches_interactions_create(
    db_session: Session, clip, recorder
) -> None:
    """The dedicated recogniser: word spans and speaker labels come back with the text."""
    config = _vertex_routes(
        api="transcription",
        model="gemini-3.5-transcribe",
        system_id="gemini-3.5-transcribe",
        language_codes=["ne-NP", "en-US"],
    )
    result = transcribe(db_session, clip, route="asr_vertex", config=config)

    assert str(recorder[-1].url).endswith("/locations/global/interactions:create")
    assert result.text == "भर्टेक्स भन्छ"
    assert result.words == [{"word": "भर्टेक्स", "start": 0.0, "end": 0.5, "speaker": "spk_1"}]


def test_a_vertex_audio_chat_route_reaches_generate_content(
    db_session: Session, clip, recorder
) -> None:
    config = _vertex_routes(api="audio_chat", model="gemini-3.8-flash", language="ne")
    result = transcribe(db_session, clip, route="asr_vertex", config=config)

    url = str(recorder[-1].url)
    assert url.endswith("/publishers/google/models/gemini-3.8-flash:generateContent")
    assert result.text == "फ्ल्यास भन्छ"
    # generateContent returns text; word spans come from the forced aligner (D32).
    assert result.words is None


def test_a_dedicated_recogniser_gets_the_policy_without_the_chat_scaffolding(
    db_session: Session, clip, recorder
) -> None:
    """Telling a speech model not to write a preamble is noise in a system instruction."""
    config = _vertex_routes(api="transcription", model="gemini-3.5-transcribe")
    transcribe(db_session, clip, route="asr_vertex", config=config)

    sent = json.loads(recorder[-1].content)["interaction"]
    assert sent["systemInstruction"] == SCRIPT_POLICY
    assert "nothing else" not in sent["systemInstruction"]


def test_a_vertex_chat_model_gets_the_whole_prompt(db_session: Session, clip, recorder) -> None:
    config = _vertex_routes(api="audio_chat", model="gemini-3.8-flash")
    transcribe(db_session, clip, route="asr_vertex", config=config)

    parts = json.loads(recorder[-1].content)["contents"][0]["parts"]
    assert ASR_PROMPT in parts[0]["text"]


def test_a_dedicated_recogniser_gets_the_same_key_terms_scribe_does(
    db_session: Session, clip, recorder
) -> None:
    config = _vertex_routes(api="transcription", model="gemini-3.5-transcribe")
    transcribe(db_session, clip, route="asr_vertex", config=config)

    sent = json.loads(recorder[-1].content)["interaction"]
    vocabulary = sent["modelInteraction"]["generationConfig"]["transcriptionConfig"][
        "customVocabulary"
    ]
    assert vocabulary == list(DEFAULT_KEYTERMS)


def test_the_policy_forbids_transliteration_in_both_directions() -> None:
    """The one rule a multilingual model gets wrong by default, in the shared policy."""
    assert "DO NOT TRANSLITERATE" in SCRIPT_POLICY
    assert "verbatim" in SCRIPT_POLICY
    assert SCRIPT_POLICY in ASR_PROMPT
