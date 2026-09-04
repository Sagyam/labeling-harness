"""Tests for the Vertex AI Gemini client.

Both speech shapes go to ``:generateContent`` on ``aiplatform.googleapis.com``; only the body
differs (D39):

- A dedicated recogniser (``gemini-3.5-transcribe-preview``) driven by
  ``generationConfig.audioTranscriptionConfig``, returning word spans and a speaker label per
  segment. It rejects ``systemInstruction``, so it must never be sent one.
- A general chat model (``gemini-3.8-flash``) that takes a ``systemInstruction`` and returns text.

The assertions that matter most here are the ones that would have caught the two bugs this module
exists to fix: the ``-preview`` model id, and the fact that Vertex has no Interactions API.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import LlmDisabledError, LlmRequestFailed, LlmRouteNotConfigured
from app.llm.vertex import (
    VertexClient,
    parse_generate_content,
    parse_transcription,
    withheld_reason,
)

TRANSCRIBE_ROUTE = "asr_gemini_composite"
FLASH_ROUTE = "asr_gemini_flash"
BASE = "https://aiplatform.googleapis.com/v1beta1"
PROJECT = "test-project"
MODELS = f"{BASE}/projects/{PROJECT}/locations/global/publishers/google/models"


def _generate_content_body(*texts: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": t} for t in texts]}}]}


def _segment(text: str, speaker: str | None, words: list[dict[str, Any]]) -> dict[str, Any]:
    """One Vertex transcription Part. The sibling ``text`` repeats the transcription verbatim."""
    transcription: dict[str, Any] = {"text": text}
    if speaker is not None:
        transcription["speakerLabel"] = speaker
    if words:
        transcription["words"] = words
    return {"text": text, "audioTranscription": transcription}


def _transcription_body(*segments: dict[str, Any]) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": list(segments)}}]}


# --- response parsing, no database ----------------------------------------------------------


def test_parse_generate_content_joins_the_parts_of_a_candidate() -> None:
    assert parse_generate_content(_generate_content_body("आजको ", "meeting मा")) == (
        "आजको meeting मा"
    )


def test_parse_generate_content_skips_a_candidate_carrying_no_text() -> None:
    """A blocked or tool-only candidate yields nothing; the next one is tried."""
    body = {
        "candidates": [
            {"finishReason": "SAFETY", "content": {"parts": []}},
            {"content": {"parts": [{"text": "real transcript"}]}},
        ]
    }
    assert parse_generate_content(body) == "real transcript"


def test_parse_generate_content_handles_empty_or_malformed() -> None:
    assert parse_generate_content({}) == ""
    assert parse_generate_content({"candidates": []}) == ""
    assert parse_generate_content({"candidates": [{}]}) == ""
    assert parse_generate_content({"candidates": [{"content": {"parts": [{"n": 1}]}}]}) == ""


def test_parse_transcription_reads_the_transcript_and_its_word_spans() -> None:
    body = _transcription_body(
        _segment(
            "आजको meeting",
            "spk:0",
            [
                {"word": "आजको", "startOffset": "0.20s", "endOffset": "0.61s"},
                {"word": "meeting", "startOffset": "0.61s", "endOffset": "1.30s"},
            ],
        )
    )
    text, words = parse_transcription(body)
    assert text == "आजको meeting"
    assert words == [
        {"word": "आजको", "start": 0.2, "end": 0.61, "speaker": "spk:0"},
        {"word": "meeting", "start": 0.61, "end": 1.3, "speaker": "spk:0"},
    ]


def test_the_segment_speaker_label_is_fanned_onto_every_word_of_that_segment() -> None:
    """Vertex reports the speaker per Part; hypothesis_words.speaker is per word (D36)."""
    body = _transcription_body(
        _segment("हो", "spk:0", [{"word": "हो", "startOffset": "0s", "endOffset": "0.4s"}]),
        _segment("yes", "spk:1", [{"word": "yes", "startOffset": "0.5s", "endOffset": "0.9s"}]),
    )
    text, words = parse_transcription(body)
    assert text == "हो yes"
    assert [w["speaker"] for w in words or []] == ["spk:0", "spk:1"]


def test_a_repeated_sibling_text_part_is_not_counted_twice() -> None:
    """Vertex sets both ``part.text`` and ``audioTranscription.text`` to the same string."""
    body = _transcription_body(_segment("आजको meeting", "spk:0", []))
    text, _ = parse_transcription(body)
    assert text == "आजको meeting"


def test_parse_transcription_returns_none_rather_than_an_empty_word_list() -> None:
    """The scorer reads None as "no signal"; an empty list would read as "no words said"."""
    text, words = parse_transcription(_transcription_body(_segment("कुनै timestamp छैन", None, [])))
    assert text == "कुनै timestamp छैन"
    assert words is None


def test_parse_transcription_survives_an_unparseable_or_missing_offset() -> None:
    body = _transcription_body(
        _segment("क", None, [{"word": "क", "startOffset": "not-a-duration", "endOffset": None}])
    )
    _, words = parse_transcription(body)
    assert words == [{"word": "क", "start": None, "end": None, "speaker": None}]


def test_parse_transcription_reads_a_part_carrying_only_plain_text() -> None:
    body = {"candidates": [{"content": {"parts": [{"text": "no transcription payload"}]}}]}
    text, words = parse_transcription(body)
    assert text == "no transcription payload"
    assert words is None


def test_parse_transcription_handles_empty_or_malformed() -> None:
    assert parse_transcription({}) == ("", None)
    assert parse_transcription({"candidates": []}) == ("", None)
    assert parse_transcription({"candidates": [{"content": {"parts": []}}]}) == ("", None)


# --- requests, with a database session ------------------------------------------------------


def routes(**kwargs: Any) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "vertex_base_url": BASE,
        "vertex_project": PROJECT,
        "vertex_location": "global",
        "routes": {
            TRANSCRIBE_ROUTE: LlmRoute(
                provider="vertex",
                api="transcription",
                model="gemini-3.5-transcribe-preview",
                system_id="gemini-3.5-transcribe",
                language_codes=["ne-NP", "en-US"],
            ),
            FLASH_ROUTE: LlmRoute(
                provider="vertex",
                api="audio_chat",
                model="gemini-3.8-flash",
                system_id="gemini-3.8-flash",
                language="ne",
                forced_align=True,
            ),
        },
    }
    return LlmRoutes(**{**base, **kwargs})


def make_client(
    session: Session,
    handler,
    *,
    config: LlmRoutes | None = None,
    api_key: str | None = "test-api-key",
) -> VertexClient:
    return VertexClient(
        session,
        config=config or routes(),
        api_key=api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.flac"
    path.write_bytes(b"mock audio bytes")
    return path


def _ok(body: dict[str, Any], seen: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=body)

    return handler


@pytest.mark.db
def test_a_disabled_vertex_client_refuses_to_transcribe(db_session: Session, clip: Path) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200), config=routes(enabled=False))
    with pytest.raises(LlmDisabledError):
        client.transcribe(clip, route=TRANSCRIBE_ROUTE)


@pytest.mark.db
def test_an_unknown_vertex_route_is_refused(db_session: Session, clip: Path) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200))
    with pytest.raises(LlmRouteNotConfigured, match="nonexistent"):
        client.transcribe(clip, route="nonexistent")


@pytest.mark.db
def test_a_missing_api_key_is_refused_before_any_request(
    db_session: Session, clip: Path, monkeypatch
) -> None:
    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = make_client(db_session, handler, api_key="")
    with pytest.raises(LlmRequestFailed, match="VERTEX_API_KEY"):
        client.transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert not called


@pytest.mark.db
def test_a_missing_project_is_refused_before_any_request(
    db_session: Session, clip: Path, monkeypatch
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = make_client(db_session, handler, config=routes(vertex_project=""))
    with pytest.raises(LlmRequestFailed, match="GOOGLE_CLOUD_PROJECT"):
        client.transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert not called


@pytest.mark.db
def test_a_vertex_dry_run_never_calls_the_network(db_session: Session, clip: Path) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = make_client(db_session, handler, config=routes(dry_run=True))
    result = client.transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert not called
    assert result.dry_run is True
    assert result.text


@pytest.mark.db
def test_a_transcription_route_targets_the_preview_model_on_generate_content(
    db_session: Session, clip: Path
) -> None:
    """The `-preview` suffix and `:generateContent` are the whole of the D39 fix.

    Vertex has no Interactions API, and the bare `gemini-3.5-transcribe` id is a 404 there.
    """
    seen: list[httpx.Request] = []
    make_client(db_session, _ok(_transcription_body(_segment("ok", None, [])), seen)).transcribe(
        clip, route=TRANSCRIBE_ROUTE
    )
    assert str(seen[0].url) == f"{MODELS}/gemini-3.5-transcribe-preview:generateContent"
    assert seen[0].method == "POST"
    assert "interactions" not in str(seen[0].url)


@pytest.mark.db
def test_the_api_key_travels_in_a_header_and_never_in_the_url(
    db_session: Session, clip: Path
) -> None:
    """httpx puts the URL in its error strings, and those are copied into llm_requests."""
    seen: list[httpx.Request] = []
    make_client(db_session, _ok(_transcription_body(_segment("ok", None, [])), seen)).transcribe(
        clip, route=TRANSCRIBE_ROUTE
    )
    assert seen[0].headers["x-goog-api-key"] == "test-api-key"
    assert "test-api-key" not in str(seen[0].url)
    assert "key=" not in str(seen[0].url)


@pytest.mark.db
def test_word_timestamps_and_diarization_are_booleans_on_the_transcription_config(
    db_session: Session, clip: Path
) -> None:
    """Vertex has no verbatim/smart mode; the deprecated spellings must not be sent."""
    seen: list[httpx.Request] = []
    make_client(db_session, _ok(_transcription_body(_segment("ok", None, [])), seen)).transcribe(
        clip, route=TRANSCRIBE_ROUTE
    )
    sent = json.loads(seen[0].content)
    config = sent["generationConfig"]["audioTranscriptionConfig"]
    assert config["wordTimestamp"] is True
    assert config["diarization"] is True
    for deprecated in (
        "mode",
        "timestampGranularities",
        "diarizationMode",
        "languageHints",
        "languageAuto",
        "adaptationPhrases",
    ):
        assert deprecated not in config


@pytest.mark.db
def test_both_language_codes_reach_a_code_switched_route(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []
    make_client(db_session, _ok(_transcription_body(_segment("ok", None, [])), seen)).transcribe(
        clip, route=TRANSCRIBE_ROUTE
    )
    sent = json.loads(seen[0].content)
    config = sent["generationConfig"]["audioTranscriptionConfig"]
    assert config["languageCodes"] == ["ne-NP", "en-US"]


@pytest.mark.db
def test_a_route_without_language_codes_falls_back_to_its_single_hint(
    db_session: Session, clip: Path
) -> None:
    seen: list[httpx.Request] = []
    table = routes()
    single = table.routes[TRANSCRIBE_ROUTE].model_copy(
        update={"language_codes": [], "language": "ne"}
    )
    make_client(
        db_session,
        _ok(_transcription_body(_segment("ok", None, [])), seen),
        config=routes(routes={**table.routes, TRANSCRIBE_ROUTE: single}),
    ).transcribe(clip, route=TRANSCRIBE_ROUTE)
    sent = json.loads(seen[0].content)
    assert sent["generationConfig"]["audioTranscriptionConfig"]["languageCodes"] == ["ne"]


@pytest.mark.db
def test_a_dedicated_transcription_route_never_sends_a_system_instruction(
    db_session: Session, clip: Path
) -> None:
    """Vertex answers one with 400 "The input system_instruction is not supported."."""
    seen: list[httpx.Request] = []
    make_client(db_session, _ok(_transcription_body(_segment("ok", None, [])), seen)).transcribe(
        clip, route=TRANSCRIBE_ROUTE, prompt="DO NOT TRANSLITERATE."
    )
    sent = json.loads(seen[0].content)
    assert "systemInstruction" not in sent
    assert "system_instruction" not in sent
    # A text part is accepted but silently ignored, so it is not sent either.
    assert all("text" not in part for part in sent["contents"][0]["parts"])


@pytest.mark.db
def test_custom_vocabulary_is_dropped_because_it_silently_costs_the_speaker_labels(
    db_session: Session, clip: Path
) -> None:
    """Vertex takes the combination with a 200 and then returns no speakerLabel at all.

    AI Studio at least answers 400. Here it fails silently, so the constraint lives in the client
    rather than in a comment somebody can re-delete (D39).
    """
    seen: list[httpx.Request] = []
    make_client(db_session, _ok(_transcription_body(_segment("ok", None, [])), seen)).transcribe(
        clip, route=TRANSCRIBE_ROUTE, custom_vocabulary=["meeting", "data"]
    )
    config = json.loads(seen[0].content)["generationConfig"]["audioTranscriptionConfig"]
    assert "customVocabulary" not in config
    assert config["diarization"] is True
    assert config["wordTimestamp"] is True


@pytest.mark.db
def test_the_clip_is_inlined_as_base64_flac(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []
    make_client(db_session, _ok(_transcription_body(_segment("ok", None, [])), seen)).transcribe(
        clip, route=TRANSCRIBE_ROUTE
    )
    sent = json.loads(seen[0].content)
    audio = sent["contents"][0]["parts"][0]["inlineData"]
    assert audio["mimeType"] == "audio/flac"
    assert base64.b64decode(audio["data"]) == b"mock audio bytes"


@pytest.mark.db
def test_an_audio_chat_route_targets_generate_content(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []
    result = make_client(
        db_session, _ok(_generate_content_body("chat transcript"), seen)
    ).transcribe(clip, route=FLASH_ROUTE)
    assert str(seen[0].url) == f"{MODELS}/gemini-3.8-flash:generateContent"
    assert seen[0].headers["x-goog-api-key"] == "test-api-key"
    assert result.text == "chat transcript"
    assert result.words is None


@pytest.mark.db
def test_an_audio_chat_route_carries_the_policy_as_a_system_instruction(
    db_session: Session, clip: Path
) -> None:
    """Flash is the only Gemini route that can be told not to transliterate."""
    seen: list[httpx.Request] = []
    make_client(db_session, _ok(_generate_content_body("ok"), seen)).transcribe(
        clip, route=FLASH_ROUTE, prompt="DO NOT TRANSLITERATE."
    )
    sent = json.loads(seen[0].content)
    assert "DO NOT TRANSLITERATE." in sent["systemInstruction"]["parts"][0]["text"]
    audio = sent["contents"][0]["parts"][0]["inlineData"]
    assert audio["mimeType"] == "audio/flac"
    assert base64.b64decode(audio["data"]) == b"mock audio bytes"


@pytest.mark.db
def test_a_429_is_retried_and_succeeds(db_session: Session, clip: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_transcription_body(_segment("recovered", None, [])))

    result = make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert calls == 2
    assert result.text == "recovered"


@pytest.mark.db
def test_exhausted_retries_fail_with_logged_row(db_session: Session, clip: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    with pytest.raises(LlmRequestFailed, match="Vertex transcription failed"):
        make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)


# --- withheld answers: the field a Gemini block actually arrives in (D45) --------------------


def _blocked_body(finish: str = "SAFETY", category: str = "HARM_CATEGORY_DANGEROUS_CONTENT"):
    """A real Vertex block, shaped as Gemini 3.8 Flash returns it.

    Captured live: no ``promptFeedback`` key at all, a ``finishReason`` on the candidate, and a
    ``content`` carrying a role and no parts. The harness watched ``promptFeedback`` for months
    and this shape sailed straight past it.
    """
    return {
        "candidates": [
            {
                "content": {"role": "model"},
                "finishReason": finish,
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"},
                    {"category": category, "probability": "LOW", "blocked": True},
                ],
            }
        ],
        "usageMetadata": {"promptTokenCount": 668, "candidatesTokenCount": 97},
    }


def test_a_normal_finish_is_not_read_as_withheld() -> None:
    assert withheld_reason(_generate_content_body("नमस्ते")) is None
    assert withheld_reason({"candidates": [{"finishReason": "STOP", "content": {}}]}) is None


def test_a_candidate_side_block_is_detected_and_names_the_category() -> None:
    reason = withheld_reason(_blocked_body())
    assert reason is not None
    assert "finishReason=SAFETY" in reason
    assert "HARM_CATEGORY_DANGEROUS_CONTENT" in reason


def test_a_prompt_side_block_is_detected_too() -> None:
    body = {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "candidates": []}
    assert withheld_reason(body) == "promptFeedback.blockReason=PROHIBITED_CONTENT"


@pytest.mark.parametrize(
    "finish", ["PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "MAX_TOKENS"]
)
def test_every_withholding_finish_reason_is_caught_not_just_safety(finish: str) -> None:
    """`OFF` cannot switch off the built-in filters, so they must be caught whatever D39 sends."""
    assert withheld_reason(_blocked_body(finish=finish)) is not None


@pytest.mark.db
def test_a_blocked_clip_fails_instead_of_becoming_an_empty_hypothesis(
    db_session: Session, clip: Path
) -> None:
    """The bug D45 exists to fix.

    `audio_chat` may legitimately answer with an empty string -- ASR_PROMPT asks for one over
    silence -- so an empty answer used to be accepted unconditionally. A safety block produces the
    same empty text, and so became a silent empty hypothesis in the corpus, logged as succeeded.
    """
    calls: list[httpx.Request] = []
    client = make_client(db_session, _ok(_blocked_body(), calls))
    with pytest.raises(LlmRequestFailed, match="finishReason=SAFETY"):
        client.transcribe(clip, route=FLASH_ROUTE)
    assert len(calls) == 3, "a withheld answer is retried before it is given up on"


@pytest.mark.db
def test_an_empty_answer_that_finished_normally_is_still_accepted(
    db_session: Session, clip: Path
) -> None:
    """Silence is not a block. ASR_PROMPT asks for an empty string when nothing was said."""
    body = {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}]}
    result = make_client(db_session, _ok(body)).transcribe(clip, route=FLASH_ROUTE)
    assert result.text == ""


# --- the thinking switch ---------------------------------------------------------------------


def _with_thinking_off(route_name: str) -> LlmRoutes:
    """A routing table with one route's thinking switched off. ``LlmRoute`` is frozen."""
    config = routes()
    config.routes[route_name] = config.routes[route_name].model_copy(
        update={"reasoning_enabled": False}
    )
    return config


@pytest.mark.db
def test_an_audio_chat_route_can_turn_thinking_off(db_session: Session, clip: Path) -> None:
    """Measured at 895 thought tokens against 88 tokens of transcript before this (D45)."""
    seen: list[httpx.Request] = []
    config = _with_thinking_off(FLASH_ROUTE)
    make_client(db_session, _ok(_generate_content_body("ok"), seen), config=config).transcribe(
        clip, route=FLASH_ROUTE
    )
    body = json.loads(seen[0].content)
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


@pytest.mark.db
def test_thinking_is_left_alone_when_the_route_says_nothing(
    db_session: Session, clip: Path
) -> None:
    seen: list[httpx.Request] = []
    make_client(db_session, _ok(_generate_content_body("ok"), seen)).transcribe(
        clip, route=FLASH_ROUTE
    )
    assert "thinkingConfig" not in json.loads(seen[0].content)["generationConfig"]


@pytest.mark.db
def test_the_recogniser_is_never_sent_a_thinking_config(db_session: Session, clip: Path) -> None:
    """Vertex answers one with `400 Thinking is not enabled for this model`, and the recogniser
    reports zero thought tokens anyway -- there is nothing there to switch off."""
    seen: list[httpx.Request] = []
    config = _with_thinking_off(TRANSCRIBE_ROUTE)
    make_client(
        db_session, _ok(_transcription_body(_segment("नमस्ते", "spk:0", [])), seen), config=config
    ).transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert "thinkingConfig" not in json.loads(seen[0].content)["generationConfig"]


# --- text completions on Vertex (D45) --------------------------------------------------------


def _text_routes(**overrides) -> LlmRoutes:
    config = routes()
    config.routes["script_restore"] = LlmRoute(
        provider="vertex",
        api="chat",
        model="gemini-3.8-flash",
        temperature=0.0,
        max_tokens=4096,
        **overrides,
    )
    return config


@pytest.mark.db
def test_a_text_route_completes_through_vertex(db_session: Session) -> None:
    seen: list[httpx.Request] = []
    client = make_client(
        db_session, _ok(_generate_content_body('["a"]'), seen), config=_text_routes()
    )
    result = client.complete("script_restore", [{"role": "user", "content": "hi"}])
    assert result.text == '["a"]'
    body = json.loads(seen[0].content)
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["generationConfig"]["maxOutputTokens"] == 4096
    assert str(seen[0].url) == f"{MODELS}/gemini-3.8-flash:generateContent"


@pytest.mark.db
def test_a_system_message_becomes_a_system_instruction(db_session: Session) -> None:
    seen: list[httpx.Request] = []
    make_client(
        db_session, _ok(_generate_content_body("ok"), seen), config=_text_routes()
    ).complete(
        "script_restore",
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
    )
    body = json.loads(seen[0].content)
    assert body["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


@pytest.mark.db
def test_a_withheld_completion_raises_rather_than_returning_empty_text(db_session: Session) -> None:
    """A text caller parses what it gets back; handing it "" for a block surfaces the failure
    somewhere far away from the cause, which is exactly how D44 went wrong."""
    client = make_client(db_session, _ok(_blocked_body()), config=_text_routes())
    with pytest.raises(LlmRequestFailed, match="finishReason=SAFETY"):
        client.complete("script_restore", [{"role": "user", "content": "hi"}])


@pytest.mark.db
def test_a_text_route_can_turn_thinking_off(db_session: Session) -> None:
    seen: list[httpx.Request] = []
    make_client(
        db_session,
        _ok(_generate_content_body("ok"), seen),
        config=_text_routes(reasoning_enabled=False),
    ).complete("script_restore", [{"role": "user", "content": "hi"}])
    gen = json.loads(seen[0].content)["generationConfig"]
    assert gen["thinkingConfig"] == {"thinkingBudget": 0}
