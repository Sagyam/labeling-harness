"""Transcriber dispatch: one call site, several vendors.

The ingestion pipeline should not know which vendor is behind a route. It asks for a route name
and gets an :class:`AsrResult`; this module decides whether that means OpenRouter's transcription
endpoint, an OpenRouter chat model with the clip attached, ElevenLabs Scribe, or a Gemini model on
Vertex AI.

The transcript policy lives here too. The corpus is code-switched, and the rule is that a word is
written in the script of the language it belongs to, so every model that can be told anything is
told both that the mixing is expected and which script each half takes. Scribe takes no prose
instruction at all, which is why it is given a language code and key terms instead. Gemini 3.5
Transcribe takes neither -- prose is refused and key terms cost it its speaker labels -- so it has
no lever at all, and renders English words in Devanagari (D39).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import AsrResult, LlmRouteNotConfigured
from app.llm.elevenlabs import ElevenLabsClient
from app.llm.openrouter import OpenRouterClient
from app.llm.script_restore import restore_script
from app.llm.vertex import VertexClient
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: The transcript policy itself, and the whole of what a dedicated speech recogniser is told.
#: The rule that matters most is the one a multilingual model gets wrong by default: it will
#: happily render a Nepali word in Latin or an English word in Devanagari, and either destroys
#: the code-switch measurement the corpus exists to make.
SCRIPT_POLICY = (
    "This is a Nepali-English code-switched conversation: the speakers mix both languages "
    "freely, often within a single sentence. Transcribe verbatim -- exactly what is said, "
    "including false starts, repetitions and filler words. Do not clean up the speech.\n\n"
    "DO NOT TRANSLITERATE. Write every word in the script its own language uses: Nepali words "
    "in Devanagari, English words in Latin. Never write a Nepali word in Latin letters, never "
    "write an English word in Devanagari, and never translate between the two languages. "
    "For example: आजको meeting मा हामीले नयाँ data हेर्यौं।"
)

#: Sent with every ASR request to a model that is not a dedicated recogniser. The policy, plus
#: what stops a chat model answering in prose or narrating what it heard.
ASR_PROMPT = (
    f"{SCRIPT_POLICY}\n\n"
    "Reply with the transcript and nothing else: no preamble, no translation, no commentary, no "
    "quotation marks. If there is no intelligible speech, reply with an empty string."
)

#: Terms biasing Scribe towards the English half of the vocabulary. Scribe has no prompt, so this
#: and the language code are the only steering it accepts.
DEFAULT_KEYTERMS = (
    "meeting",
    "podcast",
    "episode",
    "data",
    "team",
    "project",
    "schedule",
    "update",
    "software",
    "technology",
)


def asr_route_names(config: LlmRoutes) -> list[str]:
    """Every route that becomes an ASR system during ingestion, in configured order."""
    return config.asr_route_names()


def system_id_for(route_name: str, route_config: LlmRoute | None) -> str:
    """The name a route's hypotheses are attributed to in the database and in every export."""
    if route_config is not None and route_config.system_id:
        return route_config.system_id
    return route_name.removeprefix("asr_").replace("_", "-") or "asr"


def disagreement_excluded_system_ids(config: LlmRoutes) -> frozenset[str]:
    """The ``system_id`` of every route held out of the cross-system disagreement scores.

    Disagreement is recomputed in two places -- at ingest and again after a purge -- and they must
    agree, so both read the hold-out set from here rather than naming a system of their own.
    """
    return frozenset(
        system_id_for(name, route)
        for name, route in config.routes.items()
        if route.exclude_from_disagreement
    )


def _restore_script(
    session: Session,
    result: AsrResult,
    *,
    restore_route: str,
    routes: LlmRoutes,
    client: httpx.Client | None,
    dry_run: bool | None,
) -> AsrResult:
    """Put a single-script transcript back into mixed script, keeping every span (D41).

    The recogniser decided what was said and when; this only decides how it is spelled. One
    restored token per recognised token, so each word keeps the span the recogniser measured --
    no re-alignment, and `forced_align` stays off for this route (D33).

    The original Devanagari is kept in `metadata`, which becomes the hypothesis's
    `metadata_jsonb`. It is provenance only: it is never scored, never compared and never
    exported as a transcript.
    """
    if not result.words:
        return result

    tokens = [str(w["word"]) for w in result.words]
    if len(tokens) != len(result.text.split()):
        # The restored text is rebuilt from the spans, so a word list that does not cover the
        # transcript would silently truncate it. Vertex returns them 1:1; say so loudly if that
        # ever stops being true rather than shipping a short transcript.
        logger.warning(
            "restore_script_token_text_mismatch",
            route=restore_route,
            words=len(tokens),
            text_tokens=len(result.text.split()),
        )
    tokens_meta = {"script_restore_text_tokens": len(result.text.split())}
    restored, meta = restore_script(
        session, tokens, route=restore_route, config=routes, client=client, dry_run=dry_run
    )
    words = [{**word, "word": new} for word, new in zip(result.words, restored, strict=True)]
    return replace(
        result,
        text=" ".join(restored),
        words=words,
        metadata={
            **(result.metadata or {}),
            "text_devanagari": result.text,
            **tokens_meta,
            **meta,
        },
    )


def transcribe(
    session: Session,
    audio_path: Path | str,
    *,
    route: str,
    config: LlmRoutes | None = None,
    prompt: str | None = ASR_PROMPT,
    dry_run: bool | None = None,
    client: httpx.Client | None = None,
) -> AsrResult:
    """Transcribe one clip through a named route, whichever vendor is behind it.

    Args:
        session: Session the ``llm_requests`` row is written to.
        audio_path: The clip to transcribe.
        route: Route name from ``config/llm_routes.yaml``.
        config: Routing table override.
        prompt: Transcript policy, for the providers that accept one.
        dry_run: Override the configured dry-run mode for this call.
        client: Optional shared HTTPX client for connection pooling.

    Returns:
        The transcription, from whichever provider the route names.

    Raises:
        LlmRouteNotConfigured: The route does not exist.
        LlmDisabledError: Inference is disabled in configuration.
        LlmRequestFailed: No API key, or the request failed after retries.
    """
    routes = config or OpenRouterClient(session, client=client).config
    route_config = routes.routes.get(route)
    if route_config is None:
        raise LlmRouteNotConfigured(f"no route named {route!r} in llm_routes.yaml")

    if route_config.provider == "elevenlabs":
        return ElevenLabsClient(session, config=routes, client=client).transcribe(
            audio_path,
            route=route,
            keyterms=list(DEFAULT_KEYTERMS),
            dry_run=dry_run,
        )
    if route_config.provider == "vertex":
        # The dedicated recogniser takes no steering at all. Vertex answers a systemInstruction
        # with 400 "The input system_instruction is not supported.", a text part in `contents` is
        # accepted but ignored, and custom vocabulary is accepted and then silently costs the
        # route its speaker labels (D39). So it gets the audio, the language codes and nothing
        # else -- and, having no way to be told otherwise, writes English in Devanagari.
        dedicated = route_config.api == "transcription"
        result = VertexClient(session, config=routes, client=client).transcribe(
            audio_path,
            route=route,
            prompt=None if dedicated else prompt,
            dry_run=dry_run,
        )
        if route_config.restore_script_route:
            result = _restore_script(
                session,
                result,
                restore_route=route_config.restore_script_route,
                routes=routes,
                client=client,
                dry_run=dry_run,
            )
        return result
    return OpenRouterClient(session, config=routes, client=client).transcribe(
        audio_path,
        route=route,
        prompt=prompt,
        dry_run=dry_run,
    )
