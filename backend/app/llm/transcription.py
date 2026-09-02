"""Transcriber dispatch: one call site, several vendors.

The ingestion pipeline should not know which vendor is behind a route. It asks for a route name
and gets an :class:`AsrResult`; this module decides whether that means OpenRouter's transcription
endpoint, an OpenRouter chat model with the clip attached, or ElevenLabs Scribe.

The transcript policy lives here too. The corpus is code-switched, and the rule is that a word is
written in the script of the language it belongs to, so every model that can be told anything is
told both that the mixing is expected and which script each half takes. Scribe takes no prose
instruction at all, which is why it is given a language code and key terms instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import AsrResult, LlmRouteNotConfigured
from app.llm.elevenlabs import ElevenLabsClient
from app.llm.openrouter import OpenRouterClient

#: Sent with every ASR request that accepts a prompt. Two jobs: state the transcript policy, and
#: stop a chat model from answering in prose or narrating what it heard.
ASR_PROMPT = (
    "This is a Nepali-English code-switched conversation: the speakers mix both languages "
    "freely, often within a single sentence. Transcribe exactly what is said, and write every "
    "word in the script its own language uses -- Nepali words in Devanagari, English words in "
    "Latin. Do not translate between the two languages, and do not transliterate a word out of "
    "its own script. For example: आजको meeting मा हामीले नयाँ data हेर्यौं। "
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
    return OpenRouterClient(session, config=routes, client=client).transcribe(
        audio_path,
        route=route,
        prompt=prompt,
        dry_run=dry_run,
    )
