"""Google AI Studio client for Gemini speech-to-text.

Google AI Studio is the third inference provider admitted to the harness (D29). Like OpenRouter
and ElevenLabs, calls are prepaid -- monitored and balance-capped -- to adhere to the prepaid
provider guarantee (invariant 5).

The route calls ``gemini-3.8-flash`` on the ordinary ``generateContent`` endpoint with the clip
inlined, not the Live API's dedicated transcription model (D31). Two consequences follow, and
both are deliberate:

* **No word timestamps.** ``generateContent`` returns text. Word spans for this system come
  from the local forced aligner (``app/services/forced_align.py``), which is why the model is
  never asked for them and never trusted with them.
* **The corpus prompt applies.** A general model obeys a prompt, so the transcript policy --
  Nepali in Devanagari, English in Latin, verbatim, no translation -- finally reaches Gemini.
  It is also a general model being asked to transcribe, so it may editorialise or hallucinate
  over silence; that is why it is one of three independent audio-only hypotheses and never sees
  the other two.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import LlmRoutes
from app.llm.base import (
    AsrResult,
    LlmDisabledError,
    LlmRequestFailed,
    LlmRouteNotConfigured,
    ProviderClient,
    dry_run_transcript,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Environment variables checked for the Google API key.
API_KEY_ENV = "GOOGLE_API_KEY"
FALLBACK_KEY_ENV = "GEMINI_API_KEY"

#: Clips are FLAC by invariant 6.
AUDIO_MIME_TYPE = "audio/flac"


def parse_transcript(body: dict[str, Any]) -> str:
    """Join the text parts of the first candidate that has any.

    A model may split its answer over several parts, and may return a candidate carrying only
    metadata (a safety block, a tool call); such a candidate yields no text and the next one is
    tried.
    """
    for candidate in body.get("candidates") or []:
        content = candidate.get("content") or {}
        texts = [
            str(part["text"])
            for part in content.get("parts") or []
            if isinstance(part, dict) and part.get("text")
        ]
        if texts:
            return "".join(texts).strip()
    return ""


class GoogleClient(ProviderClient):
    """A logged, retrying client for Google AI Studio's Gemini models."""

    def __init__(
        self,
        session: Session,
        *,
        config: LlmRoutes | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(session, config=config, client=client)
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get(API_KEY_ENV) or os.environ.get(FALLBACK_KEY_ENV, "")

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        route: str,
        prompt: str | None = None,
        language: str | None = None,
        dry_run: bool | None = None,
        timeout_seconds: float | None = None,
    ) -> AsrResult:
        """Transcribe one audio clip by asking Gemini to write down what it hears.

        Args:
            audio_path: Path to the audio file (16 kHz mono FLAC).
            route: Route name from ``config/llm_routes.yaml``.
            prompt: Transcript policy to steer the model. Unlike the Live API's transcription
                model, ``generateContent`` obeys one.
            language: BCP-47 language hint, overriding route configuration. Appended to the
                prompt, as ``generateContent`` has no language parameter of its own.
            dry_run: Override the configured dry-run mode.
            timeout_seconds: Override default timeout.

        Returns:
            The transcription result. ``words`` is always ``None`` on a real call: word spans
            for this system come from the forced aligner, not the model.

        Raises:
            LlmRouteNotConfigured: If the route is missing from configuration.
            LlmDisabledError: If inference is globally disabled.
            LlmRequestFailed: If the API key is absent or retries are exhausted.
        """
        route_config = self.config.routes.get(route)
        if route_config is None:
            raise LlmRouteNotConfigured(f"no route named {route!r} in llm_routes.yaml")

        audio_file = Path(audio_path)
        model_id = route_config.model
        request_hash = self._audio_hash(audio_file, model_id)
        summary = f"google_transcribe: {audio_file.name} model={model_id}"
        effective_dry_run = self.config.dry_run if dry_run is None else dry_run

        if effective_dry_run:
            self._log(
                route=route,
                model=model_id,
                request_hash=request_hash,
                input_summary=summary,
                status="dry_run",
            )
            logger.info("google_dry_run", route=route, model=model_id, file=str(audio_file))
            text, words = dry_run_transcript(request_hash)
            return AsrResult(route=route, model=model_id, text=text, words=words, dry_run=True)

        if not self.config.enabled:
            raise LlmDisabledError(
                "LLM inference is disabled (config/llm_routes.yaml: enabled: false)"
            )

        if not self.api_key:
            self._log(
                route=route,
                model=model_id,
                request_hash=request_hash,
                input_summary=summary,
                status="failed",
                error="GOOGLE_API_KEY is not set",
            )
            raise LlmRequestFailed("GOOGLE_API_KEY is not set")

        timeout = (
            timeout_seconds or route_config.timeout_seconds or self.config.default_timeout_seconds
        )
        target_lang = language or route_config.language

        url = f"{self.config.google_base_url.rstrip('/')}/models/{model_id}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        with open(audio_file, "rb") as handle:
            audio_b64 = base64.b64encode(handle.read()).decode("ascii")

        instruction = prompt or ""
        if target_lang:
            hint = f"Primary language code: {target_lang}."
            instruction = f"{instruction}\n\n{hint}".strip() if instruction else hint

        parts: list[dict[str, Any]] = []
        if instruction:
            parts.append({"text": instruction})
        parts.append({"inline_data": {"mime_type": AUDIO_MIME_TYPE, "data": audio_b64}})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": route_config.temperature or 0.0,
                "responseModalities": ["TEXT"],
            },
        }

        def send() -> httpx.Response:
            return self._get_client().post(url, json=payload, headers=headers, timeout=timeout)

        response, last_error, latency_ms = self._send_with_retries(send)
        if response is None:
            self._log(
                route=route,
                model=model_id,
                request_hash=request_hash,
                input_summary=summary,
                status="failed",
                latency_ms=latency_ms,
                error=last_error,
            )
            logger.info("google_request_failed", route=route, error=last_error)
            raise LlmRequestFailed(f"Google Gemini transcription failed: {last_error}")

        body = response.json()
        result = AsrResult(
            route=route,
            model=model_id,
            text=parse_transcript(body),
            words=None,
            latency_ms=latency_ms,
            raw=body,
        )
        self._log(
            route=route,
            model=model_id,
            request_hash=request_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            latency_ms=latency_ms,
        )
        return result
