"""ElevenLabs Scribe speech-to-text client.

Called directly rather than through OpenRouter, because OpenRouter cannot reach Scribe at all --
and Scribe is the only transcriber the harness has that returns word spans *and* per-word log
probabilities (decision D21). Every call still lands in ``llm_requests``, so the billing audit
trail has no gap in it.

Scribe has no free-text prompt parameter -- unlike a chat model, it cannot be told in prose that
the audio is code-switched. It offered two levers and now uses one: ``language_code``, from the
route configuration. ``keyterms`` was measured and dropped (D48) -- it raised script violations
rather than lowering them, cost the route its best agreement with the other systems, and carries
a $0.05/hr surcharge. What remains is the language code.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import (
    AsrResult,
    LlmDisabledError,
    LlmRequestFailed,
    LlmRouteNotConfigured,
    ProviderClient,
    dry_run_transcript,
)
from app.llm.cost import calculate_elevenlabs_cost, get_audio_duration
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Environment variable holding the Scribe key.
API_KEY_ENV = "ELEVEN_LABS_API_KEY"
#: Word entries Scribe emits that are not words. Keeping them would inflate every word count and
#: every disagreement rate the queue is ordered by.
NON_WORD_TYPES = frozenset({"spacing", "audio_event"})


class ElevenLabsClient(ProviderClient):
    """A thin, logged, retrying client for the ElevenLabs speech-to-text endpoint."""

    def __init__(
        self,
        session: Session,
        *,
        config: LlmRoutes | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(session, config=config, client=client)
        self.api_key = os.environ.get(API_KEY_ENV, "") if api_key is None else api_key

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        route: str,
        language: str | None = None,
        dry_run: bool | None = None,
        timeout_seconds: float | None = None,
    ) -> AsrResult:
        """Transcribe one clip with Scribe.

        Args:
            audio_path: The clip to transcribe.
            route: Route name from ``config/llm_routes.yaml``.
            language: ISO language hint, overriding the route's.
            dry_run: Override the configured dry-run mode for this call.
            timeout_seconds: Override the route's timeout.

        Returns:
            The transcription, or a deterministic mock marked ``dry_run``.

        Raises:
            LlmRouteNotConfigured: The route does not exist.
            LlmDisabledError: Inference is disabled in configuration.
            LlmRequestFailed: No API key, or the request failed after retries.
        """
        route_config = self.config.routes.get(route)
        if route_config is None:
            raise LlmRouteNotConfigured(f"no route named {route!r} in llm_routes.yaml")

        audio_file = Path(audio_path)
        model_id = route_config.model
        request_hash = self._audio_hash(audio_file, model_id)
        summary = f"elevenlabs_transcribe: {audio_file.name} model={model_id}"
        effective_dry_run = self.config.dry_run if dry_run is None else dry_run

        if effective_dry_run:
            self._log(
                route=route,
                model=model_id,
                request_hash=request_hash,
                input_summary=summary,
                status="dry_run",
                cost=Decimal("0.0"),
            )
            logger.info("asr_dry_run", route=route, model=model_id, file=str(audio_file))
            text, words = dry_run_transcript(request_hash)
            return AsrResult(
                route=route,
                model=model_id,
                text=text,
                words=words,
                dry_run=True,
                estimated_cost_usd=Decimal("0.0"),
            )

        if not self.config.enabled:
            raise LlmDisabledError(
                "Inference is disabled (config/llm_routes.yaml: enabled: false), "
                "so no transcription can be produced."
            )
        if not self.api_key:
            self._log(
                route=route,
                model=model_id,
                request_hash=request_hash,
                input_summary=summary,
                status="failed",
                error=f"{API_KEY_ENV} is not set",
            )
            raise LlmRequestFailed(f"{API_KEY_ENV} is not set")

        timeout = (
            timeout_seconds or route_config.timeout_seconds or self.config.default_timeout_seconds
        )
        url = f"{self.config.elevenlabs_base_url.rstrip('/')}/speech-to-text"
        headers = {"xi-api-key": self.api_key}
        form = self._form_fields(route_config, language=language)

        def send() -> httpx.Response:
            with open(audio_file, "rb") as handle:
                return self._get_client().post(
                    url,
                    data=form,
                    files={"file": (audio_file.name, handle, "audio/flac")},
                    headers=headers,
                    timeout=timeout,
                )

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
            logger.info("asr_request_failed", route=route, error=last_error)
            raise LlmRequestFailed(f"Scribe transcription failed: {last_error}")

        body = response.json()
        words = _parse_words(body.get("words") or [])
        duration = get_audio_duration(audio_file)
        if duration is None and words:
            duration = max((w.get("end") or 0.0 for w in words), default=None)
        cost = calculate_elevenlabs_cost(duration or 0.0)
        result = AsrResult(
            route=route,
            model=model_id,
            text=body.get("text", ""),
            words=words,
            avg_logprob=_mean_logprob(body.get("words") or []),
            # Scribe reports no speech-absence probability. Absent means unknown, not confident.
            no_speech_prob=None,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            raw=body,
        )
        self._log(
            route=route,
            model=model_id,
            request_hash=request_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            cost=cost,
            latency_ms=latency_ms,
        )
        return result

    @staticmethod
    def _form_fields(
        route_config: LlmRoute,
        *,
        language: str | None,
    ) -> dict[str, Any]:
        """Build the multipart fields for one Scribe request."""
        form: dict[str, Any] = {
            "model_id": route_config.model,
            "timestamps_granularity": "word",
            # One speaker per clip by construction, and audio-event tags are not transcript text.
            "diarize": "false",
            "tag_audio_events": "false",
        }
        hint = language if language is not None else route_config.language
        if hint:
            form["language_code"] = hint
        if route_config.temperature is not None:
            form["temperature"] = str(route_config.temperature)
        return form


def _parse_words(raw_words: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Normalize Scribe's word entries onto the manifest's word shape.

    Scribe interleaves ``spacing`` and ``audio_event`` entries with the real words; only ``word``
    entries carry transcript text.
    """
    words = [
        {"word": w.get("text", ""), "start": w.get("start"), "end": w.get("end")}
        for w in raw_words
        if w.get("type", "word") not in NON_WORD_TYPES
    ]
    return words or None


def _mean_logprob(raw_words: list[dict[str, Any]]) -> float | None:
    """Average the per-word log probabilities Scribe reports, if it reported any.

    This is the only confidence signal any configured transcriber returns, and it feeds the
    ``low_confidence`` term of the priority score. When no word carries one the result is ``None``,
    never a default.
    """
    logprobs = [
        w["logprob"]
        for w in raw_words
        if w.get("type", "word") not in NON_WORD_TYPES and isinstance(w.get("logprob"), int | float)
    ]
    if not logprobs:
        return None
    return round(sum(logprobs) / len(logprobs), 6)
