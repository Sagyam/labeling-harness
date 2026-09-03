"""Google AI Studio client for Gemini speech-to-text models.

Google AI Studio is the third inference provider admitted to the harness (decision D29). Like
OpenRouter and ElevenLabs, calls are prepaid (monitored and balance-capped) to adhere to the
prepaid provider guarantee (invariant 5).

Gemini 3.5 Transcribe (``gemini-3.5-transcribe``) provides speech recognition with automatic
code-switching detection, verbatim mode, and word-level timestamps via the Interactions API.
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


def _parse_word_annotations(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract word-level annotations from the Interactions API response."""
    words: list[dict[str, Any]] = []
    for step in body.get("steps") or []:
        for content in step.get("content") or []:
            for annotation in content.get("annotations") or []:
                if annotation.get("type") == "word_info":
                    raw_text = annotation.get("text", "")
                    start_str = annotation.get("start_offset", "")
                    end_str = annotation.get("end_offset", "")
                    try:
                        start_sec = (
                            float(str(start_str).rstrip("s"))
                            if start_str is not None and str(start_str).strip()
                            else 0.0
                        )
                        end_sec = (
                            float(str(end_str).rstrip("s"))
                            if end_str is not None and str(end_str).strip()
                            else 0.0
                        )
                    except ValueError:
                        start_sec = 0.0
                        end_sec = 0.0
                    words.append({"word": raw_text, "start": start_sec, "end": end_sec})
    return words


class GoogleClient(ProviderClient):
    """A logged, retrying client for Google AI Studio's Gemini transcription API."""

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
        language: str | None = None,
        dry_run: bool | None = None,
        timeout_seconds: float | None = None,
    ) -> AsrResult:
        """Transcribe one audio clip with Gemini Transcribe in verbatim mode.

        Args:
            audio_path: Path to the audio file (16 kHz mono FLAC).
            route: Route name from ``config/llm_routes.yaml``.
            language: BCP-47 language hint, overriding route configuration.
            dry_run: Override the configured dry-run mode.
            timeout_seconds: Override default timeout.

        Returns:
            The ASR transcription result, with verbatim text and word timestamps.

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

        url = f"{self.config.google_base_url.rstrip('/')}/interactions"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        with open(audio_file, "rb") as handle:
            audio_bytes = handle.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        transcription_config: dict[str, Any] = {
            "mode": {
                "type": "verbatim",
                "timestamp_granularities": ["word"],
            }
        }
        if target_lang:
            transcription_config["language_codes"] = [target_lang]

        payload = {
            "model": model_id,
            "input": [
                {
                    "type": "audio",
                    "data": audio_b64,
                    "mime_type": "audio/flac",
                }
            ],
            "generation_config": {
                "transcription_config": transcription_config,
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
        output_text = body.get("output_text")
        if not output_text:
            for step in body.get("steps") or []:
                for content in step.get("content") or []:
                    if content.get("type") == "text" and "text" in content:
                        output_text = content["text"]
                        break
                if output_text:
                    break

        transcript_text = (output_text or "").strip()
        words = _parse_word_annotations(body)

        result = AsrResult(
            route=route,
            model=model_id,
            text=transcript_text,
            words=words if words else None,
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
