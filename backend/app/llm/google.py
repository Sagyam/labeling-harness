"""Google AI Studio / Gemini API client for speech-to-text models.

Google serves Gemini models directly through the Gemini Developer API (ai.google.dev),
authenticated with a single API key (GEMINI_API_KEY or GOOGLE_API_KEY).

Two request shapes reach a model here, matching the route shapes in ``config.py``:

* ``api: transcription`` posts to ``POST /v1beta/interactions`` with a ``transcription_config``.
  The model is a dedicated speech recogniser -- ``gemini-3.5-transcribe`` -- and returns
  verbatim text, word timings, and a speaker label per word.
* ``api: audio_chat`` posts to ``POST /v1beta/models/{model}:generateContent`` with the clip
  inlined. The model is a general LLM being asked to transcribe: it obeys the transcript policy,
  returns text and no timings, and may editorialise or hallucinate over silence. Spans for this
  shape come from the local forced aligner (D32).
"""

from __future__ import annotations

import base64
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

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
from app.llm.cost import calculate_gemini_cost, get_audio_duration
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Environment variables consulted for the Gemini API key.
PRIMARY_API_KEY_ENV = "GEMINI_API_KEY"
FALLBACK_API_KEY_ENV = "GOOGLE_API_KEY"

#: Clips are FLAC by invariant 6.
AUDIO_MIME_TYPE = "audio/flac"

#: Harm categories to turn off for speech recognition.
#:
#: A transcriber must not decline to write down what was said. Safety thresholds are set to OFF
#: so that audio is not blocked by prompt filters.
_HARM_CATEGORIES = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
)
SAFETY_SETTINGS = [{"category": c, "threshold": "OFF"} for c in _HARM_CATEGORIES]


def parse_generate_content(body: dict[str, Any]) -> str:
    """Join the text parts of the first ``generateContent`` candidate that has any."""
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


def _duration_seconds(value: Any) -> float | None:
    """Read a duration (``"1.75s"`` or numeric seconds) as float. Absent/unparseable means None."""
    if value is None:
        return None
    try:
        return float(str(value).rstrip("s"))
    except ValueError:
        return None


def parse_interaction(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
    """Read the transcript and word annotations out of an interaction response.

    Returns:
        ``(text, words)``. ``words`` is ``None`` when the response carried no word annotations,
        never an empty list -- the scorer reads ``None`` as "no signal".
    """
    texts: list[str] = []
    words: list[dict[str, Any]] = []

    for step in body.get("steps") or []:
        if not isinstance(step, dict):
            continue
        output = step.get("model_output") or step.get("modelOutput") or step
        content = output.get("content") or []
        if isinstance(content, dict):
            content = [content]

        for block in content:
            if not isinstance(block, dict):
                continue
            # Text block extraction
            t = block.get("text")
            if isinstance(t, str):
                texts.append(t)
            elif isinstance(t, dict) and t.get("text"):
                texts.append(str(t["text"]))

            # Word annotations extraction
            annotations = block.get("annotations")
            if annotations is None and isinstance(t, dict):
                annotations = t.get("annotations")
            for annotation in annotations or []:
                if not isinstance(annotation, dict):
                    continue
                info = annotation.get("word_info") or annotation.get("wordInfo") or annotation
                w_text = info.get("text")
                if not w_text:
                    continue
                start_raw = info.get("start_offset") or info.get("startOffset") or info.get("start")
                end_raw = info.get("end_offset") or info.get("endOffset") or info.get("end")
                words.append(
                    {
                        "word": str(w_text),
                        "start": _duration_seconds(start_raw),
                        "end": _duration_seconds(end_raw),
                        "speaker": info.get("speaker"),
                    }
                )

    full_text = "".join(texts).strip()
    if not full_text and body.get("output_text"):
        full_text = str(body["output_text"]).strip()

    return full_text, words or None


class GoogleClient(ProviderClient):
    """A logged, retrying client for Google Gemini models via Gemini Developer API."""

    def __init__(
        self,
        session: Session,
        *,
        config: LlmRoutes | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(session, config=config, client=client)
        self.api_key = (
            api_key
            or os.environ.get(PRIMARY_API_KEY_ENV)
            or os.environ.get(FALLBACK_API_KEY_ENV)
            or ""
        )
        self.base_url = (
            self.config.google_base_url.rstrip("/")
            if hasattr(self.config, "google_base_url") and self.config.google_base_url
            else "https://generativelanguage.googleapis.com/v1beta"
        )

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        route: str,
        prompt: str | None = None,
        language: str | None = None,
        custom_vocabulary: list[str] | None = None,
        dry_run: bool | None = None,
        timeout_seconds: float | None = None,
    ) -> AsrResult:
        """Transcribe one clip with whichever Gemini model and API shape the route names.

        Args:
            audio_path: Path to the audio file (16 kHz mono FLAC).
            route: Route name from ``config/llm_routes.yaml``.
            prompt: Transcript policy. Both shapes take one -- ``audio_chat`` as prompt part,
                ``transcription`` as interaction system_instruction.
            language: BCP-47 language hint overriding route config.
            custom_vocabulary: Ignored when diarization/timestamps are requested to prevent
                API 400 rejection.
            dry_run: Override configured dry-run mode.
            timeout_seconds: Override default timeout.

        Returns:
            The transcription. ``words`` is populated only on an ``api: transcription`` route;
            ``audio_chat`` returns text, and its spans come from the forced aligner (D32).
        """
        route_config = self.config.routes.get(route)
        if route_config is None:
            raise LlmRouteNotConfigured(f"no route named {route!r} in llm_routes.yaml")

        audio_file = Path(audio_path)
        model_id = route_config.model
        request_hash = self._audio_hash(audio_file, model_id)
        summary = f"google_transcribe: {audio_file.name} model={model_id} api={route_config.api}"
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
            logger.info("google_dry_run", route=route, model=model_id, file=str(audio_file))
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
                "LLM inference is disabled (config/llm_routes.yaml: enabled: false)"
            )

        if not self.api_key:
            self._fail(
                route,
                model_id,
                request_hash,
                summary,
                f"no Gemini API key: set {PRIMARY_API_KEY_ENV} in .env or environment",
            )

        timeout = (
            timeout_seconds or route_config.timeout_seconds or self.config.default_timeout_seconds
        )
        audio_b64 = base64.b64encode(audio_file.read_bytes()).decode("ascii")
        wants_words = route_config.api == "transcription"

        if wants_words:
            url = f"{self.base_url}/interactions"
            payload = self._interaction_payload(
                route_config,
                audio_b64=audio_b64,
                instruction=prompt,
                language=language,
            )
        else:
            url = f"{self.base_url}/models/{model_id}:generateContent"
            payload = self._generate_content_payload(
                route_config,
                audio_b64=audio_b64,
                instruction=prompt,
                language=language,
            )

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
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
            raise LlmRequestFailed(f"Gemini transcription failed: {last_error}")

        body = response.json()
        if wants_words:
            text, words = parse_interaction(body)
        else:
            text, words = parse_generate_content(body), None

        block_reason = (body.get("promptFeedback") or {}).get("blockReason")
        if block_reason and not text:
            logger.warning(
                "google_response_blocked", route=route, model=model_id, reason=block_reason
            )

        usage = body.get("usageMetadata") or body.get("usage_metadata") or {}
        prompt_tokens = usage.get("promptTokenCount") or usage.get("prompt_token_count")
        completion_tokens = usage.get("candidatesTokenCount") or usage.get("candidates_token_count")
        duration = get_audio_duration(audio_file)
        cost = calculate_gemini_cost(
            model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_seconds=duration,
        )

        self._log(
            route=route,
            model=model_id,
            request_hash=request_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
        )
        return AsrResult(
            route=route,
            model=model_id,
            text=text,
            words=words,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            raw=body,
        )

    def _fail(
        self, route: str, model: str, request_hash: str, summary: str, message: str
    ) -> NoReturn:
        """Log a request that never left the process, then raise. Never returns."""
        self._log(
            route=route,
            model=model,
            request_hash=request_hash,
            input_summary=summary,
            status="failed",
            error=message,
        )
        raise LlmRequestFailed(message)

    @staticmethod
    def _interaction_payload(
        route_config: LlmRoute,
        *,
        audio_b64: str,
        instruction: str | None,
        language: str | None,
    ) -> dict[str, Any]:
        """Build one flat ``POST /interactions`` body for a dedicated transcription model.

        Word timestamps and speaker diarization are configured within verbatim mode.
        custom_vocabulary is omitted because the API rejects combining custom vocabulary with
        diarization or word timestamps.
        """
        mode_config: dict[str, Any] = {
            "type": "verbatim",
            "diarization_mode": "speaker",
            "timestamp_granularities": ["word"],
        }
        transcription_config: dict[str, Any] = {"mode": mode_config}

        codes = list(route_config.language_codes)
        if not codes:
            hint = language or route_config.language
            codes = [hint] if hint else []
        if codes:
            transcription_config["language_codes"] = codes

        generation_config: dict[str, Any] = {"transcription_config": transcription_config}
        if route_config.temperature is not None:
            generation_config["temperature"] = route_config.temperature

        payload: dict[str, Any] = {
            "model": route_config.model,
            "input": [
                {
                    "type": "audio",
                    "data": audio_b64,
                    "mime_type": AUDIO_MIME_TYPE,
                }
            ],
            "generation_config": generation_config,
        }
        if instruction:
            payload["system_instruction"] = instruction

        return payload

    @staticmethod
    def _generate_content_payload(
        route_config: LlmRoute,
        *,
        audio_b64: str,
        instruction: str | None,
        language: str | None,
    ) -> dict[str, Any]:
        """Build one ``generateContent`` body for a general model asked to transcribe."""
        target_lang = language or route_config.language
        text = instruction or ""
        if target_lang:
            hint = f"Primary language code: {target_lang}."
            text = f"{text}\n\n{hint}".strip() if text else hint

        parts: list[dict[str, Any]] = []
        if text:
            parts.append({"text": text})
        parts.append({"inline_data": {"mime_type": AUDIO_MIME_TYPE, "data": audio_b64}})

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": route_config.temperature or 0.0,
                "responseModalities": ["TEXT"],
            },
            "safetySettings": SAFETY_SETTINGS,
        }
