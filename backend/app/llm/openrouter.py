"""OpenRouter client.

**All text inference from this codebase goes through OpenRouter**, and so does every ASR model
OpenRouter can reach: one key, one bill and one client for the long tail of models nobody is going
to write a client for. The providers called directly are the ones OpenRouter cannot reach at all --
ElevenLabs Scribe (D21) and Vertex AI (D35) -- not exceptions bought with an argument about
billing (D34).

Two request shapes reach an ASR model here, and they are not interchangeable:

* ``api: transcription`` posts a multipart upload to ``/audio/transcriptions``. The model is a
  speech recogniser; it returns a transcript and nothing else.
* ``api: audio_chat`` posts chat completions with the clip as an ``input_audio`` part. The model
  is a general LLM being asked to transcribe, so it takes instruction well but will also answer in
  prose, or invent speech over silence, unless told not to.
"""

from __future__ import annotations

import base64
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

import httpx
from sqlalchemy.orm import Session

from app.config import LlmRoutes
from app.llm.base import (
    INPUT_SUMMARY_LIMIT,
    RETRYABLE_STATUS,
    AsrResult,
    LlmDisabledError,
    LlmError,
    LlmRequestFailed,
    LlmResult,
    LlmRouteNotConfigured,
    ProviderClient,
    dry_run_transcript,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "INPUT_SUMMARY_LIMIT",
    "RETRYABLE_STATUS",
    "AsrResult",
    "LlmDisabledError",
    "LlmError",
    "LlmRequestFailed",
    "LlmResult",
    "LlmRouteNotConfigured",
    "OpenRouterClient",
]

#: Audio container sent with an ``audio_chat`` request. Clips are FLAC by invariant.
AUDIO_CHAT_FORMAT = "flac"


def _usage_cost(usage: dict[str, Any]) -> Decimal | None:
    """Read the charged amount from an OpenRouter ``usage`` block.

    Chat completions report ``cost``; the transcription endpoint reports ``cost`` too, while some
    responses still carry the older ``total_cost``. Reading only one of the two silently logs
    every request as costing nothing.
    """
    for key in ("cost", "total_cost"):
        value = usage.get(key)
        if value is not None:
            return Decimal(str(value))
    return None


class OpenRouterClient(ProviderClient):
    """A thin, logged, retrying client for OpenRouter's chat and transcription endpoints."""

    def __init__(
        self,
        session: Session,
        *,
        config: LlmRoutes | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(session, config=config, client=client)
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "") if api_key is None else api_key

    @staticmethod
    def _summarize(messages: list[dict[str, Any]]) -> str:
        rendered = " | ".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        return rendered[:INPUT_SUMMARY_LIMIT]

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def complete(
        self,
        route: str,
        messages: list[dict[str, Any]],
        *,
        dry_run: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
    ) -> LlmResult:
        """Run a chat completion through a named route.

        Args:
            route: Route name from ``config/llm_routes.yaml``.
            messages: OpenAI-shaped chat messages.
            dry_run: Override the configured dry-run mode for this call.
            max_tokens: Override the route's token cap.
            temperature: Override the route's temperature.
            timeout_seconds: Override the route's timeout.

        Returns:
            The completion, or an empty :class:`LlmResult` marked ``dry_run``.

        Raises:
            LlmDisabledError: Inference is disabled in configuration.
            LlmRouteNotConfigured: The route does not exist.
            LlmRequestFailed: No API key, or the request failed after retries.
        """
        if not self.config.enabled:
            raise LlmDisabledError(
                "LLM inference is disabled (config/llm_routes.yaml: enabled: false)."
            )
        route_config = self.config.routes.get(route)
        if route_config is None:
            raise LlmRouteNotConfigured(f"no route named {route!r} in llm_routes.yaml")

        payload: dict[str, Any] = {
            "model": route_config.model,
            "messages": messages,
            "max_tokens": max_tokens or route_config.max_tokens or self.config.default_max_tokens,
        }
        temperature = temperature if temperature is not None else route_config.temperature
        if temperature is not None:
            payload["temperature"] = temperature

        request_hash = self._hash(payload)
        summary = self._summarize(messages)
        effective_dry_run = self.config.dry_run if dry_run is None else dry_run

        if effective_dry_run:
            self._log(
                route=route,
                model=route_config.model,
                request_hash=request_hash,
                input_summary=summary,
                status="dry_run",
            )
            logger.info("llm_dry_run", route=route, model=route_config.model)
            return LlmResult(route=route, model=route_config.model, text="", dry_run=True)

        if not self.api_key:
            self._log(
                route=route,
                model=route_config.model,
                request_hash=request_hash,
                input_summary=summary,
                status="failed",
                error="OPENROUTER_API_KEY is not set",
            )
            raise LlmRequestFailed("OPENROUTER_API_KEY is not set")

        timeout = (
            timeout_seconds or route_config.timeout_seconds or self.config.default_timeout_seconds
        )
        body, latency_ms = self._post_json(
            route=route,
            model=route_config.model,
            payload=payload,
            request_hash=request_hash,
            summary=summary,
            timeout=timeout,
        )
        usage = body.get("usage") or {}
        result = LlmResult(
            route=route,
            model=body.get("model", route_config.model),
            text=_message_text(body),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            estimated_cost_usd=_usage_cost(usage),
            latency_ms=latency_ms,
            raw=body,
        )
        self._log(
            route=route,
            model=result.model,
            request_hash=request_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost=result.estimated_cost_usd,
            latency_ms=latency_ms,
        )
        return result

    def _post_json(
        self,
        *,
        route: str,
        model: str,
        payload: dict[str, Any],
        request_hash: str,
        summary: str,
        timeout: float,
    ) -> tuple[dict[str, Any], int]:
        """POST a JSON body to chat completions, logging and raising on exhausted retries."""
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {**self._headers(), "Content-Type": "application/json"}
        response, last_error, latency_ms = self._send_with_retries(
            lambda: self._get_client().post(url, json=payload, headers=headers, timeout=timeout)
        )
        if response is None:
            self._log(
                route=route,
                model=model,
                request_hash=request_hash,
                input_summary=summary,
                status="failed",
                latency_ms=latency_ms,
                error=last_error,
            )
            logger.info("llm_request_failed", route=route, error=last_error)
            raise LlmRequestFailed(f"route {route!r} failed: {last_error}")
        return response.json(), latency_ms

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        route: str = "asr",
        prompt: str | None = None,
        language: str | None = None,
        dry_run: bool | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AsrResult:
        """Transcribe an audio file through a named route.

        Dispatches on the route's ``api``: ``audio_chat`` sends the clip to a chat model, anything
        else sends it to the transcription endpoint.

        Always logs the attempt to ``llm_requests``. An explicit dry run returns a deterministic
        mock transcript marked ``dry_run``; a missing key or a disabled configuration raises,
        because a fabricated transcript written into the corpus as a real hypothesis is far worse
        than a failed ingest.

        Raises:
            LlmDisabledError: Inference is disabled in configuration.
            LlmRequestFailed: No API key, or the request failed after retries.
        """
        route_config = self.config.routes.get(route)
        target_model = model or (route_config.model if route_config else "google/gemini-3.8-flash")
        effective_dry_run = self.config.dry_run if dry_run is None else dry_run

        audio_file = Path(audio_path)
        request_hash = self._audio_hash(audio_file, target_model)
        summary = f"asr_transcribe: {audio_file.name} model={target_model}"

        if effective_dry_run:
            self._log(
                route=route,
                model=target_model,
                request_hash=request_hash,
                input_summary=summary,
                status="dry_run",
            )
            logger.info("asr_dry_run", route=route, model=target_model, file=str(audio_file))
            text, words = dry_run_transcript(request_hash)
            return AsrResult(route=route, model=target_model, text=text, words=words, dry_run=True)

        if not self.config.enabled:
            raise LlmDisabledError(
                "LLM inference is disabled (config/llm_routes.yaml: enabled: false), "
                "so no transcription can be produced."
            )
        if not self.api_key:
            self._log(
                route=route,
                model=target_model,
                request_hash=request_hash,
                input_summary=summary,
                status="failed",
                error="OPENROUTER_API_KEY is not set",
            )
            raise LlmRequestFailed("OPENROUTER_API_KEY is not set")

        timeout = (
            timeout_seconds
            or (route_config.timeout_seconds if route_config else None)
            or self.config.default_timeout_seconds
        )
        language = (
            language if language is not None else (route_config.language if route_config else None)
        )

        if route_config is not None and route_config.api == "audio_chat":
            return self._transcribe_via_chat(
                audio_file,
                route=route,
                route_config=route_config,
                model=target_model,
                prompt=prompt,
                request_hash=request_hash,
                summary=summary,
                timeout=timeout,
            )
        return self._transcribe_via_endpoint(
            audio_file,
            route=route,
            model=target_model,
            prompt=prompt,
            language=language,
            request_hash=request_hash,
            summary=summary,
            timeout=timeout,
        )

    def _transcribe_via_endpoint(
        self,
        audio_file: Path,
        *,
        route: str,
        model: str,
        prompt: str | None,
        language: str | None,
        request_hash: str,
        summary: str,
        timeout: float,
    ) -> AsrResult:
        """Transcribe through ``/audio/transcriptions``, a dedicated speech recogniser."""
        url = f"{self.config.base_url.rstrip('/')}/audio/transcriptions"

        def send() -> httpx.Response:
            with open(audio_file, "rb") as handle:
                files = {"file": (audio_file.name, handle, "audio/flac")}
                # verbose_json is the only shape that reports the detected language and the
                # segment spans; plain json returns the text alone.
                data = {
                    "model": model,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                }
                if prompt:
                    data["prompt"] = prompt
                if language:
                    data["language"] = language
                return self._get_client().post(
                    url, data=data, files=files, headers=self._headers(), timeout=timeout
                )

        response, last_error, latency_ms = self._send_with_retries(send)
        if response is None:
            self._fail(route, model, request_hash, summary, latency_ms, last_error)

        body = response.json()
        raw_words = body.get("words") or []
        result = AsrResult(
            route=route,
            model=body.get("model", model),
            # Some recognisers prefix transcripts with a space. Stored verbatim it would show up in
            # the editor and shift every character-level diff against it.
            text=body.get("text", "").strip(),
            # Not every recogniser returns word spans; an empty list would claim the model found no
            # words, which is a different statement.
            words=[
                {"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end")}
                for w in raw_words
            ]
            or None,
            # Absent means unknown, not confident. The scorer reads None as "no signal"; a
            # plausible-looking default would silently drive 25% of the queue ordering.
            avg_logprob=body.get("avg_logprob"),
            no_speech_prob=body.get("no_speech_prob"),
            latency_ms=latency_ms,
            estimated_cost_usd=_usage_cost(body.get("usage") or {}),
            raw=body,
        )
        self._log(
            route=route,
            model=result.model,
            request_hash=request_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            cost=result.estimated_cost_usd,
            latency_ms=latency_ms,
        )
        return result

    def _transcribe_via_chat(
        self,
        audio_file: Path,
        *,
        route: str,
        route_config: Any,
        model: str,
        prompt: str | None,
        request_hash: str,
        summary: str,
        timeout: float,
    ) -> AsrResult:
        """Transcribe by attaching the clip to a chat completion.

        The model is a general LLM, so the prompt carries the whole transcript policy and the
        instruction to answer with the transcript alone.
        """
        encoded = base64.b64encode(audio_file.read_bytes()).decode()
        content: list[dict[str, Any]] = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": AUDIO_CHAT_FORMAT},
            }
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": route_config.max_tokens or self.config.default_max_tokens,
        }
        # Transcription is not a creative task: pin the sampler when the route does not.
        payload["temperature"] = (
            route_config.temperature if route_config.temperature is not None else 0.0
        )

        body, latency_ms = self._post_json(
            route=route,
            model=model,
            payload=payload,
            request_hash=request_hash,
            summary=summary,
            timeout=timeout,
        )
        usage = body.get("usage") or {}
        result = AsrResult(
            route=route,
            model=body.get("model", model),
            text=_message_text(body).strip(),
            # A chat model returns prose, not word spans or confidences.
            words=None,
            avg_logprob=None,
            no_speech_prob=None,
            latency_ms=latency_ms,
            estimated_cost_usd=_usage_cost(usage),
            raw=body,
        )
        self._log(
            route=route,
            model=result.model,
            request_hash=request_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost=result.estimated_cost_usd,
            latency_ms=latency_ms,
        )
        return result

    def _fail(
        self,
        route: str,
        model: str,
        request_hash: str,
        summary: str,
        latency_ms: int,
        last_error: str,
    ) -> NoReturn:
        self._log(
            route=route,
            model=model,
            request_hash=request_hash,
            input_summary=summary,
            status="failed",
            latency_ms=latency_ms,
            error=last_error,
        )
        logger.info("asr_request_failed", route=route, error=last_error)
        raise LlmRequestFailed(f"ASR transcription failed: {last_error}")


def _message_text(body: dict[str, Any]) -> str:
    """Pull the assistant message out of a chat completion, tolerating a null content."""
    choices = body.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""
