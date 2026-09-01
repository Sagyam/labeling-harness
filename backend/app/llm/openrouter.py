"""OpenRouter client.

**All LLM inference from this codebase goes through OpenRouter.** No direct calls to OpenAI,
Anthropic, Google, Groq or Mistral. The reason is billing control: OpenRouter is prepaid, so there
is no possibility of a surprise invoice.

At MVP no route is wired to any pipeline stage -- ``config/llm_routes.yaml`` ships with
``enabled: false`` and ``routes: {}``. This client exists now so that when a route is added later,
the billing, logging, retry and dry-run guarantees are already in place rather than being
retrofitted around a call that is already in production.

(Upstream ASR, including any commercial transcription API, runs in the GPU pipeline outside this
codebase and is out of scope for this rule.)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import LlmRoutes, load_llm_routes
from app.models import LlmRequest
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: How much of the prompt is kept in the request log.
INPUT_SUMMARY_LIMIT = 1000
#: Status codes worth retrying: rate limits and transient server failures.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LlmError(RuntimeError):
    """Base class for OpenRouter failures."""


class LlmDisabledError(LlmError):
    """LLM inference is switched off in configuration."""


class LlmRouteNotConfigured(LlmError):
    """The named route does not exist in ``llm_routes.yaml``."""


class LlmRequestFailed(LlmError):
    """The request failed after exhausting retries, or could not be made at all."""


@dataclass(frozen=True)
class LlmResult:
    """One completion, plus what it cost."""

    route: str
    model: str
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    latency_ms: int | None = None
    dry_run: bool = False
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class AsrResult:
    """One ASR transcription, plus usage and cost metrics."""

    route: str
    model: str
    text: str
    words: list[dict[str, Any]] | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    latency_ms: int | None = None
    estimated_cost_usd: Decimal | None = None
    dry_run: bool = False
    raw: dict[str, Any] | None = None


class OpenRouterClient:
    """A thin, logged, retrying client for OpenRouter's chat completions endpoint."""

    def __init__(
        self,
        session: Session,
        *,
        config: LlmRoutes | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.session = session
        self.config = config or load_llm_routes()
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "") if api_key is None else api_key
        self._client = client

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.config.default_timeout_seconds)
        return self._client

    @staticmethod
    def _hash(payload: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    @staticmethod
    def _summarize(messages: list[dict[str, Any]]) -> str:
        rendered = " | ".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        return rendered[:INPUT_SUMMARY_LIMIT]

    def _log(
        self,
        *,
        route: str,
        model: str,
        request_hash: str,
        input_summary: str,
        status: str,
        output: dict[str, Any] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost: Decimal | None = None,
        latency_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        self.session.add(
            LlmRequest(
                route=route,
                model=model,
                request_hash=request_hash,
                input_summary=input_summary,
                output_json=output,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=cost,
                latency_ms=latency_ms,
                status=status,
                error_message=error,
            )
        )
        self.session.flush()

    def _send_with_retries(
        self, send: Callable[[], httpx.Response]
    ) -> tuple[httpx.Response | None, str, int]:
        """Call ``send`` until it returns 200, fails unretryably, or runs out of attempts.

        Args:
            send: Builds and performs one request. Called once per attempt.

        Returns:
            ``(response, last_error, latency_ms)``. ``response`` is ``None`` when every attempt
            failed, in which case ``last_error`` describes the final one.
        """
        attempts = max(1, self.config.max_retries)
        started = time.monotonic()
        last_error = "no attempt was made"
        for attempt in range(attempts):
            try:
                response = send()
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response, last_error, int((time.monotonic() - started) * 1000)
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in RETRYABLE_STATUS:
                    break
            if attempt + 1 < attempts:
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        return None, last_error, int((time.monotonic() - started) * 1000)

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
            LlmDisabledError: LLM inference is disabled in configuration.
            LlmRouteNotConfigured: The route does not exist.
            LlmRequestFailed: No API key, or the request failed after retries.
        """
        if not self.config.enabled:
            raise LlmDisabledError(
                "LLM inference is disabled (config/llm_routes.yaml: enabled: false). "
                "No route is wired to any pipeline stage at MVP."
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
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response, last_error, latency_ms = self._send_with_retries(
            lambda: self._get_client().post(url, json=payload, headers=headers, timeout=timeout)
        )
        if response is None:
            self._log(
                route=route,
                model=route_config.model,
                request_hash=request_hash,
                input_summary=summary,
                status="failed",
                latency_ms=latency_ms,
                error=last_error,
            )
            logger.info("llm_request_failed", route=route, error=last_error)
            raise LlmRequestFailed(f"route {route!r} failed: {last_error}")

        body = response.json()
        usage = body.get("usage") or {}
        cost = usage.get("total_cost")
        result = LlmResult(
            route=route,
            model=body.get("model", route_config.model),
            text=(body.get("choices") or [{}])[0].get("message", {}).get("content", ""),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            estimated_cost_usd=Decimal(str(cost)) if cost is not None else None,
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
        """Transcribe an audio file using Cloud ASR via OpenRouter.

        Always logs the attempt to ``llm_requests``. An explicit dry run returns a deterministic
        mock transcript marked ``dry_run``; a missing key or a disabled configuration raises,
        because a fabricated transcript written into the corpus as a real hypothesis is far worse
        than a failed ingest.

        Raises:
            LlmDisabledError: LLM inference is disabled in configuration.
            LlmRequestFailed: No API key, or the request failed after retries.
        """
        route_config = self.config.routes.get(route)
        target_model = model or (route_config.model if route_config else "openai/whisper-large-v3")
        effective_dry_run = self.config.dry_run if dry_run is None else dry_run

        audio_file = Path(audio_path)
        file_size = audio_file.stat().st_size if audio_file.exists() else 0
        hash_seed = f"{target_model}:{audio_file.name}:{file_size}".encode()
        payload_hash = hashlib.sha256(hash_seed).hexdigest()
        summary = f"asr_transcribe: {audio_file.name} model={target_model}"

        if effective_dry_run:
            self._log(
                route=route,
                model=target_model,
                request_hash=payload_hash,
                input_summary=summary,
                status="dry_run",
            )
            logger.info("asr_dry_run", route=route, model=target_model, file=str(audio_file))
            # Deterministic mock, chosen by file hash. Callers must key off `dry_run` and never
            # store this as a real hypothesis.
            sample_texts = [
                "हामीले यो project मा meeting गरेर data analyse गर्नु पर्छ",
                "आजको session मा machine learning र technology को कुरा भयो",
                "सबै team members ले आफ्नो schedule अनुसार task complete गर्नु होला",
                "यो software system मा नयाँ feature update add गरिएको छ",
            ]
            chosen = sample_texts[int(payload_hash[:4], 16) % len(sample_texts)]
            words = [
                {"word": w, "start": round(i * 0.4, 2), "end": round((i + 1) * 0.4, 2)}
                for i, w in enumerate(chosen.split())
            ]
            return AsrResult(
                route=route,
                model=target_model,
                text=chosen,
                words=words,
                dry_run=True,
            )

        if not self.config.enabled:
            raise LlmDisabledError(
                "LLM inference is disabled (config/llm_routes.yaml: enabled: false), "
                "so no transcription can be produced."
            )
        if not self.api_key:
            self._log(
                route=route,
                model=target_model,
                request_hash=payload_hash,
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
        url = f"{self.config.base_url.rstrip('/')}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        def send() -> httpx.Response:
            with open(audio_file, "rb") as handle:
                files = {"file": (audio_file.name, handle, "audio/flac")}
                data = {"model": target_model}
                if prompt:
                    data["prompt"] = prompt
                if language:
                    data["language"] = language
                return self._get_client().post(
                    url, data=data, files=files, headers=headers, timeout=timeout
                )

        response, last_error, latency_ms = self._send_with_retries(send)
        if response is None:
            self._log(
                route=route,
                model=target_model,
                request_hash=payload_hash,
                input_summary=summary,
                status="failed",
                latency_ms=latency_ms,
                error=last_error,
            )
            logger.info("asr_request_failed", route=route, error=last_error)
            raise LlmRequestFailed(f"ASR transcription failed: {last_error}")

        body = response.json()
        raw_words = body.get("words") or []
        usage = body.get("usage") or {}
        cost = usage.get("total_cost")
        result = AsrResult(
            route=route,
            model=body.get("model", target_model),
            text=body.get("text", ""),
            words=[
                {"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end")}
                for w in raw_words
            ],
            # Absent means unknown, not confident. The scorer reads None as "no signal"; a
            # plausible-looking default would silently drive 25% of the queue ordering.
            avg_logprob=body.get("avg_logprob"),
            no_speech_prob=body.get("no_speech_prob"),
            latency_ms=latency_ms,
            estimated_cost_usd=Decimal(str(cost)) if cost is not None else None,
            raw=body,
        )
        self._log(
            route=route,
            model=result.model,
            request_hash=payload_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            cost=result.estimated_cost_usd,
            latency_ms=latency_ms,
        )
        return result
