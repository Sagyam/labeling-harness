"""Shared machinery for every inference provider in this package.

The harness talks to more than one vendor, but the guarantees around a call are the same
everywhere: it is written to ``llm_requests`` before it can be forgotten, it retries only the
status codes that are worth retrying, and a dry run never reaches the network. Those three live
here so that adding a provider cannot quietly opt out of any of them.

Billing control is the reason the log exists. It is also the only spend record the harness keeps:
no provider is excluded for how it bills (D34), so ``llm_requests`` -- one row per attempt, with
route, model, status and latency -- is what an ingest's cost is reconstructed from afterwards.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
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
    """Base class for provider failures."""


class LlmDisabledError(LlmError):
    """Inference is switched off in configuration."""


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
    """One ASR transcription, plus usage and cost metrics.

    ``words``, ``avg_logprob`` and ``no_speech_prob`` are ``None`` when the provider does not
    report them. They are never defaulted: the scorer reads ``None`` as "no signal", and a
    plausible-looking default would silently drive a quarter of the queue ordering.
    """

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


#: Canned transcripts for dry runs, chosen by file hash. Callers must key off ``dry_run`` and
#: never store one as a real hypothesis.
DRY_RUN_TRANSCRIPTS = (
    "हामीले यो project मा meeting गरेर data analyse गर्नु पर्छ",
    "आजको session मा machine learning र technology को कुरा भयो",
    "सबै team members ले आफ्नो schedule अनुसार task complete गर्नु होला",
    "यो software system मा नयाँ feature update add गरिएको छ",
)


def dry_run_transcript(payload_hash: str) -> tuple[str, list[dict[str, Any]]]:
    """Pick a deterministic mock transcript and word list from a request hash."""
    chosen = DRY_RUN_TRANSCRIPTS[int(payload_hash[:4], 16) % len(DRY_RUN_TRANSCRIPTS)]
    words = [
        {"word": w, "start": round(i * 0.4, 2), "end": round((i + 1) * 0.4, 2)}
        for i, w in enumerate(chosen.split())
    ]
    return chosen, words


class ProviderClient:
    """Base for a logged, retrying client against one vendor's HTTP API.

    Subclasses supply the request building and response parsing; everything that must be true of
    every call -- the ``llm_requests`` row, the retry policy, the dry-run short circuit -- is
    inherited from here.
    """

    def __init__(
        self,
        session: Session,
        *,
        config: LlmRoutes | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.session = session
        self.config = config or load_llm_routes()
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
    def _audio_hash(audio_file: Path, model: str) -> str:
        """Hash an audio request by model, file name and size, without reading the clip."""
        size = audio_file.stat().st_size if audio_file.exists() else 0
        return hashlib.sha256(f"{model}:{audio_file.name}:{size}".encode()).hexdigest()

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
            response: httpx.Response | None = None
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
                backoff = self.config.retry_backoff_seconds * (2**attempt)
                if response is not None and response.status_code == 429:
                    retry_after_header = response.headers.get("retry-after")
                    if retry_after_header:
                        with contextlib.suppress(ValueError):
                            backoff = max(backoff, float(retry_after_header))
                time.sleep(backoff)
        return None, last_error, int((time.monotonic() - started) * 1000)
