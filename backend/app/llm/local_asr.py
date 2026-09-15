"""Client for a fine-tuned model on this machine's CPU: the playground sidecar (D85).

The sidecar (``playground/server.py``) holds one fine-tuned model in memory and transcribes one
recording per request with the notebook's standard decoder. The harness sends audio and logs the
answer; torch never enters the backend (D32, D79).

It is a provider like any other here: a named route, a row in ``llm_requests`` for every attempt,
and a dry run that never reaches the network (invariant 6). Nothing is billed, so every row costs
zero; the row is the record that a model was run and what it wrote.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

import httpx

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

#: The route in ``config/llm_routes.yaml`` the Models page transcribes through.
PLAYGROUND_ROUTE = "playground_transcribe"


class LocalAsrClient(ProviderClient):
    """Transcribes a recording with a fine-tuned model in the playground sidecar."""

    def transcribe(
        self,
        audio: bytes,
        *,
        model: str,
        weights: str,
        route: str = PLAYGROUND_ROUTE,
        dry_run: bool | None = None,
    ) -> AsrResult:
        """Transcribe one recording.

        Args:
            audio: 16 kHz mono FLAC.
            model: The model's folder name under ``models.root``; logged as the row's model.
            weights: The subfolder holding its CPU weights (``cpu`` or ``best``).
            route: Route name from ``config/llm_routes.yaml``.
            dry_run: Override the configured dry-run mode for this call.

        Raises:
            LlmRouteNotConfigured: The route does not exist.
            LlmDisabledError: Inference is disabled in configuration.
            LlmRequestFailed: The sidecar is down, refused the request or failed.
        """
        route_config = self.config.routes.get(route)
        if route_config is None:
            raise LlmRouteNotConfigured(f"no route named {route!r} in llm_routes.yaml")

        request_hash = hashlib.sha256(f"{model}/{weights}:".encode() + audio).hexdigest()
        summary = f"playground: {model}/{weights}, {len(audio)} bytes of FLAC"
        effective_dry_run = self.config.dry_run if dry_run is None else dry_run

        if effective_dry_run:
            self._log(
                route=route,
                model=model,
                request_hash=request_hash,
                input_summary=summary,
                status="dry_run",
                cost=Decimal("0.0"),
            )
            text, _ = dry_run_transcript(request_hash)
            return AsrResult(
                route=route, model=model, text=text, dry_run=True, estimated_cost_usd=Decimal("0")
            )

        if not self.config.enabled:
            raise LlmDisabledError(
                "Inference is disabled (config/llm_routes.yaml: enabled: false), "
                "so no transcription can be produced."
            )

        timeout = route_config.timeout_seconds or self.config.default_timeout_seconds
        url = f"{self.config.local_base_url.rstrip('/')}/transcribe"

        def send() -> httpx.Response:
            return self._get_client().post(
                url,
                params={"model": model, "weights": weights},
                content=audio,
                headers={"Content-Type": "audio/flac"},
                timeout=timeout,
            )

        response, last_error, latency_ms = self._send_with_retries(send)
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
            logger.info("playground_request_failed", model=model, error=last_error)
            raise LlmRequestFailed(f"playground transcription failed: {last_error}")

        body = response.json()
        self._log(
            route=route,
            model=model,
            request_hash=request_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            cost=Decimal("0.0"),
            latency_ms=latency_ms,
        )
        return AsrResult(
            route=route,
            model=model,
            text=body.get("text", ""),
            latency_ms=latency_ms,
            estimated_cost_usd=Decimal("0"),
            raw=body,
        )
