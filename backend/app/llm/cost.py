"""Pricing models and cost calculation for external AI providers.

Tracks cost across the 3 AI vendors used by the harness:
1. ElevenLabs: Scribe v2 billed per audio hour ($0.22/hr base + $0.05/hr keyterm prompting).
2. OpenRouter: Microsoft MAI-Transcribe-2 billed per audio hour ($0.10/hr) or from usage.cost.
3. Google Cloud Vertex AI:
   - Gemini 3.8 Flash: $0.75 / 1M prompt tokens, $3.75 / 1M completion tokens.
   - Gemini 3.5 Transcribe: $3.50 / 1M input tokens, $21.00 / 1M output tokens (or ~$0.0368/min).
   Served on Vertex AI as of D39; `gemini-3.5-transcribe-preview` still matches on 'transcribe'.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

# --- Published Pricing Constants -----------------------------------------------------------

# ElevenLabs Scribe v2
ELEVENLABS_BASE_PER_HOUR = Decimal("0.22")
ELEVENLABS_KEYTERM_PER_HOUR = Decimal("0.05")

# OpenRouter Microsoft MAI-Transcribe-2
OPENROUTER_MAI_PER_HOUR = Decimal("0.10")

# Google Cloud Vertex AI Gemini 3.8 Flash
VERTEX_GEMINI_38_FLASH_INPUT_PER_M = Decimal("0.75")
VERTEX_GEMINI_38_FLASH_OUTPUT_PER_M = Decimal("3.75")

# Google Cloud Vertex AI Gemini 3.5 Transcribe
VERTEX_GEMINI_35_TRANSCRIBE_INPUT_PER_M = Decimal("3.50")
VERTEX_GEMINI_35_TRANSCRIBE_OUTPUT_PER_M = Decimal("21.00")
VERTEX_GEMINI_35_TRANSCRIBE_PER_MIN = Decimal("0.0368")

# Pricing catalog for UI presentation and documentation
MODEL_PRICING_CATALOG: list[dict[str, Any]] = [
    {
        "vendor": "ElevenLabs",
        "route": "asr_scribe_v2",
        "model": "scribe_v2",
        "pricing_unit": "audio_hour",
        "base_rate_usd": 0.22,
        "keyterm_rate_usd": 0.05,
        "effective_rate_display": "$0.27 / hr ($0.22 base + $0.05 keyterm prompting)",
        "description": "Multilingual Scribe v2 with word-level timestamps & log probabilities.",
    },
    {
        "vendor": "OpenRouter",
        "route": "asr_mai_transcribe_2",
        "model": "microsoft/mai-transcribe-2",
        "pricing_unit": "audio_hour",
        "base_rate_usd": 0.10,
        "keyterm_rate_usd": 0.0,
        "effective_rate_display": "$0.10 / hr of audio",
        "description": "Microsoft MAI multilingual transcription with native code-switching.",
    },
    {
        "vendor": "Google Cloud Vertex AI",
        "route": "asr_gemini_flash",
        "model": "gemini-3.8-flash",
        "pricing_unit": "tokens",
        "input_per_m_usd": 0.75,
        "output_per_m_usd": 3.75,
        "effective_rate_display": "$0.75 / 1M in, $3.75 / 1M out (~25 audio tokens/s)",
        "description": "Multimodal general LLM audio transcription via generateContent.",
    },
    {
        "vendor": "Google Cloud Vertex AI",
        "route": "asr_gemini_composite",
        "model": "gemini-3.5-transcribe-preview",
        "pricing_unit": "tokens_or_minute",
        "input_per_m_usd": 3.50,
        "output_per_m_usd": 21.00,
        "effective_rate_display": "$3.50 / 1M in, $21.00 / 1M out (~$0.0368 / min audio)",
        "description": (
            "Dedicated recognizer with word timestamps & speaker diarization; a cheap "
            "OpenRouter text call restores script afterwards (D41)."
        ),
    },
]


def _quantize_usd(value: Decimal) -> Decimal:
    """Format currency to 6 decimal places (micro-dollars) for database precision."""
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def get_audio_duration(audio_file: Path | str) -> float | None:
    """Safely probe audio file duration in seconds without throwing exceptions."""
    try:
        from app.services.audio import probe

        return probe(audio_file).duration_seconds
    except Exception:
        return None


def calculate_elevenlabs_cost(duration_seconds: float, has_keyterms: bool = True) -> Decimal:
    """Calculate ElevenLabs Scribe transcription cost based on audio duration.

    Args:
        duration_seconds: Duration of the audio clip in seconds.
        has_keyterms: Whether keyterms were sent (adds $0.05/hr).
    """
    rate_per_hour = ELEVENLABS_BASE_PER_HOUR + (
        ELEVENLABS_KEYTERM_PER_HOUR if has_keyterms else Decimal("0.0")
    )
    seconds_dec = Decimal(str(max(0.0, duration_seconds)))
    cost = (seconds_dec / Decimal("3600")) * rate_per_hour
    return _quantize_usd(cost)


def calculate_openrouter_cost(
    model: str,
    duration_seconds: float | None = None,
    usage: dict[str, Any] | None = None,
) -> Decimal | None:
    """Calculate OpenRouter cost from reported usage or fallback to duration for audio endpoints.

    Args:
        model: OpenRouter model identifier.
        duration_seconds: Audio clip duration in seconds.
        usage: Response usage payload (may contain ``cost`` or ``total_cost``).
    """
    if usage:
        for key in ("cost", "total_cost"):
            val = usage.get(key)
            if val is not None:
                try:
                    return _quantize_usd(Decimal(str(val)))
                except Exception:
                    pass

    # Audio model fallback when usage doesn't report cost
    if "mai-transcribe" in model and duration_seconds is not None:
        seconds_dec = Decimal(str(max(0.0, duration_seconds)))
        cost = (seconds_dec / Decimal("3600")) * OPENROUTER_MAI_PER_HOUR
        return _quantize_usd(cost)

    return None


def calculate_vertex_cost(
    model: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    duration_seconds: float | None = None,
) -> Decimal:
    """Calculate Vertex AI cost based on token usage or audio duration.

    Args:
        model: Model name (e.g. gemini-3.8-flash or gemini-3.5-transcribe-preview).
        prompt_tokens: Input / prompt tokens consumed.
        completion_tokens: Output / candidates tokens consumed.
        duration_seconds: Audio duration in seconds (used as fallback).
    """
    if "flash" in model.lower():
        inp_rate = VERTEX_GEMINI_38_FLASH_INPUT_PER_M / Decimal("1000000")
        out_rate = VERTEX_GEMINI_38_FLASH_OUTPUT_PER_M / Decimal("1000000")
        p_tokens = prompt_tokens or 0
        c_tokens = completion_tokens or 0

        # If token count is absent but duration is known, estimate ~25 tokens per second
        if p_tokens == 0 and duration_seconds:
            p_tokens = int(duration_seconds * 25)

        cost = (Decimal(p_tokens) * inp_rate) + (Decimal(c_tokens) * out_rate)
        return _quantize_usd(cost)

    if "transcribe" in model.lower():
        if prompt_tokens or completion_tokens:
            inp_rate = VERTEX_GEMINI_35_TRANSCRIBE_INPUT_PER_M / Decimal("1000000")
            out_rate = VERTEX_GEMINI_35_TRANSCRIBE_OUTPUT_PER_M / Decimal("1000000")
            p_tokens = prompt_tokens or 0
            c_tokens = completion_tokens or 0
            cost = (Decimal(p_tokens) * inp_rate) + (Decimal(c_tokens) * out_rate)
            return _quantize_usd(cost)

        # Minute-based fallback
        if duration_seconds is not None:
            mins = Decimal(str(max(0.0, duration_seconds))) / Decimal("60")
            cost = mins * VERTEX_GEMINI_35_TRANSCRIBE_PER_MIN
            return _quantize_usd(cost)

    return Decimal("0.000000")


def vendor_for_route_or_model(route: str, model: str | None = None) -> str:
    """Resolve human-readable vendor name from route name or model slug."""
    try:
        from app.config import load_llm_routes

        cfg = load_llm_routes()
        if route in cfg.routes:
            provider = cfg.routes[route].provider
            if provider == "elevenlabs":
                return "ElevenLabs"
            if provider == "vertex":
                return "Google Cloud Vertex AI"
            if provider == "openrouter":
                return "OpenRouter"
    except Exception:
        pass

    r = (route or "").lower()
    m = (model or "").lower()

    # `scribe` must not match inside `transcribe`: every Gemini recogniser model id ends in
    # "-transcribe", so a plain substring test bills them to ElevenLabs. This only bites for a
    # route missing from the routing table -- a historical row, or one renamed since it was
    # logged -- which is exactly when the fallback is load-bearing.
    if (
        "elevenlabs" in r
        or "elevenlabs" in m
        or re.search(r"(?<!tran)scribe", m)
        or r.startswith("asr_scribe")
        or r.startswith("scribe")
    ):
        return "ElevenLabs"
    if "vertex" in r or "gemini" in r or "gemini" in m:
        return "Google Cloud Vertex AI"
    if "mai" in r or "openrouter" in r or "/" in m:
        return "OpenRouter"
    return "Unknown Vendor"


def estimate_request_cost(
    route: str,
    model: str | None = None,
    status: str = "succeeded",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    duration_seconds: float | None = None,
) -> Decimal:
    """Estimate cost for an arbitrary request (useful for historical rows or dry runs)."""
    if status in ("failed", "dry_run"):
        return Decimal("0.000000")

    model_str = model or route
    vendor = vendor_for_route_or_model(route, model)

    if vendor == "ElevenLabs":
        dur = duration_seconds if duration_seconds is not None else 5.0
        return calculate_elevenlabs_cost(dur, has_keyterms=True)
    if vendor == "OpenRouter":
        if duration_seconds is not None and "mai" in model_str:
            calculated = calculate_openrouter_cost(model_str, duration_seconds=duration_seconds)
            return calculated or Decimal("0.000000")
        if prompt_tokens or completion_tokens:
            # Fallback estimation for chat completions
            p = prompt_tokens or 0
            c = completion_tokens or 0
            cost = (Decimal(p) * Decimal("0.00000075")) + (Decimal(c) * Decimal("0.00000375"))
            return _quantize_usd(cost)
        return Decimal("0.000100")
    if vendor == "Google Cloud Vertex AI":
        return calculate_vertex_cost(
            model_str,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_seconds=duration_seconds,
        )

    return Decimal("0.000000")
