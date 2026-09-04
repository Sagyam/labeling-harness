"""Tests for model pricing and cost estimation across the 3 AI vendors.

ElevenLabs, OpenRouter, and Google Vertex AI have distinct billing shapes:
- ElevenLabs Scribe v2: billed per audio hour ($0.22/hr + $0.05/hr keyterm prompting = $0.27/hr).
- OpenRouter MAI-Transcribe-2: billed per audio hour ($0.10/hr) or from usage.cost.
- Vertex AI Gemini 3.8 Flash: token-based ($0.75/1M input, $3.75/1M output).
- Vertex AI Gemini 3.5 Transcribe: token-based ($3.50/1M input, $21.00/1M output) or ~$0.0368/min.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.llm.cost import (
    MODEL_PRICING_CATALOG,
    calculate_elevenlabs_cost,
    calculate_openrouter_cost,
    calculate_vertex_cost,
    estimate_request_cost,
    get_audio_duration,
)


def test_elevenlabs_scribe_v2_pricing_with_and_without_keyterms() -> None:
    # 3600 seconds = 1 hour
    cost_with_keyterms = calculate_elevenlabs_cost(3600.0, has_keyterms=True)
    assert cost_with_keyterms == Decimal("0.270000")

    cost_base_only = calculate_elevenlabs_cost(3600.0, has_keyterms=False)
    assert cost_base_only == Decimal("0.220000")

    # 10 second clip
    cost_10s = calculate_elevenlabs_cost(10.0, has_keyterms=True)
    # 10 * 0.27 / 3600 = 0.00075
    assert cost_10s == Decimal("0.000750")


def test_openrouter_mai_transcribe_2_pricing() -> None:
    # 3600 seconds = 1 hour at $0.10/hr
    cost = calculate_openrouter_cost("microsoft/mai-transcribe-2", duration_seconds=3600.0)
    assert cost == Decimal("0.100000")

    # 10 second clip: 10 * 0.10 / 3600 = 0.000278
    cost_10s = calculate_openrouter_cost("microsoft/mai-transcribe-2", duration_seconds=10.0)
    assert cost_10s == Decimal("0.000278")


def test_openrouter_usage_cost_precedence() -> None:
    # When OpenRouter returns a cost in usage dict, it takes precedence over duration estimate
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.00042}
    cost = calculate_openrouter_cost(
        "microsoft/mai-transcribe-2", duration_seconds=3600.0, usage=usage
    )
    assert cost == Decimal("0.000420")


def test_vertex_gemini_38_flash_token_pricing() -> None:
    # Gemini 3.8 Flash: $0.75 per 1M input tokens, $3.75 per 1M output tokens
    # 1,000,000 input tokens = $0.75
    cost_input = calculate_vertex_cost(
        "gemini-3.8-flash", prompt_tokens=1_000_000, completion_tokens=0
    )
    assert cost_input == Decimal("0.750000")

    # 1,000,000 output tokens = $3.75
    cost_output = calculate_vertex_cost(
        "gemini-3.8-flash", prompt_tokens=0, completion_tokens=1_000_000
    )
    assert cost_output == Decimal("3.750000")

    # 1000 prompt tokens + 200 completion tokens
    # 1000 * 0.00000075 + 200 * 0.00000375 = 0.00075 + 0.00075 = 0.0015
    cost_mixed = calculate_vertex_cost(
        "gemini-3.8-flash", prompt_tokens=1000, completion_tokens=200
    )
    assert cost_mixed == Decimal("0.001500")


def test_vertex_gemini_35_transcribe_pricing() -> None:
    # Gemini 3.5 Transcribe: $3.50/1M input, $21.00/1M output
    cost = calculate_vertex_cost(
        "gemini-3.5-transcribe", prompt_tokens=10_000, completion_tokens=500
    )
    # 10,000 * 0.0000035 + 500 * 0.000021 = 0.035 + 0.0105 = 0.0455
    assert cost == Decimal("0.045500")

    # Fallback to duration (approx $0.0368/min = $0.0006133/sec)
    cost_duration = calculate_vertex_cost("gemini-3.5-transcribe", duration_seconds=60.0)
    assert cost_duration == pytest.approx(Decimal("0.036800"), abs=Decimal("0.0001"))


def test_estimate_request_cost_handles_dry_run_and_failed() -> None:
    assert estimate_request_cost(
        route="asr_scribe_v2", model="scribe_v2", status="failed", duration_seconds=10.0
    ) == Decimal("0.000000")
    assert estimate_request_cost(
        route="asr_scribe_v2", model="scribe_v2", status="dry_run", duration_seconds=10.0
    ) == Decimal("0.000000")


def test_get_audio_duration_handles_missing_or_invalid_file(tmp_path) -> None:
    missing = tmp_path / "missing.flac"
    assert get_audio_duration(missing) is None

    invalid = tmp_path / "not_audio.flac"
    invalid.write_bytes(b"some non audio bytes")
    assert get_audio_duration(invalid) is None


def test_pricing_catalog_contains_all_active_routes() -> None:
    models = {entry["model"] for entry in MODEL_PRICING_CATALOG}
    assert "scribe_v2" in models
    assert "microsoft/mai-transcribe-2" in models
    assert "gemini-3.5-transcribe" in models
    assert "gemini-3.8-flash" in models

    vendors = {entry["vendor"] for entry in MODEL_PRICING_CATALOG}
    assert vendors == {"ElevenLabs", "OpenRouter", "Google Cloud Vertex AI"}
