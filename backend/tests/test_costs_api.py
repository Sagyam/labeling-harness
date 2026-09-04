"""Tests for the Cost Tracker API and reporting service."""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import LlmRequest

pytestmark = pytest.mark.db


@pytest.fixture
def sample_requests(db_session: Session) -> list[LlmRequest]:
    """Create sample request rows spanning ElevenLabs, OpenRouter, and Vertex AI."""
    requests = [
        # ElevenLabs Scribe v2 succeeded
        LlmRequest(
            route="asr_scribe_v2",
            model="scribe_v2",
            input_summary="elevenlabs_transcribe: clip1.flac",
            prompt_tokens=None,
            completion_tokens=None,
            estimated_cost_usd=Decimal("0.001350"),
            latency_ms=1200,
            status="succeeded",
        ),
        # OpenRouter MAI-Transcribe-2 succeeded
        LlmRequest(
            route="asr_mai_transcribe_2",
            model="microsoft/mai-transcribe-2",
            input_summary="asr_transcribe: clip1.flac",
            prompt_tokens=None,
            completion_tokens=None,
            estimated_cost_usd=Decimal("0.000500"),
            latency_ms=1500,
            status="succeeded",
        ),
        # Vertex AI Gemini 3.8 Flash succeeded
        LlmRequest(
            route="asr_gemini_flash",
            model="gemini-3.8-flash",
            input_summary="vertex_transcribe: clip1.flac",
            prompt_tokens=500,
            completion_tokens=50,
            estimated_cost_usd=Decimal("0.000563"),
            latency_ms=800,
            status="succeeded",
        ),
        # Vertex AI Gemini 3.5 Transcribe failed
        LlmRequest(
            route="asr_gemini_transcribe",
            model="gemini-3.5-transcribe",
            input_summary="vertex_transcribe: clip1.flac",
            prompt_tokens=None,
            completion_tokens=None,
            estimated_cost_usd=Decimal("0.000000"),
            latency_ms=150,
            status="failed",
            error_message="RESOURCE_PROJECT_INVALID: Interactions API not allowlisted",
        ),
        # ElevenLabs dry run
        LlmRequest(
            route="asr_scribe_v2",
            model="scribe_v2",
            input_summary="elevenlabs_transcribe: clip2.flac",
            prompt_tokens=None,
            completion_tokens=None,
            estimated_cost_usd=Decimal("0.000000"),
            latency_ms=10,
            status="dry_run",
        ),
        # Historical row with NULL estimated_cost_usd to verify on-the-fly estimation
        LlmRequest(
            route="asr_gemini_flash",
            model="gemini-3.8-flash",
            input_summary="vertex_transcribe: clip3.flac",
            prompt_tokens=1000,
            completion_tokens=100,
            estimated_cost_usd=None,
            latency_ms=900,
            status="succeeded",
        ),
    ]
    db_session.add_all(requests)
    db_session.commit()
    return requests


def test_get_costs_summary_and_breakdowns(client: TestClient, sample_requests) -> None:
    res = client.get("/costs")
    assert res.status_code == 200
    data = res.json()

    # Summary
    summary = data["summary"]
    assert summary["total_requests"] == 6
    assert summary["successful_requests"] == 4
    assert summary["failed_requests"] == 1
    assert summary["dry_run_requests"] == 1
    assert summary["total_cost_usd"] > 0
    assert summary["average_latency_ms"] is not None

    # Vendor breakdown includes all 3 vendors
    vendors = {v["vendor"]: v for v in data["vendor_breakdown"]}
    assert "ElevenLabs" in vendors
    assert "OpenRouter" in vendors
    assert "Google Cloud Vertex AI" in vendors

    assert vendors["ElevenLabs"]["requests"] == 2
    assert vendors["ElevenLabs"]["successful"] == 1
    assert vendors["OpenRouter"]["requests"] == 1
    assert vendors["Google Cloud Vertex AI"]["requests"] == 3
    assert vendors["Google Cloud Vertex AI"]["failed"] == 1

    # Model breakdown
    models = {m["route"]: m for m in data["model_breakdown"]}
    assert "asr_scribe_v2" in models
    assert "asr_mai_transcribe_2" in models
    assert "asr_gemini_flash" in models
    assert "asr_gemini_transcribe" in models

    # Pricing catalog
    catalog = data["pricing_catalog"]
    assert len(catalog) >= 4


def test_get_cost_requests_ledger_with_filters(client: TestClient, sample_requests) -> None:
    # All requests
    res = client.get("/costs/requests")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 6
    assert len(data["items"]) == 6

    # Filter by vendor
    res_el = client.get("/costs/requests", params={"vendor": "ElevenLabs"})
    assert res_el.status_code == 200
    data_el = res_el.json()
    assert data_el["total"] == 2
    assert all(r["vendor"] == "ElevenLabs" for r in data_el["items"])

    # Filter by status
    res_failed = client.get("/costs/requests", params={"status": "failed"})
    assert res_failed.status_code == 200
    data_failed = res_failed.json()
    assert data_failed["total"] == 1
    assert data_failed["items"][0]["status"] == "failed"
    assert "RESOURCE_PROJECT_INVALID" in data_failed["items"][0]["error_message"]

    # Search filter
    res_search = client.get("/costs/requests", params={"search": "clip3"})
    assert res_search.status_code == 200
    data_search = res_search.json()
    assert data_search["total"] == 1
    assert "clip3" in data_search["items"][0]["input_summary"]
