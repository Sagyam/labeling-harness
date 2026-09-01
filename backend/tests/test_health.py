"""Tests for the application factory and health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_ok_without_dependencies() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "database" in body["checks"]
    assert "storage" in body["checks"]


def test_health_is_degraded_when_database_is_unreachable(monkeypatch) -> None:
    from app.api import health as health_module

    monkeypatch.setattr(health_module, "_check_database", lambda: (False, "connection refused"))
    client = TestClient(create_app())
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"]["ok"] is False


def test_app_exposes_openapi_schema() -> None:
    client = TestClient(create_app())
    assert client.get("/openapi.json").status_code == 200
