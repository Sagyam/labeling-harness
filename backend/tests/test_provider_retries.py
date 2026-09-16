"""How a provider client waits out rate limits behind its service's gate (D88)."""

from __future__ import annotations

import threading
import time

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import ProviderLimit
from app.llm.base import LlmRequestFailed
from app.models import LlmRequest
from app.utils.rate_limit import gate_snapshots
from tests.test_openrouter import completion, make_client, routes

pytestmark = pytest.mark.db


def _message() -> list[dict[str, str]]:
    return [{"role": "user", "content": "x"}]


def test_rate_limits_do_not_spend_the_ordinary_retry_budget(db_session: Session) -> None:
    """Five refusals in a row still succeed with max_retries=2: a 429 is a wait, not a failure."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 5:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=completion())

    result = make_client(db_session, handler, config=routes(max_retries=2)).complete(
        "check", _message()
    )
    assert result.text == "ok"
    assert calls["n"] == 6
    (gate,) = gate_snapshots()
    assert gate["name"] == "openrouter"
    assert gate["throttled_total"] == 5


def test_a_service_that_keeps_refusing_fails_and_is_logged(db_session: Session) -> None:
    limit = ProviderLimit(rate_limit_retries=2, cooldown_seconds=0.001, max_cooldown_seconds=0.01)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    client = make_client(db_session, handler, config=routes(limits={"openrouter": limit}))
    with pytest.raises(LlmRequestFailed, match="429"):
        client.complete("check", _message())
    row = db_session.scalars(sa.select(LlmRequest)).one()
    assert row.status == "failed"
    assert "rate limited" in (row.error_message or "")


def test_retry_after_is_waited_out(db_session: Session) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0.2"}, text="slow down")
        return httpx.Response(200, json=completion())

    limit = ProviderLimit(cooldown_seconds=0.001, max_cooldown_seconds=1.0)
    started = time.monotonic()
    make_client(db_session, handler, config=routes(limits={"openrouter": limit})).complete(
        "check", _message()
    )
    assert time.monotonic() - started >= 0.19


def test_concurrent_calls_share_the_services_in_flight_cap(db_session: Session) -> None:
    limit = ProviderLimit(max_in_flight=2)
    current = 0
    peak = 0
    guard = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal current, peak
        with guard:
            current += 1
            peak = max(peak, current)
        time.sleep(0.03)
        with guard:
            current -= 1
        return httpx.Response(200, json=completion())

    config = routes(limits={"openrouter": limit})
    # One client per thread, as the ingest has; only the gate is shared.
    clients = [make_client(db_session, handler, config=config) for _ in range(6)]

    def call(client) -> None:
        response, _, _ = client._send_with_retries(
            lambda: client._get_client().post("https://example.invalid/x")
        )
        assert response is not None

    threads = [threading.Thread(target=call, args=(c,)) for c in clients]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert peak == 2
