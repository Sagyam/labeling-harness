"""The per-service gates every concurrent ingest shares (D88)."""

from __future__ import annotations

import threading
import time

import pytest

from app.config import ProviderLimit
from app.utils.rate_limit import ProviderGate, provider_gate, reset_gates, retry_after_seconds


def _limit(**kwargs) -> ProviderLimit:
    base = {"max_in_flight": 2, "cooldown_seconds": 0.05, "max_cooldown_seconds": 0.2}
    return ProviderLimit(**{**base, **kwargs})


def test_in_flight_never_exceeds_the_limit() -> None:
    gate = ProviderGate("svc", _limit(max_in_flight=3))
    peak = 0
    current = 0
    guard = threading.Lock()

    def worker() -> None:
        nonlocal peak, current
        with gate.slot():
            with guard:
                current += 1
                peak = max(peak, current)
            time.sleep(0.02)
            with guard:
                current -= 1

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert peak == 3


def test_a_refusal_halves_the_limit_and_successes_win_it_back() -> None:
    gate = ProviderGate("svc", _limit(max_in_flight=8))
    gate.throttled()
    assert gate.snapshot()["allowed"] == 4

    for _ in range(4):
        gate.succeeded()
    assert gate.snapshot()["allowed"] == 5


def test_refusals_inside_one_cooldown_count_once() -> None:
    """Twenty requests refused together are one signal, not twenty halvings."""
    gate = ProviderGate(
        "svc", _limit(max_in_flight=8, cooldown_seconds=1.0, max_cooldown_seconds=5)
    )
    for _ in range(20):
        gate.throttled()
    assert gate.snapshot()["allowed"] == 4
    assert gate.snapshot()["throttled_total"] == 20


def test_a_cooldown_holds_every_caller_back() -> None:
    gate = ProviderGate("svc", _limit(cooldown_seconds=0.15, max_cooldown_seconds=1))
    gate.throttled()
    started = time.monotonic()
    with gate.slot():
        waited = time.monotonic() - started
    assert waited >= 0.14


def test_retry_after_is_honoured_up_to_the_ceiling() -> None:
    gate = ProviderGate("svc", _limit(cooldown_seconds=0.01, max_cooldown_seconds=0.3))
    assert gate.throttled(retry_after=0.2) == pytest.approx(0.2, abs=0.05)
    reset = ProviderGate("svc2", _limit(cooldown_seconds=0.01, max_cooldown_seconds=0.3))
    assert reset.throttled(retry_after=60) == pytest.approx(0.3, abs=0.01)


def test_repeated_cooldowns_back_off_exponentially() -> None:
    gate = ProviderGate("svc", _limit(cooldown_seconds=0.02, max_cooldown_seconds=10))
    first = gate.throttled()
    time.sleep(first + 0.01)
    second = gate.throttled()
    assert second > first * 1.5


def test_minimum_interval_spaces_out_starts() -> None:
    gate = ProviderGate("svc", _limit(max_in_flight=4, min_interval_seconds=0.05))
    starts: list[float] = []
    for _ in range(3):
        with gate.slot():
            starts.append(time.monotonic())
    assert starts[2] - starts[0] >= 0.09


def test_on_wait_is_called_once_when_a_caller_is_held() -> None:
    gate = ProviderGate("svc", _limit(cooldown_seconds=0.05))
    gate.throttled()
    calls: list[float] = []
    with gate.slot(on_wait=calls.append):
        pass
    assert len(calls) == 1


def test_the_registry_shares_one_gate_per_service() -> None:
    reset_gates()
    assert provider_gate("vertex", _limit()) is provider_gate("vertex", _limit())
    assert provider_gate("vertex", _limit()) is not provider_gate("openrouter", _limit())
    reset_gates()


def test_retry_after_parsing() -> None:
    now = 1_800_000_000.0
    assert retry_after_seconds({"retry-after": "7"}, now=now) == 7.0
    assert (
        retry_after_seconds({"retry-after": "Fri, 15 Jan 2027 08:00:10 GMT"}, now=now) is not None
    )
    # OpenRouter's reset is epoch milliseconds.
    assert retry_after_seconds({"x-ratelimit-reset": str(int((now + 12) * 1000))}, now=now) == (
        pytest.approx(12.0)
    )
    assert retry_after_seconds({"x-ratelimit-reset": str(int(now + 3))}, now=now) == pytest.approx(
        3
    )
    assert retry_after_seconds({}, now=now) is None
    assert retry_after_seconds({"retry-after": "soon"}, now=now) is None
