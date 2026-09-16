"""Process-wide gates in front of every rate-limited service the ingest calls (D88).

Several episodes ingest at once, and each fans its clips out across every recogniser. Left alone
that multiplies the request rate by the number of jobs, and a 429 on one clip discards it (D46).
So every call to a service passes through that service's one :class:`ProviderGate`:

* at most ``max_in_flight`` requests are open at once, whichever job sent them;
* starts are spaced by ``min_interval_seconds``;
* a rate-limit refusal puts the whole service into a cooldown that every caller waits out, and
  halves the working limit (additive increase, multiplicative decrease). Refusals that land
  inside a cooldown already running are the same signal and count once;
* the limit climbs back one slot per ``allowed`` successes.
"""

from __future__ import annotations

import email.utils
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from app.config import ProviderLimit
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ProviderGate:
    """Concurrency, spacing and shared cooldown for one service."""

    def __init__(self, name: str, limit: ProviderLimit) -> None:
        self.name = name
        self._limit = limit
        self._cond = threading.Condition()
        self._in_flight = 0
        self._allowed = limit.max_in_flight
        self._successes = 0
        self._strikes = 0
        self._blocked_until = 0.0
        self._next_start = 0.0
        self._throttled_total = 0

    def configure(self, limit: ProviderLimit) -> None:
        """Adopt a changed limit; a no-op when it is the one already in force."""
        with self._cond:
            if limit == self._limit:
                return
            self._limit = limit
            self._allowed = limit.max_in_flight
            self._cond.notify_all()

    @contextmanager
    def slot(self, on_wait: Callable[[float], None] | None = None) -> Iterator[None]:
        """Hold one request slot, waiting for a free slot, the spacing and any cooldown.

        Args:
            on_wait: Called once, with the known delay in seconds (0 when only waiting for a free
                slot), if the caller has to wait at all.
        """
        notified = False
        with self._cond:
            while True:
                now = time.monotonic()
                delay = max(self._blocked_until, self._next_start) - now
                if delay <= 0 and self._in_flight < self._allowed:
                    break
                if on_wait is not None and not notified:
                    notified = True
                    on_wait(max(delay, 0.0))
                self._cond.wait(timeout=delay if delay > 0 else None)
            self._in_flight += 1
            self._next_start = now + self._limit.min_interval_seconds
        try:
            yield
        finally:
            with self._cond:
                self._in_flight -= 1
                self._cond.notify_all()

    def succeeded(self) -> None:
        """A request got through: end the backoff streak and win back capacity slowly."""
        with self._cond:
            self._strikes = 0
            if self._allowed >= self._limit.max_in_flight:
                return
            self._successes += 1
            if self._successes >= self._allowed:
                self._allowed += 1
                self._successes = 0
                self._cond.notify_all()

    def throttled(self, retry_after: float | None = None) -> float:
        """The service refused for rate: start (or extend) a cooldown every caller waits out.

        Args:
            retry_after: The wait the service asked for, if it named one.

        Returns:
            Seconds until the cooldown ends.
        """
        with self._cond:
            now = time.monotonic()
            self._throttled_total += 1
            if now >= self._blocked_until:
                self._strikes += 1
                self._allowed = max(1, self._allowed // 2)
                self._successes = 0
                backoff = self._limit.cooldown_seconds * 2 ** (self._strikes - 1)
                # Jitter, so callers released together by one cooldown do not collide again.
                wait = max(retry_after or 0.0, backoff * random.uniform(1.0, 1.25))
                wait = min(wait, self._limit.max_cooldown_seconds)
                self._blocked_until = now + wait
                logger.warning(
                    "provider_rate_limited",
                    provider=self.name,
                    cooldown_seconds=round(wait, 2),
                    allowed=self._allowed,
                    strikes=self._strikes,
                )
            elif retry_after:
                self._blocked_until = max(
                    self._blocked_until, now + min(retry_after, self._limit.max_cooldown_seconds)
                )
            self._cond.notify_all()
            return self._blocked_until - now

    def cooling_down_for(self) -> float:
        """Seconds left in the current cooldown, 0 when there is none."""
        with self._cond:
            return max(0.0, self._blocked_until - time.monotonic())

    def snapshot(self) -> dict[str, Any]:
        """The gate's state, for the queue endpoint."""
        with self._cond:
            return {
                "name": self.name,
                "in_flight": self._in_flight,
                "allowed": self._allowed,
                "max_in_flight": self._limit.max_in_flight,
                "cooling_down_seconds": round(max(0.0, self._blocked_until - time.monotonic()), 1),
                "throttled_total": self._throttled_total,
            }


_gates: dict[str, ProviderGate] = {}
_registry_lock = threading.Lock()


def provider_gate(name: str, limit: ProviderLimit) -> ProviderGate:
    """The one gate for service ``name``, created on first use and kept to ``limit``."""
    with _registry_lock:
        gate = _gates.get(name)
        if gate is None:
            gate = _gates[name] = ProviderGate(name, limit)
            return gate
    gate.configure(limit)
    return gate


def gate_snapshots() -> list[dict[str, Any]]:
    """Every gate used so far, by name."""
    with _registry_lock:
        gates = sorted(_gates.values(), key=lambda g: g.name)
    return [gate.snapshot() for gate in gates]


def reset_gates() -> None:
    """Forget every gate (tests)."""
    with _registry_lock:
        _gates.clear()


def retry_after_seconds(headers: Mapping[str, str], *, now: float | None = None) -> float | None:
    """How long a rate-limited response asks the caller to wait, or ``None`` if it does not say.

    Reads ``Retry-After`` (seconds or an HTTP date) and ``X-RateLimit-Reset`` (OpenRouter sends
    epoch milliseconds; epoch seconds and a plain delta are accepted too).
    """
    now = time.time() if now is None else now
    lowered = {k.lower(): v for k, v in headers.items()}

    raw = lowered.get("retry-after")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(raw) if _looks_like_date(raw) else None
            if parsed is not None:
                return max(0.0, parsed.timestamp() - now)

    raw = lowered.get("x-ratelimit-reset")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            return None
        if value > 1e12:
            return max(0.0, value / 1000 - now)
        if value > 1e9:
            return max(0.0, value - now)
        return max(0.0, value)
    return None


def _looks_like_date(raw: str) -> bool:
    try:
        return email.utils.parsedate_to_datetime(raw) is not None
    except (TypeError, ValueError):
        return False
