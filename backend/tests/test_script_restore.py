"""Tests for the script-restoration half of ``gemini-composite`` (D41).

The recogniser writes everything in Devanagari and cannot be told not to, so a reasoning model
puts each token back into its own language's script afterwards. Every test here exists to protect
one invariant: **one output token per input token**. That is what lets each restored word keep the
span the recogniser measured for it, and a rewrite that breaks it must fail loudly rather than
produce a transcript whose timings are quietly wrong from the first divergence onward.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import LlmRequestFailed
from app.llm.script_restore import (
    count_same_script_edits,
    restore_script,
)

RESTORE_ROUTE = "script_restore"
TOKENS = ["आजको", "एक्टिभ", "मा", "रेन्ज"]
RESTORED = ["आजको", "active", "मा", "range"]


def routes(**kwargs: Any) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "routes": {
            RESTORE_ROUTE: LlmRoute(
                provider="openrouter", api="chat", model="google/gemini-3.8-flash"
            )
        },
    }
    return LlmRoutes(**{**base, **kwargs})


def completion(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "model": "google/gemini-3.8-flash",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def client_returning(*bodies: dict[str, Any]) -> httpx.Client:
    """A client answering each successive call with the next body, repeating the last."""
    seq = list(bodies)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=seq.pop(0) if len(seq) > 1 else seq[0])

    return httpx.Client(transport=httpx.MockTransport(handler))


# --- the same-script guard, no database -----------------------------------------------------


def test_a_script_change_is_not_counted_as_an_edit() -> None:
    assert count_same_script_edits(["एक्टिभ"], ["active"]) == 0


def test_a_devanagari_to_devanagari_rewrite_is_counted() -> None:
    """That is the model correcting the recogniser, which is not its job (rule 2)."""
    assert count_same_script_edits(["मिटिङ"], ["बैठक"]) == 1


def test_an_unchanged_token_is_not_counted() -> None:
    assert count_same_script_edits(["आजको", "मा"], ["आजको", "मा"]) == 0


# --- the rewrite itself ---------------------------------------------------------------------


@pytest.mark.db
def test_a_well_formed_rewrite_is_returned_with_its_counts(db_session: Session) -> None:
    restored, meta = restore_script(
        db_session,
        TOKENS,
        route=RESTORE_ROUTE,
        config=routes(),
        client=client_returning(completion(json.dumps(RESTORED, ensure_ascii=False))),
    )
    assert restored == RESTORED
    assert meta["script_restore_tokens"] == 4
    assert meta["script_restore_changed"] == 2
    assert meta["script_restore_same_script_edits"] == 0


@pytest.mark.db
def test_a_fenced_or_chatty_response_is_still_parsed(db_session: Session) -> None:
    """Models wrap JSON in prose and code fences; that is not a reason to fail a segment."""
    content = "Sure! Here you go:\n```json\n" + json.dumps(RESTORED, ensure_ascii=False) + "\n```"
    restored, _ = restore_script(
        db_session,
        TOKENS,
        route=RESTORE_ROUTE,
        config=routes(),
        client=client_returning(completion(content)),
    )
    assert restored == RESTORED


@pytest.mark.db
def test_a_short_rewrite_is_retried_and_then_accepted(db_session: Session) -> None:
    short = completion(json.dumps(["आजको", "active"], ensure_ascii=False))
    good = completion(json.dumps(RESTORED, ensure_ascii=False))
    restored, _ = restore_script(
        db_session,
        TOKENS,
        route=RESTORE_ROUTE,
        config=routes(),
        client=client_returning(short, good),
    )
    assert restored == RESTORED


@pytest.mark.db
def test_a_persistently_misaligned_rewrite_fails_rather_than_being_patched(
    db_session: Session,
) -> None:
    """Padding or truncating would give every word after the divergence someone else's timing."""
    short = completion(json.dumps(["आजको", "active"], ensure_ascii=False))
    with pytest.raises(LlmRequestFailed, match="script restoration failed"):
        restore_script(
            db_session,
            TOKENS,
            route=RESTORE_ROUTE,
            config=routes(),
            client=client_returning(short),
        )


@pytest.mark.db
def test_a_longer_rewrite_fails_too(db_session: Session) -> None:
    long = completion(json.dumps([*RESTORED, "extra"], ensure_ascii=False))
    with pytest.raises(LlmRequestFailed):
        restore_script(
            db_session, TOKENS, route=RESTORE_ROUTE, config=routes(), client=client_returning(long)
        )


@pytest.mark.db
def test_an_unparseable_response_fails_rather_than_guessing(db_session: Session) -> None:
    with pytest.raises(LlmRequestFailed):
        restore_script(
            db_session,
            TOKENS,
            route=RESTORE_ROUTE,
            config=routes(),
            client=client_returning(completion("I could not do that.")),
        )


@pytest.mark.db
def test_same_script_edits_are_reported_but_do_not_fail_the_segment(db_session: Session) -> None:
    """One disputed token must not cost an episode; it is counted so it can be found again."""
    meddled = completion(json.dumps(["आजको", "active", "मा", "दायरा"], ensure_ascii=False))
    restored, meta = restore_script(
        db_session,
        TOKENS,
        route=RESTORE_ROUTE,
        config=routes(),
        client=client_returning(meddled),
    )
    assert restored[3] == "दायरा"
    assert meta["script_restore_same_script_edits"] == 1


@pytest.mark.db
def test_an_empty_token_list_never_calls_the_model(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("the model was called for an empty transcript")

    restored, meta = restore_script(
        db_session,
        [],
        route=RESTORE_ROUTE,
        config=routes(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert restored == []
    assert meta["script_restore_tokens"] == 0
