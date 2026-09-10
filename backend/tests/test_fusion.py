"""Tests for transcript fusion: windows, prompt, response contract, retry and bisection (D72)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.llm.base import LlmRequestFailed, LlmResult
from app.llm.fusion import (
    SYSTEM_INSTRUCTION,
    FusionSegment,
    fuse,
    parse_response,
    plan_windows,
    render_window,
)


def segs(
    durations: list[float], *, texts: tuple[str, str, str] | None = None
) -> list[FusionSegment]:
    out, cursor = [], 0.0
    for index, duration in enumerate(durations):
        hyps = texts or (f"scribe {index}", f"mai {index}", f"gemini {index}")
        out.append(
            FusionSegment(
                key=f"seg{index:04d}", start=cursor, end=cursor + duration, hypotheses=hyps
            )
        )
        cursor += duration + 0.5
    return out


# --- windows ---------------------------------------------------------------------------------


def test_a_short_episode_is_one_window_with_no_seams() -> None:
    windows = plan_windows(segs([12.0] * 50), target_seconds=1800, lookahead_seconds=120)
    assert len(windows) == 1
    assert len(windows[0].targets) == 50
    assert windows[0].after == []


def test_windows_are_balanced_rather_than_full_plus_remainder() -> None:
    # 45 minutes at a 30-minute target is two windows of ~22.5, not 30 + 15: a short last
    # window pays the same thinking cost for half the work.
    windows = plan_windows(segs([15.0] * 180), target_seconds=1800, lookahead_seconds=120)
    assert len(windows) == 2
    sizes = [len(w.targets) for w in windows]
    assert abs(sizes[0] - sizes[1]) <= 1


def test_every_segment_is_a_target_exactly_once_and_in_order() -> None:
    segments = segs([7.0, 19.0, 3.0, 11.0] * 60)
    windows = plan_windows(segments, target_seconds=600, lookahead_seconds=60)
    flattened = [s.key for w in windows for s in w.targets]
    assert flattened == [s.key for s in segments]


def test_lookahead_is_the_segments_that_follow_up_to_its_duration() -> None:
    segments = segs([10.0] * 200)
    first, *_ = plan_windows(segments, target_seconds=1000, lookahead_seconds=35)
    last_target = first.targets[-1].key
    following = [s.key for s in segments]
    start = following.index(last_target) + 1
    assert [s.key for s in first.after] == following[start : start + 4]


def test_a_window_is_capped_at_a_segment_count() -> None:
    windows = plan_windows(
        segs([2.0] * 500), target_seconds=3600, lookahead_seconds=0, max_segments=200
    )
    assert max(len(w.targets) for w in windows) <= 200


def test_no_segments_no_windows() -> None:
    assert plan_windows([], target_seconds=1800, lookahead_seconds=120) == []


# --- prompt ----------------------------------------------------------------------------------


def test_the_prompt_names_no_system() -> None:
    """The fuser is a Gemini model arbitrating a slate with a Gemini transcript in it (§3.5).
    Nothing tells it which one is its own."""
    import re

    window = plan_windows(
        segs([5.0] * 3, texts=("एक", "दुई", "तीन")), target_seconds=1800, lookahead_seconds=0
    )[0]
    rendered = render_window(window, carryover=[])
    for text in (SYSTEM_INSTRUCTION, rendered):
        assert not re.search(r"\b(scribe|mai|gemini|elevenlabs|microsoft|google)\b", text, re.I)
    assert "A: एक" in rendered and "B: दुई" in rendered and "C: तीन" in rendered


def test_targets_carry_ids_and_context_does_not() -> None:
    segments = segs([5.0] * 6)
    windows = plan_windows(segments, target_seconds=12, lookahead_seconds=6)
    rendered = render_window(windows[1], carryover=[(segments[0], "पहिलेको text")])
    assert "पहिलेको text" in rendered
    ids = [int(m) for m in __import__("re").findall(r"^\[(\d+)\]", rendered, flags=8)]
    assert ids == [windows[1].ids[s.key] for s in windows[1].targets]


# --- response contract -----------------------------------------------------------------------


def test_parse_reads_the_array() -> None:
    parsed = parse_response('[{"id": 3, "t": "नमस्ते", "c": "k"}]', [3])
    assert parsed.texts == {3: "नमस्ते"}
    assert parsed.codes == {3: "k"}
    assert parsed.problems == []


def test_parse_tolerates_a_fence() -> None:
    parsed = parse_response('```json\n[{"id": 1, "t": "a", "c": "m"}]\n```', [1])
    assert parsed.problems == []


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        ('[{"id": 1, "t": "a", "c": "k"}]', "missing"),
        ('[{"id": 1, "t": "a"}, {"id": 2, "t": "b"}, {"id": 9, "t": "x"}]', "unexpected"),
        ('[{"id": 1, "t": "a"}, {"id": 1, "t": "b"}, {"id": 2, "t": "c"}]', "duplicate"),
        ("I think the transcript is", "no JSON array"),
        ('[{"id": 1, "t": "a"', "no JSON array"),
    ],
)
def test_parse_reports_a_broken_contract(body: str, problem: str) -> None:
    assert any(problem in p for p in parse_response(body, [1, 2]).problems)


def test_an_unknown_code_is_kept_as_unknown_not_fatal() -> None:
    parsed = parse_response('[{"id": 1, "t": "a", "c": "zz"}]', [1])
    assert parsed.codes == {1: "?"}
    assert parsed.problems == []


# --- the loop --------------------------------------------------------------------------------


def _answer(targets: list[tuple[int, str]], *, finish: str = "STOP") -> LlmResult:
    body = json.dumps([{"id": i, "t": t, "c": "k"} for i, t in targets], ensure_ascii=False)
    return LlmResult(
        route="fuse_transcript",
        model="gemini-3.8-flash",
        text=body,
        raw={
            "candidates": [{"finishReason": finish}],
            "modelVersion": "gemini-3.8-flash-001",
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 10,
                "thoughtsTokenCount": 50,
            },
        },
    )


def _ids_in(messages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """The target ids of a rendered window, paired with hypothesis A -- an echoing fuser."""
    import re

    user = messages[-1]["content"]
    block = user.split("### TRANSCRIBE THESE", 1)[1].split("###", 1)[0]
    return [
        (int(i), a.strip())
        for i, a in re.findall(r"^\[(\d+)\][^\n]*\n  A: ([^\n]*)", block, flags=re.MULTILINE)
    ]


def echo(messages: list[dict[str, Any]]) -> LlmResult:
    return _answer(_ids_in(messages))


def test_every_segment_is_fused_through_an_echoing_model() -> None:
    segments = segs([10.0] * 30)
    outcome = fuse(segments, complete=echo, target_seconds=100, lookahead_seconds=20)
    assert outcome.unfused == []
    assert {k: v.text for k, v in outcome.fused.items()} == {
        s.key: s.hypotheses[0] for s in segments
    }
    assert all(v.code == "k" for v in outcome.fused.values())


def test_a_later_window_is_shown_the_earlier_windows_output() -> None:
    seen: list[str] = []

    def recording(messages: list[dict[str, Any]]) -> LlmResult:
        seen.append(messages[-1]["content"])
        answer = [(i, f"FUSED {i}") for i, _ in _ids_in(messages)]
        return _answer(answer)

    fuse(
        segs([10.0] * 20),
        complete=recording,
        target_seconds=100,
        lookahead_seconds=0,
        carryover_seconds=30,
    )
    assert len(seen) == 2
    assert "ALREADY TRANSCRIBED" not in seen[0]
    assert "ALREADY TRANSCRIBED" in seen[1] and "FUSED" in seen[1].split("### TRANSCRIBE")[0]


def test_a_window_with_a_missing_id_is_retried_then_bisected() -> None:
    calls: list[int] = []

    def drops_one(messages: list[dict[str, Any]]) -> LlmResult:
        targets = _ids_in(messages)
        calls.append(len(targets))
        # Any window bigger than four drops its last segment; halves of it are answered whole.
        return _answer(targets[:-1] if len(targets) > 4 else targets)

    outcome = fuse(segs([10.0] * 8), complete=drops_one, target_seconds=1000, lookahead_seconds=0)
    assert outcome.unfused == []
    assert calls == [8, 8, 4, 4]
    assert any(c.problems for c in outcome.calls)


def test_a_truncated_answer_is_bisected_not_patched() -> None:
    def truncates_big(messages: list[dict[str, Any]]) -> LlmResult:
        targets = _ids_in(messages)
        return _answer(targets, finish="MAX_TOKENS" if len(targets) > 3 else "STOP")

    outcome = fuse(
        segs([10.0] * 6), complete=truncates_big, target_seconds=1000, lookahead_seconds=0
    )
    assert outcome.unfused == []
    assert all(v.window_depth == 1 for v in outcome.fused.values())


def test_a_window_that_never_answers_leaves_its_segments_unfused_and_the_rest_fused() -> None:
    def fails_second_window(messages: list[dict[str, Any]]) -> LlmResult:
        targets = _ids_in(messages)
        if targets and targets[0][0] >= 10:
            raise LlmRequestFailed("route 'fuse_transcript' failed: HTTP 500")
        return _answer(targets)

    outcome = fuse(
        segs([10.0] * 20),
        complete=fails_second_window,
        target_seconds=100,
        lookahead_seconds=0,
        max_depth=1,
    )
    assert sorted(outcome.unfused) == [f"seg{i:04d}" for i in range(10, 20)]
    assert len(outcome.fused) == 10


def test_a_quota_refusal_waits_and_tries_again() -> None:
    waits: list[float] = []
    attempts = {"n": 0}

    def quota_then_ok(messages: list[dict[str, Any]]) -> LlmResult:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LlmRequestFailed("route 'fuse_transcript' failed: HTTP 429 Too Many Requests")
        return echo(messages)

    outcome = fuse(
        segs([10.0] * 3),
        complete=quota_then_ok,
        target_seconds=1000,
        lookahead_seconds=0,
        sleep=waits.append,
    )
    assert outcome.unfused == []
    assert waits and waits[0] >= 30


def test_a_stop_request_ends_the_run_between_windows() -> None:
    calls = {"n": 0}

    def counting(messages: list[dict[str, Any]]) -> LlmResult:
        calls["n"] += 1
        return echo(messages)

    outcome = fuse(
        segs([10.0] * 30),
        complete=counting,
        target_seconds=100,
        lookahead_seconds=0,
        should_stop=lambda: calls["n"] >= 1,
    )
    assert calls["n"] == 1
    assert len(outcome.unfused) == 20


def test_calls_record_what_they_cost_in_thought() -> None:
    outcome = fuse(segs([10.0] * 3), complete=echo, target_seconds=1000, lookahead_seconds=0)
    (call,) = outcome.calls
    assert (call.prompt_tokens, call.thought_tokens, call.output_tokens) == (100, 50, 10)
    assert call.finish_reason == "STOP"
    assert call.model_version == "gemini-3.8-flash-001"
