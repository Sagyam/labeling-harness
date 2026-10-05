"""Early stopping in the notebooks' training loop (notebooks/src/ftkit.py, `patience_step`).

ftkit imports torch, which never enters the backend, so the rule is read out of its source and
run alone: it is a pure function of numbers."""

from __future__ import annotations

import ast
from pathlib import Path

_FTKIT = Path(__file__).resolve().parents[2] / "notebooks" / "src" / "ftkit.py"


def _patience_step():
    tree = ast.parse(_FTKIT.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "patience_step")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_FTKIT), "exec"), ns)
    return ns["patience_step"]


def _run(wers: list[float], min_delta: float = 0.2) -> list[int]:
    """`bad` after each evaluation, from a fresh run."""
    step, counted, bad, out = _patience_step(), float("inf"), 0, []
    for wer in wers:
        counted, bad = step(wer, counted, bad, min_delta)
        out.append(bad)
    return out


def test_a_gain_resets_and_a_crawl_counts() -> None:
    assert _run([20.0, 15.0, 14.9, 14.85, 14.81]) == [0, 0, 1, 2, 3]


def test_a_gain_is_measured_from_the_last_counted_evaluation() -> None:
    # 14.9 and 14.85 do not reset; 14.75 is 0.25 below 15.0, the last WER that did
    assert _run([15.0, 14.9, 14.85, 14.75]) == [0, 1, 2, 0]


def test_a_model_that_writes_nothing_has_not_started() -> None:
    """06f, 2026-10-05: from random weights, val sat at 100.00 (all deletions) for four epochs while
    the train loss fell from 29 to 4.5, and patience stopped the run after epoch 4."""
    assert _run([100.0, 100.0, 100.0, 100.0, 100.0]) == [0, 0, 0, 0, 0]


def test_garbage_longer_than_the_reference_has_not_started_either() -> None:
    assert _run([632.4, 140.0, 100.0]) == [0, 0, 0]


def test_patience_starts_at_the_first_evaluation_below_100() -> None:
    assert _run([100.0, 100.0, 100.0, 80.0, 79.9, 79.9, 79.9]) == [0, 0, 0, 0, 1, 2, 3]
