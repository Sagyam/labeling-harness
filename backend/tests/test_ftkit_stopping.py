"""Early stopping in the notebooks' training loop (notebooks/src/ftkit.py, `patience_step`).

ftkit imports torch, which never enters the backend, so the rule is read out of its source and
run alone: it is a pure function of numbers."""

from __future__ import annotations

import ast
from pathlib import Path

_FTKIT = Path(__file__).resolve().parents[2] / "notebooks" / "src" / "ftkit.py"


def _ftkit(*names: str) -> dict:
    tree = ast.parse(_FTKIT.read_text(encoding="utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns: dict = {}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(_FTKIT), "exec"), ns)
    return ns


def _patience_step():
    return _ftkit("patience_step")["patience_step"]


def _run(wers: list[float], min_delta: float = 0.2, warming: int = 0) -> list[int]:
    """`bad` after each evaluation, from a fresh run; the first `warming` end inside warmup."""
    step, counted, bad, out = _patience_step(), float("inf"), 0, []
    for i, wer in enumerate(wers):
        counted, bad = step(wer, counted, bad, min_delta, warming=i < warming)
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


def test_an_evaluation_inside_warmup_counts_for_nothing() -> None:
    """06f stage 2, 2026-10-05: it starts from stage 1's weights and re-warms its learning rate
    over three epochs. Val went 31.68, 31.84, 32.28 as the rate rose, and patience counted the
    climb, so the first evaluation at peak rate would have had to beat 31.48 or end the stage."""
    assert _run([31.68, 31.84, 32.28, 31.9, 31.5, 31.0], warming=3) == [0, 0, 0, 0, 0, 0]


def test_patience_starts_at_the_first_evaluation_after_warmup() -> None:
    # 31.9 sets the mark, not 31.68 from inside warmup
    assert _run([31.68, 31.84, 32.28, 31.9, 31.8, 31.75, 31.71], warming=3) == [0, 0, 0, 0, 1, 2, 3]


def _replay(history: list[dict], warm: int, min_delta: float = 0.2) -> tuple[float, int]:
    return _ftkit("patience_step", "patience_replay")["patience_replay"](history, warm, min_delta)


def test_a_resumed_run_counts_its_history_by_the_current_rule() -> None:
    """A resume point saved under the old rule carries `bad` 2 for the warmup climb; the resumed
    run rebuilds the count from the history it stored, so the rule in force is the one applied."""
    history = [
        {"epoch": 1, "step": 2506, "val_wer": 31.68},
        {"epoch": 2, "step": 5012, "val_wer": 31.84},
        {"epoch": 3, "step": 7518, "val_wer": 32.28},
    ]
    assert _replay(history, warm=7519) == (float("inf"), 0)
    assert _replay(history, warm=7518) == (32.28, 0)  # epoch 3 ends at peak rate: it counts
    assert _replay(history, warm=1) == (31.68, 2)  # the old count, when nothing is warmup


def test_a_replay_skips_blank_evaluations_too() -> None:
    history = [
        {"epoch": e, "step": 100 * e, "val_wer": w} for e, w in enumerate([100.0, 80.0, 79.9], 1)
    ]
    assert _replay(history, warm=1) == (80.0, 1)


def test_a_replay_reads_only_the_evaluations_among_the_step_logs() -> None:
    """`train`'s history holds a record per logged step (loss, lr, ...) besides one per evaluation;
    the first 06f stage 2 resume (2026-10-06) failed on a step record with KeyError 'val_wer'."""
    history = [
        {"step": 10, "epoch": 1, "lr": 1e-5, "loss": 3.1},
        {"epoch": 1, "step": 2506, "val_wer": 31.68},
        {"step": 2510, "epoch": 2, "lr": 3e-4, "loss": 2.9},
        {"epoch": 2, "step": 5012, "val_wer": 31.84},
    ]
    assert _replay(history, warm=1) == (31.68, 1)
