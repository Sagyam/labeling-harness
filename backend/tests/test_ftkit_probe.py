"""Which clip the micro-batch probe also measures (notebooks/src/ftkit.py, `crossover_clip`).

ftkit imports torch, which never enters the backend, so the rule is read out of its source and
run alone: it is a pure function of rows."""

from __future__ import annotations

import ast
from pathlib import Path

_FTKIT = Path(__file__).resolve().parents[2] / "notebooks" / "src" / "ftkit.py"


def _crossover_clip():
    tree = ast.parse(_FTKIT.read_text(encoding="utf-8"))
    names = {"duration", "crossover_clip"}
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns: dict = {}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(_FTKIT), "exec"), ns)
    return ns["crossover_clip"]


def _row(seconds: float, text: str) -> dict:
    return {"start_time": 0.0, "end_time": seconds, "text": text}


def test_the_probe_also_measures_the_length_where_both_limits_meet() -> None:
    """A budget of 720 s and 163 clips meet at clips of 4.4 s: a batch of 163 of those is legal
    under both limits, and neither the longest nor the shortest clip measures it."""
    pick = _crossover_clip()
    rows = [_row(2.0, "a"), _row(4.3, "a b"), _row(4.5, "a b c d e"), _row(9.0, "a b c d e f g")]
    assert pick(rows, budget_s=720.0, items=163) == rows[2]  # the wordiest near 4.4 s


def test_the_crossover_clip_is_the_wordiest_within_half_a_second() -> None:
    pick = _crossover_clip()
    rows = [_row(4.0, "a b c d e f"), _row(5.5, "a b c d e f g h i"), _row(4.9, "a")]
    # 4.4 s: 4.0 and 4.9 are within half a second, 5.5 is not
    assert pick(rows, budget_s=440.0, items=100) == rows[0]


def test_with_no_clip_near_the_crossover_the_nearest_is_measured() -> None:
    pick = _crossover_clip()
    rows = [_row(2.0, "a b c"), _row(12.0, "a")]
    assert pick(rows, budget_s=500.0, items=100) == rows[0]  # 5 s: 2.0 is nearer than 12.0


def test_a_clip_count_budget_has_no_crossover() -> None:
    """Whisper pays for 30 s whatever a clip's length: its budget is a count, not seconds."""
    pick = _crossover_clip()
    assert pick([_row(3.0, "a")], budget_s=float("inf"), items=40) is None


def test_the_wordiest_is_measured_in_the_students_own_tokens() -> None:
    """Characters are the wrong measure: a Devanagari character costs a byte-level tokenizer
    several tokens and a Latin one about a quarter of one, so the clip with the most characters
    was not the one with the most tokens, and 06b's probe under-measured its batches."""
    pick = _crossover_clip()
    rows = [_row(4.4, "a long english sentence here"), _row(4.4, "नमस्ते")]  # namaste
    tokens = {rows[0]["text"]: 5, rows[1]["text"]: 18}
    assert pick(rows, budget_s=440.0, items=100) == rows[0]  # by characters
    chosen = pick(rows, budget_s=440.0, items=100, size=lambda r: tokens[r["text"]])
    assert chosen == rows[1]
