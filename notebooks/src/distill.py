"""Pure helpers for the distillation notebook (roadmap §B).

No torch and no NeMo, so the rules that decide what shapes the students' tokenizer and what their
scores are paired against are tested from backend/tests. Keep it importable on Python 3.10+.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def tokenizer_texts(splits: Mapping[str, Sequence[dict]]) -> list[str]:
    """The labels the shared student tokenizer is trained on: train only.

    Val and gold text never shapes the vocabulary, so a word only they contain is scored as the
    student would meet it on new audio. Raises if a val or gold clip is in train."""
    held = {r["segment_id"] for name in ("val", "gold") for r in splits.get(name, ())}
    leaked = sorted(held & {r["segment_id"] for r in splits["train"]})
    if leaked:
        raise ValueError(f"held-out clips in train: {leaked[:5]}")
    return [r["text"] for r in splits["train"]]


def reference_texts(rows: Sequence[dict], by_id: Mapping[str, str]) -> list[str]:
    """Another system's transcripts in `rows`' order, for pairing clip by clip. Raises, naming
    them, if any row has none: a pairing over a subset would not be the same clips."""
    missing = [r["segment_id"] for r in rows if r["segment_id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} clip(s) have no reference transcript: {missing[:5]}")
    return [by_id[r["segment_id"]] for r in rows]


def by_class(
    rows: Sequence[dict],
    refs: Sequence[str],
    hyps: Sequence[str],
    score: Callable[[Sequence[str], Sequence[str]], Any],
    keys: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """`score` per value of each clip class (D87), as {key: {value: score}}.

    `keys` defaults to every class key any row carries. A clip without a key (or with no classes)
    is left out of that key's groups, not counted under a made-up value."""
    if not (len(rows) == len(refs) == len(hyps)):
        raise ValueError("rows, refs and hyps must describe the same clips")
    if keys is None:
        keys = sorted({k for r in rows for k in (r.get("classes") or {})})
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            value = (r.get("classes") or {}).get(key)
            if value is not None:
                groups.setdefault(str(value), []).append(i)
        if groups:
            out[key] = {
                v: score([refs[i] for i in idx], [hyps[i] for i in idx])
                for v, idx in sorted(groups.items())
            }
    return out
