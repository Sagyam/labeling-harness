"""Pure helpers for the distillation notebook (roadmap §B).

No torch and no NeMo, so the rules that decide what shapes the students' tokenizer and what their
scores are paired against are tested from backend/tests. Keep it importable on Python 3.10+.
"""

from __future__ import annotations

import re
import unicodedata
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
    clips: Sequence[Any],
    summarize: Callable[[list[Any]], Any],
    keys: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """`summarize` per value of each clip class (D87), as {key: {value: summary}}.

    `clips` holds one per-clip count per row (`score.per_clip`), aligned once by the caller: a
    group's score is its clips' counts added up, so no clip is aligned again per key. `keys`
    defaults to every class key any row carries. A clip without a key (or with no classes) is left
    out of that key's groups, not counted under a made-up value."""
    if len(rows) != len(clips):
        raise ValueError("rows and clips must describe the same clips")
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
            out[key] = {v: summarize([clips[i] for i in idx]) for v, idx in sorted(groups.items())}
    return out


def filter_pseudo(
    rows: Sequence[dict],
    *,
    tokens_per_s: tuple[float, float],
    drop_fraction: float,
) -> tuple[list[dict], dict[str, Any]]:
    """Roadmap §B step 3: keep the teacher's labels worth training on, cheapest rule first.

    Each row carries ``segment_id``, ``text``, ``n_tokens`` (teacher tokens), ``duration`` (s),
    ``mean_logprob`` and ``looped`` (the greedy output loops, ``ftkit.is_loop``). Dropped, in
    order: a clip whose output loops; an empty transcript; a rate outside ``tokens_per_s`` (the
    range train's labels span, in the teacher's tokens); then the least confident
    ``drop_fraction`` of what is left, by mean log-prob. Returns the kept rows, in input order,
    and a report of what each rule dropped.
    """
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError(f"drop_fraction must be in [0, 1), not {drop_fraction}")
    lo, hi = tokens_per_s
    dropped = {"loop": 0, "empty": 0, "rate": 0, "confidence": 0}
    left = []
    for r in rows:
        if r["looped"]:
            dropped["loop"] += 1
        elif not r["text"].strip():
            dropped["empty"] += 1
        elif not lo <= r["n_tokens"] / r["duration"] <= hi:
            dropped["rate"] += 1
        else:
            left.append(r)
    cut = None
    n_drop = int(len(left) * drop_fraction)
    if n_drop:
        least = sorted(left, key=lambda r: r["mean_logprob"])[:n_drop]
        cut = max(r["mean_logprob"] for r in least)
        gone = {r["segment_id"] for r in least}
        left = [r for r in left if r["segment_id"] not in gone]
        dropped["confidence"] = n_drop
    report = {
        "total": len(rows),
        "kept": len(left),
        "kept_hours": round(sum(r["duration"] for r in left) / 3600, 3),
        "dropped": dropped,
        "logprob_cut": cut,
        "tokens_per_s": [lo, hi],
        "drop_fraction": drop_fraction,
    }
    return left, report


def latin_share(text: str) -> float:
    """The share of a transcript's words written in Latin script: a code-mixing proxy that needs
    only the text, for pseudo-labels, which have no CMI class."""
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not words:
        return 0.0
    return sum(any("a" <= c.lower() <= "z" for c in w) for w in words) / len(words)


def mixing_bucket(share: float) -> str:
    """``latin_share`` in the corpus's CMI buckets (D87): 0, <15, 15-30, 30+ percent."""
    if share == 0:
        return "0"
    if share < 0.15:
        return "<15"
    return "15-30" if share < 0.30 else "30+"


# --- PreDistill: the owner's zip of recordings (D101) ------------------------------------------

_NUMBER = re.compile(r"_\d+$")


def _path_safe(text: str) -> str:
    """Letters and digits of any script, combining marks (Devanagari's vowel signs, which regex's
    word class misses), ``.``, ``-`` and ``_`` kept; every other run of characters becomes one
    ``_``."""
    kept = "".join(
        c if c.isalnum() or c in "._-" or unicodedata.category(c).startswith("M") else "\0"
        for c in text
    )
    return re.sub("\0+", "_", kept).strip("_")


def source_from_filename(name: str) -> tuple[str, str]:
    """A recording named ``<channel_name>_<NN>.mp3`` as ``(source_id, channel)``.

    The source id is the file's stem with every run of characters unsafe in a path turned into
    one ``_`` (letters of any script, digits, ``.``, ``-`` and ``_`` are kept); the channel is the
    id less its trailing ``_<number>``.
    """
    stem = name.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    source_id = _path_safe(stem)
    return source_id, _NUMBER.sub("", source_id) or source_id


def channel_blocked(channel: str, blocked: Sequence[str]) -> bool:
    """Whether a channel's name contains a blocked name, ignoring case and ``_`` against spaces."""
    name = channel.replace("_", " ").casefold()
    return any(b.strip().casefold() in name for b in blocked if b.strip())


def slice_rows(source_id: str, channel: str, slices: Sequence[tuple[float, float]]) -> list[dict]:
    """The manifest rows of one recording's VAD slices, named and timed as the labelled export's
    rows are, so ``ftkit.AudioStore`` cuts them from the whole recording."""
    return [
        {
            "segment_id": f"{source_id}_{i:05d}",
            "episode_id": source_id,
            "source_id": source_id,
            "channel": channel,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": round(end - start, 3),
        }
        for i, (start, end) in enumerate(slices)
    ]
