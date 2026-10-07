"""The speech-LLM bake-off's rules (D121): which audio LLMs are worth training as students?

The search is breadth-first: every candidate is swept zero-shot on a val sample first, and only
the shortlist has its training step timed and its full val and gold decoded.
Zero-shot WER says little about how a model trains (Whisper-turbo read 123% before fine-tuning
and 15.02 after it), so the transcripts serve as a gate, not a ranking:

  * **The gate.** Among the clips whose reference is mostly Devanagari, the share whose answer is
    mostly Devanagari too must be at least `GATE_SHARE`. A model that romanises Nepali, answers in
    another script, or stays silent has to learn a script before it can learn the language.
    Hindi passes the script check, being Devanagari too; its WER shows it.
  * **The failures** an LLM decoder adds, counted beside the WER: empty answers, runaways (far
    more words than the reference) and answers in another script.
  * **The budget.** Training throughput, projected over both student stages, must fit
    `BUDGET_H` GPU-hours.

The WER itself comes from evalkit, like every other number (D105). Pure Python; importable
without torch, on Python 3.10+.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

#: The gate: this share of the mostly-Nepali clips must be answered mostly in Devanagari (D121).
GATE_SHARE = 0.90
#: A clip counts toward the gate when at least this share of its reference's letters are Devanagari.
NEPALI_REF_SHARE = 0.5
#: An answer writes Nepali when at least this share of its letters are Devanagari.
WRITES_SHARE = 0.5
#: An answer is a runaway when it has more than `RUNAWAY_FACTOR` times the reference's words plus
#: `RUNAWAY_SLACK`: the decoder kept going after the speech ended.
RUNAWAY_FACTOR, RUNAWAY_SLACK = 2, 10

_DEVANAGARI = (0x0900, 0x097F)
_QUOTES = {'"': '"', "'": "'", "`": "`", "\u201c": "\u201d", "\u2018": "\u2019", "\u00ab": "\u00bb"}


def _script(ch: str) -> str | None:
    """'deva', 'latin' or 'other' for a letter or combining sign; None for anything else."""
    if unicodedata.category(ch)[0] not in "LM":
        return None
    if _DEVANAGARI[0] <= ord(ch) <= _DEVANAGARI[1]:
        return "deva"
    if unicodedata.name(ch, "").startswith("LATIN"):
        return "latin"
    return "other"


def script_letters(text: str) -> dict[str, int]:
    """Letters and combining signs of `text` by script: Devanagari (vowel signs and the virama
    included), Latin, and every other script. Digits, punctuation and spaces are not counted."""
    counts = {"deva": 0, "latin": 0, "other": 0}
    for ch in text:
        script = _script(ch)
        if script is not None:
            counts[script] += 1
    return counts


def script_shares(text: str) -> dict[str, float]:
    """`script_letters` as shares of all letters; all zero when there are none."""
    counts = script_letters(text)
    total = sum(counts.values())
    return {k: (v / total if total else 0.0) for k, v in counts.items()}


def clean(text: str) -> str:
    """The answer as scored: whitespace (newlines included) collapsed to single spaces, and one
    pair of quotes or backticks around the whole answer removed. Nothing else: a preamble such as
    "Here is the transcription:" stays, and is charged as insertions."""
    text = " ".join(text.split())
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] == _QUOTES[text[0]]:
        inner = text[1:-1]
        if text[0] not in inner and _QUOTES[text[0]] not in inner:
            text = inner.strip()
    return text


def is_runaway(hyp: str, ref: str) -> bool:
    return len(hyp.split()) > RUNAWAY_FACTOR * len(ref.split()) + RUNAWAY_SLACK


def zero_shot_report(
    refs: Sequence[str], hyps: Sequence[str], gate_share: float = GATE_SHARE
) -> dict[str, Any]:
    """The gate and the failure counts of one candidate's answers on one set.

    `writes_nepali` is the share of the mostly-Nepali clips (`nepali_clips`) answered mostly in
    Devanagari; `passes_gate` is whether it reaches `gate_share`. `latin_ref` and `latin_hyp` are
    the Latin share of all letters, over the whole set, in the references and in the answers."""
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references against {len(hyps)} answers")
    nepali = writes = empty = runaway = other = 0
    ref_letters = {"deva": 0, "latin": 0, "other": 0}
    hyp_letters = {"deva": 0, "latin": 0, "other": 0}
    for ref, hyp in zip(refs, hyps, strict=True):
        r, h = script_letters(ref), script_letters(hyp)
        for k in ref_letters:
            ref_letters[k] += r[k]
            hyp_letters[k] += h[k]
        r_total, h_total = sum(r.values()), sum(h.values())
        if not hyp.strip():
            empty += 1
        if is_runaway(hyp, ref):
            runaway += 1
        if h_total and h["other"] / h_total >= WRITES_SHARE:
            other += 1
        if r_total and r["deva"] / r_total >= NEPALI_REF_SHARE:
            nepali += 1
            if h_total and h["deva"] / h_total >= WRITES_SHARE:
                writes += 1
    share = writes / nepali if nepali else 0.0
    ref_total, hyp_total = sum(ref_letters.values()), sum(hyp_letters.values())
    return {
        "clips": len(refs),
        "nepali_clips": nepali,
        "writes_nepali": share,
        "passes_gate": bool(nepali) and share >= gate_share,
        "empty": empty,
        "runaway": runaway,
        "other_script_clips": other,
        "latin_ref": ref_letters["latin"] / ref_total if ref_total else 0.0,
        "latin_hyp": hyp_letters["latin"] / hyp_total if hyp_total else 0.0,
    }


def pick_prompt(reports: Mapping[str, Mapping[str, Any]]) -> str:
    """A candidate's prompt for the full run, from its reports on the val sample: the lowest WER
    among the prompts that pass the gate, or, when none does, the one closest to it. A tie keeps
    the prompt named first. Gold never takes part (D105)."""
    names = list(reports)
    passing = [n for n in names if reports[n]["passes_gate"]]
    if passing:
        return min(passing, key=lambda n: (reports[n]["wer"], names.index(n)))
    return max(names, key=lambda n: (reports[n]["writes_nepali"], -names.index(n)))


def shortlist(sweeps: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[str]:
    """The candidates that go on from the sweep (round 1) by default: those whose kept prompt
    passes the gate on the val sample, in the sweep's order. One not yet swept is left out."""
    return [
        k
        for k, prompts in sweeps.items()
        if prompts and prompts[pick_prompt(prompts)]["passes_gate"]
    ]


def projection(
    *,
    train_x_realtime: float,
    val_x_realtime: float,
    human_h: float,
    human_epochs: int,
    distill_h: float,
    distill_epochs: int,
    val_h: float,
) -> dict[str, float]:
    """GPU-hours for the two student stages at the measured speeds, every epoch run (early
    stopping can only cut it): stage 1 trains on `human_h` of audio for `human_epochs`, stage 2 on
    a `distill_h` epoch for `distill_epochs`, and each epoch ends with a val pass of `val_h`."""
    if train_x_realtime <= 0 or val_x_realtime <= 0:
        raise ValueError("a projection needs measured speeds above zero")
    val_pass = val_h / val_x_realtime
    stage1 = human_epochs * (human_h / train_x_realtime + val_pass)
    stage2 = distill_epochs * (distill_h / train_x_realtime + val_pass)
    return {"stage1_h": stage1, "stage2_h": stage2, "total_h": stage1 + stage2}


def verdict(
    gate: Mapping[str, Any], cost: Mapping[str, float] | None, budget_h: float
) -> dict[str, Any]:
    """Whether a candidate goes on to be a student (D121): it passes the gate on val, and its
    projected training fits `budget_h`. `reasons` names each rule it fails, in that order."""
    reasons = []
    if not gate.get("passes_gate"):
        reasons.append("gate: writes Devanagari on too few of the Nepali clips")
    if cost is None:
        reasons.append("budget: training was not measured")
    elif cost["total_h"] > budget_h:
        reasons.append(f"budget: {cost['total_h']:.1f} h projected against {budget_h:g} h")
    return {"continue": not reasons, "reasons": reasons}


def token_cap(
    tokens_per_s: float, seconds: float, slack: float = 1.5, extra: int = 8, ceiling: int = 1024
) -> int:
    """The most tokens a zero-shot answer may run to: `slack` times the densest training label's
    rate (in the candidate's own tokens) over the clip, plus `extra`, at most `ceiling`. A runaway
    is cut there rather than left to fill the context."""
    return min(ceiling, math.ceil(slack * tokens_per_s * seconds) + extra)
