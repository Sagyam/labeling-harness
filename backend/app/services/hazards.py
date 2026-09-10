"""Hazard gates and ranking evidence for a fused seed (D74).

The seed the annotator sees is now written by a model that read the recognisers and never heard
the audio. Every term of the old score measured the seed *against* the recognisers, and a
transcript built to reconcile them agrees with them by construction -- that score would collapse
toward zero and read as quality. So this module asks a different question: where does the fused
text say something the evidence does not support, or leave out something it does?

The evidence is the three recognisers (independent, and compared by sound via ``fold.py``, so a
respelling is never a disagreement), the neighbouring clips (for text moved across a seam), the
fuser's own code, and the waveform (the aligner's :class:`AcousticFit`).

**Gates** are the catastrophic failure shapes. Any one of them makes a clip unscreenable and puts
it at the top of the queue; they are deliberately about *shape*, not size, because a three-word
invention in a forty-word clip barely moves an error rate:

* ``invention`` -- a contiguous run of fused words that no recogniser heard, even approximately.
* ``dropped`` -- a contiguous run that at least two recognisers agree on and the fused text lacks.
  Every orphan-style metric is blind to deletion by construction, and deletion is what a
  fluency-driven model does most.
* ``seam_bleed`` -- unheard fused words that the neighbouring clip's recognisers did hear.
* ``length_outlier``, ``emptied``, ``speech_over_silence`` -- the fused text is the wrong size for
  what the recognisers heard.
* ``unaligned`` -- the aligner cannot fit the text into the clip at all.
* ``fuser_uncertain`` -- the fuser said so itself. Its self-report is weak on recall (3 of 896 in
  the pilot) and precise, so it routes and never approves.
* ``unfused`` -- no fused text; the seed is a bare recogniser.

What no gate can see, and the decision entry says so: a knowledgeable correction of something the
speaker actually misstated sounds right, aligns fine and agrees with nothing -- or with one
recogniser -- and passes every check here.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from app.config import HazardSettings
from app.services.fold import align, fold_tokens, romanized
from app.services.normalize import Ruleset

GATES = (
    "unfused",
    "invention",
    "dropped",
    "seam_bleed",
    "length_outlier",
    "emptied",
    "speech_over_silence",
    "unaligned",
    "fuser_uncertain",
)


@dataclass(frozen=True)
class FusionEvidence:
    """Everything one clip's assessment reads."""

    #: The fused transcript, or None when the fuser never answered for this clip.
    fused_text: str | None
    #: Each recogniser's transcript of this clip, in any order.
    asr_texts: Sequence[str]
    #: Recogniser transcripts of the clips immediately before and after.
    neighbour_texts: Sequence[str] = ()
    #: The fuser's one-letter code for this clip.
    fused_code: str | None = None
    #: :meth:`AcousticFit.as_dict` for the fused text, or None when no aligner ran.
    acoustic: dict[str, Any] | None = None


@dataclass
class HazardReport:
    """Gates that fired, the ranking components, and the words behind each gate."""

    hazards: list[str] = field(default_factory=list)
    #: Share of fused words no recogniser heard, by sound. 0-1.
    unsupported_rate: float = 0.0
    #: Words two recognisers agree on and the fused text lacks, against the recognisers' median
    #: length. 0-1.
    dropped_rate: float = 0.0
    #: Mean pairwise script-folded word error among the recognisers: how hard the audio is. 0-1.
    asr_disagreement: float = 0.0
    #: The acoustic gap on a 0-1 scale, or None when unmeasured.
    acoustic_gap: float | None = None
    #: ``{gate: the words that fired it}``, for the tooltip.
    details: dict[str, str] = field(default_factory=dict)

    def gate(self, name: str, detail: str | None = None) -> None:
        if name not in self.hazards:
            self.hazards.append(name)
        if detail:
            self.details.setdefault(name, detail)


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive ``True``."""
    out: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            out.append((start, index))
            start = None
    return out


def _support(fused: list[str], other: list[str], near: float) -> tuple[list[bool], list[bool]]:
    """Which fused words ``other`` heard, and which of ``other``'s words the fused text lacks."""
    heard = [False] * len(fused)
    lacked = [False] * len(other)
    f = o = 0
    for op in align(fused, other).ops:
        if op.kind in ("match", "fold", "merge") or (op.kind == "sub" and op.similarity >= near):
            for offset in range(len(op.ref)):
                heard[f + offset] = True
        elif op.kind == "ins":
            lacked[o] = True
        f += len(op.ref)
        o += len(op.hyp)
    return heard, lacked


def _matched(a: list[str], b: list[str]) -> int:
    return sum(len(op.ref) for op in align(a, b).ops if op.kind in ("match", "fold", "merge"))


def _chars(tokens: list[str]) -> int:
    return sum(len(romanized(t)) for t in tokens)


def assess(
    evidence: FusionEvidence,
    *,
    config: HazardSettings,
    ruleset: Ruleset | None = None,
) -> HazardReport:
    """Assess one clip's fused seed. Pure: no database, no audio."""
    report = HazardReport()
    heard_asr = [
        tokens for tokens in (fold_tokens(t, ruleset) for t in evidence.asr_texts) if tokens
    ]
    pairs = list(combinations(heard_asr, 2))
    if pairs:
        report.asr_disagreement = round(
            statistics.fmean(min(1.0, align(a, b).errors / max(len(a), len(b))) for a, b in pairs),
            4,
        )

    if evidence.fused_text is None:
        report.gate("unfused")
        return report

    fused = fold_tokens(evidence.fused_text, ruleset)
    speaking = [tokens for tokens in heard_asr if len(tokens) >= 2]

    if not fused:
        if len(speaking) >= 2:
            report.gate("emptied", " / ".join(" ".join(t) for t in speaking[:2]))
        return report
    if not heard_asr:
        if len(fused) >= 2:
            report.gate("speech_over_silence", " ".join(fused))
        report.unsupported_rate = 1.0
        return report

    # Support: which fused words each recogniser heard, and what each heard that fusion lacks.
    heard_by_any = [False] * len(fused)
    lacked_by_system: list[list[list[str]]] = []
    for tokens in heard_asr:
        heard, lacked = _support(fused, tokens, config.near_similarity)
        heard_by_any = [a or b for a, b in zip(heard_by_any, heard, strict=True)]
        lacked_by_system.append([tokens[start:end] for start, end in _runs(lacked)])
    unheard = [not h for h in heard_by_any]
    report.unsupported_rate = round(sum(unheard) / len(fused), 4)

    # Invention, or text moved across a seam: a contiguous unheard run, classified by whether the
    # neighbouring clips' recognisers heard it.
    neighbours = [fold_tokens(t, ruleset) for t in evidence.neighbour_texts]
    for start, end in _runs(unheard):
        run = fused[start:end]
        bled = len(run) >= config.bleed_words and any(
            _matched(run, n) >= max(config.bleed_words, round(0.6 * len(run)))
            for n in neighbours
            if n
        )
        if bled:
            report.gate("seam_bleed", " ".join(run))
        elif len(run) >= config.invention_run:
            report.gate("invention", " ".join(run))

    # Dropped: a run one recogniser heard and fusion lacks, corroborated by a second recogniser.
    corroborated = 0
    for runs_i, runs_j in combinations(lacked_by_system, 2):
        for run_i in runs_i:
            for run_j in runs_j:
                shared = _matched(run_i, run_j)
                if shared >= config.dropped_run:
                    report.gate("dropped", " ".join(run_i))
                corroborated = max(corroborated, shared)
    median_len = statistics.median(len(t) for t in heard_asr)
    report.dropped_rate = round(min(1.0, corroborated / median_len), 4) if median_len else 0.0

    # Size: romanized characters, so a script difference is not a length difference.
    fused_chars = _chars(fused)
    median_chars = statistics.median(_chars(t) for t in heard_asr)
    if median_chars and abs(fused_chars - median_chars) >= config.length_min_chars:
        ratio = fused_chars / median_chars
        if ratio > config.length_ratio or ratio < 1 / config.length_ratio:
            report.gate("length_outlier", f"{fused_chars} chars against {median_chars:g}")

    acoustic = evidence.acoustic
    if acoustic is not None:
        if acoustic.get("aligned") is False:
            report.gate("unaligned")
        elif acoustic.get("gap") is not None:
            report.acoustic_gap = round(
                min(1.0, max(0.0, float(acoustic["gap"]) / config.acoustic_gap_full_scale)), 4
            )

    if evidence.fused_code == "u":
        report.gate("fuser_uncertain")
    return report
