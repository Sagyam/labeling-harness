"""Acoustic timestamp cross-verification and word boundary alignment service.

Compares word-level boundaries from two *independent* timing sources over the same clip:

1. Matches the two systems' word lists against each other by normalized text.
2. Computes per-boundary delta (|Δ_start|, |Δ_end|) in milliseconds on matched pairs only.
3. Evaluates publication sanity-check metrics: % within ±25ms, ±50ms, and ±100ms.
4. Stratifies boundary precision across Monolingual vs. Code-Switched tokens.
5. Flags divergent segments (Δ > 200ms) for human-review triage in the harness.

The two sources must be genuinely independent for the numbers to mean anything (D33). Today
that is ElevenLabs Scribe's native word spans as the reference, against the local CTC forced
aligner's spans over Gemini Flash's transcript as the comparison. A token present on only one
side is counted but never scored: two systems that disagree about *which word was said* have
no meaningful boundary delta to report.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEV_RE = re.compile(r"[ऀ-ॿ]")
WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[ऀ-ॿ]+")

#: Systems whose word spans are compared on an analytics export. The reference is the system
#: trusted to be right; the comparison is the one being measured against it.
DEFAULT_REFERENCE_SYSTEM_ID = "elevenlabs-scribe-v2"
DEFAULT_COMPARISON_SYSTEM_ID = "gemini-3.8-flash"


def normalize_token(tok: str) -> str:
    """Normalize word for robust alignment (NFKC, lowercase, strip punctuation)."""
    norm = unicodedata.normalize("NFKC", tok).strip().lower()
    return re.sub(r"[^\wऀ-ॿ]", "", norm)


@dataclass
class WordSpan:
    """Word token with acoustic boundary timestamps in seconds."""

    word: str
    start: float
    end: float
    confidence: float | None = None
    language: str | None = None

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.end - self.start) * 1000.0)

    @property
    def is_devanagari(self) -> bool:
        return bool(DEV_RE.search(self.word))


@dataclass
class TokenBoundaryDiff:
    """Boundary difference between two independent timing sources for one matched token."""

    word: str
    segment_id: str
    ref_start: float
    ref_end: float
    comp_start: float
    comp_end: float
    delta_start_ms: float
    delta_end_ms: float
    max_delta_ms: float
    is_code_switched: bool
    is_devanagari: bool


@dataclass
class VerificationSummary:
    """Aggregate statistics for acoustic boundary agreement."""

    #: Which two systems were compared. Recorded so a report can never be misread as a
    #: system being compared against itself.
    reference_system_id: str = DEFAULT_REFERENCE_SYSTEM_ID
    comparison_system_id: str = DEFAULT_COMPARISON_SYSTEM_ID

    total_segments: int = 0
    #: Segments skipped because one of the two systems had no word spans on that record.
    segments_missing_source: int = 0
    total_tokens_evaluated: int = 0
    devanagari_tokens: int = 0
    latin_tokens: int = 0
    code_switched_tokens: int = 0

    # Coverage: tokens each side offered, and how many found no counterpart.
    reference_tokens: int = 0
    comparison_tokens: int = 0
    unmatched_reference_tokens: int = 0
    unmatched_comparison_tokens: int = 0

    # Agreement tolerances
    within_25ms_count: int = 0
    within_50ms_count: int = 0
    within_100ms_count: int = 0
    within_200ms_count: int = 0

    # Rates (%)
    rate_within_25ms: float = 0.0
    rate_within_50ms: float = 0.0
    rate_within_100ms: float = 0.0
    rate_within_200ms: float = 0.0

    # Error metrics (ms)
    mae_start_ms: float = 0.0
    mae_end_ms: float = 0.0
    mae_overall_ms: float = 0.0

    # Sub-strata rates (<= 50ms)
    rate_within_50ms_devanagari: float = 0.0
    rate_within_50ms_latin: float = 0.0
    rate_within_50ms_code_switch: float = 0.0

    # Human review triage
    flagged_segments: list[dict[str, Any]] = field(default_factory=list)


def align_tokens_dynamic(
    ref_tokens: list[str],
    acoustic_spans: list[WordSpan],
) -> list[tuple[str, WordSpan | None]]:
    """Align reference words with acoustic word spans using Levenshtein distance.

    Returns exactly one entry per reference token, in reference order. A token with no
    counterpart is paired with ``None``.
    """
    n = len(ref_tokens)
    m = len(acoustic_spans)

    if n == 0 or m == 0:
        return [(t, None) for t in ref_tokens]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * 1.0
    for j in range(m + 1):
        dp[0][j] = j * 1.0

    for i in range(1, n + 1):
        t_ref = normalize_token(ref_tokens[i - 1])
        for j in range(1, m + 1):
            t_ac = normalize_token(acoustic_spans[j - 1].word)
            cost = 0.0 if t_ref == t_ac else 1.0
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost,
            )

    i, j = n, m
    alignment: list[tuple[str, WordSpan | None]] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            t_ref = normalize_token(ref_tokens[i - 1])
            t_ac = normalize_token(acoustic_spans[j - 1].word)
            cost = 0.0 if t_ref == t_ac else 1.0
            if abs(dp[i][j] - (dp[i - 1][j - 1] + cost)) < 1e-6:
                if cost == 0.0:
                    alignment.append((ref_tokens[i - 1], acoustic_spans[j - 1]))
                else:
                    alignment.append((ref_tokens[i - 1], None))
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + 1.0)) < 1e-6:
            alignment.append((ref_tokens[i - 1], None))
            i -= 1
        else:
            j -= 1

    alignment.reverse()
    return alignment


def project_missing_spans(
    aligned: list[tuple[str, WordSpan | None]],
    segment_start: float,
    segment_end: float,
) -> list[WordSpan]:
    """Interpolate spans for unaligned or edited tokens between acoustic anchor words."""
    result: list[WordSpan] = []
    n = len(aligned)

    for idx, (token, span) in enumerate(aligned):
        if span is not None:
            result.append(span)
            continue

        prev_end = segment_start
        for p in range(idx - 1, -1, -1):
            if aligned[p][1] is not None:
                prev_end = aligned[p][1].end  # type: ignore[union-attr]
                break

        next_start = segment_end
        for q in range(idx + 1, n):
            if aligned[q][1] is not None:
                next_start = aligned[q][1].start  # type: ignore[union-attr]
                break

        gap = max(0.05, next_start - prev_end)
        m_start = idx
        while m_start > 0 and aligned[m_start - 1][1] is None:
            m_start -= 1
        m_end = idx
        while m_end + 1 < n and aligned[m_end + 1][1] is None:
            m_end += 1

        total_missing = m_end - m_start + 1
        pos_in_missing = idx - m_start
        step = gap / total_missing

        t_s = round(prev_end + pos_in_missing * step, 3)
        t_e = round(t_s + step, 3)
        result.append(WordSpan(word=token, start=t_s, end=t_e, confidence=0.5))

    return result


def spans_from_dicts(words: list[dict[str, Any]] | None) -> list[WordSpan]:
    """Build ``WordSpan``s from exported word dicts, dropping any without both boundaries."""
    return [
        WordSpan(
            word=str(w.get("word") or ""),
            start=float(w.get("start") or 0.0),
            end=float(w.get("end") or 0.0),
            confidence=w.get("confidence"),
        )
        for w in (words or [])
        if w.get("start") is not None and w.get("end") is not None
    ]


def _code_switch_flags(tokens: list[str]) -> list[bool]:
    """Mark tokens that sit on a script boundary with a neighbour."""
    tags = ["ne" if DEV_RE.search(t) else "en" for t in tokens]
    flags = [False] * len(tokens)
    for k in range(len(tokens)):
        if (k > 0 and tags[k] != tags[k - 1]) or (k + 1 < len(tags) and tags[k] != tags[k + 1]):
            flags[k] = True
    return flags


def verify_segment_timestamps(
    segment_id: str,
    reference_words: list[dict[str, Any]],
    comparison_words: list[dict[str, Any]],
) -> list[TokenBoundaryDiff]:
    """Cross-verify one segment's word boundaries between two independent timing sources.

    Args:
        segment_id: Segment the words belong to, carried through onto every diff.
        reference_words: Word dicts from the system trusted as the boundary reference.
        comparison_words: Word dicts from the system being measured against it.

    Returns:
        One ``TokenBoundaryDiff`` per token matched on both sides, in reference order.
        Tokens appearing on only one side are omitted: they carry no boundary delta.
    """
    ref_spans = spans_from_dicts(reference_words)
    comp_spans = spans_from_dicts(comparison_words)
    if not ref_spans or not comp_spans:
        return []

    ref_tokens = [s.word for s in ref_spans]
    is_cs = _code_switch_flags(ref_tokens)

    aligned = align_tokens_dynamic(ref_tokens, comp_spans)
    diffs: list[TokenBoundaryDiff] = []

    for idx, (tok, comp) in enumerate(aligned):
        if comp is None:
            continue

        ref = ref_spans[idx]
        d_start = abs(ref.start - comp.start) * 1000.0
        d_end = abs(ref.end - comp.end) * 1000.0

        diffs.append(
            TokenBoundaryDiff(
                word=tok,
                segment_id=segment_id,
                ref_start=ref.start,
                ref_end=ref.end,
                comp_start=comp.start,
                comp_end=comp.end,
                delta_start_ms=round(d_start, 1),
                delta_end_ms=round(d_end, 1),
                max_delta_ms=round(max(d_start, d_end), 1),
                is_code_switched=is_cs[idx],
                is_devanagari=bool(DEV_RE.search(tok)),
            )
        )

    return diffs


def _words_for_system(hypotheses: list[dict[str, Any]], system_id: str) -> list[dict[str, Any]]:
    """Word list of the named system, tolerating the ``mock-`` prefix a dry run writes."""
    accepted = {system_id, f"mock-{system_id}"}
    for hyp in hypotheses:
        if hyp.get("system_id") in accepted:
            return hyp.get("words") or []
    return []


def run_cross_verification_on_records(
    records: list[dict[str, Any]],
    divergence_threshold_ms: float = 200.0,
    reference_system_id: str = DEFAULT_REFERENCE_SYSTEM_ID,
    comparison_system_id: str = DEFAULT_COMPARISON_SYSTEM_ID,
) -> VerificationSummary:
    """Run full acoustic boundary verification over in-memory export records."""
    summary = VerificationSummary(
        reference_system_id=reference_system_id,
        comparison_system_id=comparison_system_id,
    )
    all_diffs: list[TokenBoundaryDiff] = []
    segment_diffs: dict[str, list[TokenBoundaryDiff]] = {}

    for rec in records:
        segment_id = rec.get("segment_id", "unknown")
        hyps = rec.get("hypotheses", [])

        ref_words = _words_for_system(hyps, reference_system_id)
        comp_words = _words_for_system(hyps, comparison_system_id)
        if not ref_words or not comp_words:
            summary.segments_missing_source += 1
            continue

        summary.total_segments += 1
        summary.reference_tokens += len(ref_words)
        summary.comparison_tokens += len(comp_words)

        diffs = verify_segment_timestamps(
            segment_id=segment_id,
            reference_words=ref_words,
            comparison_words=comp_words,
        )
        summary.unmatched_reference_tokens += len(ref_words) - len(diffs)
        summary.unmatched_comparison_tokens += len(comp_words) - len(diffs)

        if diffs:
            all_diffs.extend(diffs)
            segment_diffs[segment_id] = diffs

    if not all_diffs:
        return summary

    summary.total_tokens_evaluated = len(all_diffs)
    summary.devanagari_tokens = sum(1 for d in all_diffs if d.is_devanagari)
    summary.latin_tokens = sum(1 for d in all_diffs if not d.is_devanagari)
    summary.code_switched_tokens = sum(1 for d in all_diffs if d.is_code_switched)

    summary.within_25ms_count = sum(1 for d in all_diffs if d.max_delta_ms <= 25.0)
    summary.within_50ms_count = sum(1 for d in all_diffs if d.max_delta_ms <= 50.0)
    summary.within_100ms_count = sum(1 for d in all_diffs if d.max_delta_ms <= 100.0)
    summary.within_200ms_count = sum(1 for d in all_diffs if d.max_delta_ms <= 200.0)

    n = summary.total_tokens_evaluated
    summary.rate_within_25ms = round(100.0 * summary.within_25ms_count / n, 2)
    summary.rate_within_50ms = round(100.0 * summary.within_50ms_count / n, 2)
    summary.rate_within_100ms = round(100.0 * summary.within_100ms_count / n, 2)
    summary.rate_within_200ms = round(100.0 * summary.within_200ms_count / n, 2)

    summary.mae_start_ms = round(sum(d.delta_start_ms for d in all_diffs) / n, 2)
    summary.mae_end_ms = round(sum(d.delta_end_ms for d in all_diffs) / n, 2)
    summary.mae_overall_ms = round(sum(d.max_delta_ms for d in all_diffs) / n, 2)

    dev_n = max(1, summary.devanagari_tokens)
    lat_n = max(1, summary.latin_tokens)
    cs_n = max(1, summary.code_switched_tokens)

    summary.rate_within_50ms_devanagari = round(
        100.0 * sum(1 for d in all_diffs if d.is_devanagari and d.max_delta_ms <= 50.0) / dev_n, 2
    )
    summary.rate_within_50ms_latin = round(
        100.0 * sum(1 for d in all_diffs if not d.is_devanagari and d.max_delta_ms <= 50.0) / lat_n,
        2,
    )
    summary.rate_within_50ms_code_switch = round(
        100.0 * sum(1 for d in all_diffs if d.is_code_switched and d.max_delta_ms <= 50.0) / cs_n,
        2,
    )

    for seg_id, s_diffs in segment_diffs.items():
        divergent_tokens = [d for d in s_diffs if d.max_delta_ms > divergence_threshold_ms]
        if divergent_tokens:
            summary.flagged_segments.append(
                {
                    "segment_id": seg_id,
                    "divergent_token_count": len(divergent_tokens),
                    "total_tokens": len(s_diffs),
                    "divergence_ratio": round(len(divergent_tokens) / len(s_diffs), 3),
                    "max_delta_ms": max(d.max_delta_ms for d in divergent_tokens),
                    "flagged_words": [d.word for d in divergent_tokens[:5]],
                }
            )

    return summary


def run_cross_verification(
    analytics_jsonl_path: Path,
    divergence_threshold_ms: float = 200.0,
    reference_system_id: str = DEFAULT_REFERENCE_SYSTEM_ID,
    comparison_system_id: str = DEFAULT_COMPARISON_SYSTEM_ID,
) -> VerificationSummary:
    """Run full acoustic boundary verification over an exported analytics.jsonl."""
    records: list[dict[str, Any]] = []
    with open(analytics_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return run_cross_verification_on_records(
        records,
        divergence_threshold_ms=divergence_threshold_ms,
        reference_system_id=reference_system_id,
        comparison_system_id=comparison_system_id,
    )


def format_verification_report(summary: VerificationSummary) -> str:
    """Format an academic verification summary table."""
    tot = summary.total_tokens_evaluated
    flagged_len = len(summary.flagged_segments)
    lines = [
        "=" * 72,
        "      ACOUSTIC TIMESTAMP CROSS-VERIFICATION & BOUNDARY GATING REPORT",
        "=" * 72,
        f"Reference System:          {summary.reference_system_id}",
        f"Comparison System:         {summary.comparison_system_id}",
        f"Segments Evaluated:        {summary.total_segments}",
        f"Segments Skipped:          {summary.segments_missing_source} (a source had no words)",
        f"Total Words Matched:       {summary.total_tokens_evaluated}",
        f"  • Devanagari (Nepali):   {summary.devanagari_tokens}",
        f"  • Latin (English/Loan):  {summary.latin_tokens}",
        f"  • Code-Switch Boundary:  {summary.code_switched_tokens}",
        (
            f"Unmatched (reference):     {summary.unmatched_reference_tokens}"
            f" of {summary.reference_tokens}"
        ),
        (
            f"Unmatched (comparison):    {summary.unmatched_comparison_tokens}"
            f" of {summary.comparison_tokens}"
        ),
        "-" * 72,
        "BOUNDARY AGREEMENT RATES:",
        (
            f"  • Within ±25 ms:         {summary.rate_within_25ms}%  "
            f"({summary.within_25ms_count}/{tot})"
        ),
        (
            f"  • Within ±50 ms (Gold):  {summary.rate_within_50ms}%  "
            f"({summary.within_50ms_count}/{tot})"
        ),
        (
            f"  • Within ±100 ms:        {summary.rate_within_100ms}%  "
            f"({summary.within_100ms_count}/{tot})"
        ),
        (
            f"  • Within ±200 ms:        {summary.rate_within_200ms}%  "
            f"({summary.within_200ms_count}/{tot})"
        ),
        "-" * 72,
        "MEAN BOUNDARY DELTA (MAE):",
        f"  • Start Boundary MAE:    {summary.mae_start_ms} ms",
        f"  • End Boundary MAE:      {summary.mae_end_ms} ms",
        f"  • Overall Boundary MAE:  {summary.mae_overall_ms} ms",
        "-" * 72,
        "STRATIFIED PRECISION (≤ 50 ms Agreement):",
        f"  • Devanagari Tokens:     {summary.rate_within_50ms_devanagari}%",
        f"  • Latin Script Tokens:   {summary.rate_within_50ms_latin}%",
        f"  • Code-Switched Tokens:  {summary.rate_within_50ms_code_switch}%",
        "-" * 72,
        f"HUMAN REVIEW AUDIT QUEUE (Divergence > 200 ms): {flagged_len} segments flagged",
    ]
    for item in summary.flagged_segments[:5]:
        words_str = ", ".join(item["flagged_words"])
        max_d = item["max_delta_ms"]
        lines.append(f"  • Segment {item['segment_id']}: max Δ={max_d}ms (tokens: {words_str})")
    if flagged_len > 5:
        lines.append(f"  ... and {flagged_len - 5} more segments")
    lines.append("=" * 72)
    return "\n".join(lines)
