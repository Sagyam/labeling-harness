"""Unit tests for acoustic timestamp cross-verification service."""

from __future__ import annotations

from app.services.alignment import (
    WordSpan,
    align_tokens_dynamic,
    project_missing_spans,
    run_cross_verification_on_records,
    verify_segment_timestamps,
)

SCRIBE = "elevenlabs-scribe-v2"
FLASH = "gemini-3.8-flash"


def test_align_tokens_dynamic_exact_match() -> None:
    ref = ["हामी", "नेपाली", "हौँ"]
    spans = [
        WordSpan(word="हामी", start=0.1, end=0.5),
        WordSpan(word="नेपाली", start=0.6, end=1.2),
        WordSpan(word="हौँ", start=1.3, end=1.6),
    ]
    aligned = align_tokens_dynamic(ref, spans)
    assert len(aligned) == 3
    assert aligned[0][1] is not None
    assert aligned[0][1].word == "हामी"
    assert aligned[1][1] is not None
    assert aligned[1][1].word == "नेपाली"
    assert aligned[2][1] is not None
    assert aligned[2][1].word == "हौँ"


def test_align_tokens_dynamic_with_edit() -> None:
    ref = ["हामी", "सबै", "नेपाली"]
    spans = [
        WordSpan(word="हामी", start=0.1, end=0.5),
        WordSpan(word="नेपाली", start=0.6, end=1.2),
    ]
    aligned = align_tokens_dynamic(ref, spans)
    assert len(aligned) == 3
    # "सबै" was inserted/edited, so its matched span should be None
    assert aligned[1][0] == "सबै"
    assert aligned[1][1] is None


def test_project_missing_spans_interpolates_gap() -> None:
    aligned = [
        ("हामी", WordSpan(word="हामी", start=0.0, end=0.5)),
        ("सबै", None),
        ("नेपाली", WordSpan(word="नेपाली", start=1.0, end=1.5)),
    ]
    projected = project_missing_spans(aligned, segment_start=0.0, segment_end=2.0)
    assert len(projected) == 3
    assert projected[0].word == "हामी"
    assert projected[1].word == "सबै"
    # Interpolated span should fall between 0.5s and 1.0s
    assert projected[1].start >= 0.5
    assert projected[1].end <= 1.0
    assert projected[2].word == "नेपाली"


def test_verify_segment_timestamps_scores_a_real_offset() -> None:
    """The delta must come from two independent sources, not one list against itself.

    Regression guard for the defect this replaced: the old implementation compared each
    aligned span against ``spans[idx]`` -- an element of the same list -- so every delta
    was 0.0 whenever the alignment was one-to-one.
    """
    reference = [
        {"word": "हामी", "start": 0.100, "end": 0.500},
        {"word": "meeting", "start": 0.600, "end": 1.100},
    ]
    comparison = [
        {"word": "हामी", "start": 0.180, "end": 0.545},
        {"word": "meeting", "start": 0.640, "end": 1.100},
    ]
    diffs = verify_segment_timestamps(
        segment_id="seg_01",
        reference_words=reference,
        comparison_words=comparison,
    )
    assert len(diffs) == 2

    assert diffs[0].word == "हामी"
    assert diffs[0].ref_start == 0.100
    assert diffs[0].comp_start == 0.180
    assert diffs[0].delta_start_ms == 80.0
    assert diffs[0].delta_end_ms == 45.0
    assert diffs[0].max_delta_ms == 80.0

    assert diffs[1].word == "meeting"
    assert diffs[1].delta_start_ms == 40.0
    assert diffs[1].delta_end_ms == 0.0
    assert diffs[1].is_code_switched is True


def test_verify_segment_timestamps_identical_sources_are_zero() -> None:
    words = [
        {"word": "हामी", "start": 0.1, "end": 0.5},
        {"word": "meeting", "start": 0.6, "end": 1.1},
    ]
    diffs = verify_segment_timestamps(
        segment_id="seg_01",
        reference_words=words,
        comparison_words=list(words),
    )
    assert [d.max_delta_ms for d in diffs] == [0.0, 0.0]


def test_verify_segment_timestamps_skips_tokens_present_on_one_side_only() -> None:
    reference = [
        {"word": "हामी", "start": 0.10, "end": 0.50},
        {"word": "सबै", "start": 0.55, "end": 0.80},
        {"word": "meeting", "start": 0.90, "end": 1.40},
    ]
    comparison = [
        {"word": "हामी", "start": 0.12, "end": 0.50},
        {"word": "meeting", "start": 0.95, "end": 1.40},
    ]
    diffs = verify_segment_timestamps(
        segment_id="seg_01",
        reference_words=reference,
        comparison_words=comparison,
    )
    # "सबै" has no counterpart, so it carries no meaningful boundary delta.
    assert [d.word for d in diffs] == ["हामी", "meeting"]
    assert diffs[0].delta_start_ms == 20.0
    assert diffs[1].delta_start_ms == 50.0


def test_run_cross_verification_selects_the_two_named_systems() -> None:
    records = [
        {
            "segment_id": "seg_01",
            "duration_seconds": 2.0,
            "hypotheses": [
                {
                    "system_id": "mai-transcribe-2",
                    "words": [{"word": "हामी", "start": 9.0, "end": 9.5}],
                },
                {
                    "system_id": FLASH,
                    "words": [
                        {"word": "हामी", "start": 0.120, "end": 0.500},
                        {"word": "meeting", "start": 0.600, "end": 1.100},
                    ],
                },
                {
                    "system_id": SCRIBE,
                    "words": [
                        {"word": "हामी", "start": 0.100, "end": 0.500},
                        {"word": "meeting", "start": 0.600, "end": 1.100},
                    ],
                },
            ],
        }
    ]
    summary = run_cross_verification_on_records(records)
    assert summary.reference_system_id == SCRIBE
    assert summary.comparison_system_id == FLASH
    assert summary.total_segments == 1
    assert summary.total_tokens_evaluated == 2
    # 20 ms on one token, 0 ms on the other: both inside 25 ms.
    assert summary.rate_within_25ms == 100.0
    assert summary.mae_start_ms == 10.0
    assert summary.flagged_segments == []


def test_run_cross_verification_skips_records_missing_a_source() -> None:
    records = [
        {
            "segment_id": "seg_01",
            "duration_seconds": 2.0,
            "hypotheses": [
                {"system_id": SCRIBE, "words": [{"word": "हामी", "start": 0.1, "end": 0.5}]},
            ],
        },
        {
            "segment_id": "seg_02",
            "duration_seconds": 2.0,
            "hypotheses": [
                {"system_id": SCRIBE, "words": [{"word": "हामी", "start": 0.1, "end": 0.5}]},
                {"system_id": FLASH, "words": [{"word": "हामी", "start": 0.4, "end": 0.5}]},
            ],
        },
    ]
    summary = run_cross_verification_on_records(records)
    assert summary.total_segments == 1
    assert summary.segments_missing_source == 1
    assert summary.total_tokens_evaluated == 1
    assert summary.mae_start_ms == 300.0


def test_run_cross_verification_flags_divergent_segments() -> None:
    records = [
        {
            "segment_id": "seg_bad",
            "duration_seconds": 2.0,
            "hypotheses": [
                {"system_id": SCRIBE, "words": [{"word": "meeting", "start": 0.10, "end": 0.50}]},
                {"system_id": FLASH, "words": [{"word": "meeting", "start": 0.85, "end": 1.20}]},
            ],
        }
    ]
    summary = run_cross_verification_on_records(records, divergence_threshold_ms=200.0)
    assert len(summary.flagged_segments) == 1
    assert summary.flagged_segments[0]["segment_id"] == "seg_bad"
    assert summary.flagged_segments[0]["max_delta_ms"] == 750.0


def test_project_missing_spans_never_overlaps_its_neighbours() -> None:
    """With no acoustic room between two anchors, a projected span must be degenerate.

    The forced aligner leans on this: a token it could not place -- bare digits, punctuation --
    sits between two words that may be exactly adjacent. A minimum-width span there would run
    past the next word's start.
    """
    aligned = [
        ("a", WordSpan(word="a", start=0.0, end=0.5)),
        ("123", None),
        ("b", WordSpan(word="b", start=0.5, end=1.0)),
    ]
    projected = project_missing_spans(aligned, segment_start=0.0, segment_end=1.0)
    assert projected[1].start >= projected[0].end
    assert projected[1].end <= projected[2].start
