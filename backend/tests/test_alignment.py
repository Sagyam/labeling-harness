"""Unit tests for acoustic timestamp cross-verification service."""

from __future__ import annotations

from app.services.alignment import (
    WordSpan,
    align_tokens_dynamic,
    project_missing_spans,
    run_cross_verification_on_records,
    verify_segment_timestamps,
)


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


def test_verify_segment_timestamps_detects_deltas() -> None:
    words = [
        {"word": "हामी", "start": 0.1, "end": 0.5},
        {"word": "meeting", "start": 0.6, "end": 1.1},
    ]
    diffs = verify_segment_timestamps(
        segment_id="seg_01",
        gold_text="हामी meeting",
        acoustic_words=words,
    )
    assert len(diffs) == 2
    assert diffs[0].word == "हामी"
    assert diffs[0].delta_start_ms == 0.0
    assert diffs[1].word == "meeting"
    assert diffs[1].is_code_switched is True


def test_run_cross_verification_on_records_summary() -> None:
    records = [
        {
            "segment_id": "seg_01",
            "text": "हामी meeting",
            "duration_seconds": 2.0,
            "hypotheses": [
                {
                    "system_id": "scribe",
                    "words": [
                        {"word": "हामी", "start": 0.1, "end": 0.5},
                        {"word": "meeting", "start": 0.6, "end": 1.1},
                    ],
                }
            ],
        }
    ]
    summary = run_cross_verification_on_records(records)
    assert summary.total_segments == 1
    assert summary.total_tokens_evaluated == 2
    assert summary.rate_within_25ms == 100.0
    assert summary.rate_within_50ms == 100.0
    assert summary.flagged_segments == []
