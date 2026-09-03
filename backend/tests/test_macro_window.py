"""Unit tests for macro-window clustering and word timestamp demultiplexing.

These tests run without a live database (no `db` mark) to verify the pure algorithmic
components of Gemini 3.5 Transcribe macro-windowing under 10 RPM / 10K TPM / 100 RPD limits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from app.llm.base import AsrResult
from app.services.ingest import (
    cluster_segments_into_windows,
    demux_window_words_to_segments,
    extract_window_clip,
)
from app.services.silero_vad import CutSegment


def _make_cut_segment(
    seg_id: str, start: float, end: float, tmp_path: Path | None = None
) -> CutSegment:
    clip_path = (tmp_path or Path("/tmp")) / f"{seg_id}.flac"
    return CutSegment(
        segment_id=seg_id,
        start_time=start,
        end_time=end,
        duration=round(end - start, 3),
        clip_path=clip_path,
        clip_rel_path=f"clips/{seg_id}.flac",
        clip_checksum="fake_checksum",
    )


def test_cluster_segments_into_windows_groups_by_target_duration() -> None:
    # 6 segments of 10s each, with 0.5s gaps between them
    segments = [
        _make_cut_segment("seg_0", 0.0, 10.0),
        _make_cut_segment("seg_1", 10.5, 20.5),
        _make_cut_segment("seg_2", 21.0, 31.0),
        _make_cut_segment("seg_3", 31.5, 41.5),
        _make_cut_segment("seg_4", 42.0, 52.0),
        _make_cut_segment("seg_5", 52.5, 62.5),
    ]

    # With target_duration = 32.0s:
    # Window 0: seg0 (0..10), seg1 (0..20.5), seg2 (0..31.0 <= 32.0).
    # Adding seg3 gives span 41.5 > 32.0 -> breaks into Window 1
    # Window 1: seg3 (31.5..41.5), seg4 (31.5..52.0), seg5 (31.5..62.5, span=31.0 <= 32.0)
    windows = cluster_segments_into_windows(segments, target_duration=32.0)
    assert len(windows) == 2
    assert [s.segment_id for s in windows[0]] == ["seg_0", "seg_1", "seg_2"]
    assert [s.segment_id for s in windows[1]] == ["seg_3", "seg_4", "seg_5"]


def test_cluster_segments_into_windows_empty_and_oversized() -> None:
    assert cluster_segments_into_windows([]) == []

    # Single segment larger than target
    oversized = [_make_cut_segment("seg_big", 0.0, 200.0)]
    windows = cluster_segments_into_windows(oversized, target_duration=150.0)
    assert len(windows) == 1
    assert len(windows[0]) == 1
    assert windows[0][0].segment_id == "seg_big"


def test_extract_window_clip_creates_valid_flac(tmp_path: Path) -> None:
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    src_flac = tmp_path / "source.flac"
    sf.write(str(src_flac), audio, sr, format="FLAC")

    out_clip = tmp_path / "window_slice.flac"
    extract_window_clip(src_flac, start_sec=1.5, end_sec=3.5, output_path=out_clip)

    assert out_clip.is_file()
    info = sf.info(str(out_clip))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.format == "FLAC"
    assert abs(info.duration - 2.0) < 0.01


def test_demux_window_words_to_segments_clip_relative_contract() -> None:
    seg1 = _make_cut_segment("seg_001", 10.0, 20.0)
    seg2 = _make_cut_segment("seg_002", 20.5, 30.0)
    window = [seg1, seg2]

    window_result = AsrResult(
        route="asr_gemini_transcribe",
        model="gemini-3.5-transcribe",
        text="hello world nepal kathmandu",
        words=[
            {"word": "hello", "start": 1.0, "end": 1.5},
            {"word": "world", "start": 5.0, "end": 5.5},
            {"word": "nepal", "start": 11.0, "end": 11.6},
            {"word": "kathmandu", "start": 15.0, "end": 15.8},
        ],
        latency_ms=120,
    )

    demuxed = demux_window_words_to_segments(window, window_result)
    assert set(demuxed.keys()) == {"seg_001", "seg_002"}

    res1 = demuxed["seg_001"]
    assert res1.text == "hello world"
    assert res1.words == [
        {"word": "hello", "start": 1.0, "end": 1.5},
        {"word": "world", "start": 5.0, "end": 5.5},
    ]

    res2 = demuxed["seg_002"]
    assert res2.text == "nepal kathmandu"
    assert res2.words == [
        {"word": "nepal", "start": 0.5, "end": 1.1},
        {"word": "kathmandu", "start": 4.5, "end": 5.3},
    ]


def test_demux_window_words_handles_dry_run_population() -> None:
    seg1 = _make_cut_segment("seg_d1", 0.0, 10.0)
    seg2 = _make_cut_segment("seg_d2", 10.0, 20.0)
    window = [seg1, seg2]

    window_result = AsrResult(
        route="asr_gemini_transcribe",
        model="gemini-3.5-transcribe",
        text="mock transcript",
        words=[{"word": "mock", "start": 0.0, "end": 0.5}],
        dry_run=True,
    )

    demuxed = demux_window_words_to_segments(window, window_result)
    assert demuxed["seg_d1"].dry_run is True
    assert demuxed["seg_d2"].dry_run is True
    assert len(demuxed["seg_d1"].text) > 0
    assert len(demuxed["seg_d2"].text) > 0
