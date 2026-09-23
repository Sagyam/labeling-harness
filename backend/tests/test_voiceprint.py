"""Tests for voiceprints: the fbank, the offline model contract, and suggestions (D99)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.voiceprint import (
    VoiceEmbedder,
    fbank,
    suggest,
    suggestion_report,
    unit,
    word_window,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "reference" / "kaldi_fbank_reference.json").read_text()
)


def reference_signal() -> np.ndarray:
    t = np.arange(8000) / 16000
    return (
        0.3 * np.sin(2 * np.pi * 220 * t)
        + 0.1 * np.sin(2 * np.pi * 1750 * t + 3 * t)
        + 0.05 * np.sin(2 * np.pi * 5300 * t * t)
    ).astype(np.float32)


# --- fbank ------------------------------------------------------------------------------------


def test_the_fbank_matches_kaldi_as_torchaudio_computes_it() -> None:
    feats = fbank(reference_signal())
    assert list(feats.shape) == REFERENCE["shape"]
    for frame, values in REFERENCE["frames"].items():
        np.testing.assert_allclose(feats[int(frame)], values, atol=2e-3)


def test_the_fbank_is_mean_normalised() -> None:
    np.testing.assert_allclose(fbank(reference_signal()).mean(axis=0), 0.0, atol=1e-4)


def test_audio_shorter_than_one_frame_has_no_features() -> None:
    assert fbank(np.zeros(100, dtype=np.float32)).shape == (0, 80)


# --- the model is optional --------------------------------------------------------------------


def test_a_missing_model_means_no_embedder_not_an_error(tmp_path: Path) -> None:
    embedder = VoiceEmbedder(tmp_path / "absent.onnx")
    assert not embedder.available
    with pytest.raises(RuntimeError):
        embedder.embed([np.zeros(16000, dtype=np.float32)])


def test_unit_has_length_one_and_refuses_a_zero_vector() -> None:
    assert np.linalg.norm(unit([3.0, 4.0])) == pytest.approx(1.0)
    assert unit([0.0, 0.0]) is None
    assert unit([]) is None


# --- windows ----------------------------------------------------------------------------------


def test_a_window_is_centred_on_the_word() -> None:
    assert word_window(2.0, 2.2, clip_seconds=10.0) == pytest.approx((1.6, 2.6))


def test_a_window_slides_to_stay_inside_the_clip() -> None:
    assert word_window(0.0, 0.1, clip_seconds=10.0) == pytest.approx((0.0, 1.0))
    assert word_window(9.9, 10.0, clip_seconds=10.0) == pytest.approx((9.0, 10.0))


def test_a_window_is_never_longer_than_the_clip() -> None:
    assert word_window(0.1, 0.3, clip_seconds=0.6) == pytest.approx((0.0, 0.6))


# --- suggestions ------------------------------------------------------------------------------


def test_a_clear_other_voice_is_suggested() -> None:
    s = suggest({1: 0.31, 2: 0.62}, current=1, overlapped=False)
    assert s is not None and s.speaker == 2 and s.margin == pytest.approx(0.31)


def test_nothing_is_suggested_in_crosstalk() -> None:
    """There the print follows whichever voice is louder, and is confident when wrong."""
    assert suggest({1: 0.1, 2: 0.9}, current=1, overlapped=True) is None


def test_nothing_is_suggested_when_the_lane_already_agrees() -> None:
    assert suggest({1: 0.7, 2: 0.2}, current=1, overlapped=False) is None


def test_nothing_is_suggested_on_a_narrow_margin() -> None:
    assert suggest({1: 0.50, 2: 0.55}, current=1, overlapped=False) is None


def test_a_word_on_no_lane_is_placed_by_the_closer_voice() -> None:
    s = suggest({1: 0.2, 2: 0.6}, current=None, overlapped=False)
    assert s is not None and s.speaker == 2


def test_a_word_on_no_lane_with_one_candidate_gets_that_candidate() -> None:
    s = suggest({3: 0.4}, current=None, overlapped=False)
    assert s is not None and s.speaker == 3


def test_one_candidate_that_is_already_the_lane_suggests_nothing() -> None:
    assert suggest({1: 0.4}, current=1, overlapped=False) is None


def test_no_similarities_suggest_nothing() -> None:
    assert suggest(None, current=1, overlapped=False) is None
    assert suggest({}, current=1, overlapped=False) is None


# --- how suggestions fared --------------------------------------------------------------------


def test_the_report_counts_precision_and_the_moves_it_foresaw() -> None:
    report = suggestion_report(
        [
            ("A", "A", None, "label"),  # left alone, no suggestion
            ("B", "A", "B", "label"),  # moved where suggested
            ("A", "A", "B", "label"),  # suggestion refused
            ("C", "A", None, "label"),  # moved, unforeseen
            ("B", None, None, "typed"),  # not a proposal at all
        ]
    )
    assert (report.words, report.suggested, report.taken) == (4, 2, 1)
    assert (report.moved, report.moved_suggested) == (2, 1)
    assert report.precision == pytest.approx(0.5)
    assert report.recall == pytest.approx(0.5)


def test_an_empty_report_has_no_rates() -> None:
    report = suggestion_report([])
    assert report.precision is None and report.recall is None
