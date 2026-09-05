"""Tests for the numpy reimplementation of Kaldi's filterbank front end (D58).

There is no reference implementation to diff against -- torchaudio is the reference and it is not
a dependency (D32) -- so these pin the properties that make the output *Kaldi's* rather than
merely a reasonable log-mel: the frame arithmetic, the window shape, the filterbank layout, and
the invariants a wrong implementation would break silently.

The end-to-end evidence that it is right lives outside the suite: run against the real speaker
embedding model on Nepali speech, two turns of one speaker score a mean 0.28 cosine distance
against 0.66 for different speakers, which a mismatched front end does not produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.fbank import (
    NUM_MEL_BINS,
    compute_fbank,
    frame_count,
    mel_filterbank,
    mel_scale,
    povey_window,
)

SR = 16000


def speech_like(seconds: float, f0: float = 120.0, seed: int = 0) -> np.ndarray:
    """A harmonic-plus-noise signal: not speech, but broadband and non-degenerate like it."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20))
    return (0.3 * wave + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


# --- frame arithmetic -------------------------------------------------------------------------


def test_frame_count_snips_a_trailing_partial_frame() -> None:
    """Kaldi's snip_edges=True: a partial frame is dropped, never zero-padded."""
    assert frame_count(400, 400, 160) == 1
    assert frame_count(559, 400, 160) == 1  # 159 spare samples are not a frame
    assert frame_count(560, 400, 160) == 2
    assert frame_count(399, 400, 160) == 0


def test_a_clip_shorter_than_one_frame_gives_no_features() -> None:
    assert compute_fbank(np.zeros(100, dtype=np.float32)).shape == (0, NUM_MEL_BINS)


def test_frame_rate_is_one_per_ten_milliseconds() -> None:
    features = compute_fbank(speech_like(1.0))
    # 1 + (16000 - 400) // 160: one second at a 10 ms shift, less the trailing 96 samples that
    # cannot complete a 25 ms window and are snipped rather than padded.
    assert features.shape == (98, NUM_MEL_BINS)


# --- the window -------------------------------------------------------------------------------


def test_povey_window_is_hann_to_the_point_eight_five() -> None:
    window = povey_window(400)
    n = np.arange(400)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / 399)
    assert np.allclose(window, hann**0.85)


def test_the_window_is_symmetric_and_zero_at_both_ends() -> None:
    window = povey_window(400)
    assert window[0] == pytest.approx(0.0)
    assert window[-1] == pytest.approx(0.0)
    assert np.allclose(window, window[::-1])


# --- the mel filterbank -----------------------------------------------------------------------


def test_mel_scale_matches_kaldis_constants() -> None:
    assert mel_scale(0.0) == pytest.approx(0.0)
    assert float(mel_scale(700.0)) == pytest.approx(1127.0 * np.log(2.0))


def test_filters_are_triangular_and_ordered_up_the_spectrum() -> None:
    filters = mel_filterbank(fft_length=512)
    assert filters.shape == (NUM_MEL_BINS, 257)
    assert (filters >= 0).all()
    # The bank climbs the spectrum. Non-decreasing rather than strictly increasing: below about
    # 200 Hz the mel spacing is finer than the FFT resolution, so two neighbouring filters can
    # legitimately peak on the same bin -- Kaldi included.
    peaks = filters.argmax(axis=1)
    assert (np.diff(peaks) >= 0).all()
    assert peaks[-1] > peaks[0]


def test_every_filter_carries_weight() -> None:
    """An empty bin means a mel edge collapsed onto an FFT bin and the layout is wrong."""
    assert (mel_filterbank(fft_length=512).sum(axis=1) > 0).all()


# --- invariants a wrong front end would break --------------------------------------------------


def test_features_are_float32_and_finite() -> None:
    features = compute_fbank(speech_like(0.5))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_digital_silence_does_not_produce_infinities() -> None:
    """The energy floor is what keeps a silent frame from becoming -inf and poisoning the mean."""
    features = compute_fbank(np.zeros(SR, dtype=np.float32))
    assert np.isfinite(features).all()


def test_cepstral_mean_normalisation_centres_each_bin() -> None:
    features = compute_fbank(speech_like(1.0), cmn=True)
    assert np.allclose(features.mean(axis=0), 0.0, atol=1e-4)


def test_without_cmn_the_bins_are_not_centred() -> None:
    features = compute_fbank(speech_like(1.0), cmn=False)
    assert abs(float(features.mean())) > 1.0


def test_cmn_makes_the_output_invariant_to_gain() -> None:
    """A quieter recording of the same voice must not embed differently for being quieter."""
    signal = speech_like(1.0)
    assert np.allclose(compute_fbank(signal), compute_fbank(signal * 0.25), atol=1e-3)


def test_a_higher_pitch_moves_energy_up_the_filterbank() -> None:
    """A sanity check that the bank is oriented low-to-high rather than reversed."""

    def centroid(bins: np.ndarray) -> float:
        weights = np.exp(bins)
        return float((np.arange(bins.size) * weights).sum() / weights.sum())

    low = compute_fbank(speech_like(1.0, f0=100.0), cmn=False).mean(axis=0)
    high = compute_fbank(speech_like(1.0, f0=300.0), cmn=False).mean(axis=0)
    assert centroid(high) > centroid(low)
