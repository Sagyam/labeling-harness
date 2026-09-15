"""Acoustic measurements of a clip, from the episode audio (roadmap item 1, D87)."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.acoustics import ACOUSTICS_VERSION, AcousticMeter, bandwidth_hz

SR = 16_000


def _noise(seconds: float, *, cutoff_hz: float | None = None, seed: int = 0) -> np.ndarray:
    """White noise, brick-wall low-passed at ``cutoff_hz`` -- a band-limited "voice"."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * SR)).astype(np.float32)
    if cutoff_hz is None:
        return x * 0.1
    spectrum = np.fft.rfft(x)
    spectrum[np.fft.rfftfreq(len(x), 1 / SR) > cutoff_hz] = 0
    return (np.fft.irfft(spectrum, n=len(x)) * 0.1).astype(np.float32)


@pytest.mark.parametrize("cutoff", [3400.0, 5300.0])
def test_bandwidth_finds_where_a_band_limited_signal_stops(cutoff: float) -> None:
    assert bandwidth_hz(_noise(3.0, cutoff_hz=cutoff), SR) == pytest.approx(cutoff, abs=100)


def test_full_band_audio_reaches_nyquist() -> None:
    assert bandwidth_hz(_noise(3.0), SR) > 7500


def test_only_speech_is_read_when_spans_are_given() -> None:
    """Silence between words carries the room, not the voice; a narrow voice between full-band
    bursts outside its spans is still narrow."""
    burst, voice = _noise(1.0), _noise(2.0, cutoff_hz=3400.0, seed=1)
    audio = np.concatenate([burst, voice, burst])
    assert bandwidth_hz(audio, SR) > 7500
    assert bandwidth_hz(audio, SR, spans=[[1.0, 3.0]]) == pytest.approx(3400, abs=100)


def test_too_little_audio_is_not_measured() -> None:
    assert bandwidth_hz(np.zeros(100, dtype=np.float32), SR) is None
    assert bandwidth_hz(np.zeros(SR, dtype=np.float32), SR) is None  # digital silence


def test_the_meter_measures_every_clip_of_an_episode_and_versions_the_result() -> None:
    audio = np.concatenate([_noise(4.0, cutoff_hz=3400.0), _noise(4.0, seed=2)])
    meter = AcousticMeter()
    narrow, full = meter.measure(audio, SR, [(0.0, 4.0, [[0.0, 4.0]]), (4.0, 8.0, None)])
    assert narrow["version"] == full["version"] == ACOUSTICS_VERSION
    assert narrow["bandwidth_hz"] < 4500 < 7500 < full["bandwidth_hz"]


def test_the_meter_wants_16_khz() -> None:
    meter = AcousticMeter()
    with pytest.raises(ValueError, match="16000"):
        meter.measure(np.zeros(8000, dtype=np.float32), 8000, [(0.0, 1.0, None)])
