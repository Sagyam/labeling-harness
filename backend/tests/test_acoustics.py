"""Acoustic measurements of a clip, from the episode audio (roadmap item 1, D87)."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.acoustics import (
    ACOUSTICS_VERSION,
    BANDWIDTH_ONLY_VERSION,
    AcousticMeter,
    bandwidth_hz,
)
from app.services.brouhaha import FRAME_STEP

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
    assert narrow["version"] == full["version"] == BANDWIDTH_ONLY_VERSION  # no Brouhaha given
    assert narrow["bandwidth_hz"] < 4500 < 7500 < full["bandwidth_hz"]


def test_the_meter_wants_16_khz() -> None:
    meter = AcousticMeter()
    with pytest.raises(ValueError, match="16000"):
        meter.measure(np.zeros(8000, dtype=np.float32), 8000, [(0.0, 1.0, None)])


# --- SNR and reverb, from Brouhaha's frames -------------------------------------------------------


class _FakeBrouhaha:
    """Speech in the first half of the audio at 30 dB SNR, silence after; C50 40 dB throughout."""

    available = True

    def __init__(self) -> None:
        self.calls = 0

    def frames(self, audio, sample_rate):
        self.calls += 1
        n = int(np.ceil(len(audio) / sample_rate / FRAME_STEP))
        out = np.zeros((n, 3))
        out[: n // 2, 0] = 0.9
        out[:, 1] = 30.0
        out[n // 2 :, 1] = -10.0  # noise-only frames: must not drag a speech mean down
        out[:, 2] = 40.0
        return out


def test_each_clip_gets_the_mean_snr_and_c50_of_its_speech_frames() -> None:
    brouhaha = _FakeBrouhaha()
    meter = AcousticMeter(brouhaha)
    audio = _noise(8.0)
    speech, spanning, silent = meter.measure(
        audio, SR, [(0.0, 3.0, None), (2.0, 6.0, None), (5.0, 8.0, None)]
    )
    assert brouhaha.calls == 1, "the whole episode is read once"
    assert speech["version"] == ACOUSTICS_VERSION
    assert (speech["snr_db"], speech["c50_db"]) == (30.0, 40.0)
    assert spanning["snr_db"] == 30.0  # only its speech half counts
    assert "snr_db" not in silent and "c50_db" not in silent
    assert silent["bandwidth_hz"] > 0


def test_a_model_that_did_not_load_is_no_model() -> None:
    class Absent:
        available = False

    assert AcousticMeter(Absent()).version == BANDWIDTH_ONLY_VERSION


def test_a_backfill_never_measures_less_than_it_found() -> None:
    full, partial = AcousticMeter(_FakeBrouhaha()), AcousticMeter()
    assert full.is_current({"version": ACOUSTICS_VERSION})
    assert not full.is_current({"version": BANDWIDTH_ONLY_VERSION})  # the model can add SNR
    assert partial.is_current({"version": BANDWIDTH_ONLY_VERSION})
    assert partial.is_current({"version": ACOUSTICS_VERSION})  # already more than it can do
    assert not partial.is_current({"version": "acoustics-v1"})
    assert not partial.is_current(None)
