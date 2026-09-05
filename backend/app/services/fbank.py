"""Kaldi-compatible log-mel filterbank features, in numpy.

The speaker-embedding graph takes ``[batch, frames, 80]`` filterbank features rather than raw
audio, so something has to compute them. Every published implementation of the exact recipe it was
trained on lives in ``torchaudio.compliance.kaldi``, and torch is deliberately not a dependency of
this project (D32) -- so the recipe is reimplemented here against numpy, which is.

"Kaldi-compatible" is a specific claim and every default below is part of it. A speaker embedding
is a comparison against a model's training distribution: features that are close but not the same
do not fail, they quietly produce embeddings that cluster worse, which is the kind of defect that
shows up as a mediocre diarization result and never as an error. The details that matter:

* the waveform is scaled to the int16 range, which is what Kaldi's front end assumes;
* DC offset is removed per frame, then pre-emphasis is applied per frame, in that order;
* the window is Povey (a Hann window raised to 0.85), not Hann or Hamming;
* the FFT length is the window length rounded up to a power of two -- 512 for a 25 ms window;
* triangular mel bins are laid out in the mel domain between 20 Hz and the Nyquist frequency;
* frames are "snipped": a trailing partial frame is dropped rather than padded.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0
PREEMPHASIS = 0.97
LOW_FREQ_HZ = 20.0
#: Kaldi floors the filterbank energy at float epsilon before taking its log, so a silent frame
#: becomes a large negative number rather than ``-inf``.
LOG_FLOOR = float(np.finfo(np.float32).eps)


def mel_scale(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """Kaldi's mel scale: ``1127 * ln(1 + f / 700)``."""
    return 1127.0 * np.log(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)


def povey_window(length: int) -> np.ndarray:
    """Kaldi's default window: a Hann window raised to the power 0.85."""
    n = np.arange(length, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
    return np.power(hann, 0.85)


def mel_filterbank(
    *,
    num_bins: int = NUM_MEL_BINS,
    fft_length: int,
    sample_rate: int = SAMPLE_RATE,
    low_freq: float = LOW_FREQ_HZ,
    high_freq: float | None = None,
) -> np.ndarray:
    """Triangular mel filters laid out in the mel domain.

    Args:
        num_bins: Number of mel bins.
        fft_length: Length of the FFT the filters are applied to.
        sample_rate: Sample rate of the audio.
        low_freq: Lower edge of the lowest filter, in Hz.
        high_freq: Upper edge of the highest filter. Defaults to the Nyquist frequency.

    Returns:
        ``[num_bins, fft_length // 2 + 1]``, each row summing the FFT bins under one triangle.
    """
    nyquist = sample_rate / 2.0
    high = nyquist if high_freq is None else high_freq

    # num_bins triangles need num_bins + 2 edges: each triangle spans two intervals.
    mel_edges = np.linspace(mel_scale(low_freq), mel_scale(high), num_bins + 2)
    bin_hz = np.arange(fft_length // 2 + 1, dtype=np.float64) * sample_rate / fft_length
    bin_mel = mel_scale(bin_hz)

    filters = np.zeros((num_bins, fft_length // 2 + 1), dtype=np.float64)
    for i in range(num_bins):
        left, centre, right = mel_edges[i], mel_edges[i + 1], mel_edges[i + 2]
        rising = (bin_mel - left) / (centre - left)
        falling = (right - bin_mel) / (right - centre)
        filters[i] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


def frame_count(num_samples: int, frame_length: int, frame_shift: int) -> int:
    """How many whole frames fit, with Kaldi's ``snip_edges=True`` behaviour."""
    if num_samples < frame_length:
        return 0
    return 1 + (num_samples - frame_length) // frame_shift


def compute_fbank(
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    num_mel_bins: int = NUM_MEL_BINS,
    cmn: bool = True,
) -> np.ndarray:
    """Log-mel filterbank features for one mono waveform.

    Args:
        samples: Mono audio in ``[-1, 1]``, as ``soundfile`` returns it.
        sample_rate: Sample rate of ``samples``.
        num_mel_bins: Number of mel bins.
        cmn: Subtract the per-utterance mean of each bin, which is what the embedding model was
            trained with. Turn it off only to inspect raw features.

    Returns:
        ``[frames, num_mel_bins]`` float32. Empty when the audio is shorter than one frame.
    """
    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    # Kaldi's front end works on int16-ranged values; feeding it [-1, 1] shifts every log energy
    # by a constant, which CMN would hide and the mel-bin floor would not.
    audio = audio * 32768.0

    frame_length = round(sample_rate * FRAME_LENGTH_MS / 1000.0)
    frame_shift = round(sample_rate * FRAME_SHIFT_MS / 1000.0)
    frames = frame_count(audio.size, frame_length, frame_shift)
    if frames == 0:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    # One strided view instead of a Python loop: an hour of audio is ~360k frames.
    windows = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frames, frame_length),
        strides=(audio.strides[0] * frame_shift, audio.strides[0]),
    ).copy()

    windows -= windows.mean(axis=1, keepdims=True)
    # Pre-emphasis, per frame, with the first sample held against itself as Kaldi does.
    windows[:, 1:] -= PREEMPHASIS * windows[:, :-1]
    windows[:, 0] -= PREEMPHASIS * windows[:, 0]
    windows *= povey_window(frame_length)

    fft_length = 1 << (frame_length - 1).bit_length()
    power = np.abs(np.fft.rfft(windows, n=fft_length)) ** 2
    energies = (
        power
        @ mel_filterbank(num_bins=num_mel_bins, fft_length=fft_length, sample_rate=sample_rate).T
    )

    features = np.log(np.maximum(energies, LOG_FLOOR))
    if cmn:
        features -= features.mean(axis=0, keepdims=True)
    return features.astype(np.float32)
