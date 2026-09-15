"""Acoustic measurements of each clip, read from the episode audio (roadmap item 1, D87).

**Bandwidth.** The highest frequency at which the clip's long-term speech spectrum is still
within :data:`BANDWIDTH_DROP_DB` of its peak in the voice band (200 Hz - 2 kHz). A phone line
stops near 3.4 kHz, some codecs and microphones near 5 kHz, and full-band audio runs to the 8 kHz
Nyquist limit of the stored 16 kHz clips. Only the clip's VAD speech is read: the pauses between
words carry the room, not the voice. Over the 44 episodes of 2026-09-15 most podcasts sat near
7.8 kHz and many tech reviews near 5.3 kHz.

**Not measured: clipping.** The stored audio is loudness-normalised and resampled to 16 kHz
before anything reads it (ingest stage 1), and resampling rounds off the flat tops a clipping
detector looks for. The source audio is not retained, so there is nothing honest to measure.

Every result carries :data:`ACOUSTICS_VERSION`, so a backfill can tell a clip measured under
older rules from one measured under the current ones.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SAMPLE_RATE = 16_000
#: Bump whenever a measurement's rule changes or one is added.
ACOUSTICS_VERSION = "acoustics-v1"

_FRAME = 512
_HOP = 256
_WINDOW = np.hanning(_FRAME).astype(np.float32)
_FREQS = np.fft.rfftfreq(_FRAME, 1 / SAMPLE_RATE)
_VOICE_BAND = (_FREQS >= 200) & (_FREQS <= 2000)
#: How far below the voice-band peak the spectrum may fall and still count as signal. 40 dB
#: separates a 3.4 kHz phone line and a 5 kHz codec from full-band audio on this corpus without
#: reading the noise floor of a quiet room as bandwidth.
BANDWIDTH_DROP_DB = 40.0

#: A clip-relative span of speech, ``[start, end]`` seconds.
Span = Sequence[float]
#: One clip to measure: episode-relative ``start`` and ``end``, and its clip-relative VAD spans.
ClipSpec = tuple[float, float, Sequence[Span] | None]


def _speech(audio: np.ndarray, sample_rate: int, spans: Sequence[Span] | None) -> np.ndarray:
    if not spans:
        return audio
    pieces = [audio[int(a * sample_rate) : int(b * sample_rate)] for a, b in spans]
    pieces = [p for p in pieces if len(p)]
    return np.concatenate(pieces) if pieces else audio


def bandwidth_hz(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, *, spans: Sequence[Span] | None = None
) -> float | None:
    """The clip's audio bandwidth in Hz, or ``None`` when there is too little signal to say.

    Args:
        audio: The clip, 16 kHz mono float.
        sample_rate: Must be :data:`SAMPLE_RATE`.
        spans: Clip-relative speech spans; the whole clip when omitted or empty.
    """
    x = _speech(np.asarray(audio, dtype=np.float32), sample_rate, spans)
    if len(x) < _FRAME:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(x, _FRAME)[::_HOP] * _WINDOW
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    if not np.any(power[_VOICE_BAND] > 0):
        return None
    db = 10 * np.log10(power + 1e-20)
    audible = np.flatnonzero(db >= db[_VOICE_BAND].max() - BANDWIDTH_DROP_DB)
    return float(_FREQS[audible.max()])


class AcousticMeter:
    """Measures every clip of an episode from the whole episode's audio, which is what a model
    reading long windows (SNR, reverb) will need; bandwidth itself only reads the clip."""

    def measure(
        self, audio: np.ndarray, sample_rate: int, clips: Sequence[ClipSpec]
    ) -> list[dict[str, object]]:
        """One result per clip, in order: ``{"version": ..., "bandwidth_hz": ...}``.

        A value that could not be measured is absent from its clip's result, never ``None``,
        so a consumer reads "missing" the same way whichever measurement it asks for.
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"acoustic measurement needs {SAMPLE_RATE} Hz audio, got {sample_rate}"
            )
        audio = np.asarray(audio, dtype=np.float32)
        results: list[dict[str, object]] = []
        for start, end, spans in clips:
            clip = audio[int(start * sample_rate) : int(end * sample_rate)]
            result: dict[str, object] = {"version": ACOUSTICS_VERSION}
            if (hz := bandwidth_hz(clip, sample_rate, spans=spans)) is not None:
                result["bandwidth_hz"] = round(hz, 1)
            results.append(result)
        return results
