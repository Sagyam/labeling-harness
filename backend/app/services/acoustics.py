"""Acoustic measurements of each clip, read from the episode audio (roadmap item 1, D87).

**Bandwidth.** The highest frequency at which the clip's long-term speech spectrum is still
within :data:`BANDWIDTH_DROP_DB` of its peak in the voice band (200 Hz - 2 kHz). A phone line
stops near 3.4 kHz, some codecs and microphones near 5 kHz, and full-band audio runs to the 8 kHz
Nyquist limit of the stored 16 kHz clips. Only the clip's VAD speech is read: the pauses between
words carry the room, not the voice. Over the 44 episodes of 2026-09-15 most podcasts sat near
7.8 kHz and many tech reviews near 5.3 kHz.

**SNR and reverb.** Brouhaha (:mod:`app.services.brouhaha`) reads the whole episode, and each
clip gets the mean speech-to-noise ratio and C50 over its own frames that Brouhaha calls speech.
The model is gated and never downloaded; without it a clip gets bandwidth alone, and is marked
so (:data:`BANDWIDTH_ONLY_VERSION`) that a backfill with the model finishes the job.

**Not measured: clipping.** The stored audio is loudness-normalised and resampled to 16 kHz
before anything reads it (ingest stage 1), and resampling rounds off the flat tops a clipping
detector looks for. The source audio is not retained, so there is nothing honest to measure.

Every result carries :data:`ACOUSTICS_VERSION`, so a backfill can tell a clip measured under
older rules from one measured under the current ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
#: Bump whenever a measurement's rule changes or one is added.
ACOUSTICS_VERSION = "acoustics-v2"
#: The same rules measured without Brouhaha: bandwidth only, to be completed when it is present.
BANDWIDTH_ONLY_VERSION = f"{ACOUSTICS_VERSION}-bandwidth-only"
#: A Brouhaha frame counts as speech above this probability, as in brouhaha-vad's own pipeline.
SPEECH_THRESHOLD = 0.5

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


class FrameModel(Protocol):
    """What the meter needs of :class:`~app.services.brouhaha.Brouhaha`."""

    @property
    def available(self) -> bool: ...

    def frames(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...


class AcousticMeter:
    """Measures every clip of an episode from the whole episode's audio.

    Args:
        brouhaha: The SNR/reverb model; ``None`` measures bandwidth only. :meth:`default` loads
            the exported model when it is on disk.
    """

    def __init__(self, brouhaha: FrameModel | None = None) -> None:
        self._brouhaha = brouhaha if brouhaha is not None and brouhaha.available else None

    @classmethod
    def default(cls) -> AcousticMeter:
        """The meter ingest and the backfill use: Brouhaha from its default path, if there."""
        from app.services.brouhaha import Brouhaha

        return cls(Brouhaha())

    @property
    def version(self) -> str:
        """What this meter's results are stamped with."""
        return ACOUSTICS_VERSION if self._brouhaha is not None else BANDWIDTH_ONLY_VERSION

    def is_current(self, result: Mapping[str, object] | None) -> bool:
        """Whether a stored result needs nothing this meter could add. A full result is current
        even for a meter without Brouhaha: a backfill never measures less than it found."""
        version = (result or {}).get("version")
        return version in (ACOUSTICS_VERSION, self.version)

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
        frames = self._brouhaha.frames(audio, sample_rate) if self._brouhaha is not None else None
        results: list[dict[str, object]] = []
        for start, end, spans in clips:
            clip = audio[int(start * sample_rate) : int(end * sample_rate)]
            result: dict[str, object] = {"version": self.version}
            if (hz := bandwidth_hz(clip, sample_rate, spans=spans)) is not None:
                result["bandwidth_hz"] = round(hz, 1)
            if frames is not None:
                result |= _speech_means(frames, start, end)
            results.append(result)
        return results


def _speech_means(frames: np.ndarray, start: float, end: float) -> dict[str, object]:
    """Mean SNR and C50 over the clip's speech frames; nothing when Brouhaha heard no speech."""
    from app.services.brouhaha import C50, FRAME_STEP, SNR, VAD

    clip = frames[int(start / FRAME_STEP) : int(np.ceil(end / FRAME_STEP))]
    speech = clip[clip[:, VAD] > SPEECH_THRESHOLD]
    if not len(speech):
        return {}
    return {
        "snr_db": round(float(speech[:, SNR].mean()), 1),
        "c50_db": round(float(speech[:, C50].mean()), 1),
    }
