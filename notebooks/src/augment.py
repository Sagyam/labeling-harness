"""Training-time augmentation for the students and Flex (roadmap C, 2026-09-26), in one switchboard.

Every stage is off until its `p` (the chance a clip gets it) is raised, so a notebook turns on
exactly what an experiment is about and nothing else. The waveform stages run in a fixed order,
one clip at a time in the DataLoader workers:

    crosstalk -> speed -> reverb -> channel -> noise -> gain -> codec

- **crosstalk**: a second voice over the clip (`xtalk.py`), with lengths, level gaps and shares
  chosen in `xtalk.CrosstalkConfig`. Before it is mixed, the second voice can get a room
  (`donor_reverb`) and a microphone (`donor_channel`) of its own: in real crosstalk it reaches the
  target's microphone from further away. The corpus's measured overlap is near-symmetric
  (findings.md, 2026-09-15: within 3 dB, either louder), so always making it the wetter, thinner
  voice would teach a shortcut; keep both below p = 1 unless that is the experiment.
- **speed**: resampled to a factor, which changes tempo and pitch together (Kaldi's 0.9/1.1).
- **reverb**: a room impulse response, synthetic (exponential decay at a drawn RT60 and
  direct-to-reverberant ratio) or from a bank the notebook loads. Findings: 591 of 706 gold clips
  are dry, so expect nothing on gold.
- **channel**: a microphone's frequency response -- band edges, a few peaks and dips, optional
  soft clipping.
- **noise**: an excerpt from a `NoiseBank` (MUSAN's music and noise, say) at a drawn SNR against
  the clip's speech level. Findings: noise does not split WER on clips without crosstalk.
- **gain**: a level change, scaled back if it would clip.
- **codec**: an MP3, Opus, AAC or 8 kHz mu-law round trip through ffmpeg, realigned to the input.

`spec_augment` is separate: it masks a feature matrix after the model's own front end.

The label never changes: every stage keeps the words the clip's speaker said. A clip from val or
gold is refused outright -- augmenting what is scored is always a bug. Noise and impulse responses
come from outside the corpus, and crosstalk donors from train clips only (D76). Pure numpy plus an
ffmpeg subprocess for the codec; importable without torch.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from typing import Literal

import numpy as np
import xtalk

SR = xtalk.SR
_I16 = 32767


def _check_p(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")


def _check_range(name: str, lo_hi: tuple[float, float]) -> None:
    if not lo_hi[0] <= lo_hi[1]:
        raise ValueError(f"{name} must be (lo, hi) with lo <= hi, got {lo_hi}")


def _to_int16(x: np.ndarray) -> np.ndarray:
    """Round to int16, scaling the whole clip down if it would clip rather than clipping it."""
    peak = np.abs(x).max() if len(x) else 0.0
    if peak > _I16:
        x = x * (_I16 / peak)
    return np.round(x).astype(np.int16)


def _match_level(x: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Scale `x` to `like`'s speech level, so a filter or a room is not also a gain change."""
    have, want = xtalk.active_level_db(x), xtalk.active_level_db(like)
    if not (np.isfinite(have) and np.isfinite(want)):
        return x
    return x * 10 ** ((want - have) / 20)


def _resample(x: np.ndarray, n_out: int) -> np.ndarray:
    """Band-limited resampling to `n_out` samples by truncating or zero-padding the spectrum."""
    spectrum = np.fft.rfft(x)
    out = np.zeros(n_out // 2 + 1, dtype=complex)
    k = min(len(spectrum), len(out))
    out[:k] = spectrum[:k]
    return np.fft.irfft(out, n_out) * (n_out / len(x))


# --- stage configs --------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecConfig:
    """SpecAugment (Park et al., 2019), NeMo's defaults: masks on a (features, frames) matrix.
    `time_width` below 1 is a share of the frames (adaptive), from 1 up a frame count."""

    p: float = 0.0
    freq_masks: int = 2
    freq_width: int = 27
    time_masks: int = 10
    time_width: float = 0.05
    fill: float = 0.0

    def __post_init__(self) -> None:
        _check_p(self.p)


@dataclass(frozen=True)
class SpeedConfig:
    """One of `factors` per augmented clip; above 1 is faster and higher."""

    p: float = 0.0
    factors: tuple[float, ...] = (0.9, 1.1)

    def __post_init__(self) -> None:
        _check_p(self.p)
        if not self.factors or min(self.factors) <= 0:
            raise ValueError(f"factors must be positive, got {self.factors}")


@dataclass(frozen=True)
class ReverbConfig:
    """A room: synthetic at a drawn RT60 (s) and direct-to-reverberant ratio (dB), or an impulse
    response drawn from the bank passed to `Augmenter` (`source="bank"`)."""

    p: float = 0.0
    rt60: tuple[float, float] = (0.2, 0.8)
    drr_db: tuple[float, float] = (0.0, 10.0)
    source: Literal["synthetic", "bank"] = "synthetic"

    def __post_init__(self) -> None:
        _check_p(self.p)
        _check_range("rt60", self.rt60)
        _check_range("drr_db", self.drr_db)
        if self.rt60[0] <= 0:
            raise ValueError(f"rt60 must be positive, got {self.rt60}")
        if self.source not in ("synthetic", "bank"):
            raise ValueError(f"source must be 'synthetic' or 'bank', got {self.source!r}")


@dataclass(frozen=True)
class ChannelConfig:
    """A microphone: a high-pass at a drawn `low_cut_hz`, a low-pass at `high_cut_hz`, `peaks`
    bell-shaped boosts or cuts of `peak_db`, and, when `drive` is a range, tanh soft clipping of
    that strength."""

    p: float = 0.0
    low_cut_hz: tuple[float, float] = (50.0, 300.0)
    high_cut_hz: tuple[float, float] = (3400.0, 8000.0)
    peaks: int = 2
    peak_db: tuple[float, float] = (-6.0, 6.0)
    drive: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _check_p(self.p)
        for name in ("low_cut_hz", "high_cut_hz", "peak_db"):
            _check_range(name, getattr(self, name))
        if self.drive is not None:
            _check_range("drive", self.drive)


@dataclass(frozen=True)
class NoiseConfig:
    """An excerpt of a `NoiseBank` source at a drawn SNR against the clip's speech level; only
    sources of `categories` when it is set."""

    p: float = 0.0
    snr_db: tuple[float, float] = (5.0, 25.0)
    categories: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _check_p(self.p)
        _check_range("snr_db", self.snr_db)


@dataclass(frozen=True)
class GainConfig:
    p: float = 0.0
    db: tuple[float, float] = (-6.0, 6.0)

    def __post_init__(self) -> None:
        _check_p(self.p)
        _check_range("db", self.db)


CODECS = ("mp3", "opus", "aac", "mulaw")


@dataclass(frozen=True)
class CodecConfig:
    """A lossy round trip: one of `codecs` at one of `kbps` (mu-law is 8 kHz and ignores it)."""

    p: float = 0.0
    codecs: tuple[str, ...] = CODECS
    kbps: tuple[int, ...] = (16, 24, 32, 64)

    def __post_init__(self) -> None:
        _check_p(self.p)
        unknown = set(self.codecs) - set(CODECS)
        if unknown or not self.codecs:
            raise ValueError(f"codecs must be among {CODECS}, got {self.codecs}")


@dataclass(frozen=True)
class AugmentConfig:
    """Every stage, all off by default. `from_dict` builds one from a notebook's config cell."""

    spec: SpecConfig = field(default_factory=SpecConfig)
    crosstalk: xtalk.CrosstalkConfig = field(default_factory=xtalk.CrosstalkConfig)
    donor_reverb: ReverbConfig = field(default_factory=ReverbConfig)
    donor_channel: ChannelConfig = field(default_factory=ChannelConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    reverb: ReverbConfig = field(default_factory=ReverbConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    gain: GainConfig = field(default_factory=GainConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)

    @classmethod
    def from_dict(cls, d: dict) -> AugmentConfig:
        """{"speed": {"p": 0.3}, "crosstalk": {"p": 0.5, "seconds": [2, 8]}, ...}; lists become
        tuples. An unknown stage is a ValueError, an unknown setting a TypeError."""
        kinds = {f.name: f.default_factory for f in fields(cls)}
        unknown = set(d) - set(kinds)
        if unknown:
            raise ValueError(f"unknown augmentation stage(s): {sorted(unknown)}")
        return cls(
            **{
                name: kinds[name](
                    **{k: tuple(v) if isinstance(v, list) else v for k, v in settings.items()}
                )
                for name, settings in d.items()
            }
        )


# --- the stages -----------------------------------------------------------------------------------


def spec_augment(
    feats: np.ndarray, rng: np.random.Generator, cfg: SpecConfig
) -> tuple[np.ndarray, dict | None]:
    """Frequency and time masks on a (features, frames) matrix, with probability `cfg.p`. The
    input is never modified."""
    if not cfg.p or rng.random() >= cfg.p:
        return feats, None
    out = np.array(feats, copy=True)
    n_freq, n_time = out.shape[-2:]
    info: dict = {"freq": [], "time": []}
    for _ in range(cfg.freq_masks):
        w = int(rng.integers(1, min(cfg.freq_width, n_freq) + 1))
        f0 = int(rng.integers(0, n_freq - w + 1))
        out[..., f0 : f0 + w, :] = cfg.fill
        info["freq"].append((f0, w))
    width = cfg.time_width * n_time if cfg.time_width < 1 else cfg.time_width
    width = max(1, min(int(width), n_time))
    for _ in range(cfg.time_masks):
        w = int(rng.integers(1, width + 1))
        t0 = int(rng.integers(0, n_time - w + 1))
        out[..., t0 : t0 + w] = cfg.fill
        info["time"].append((t0, w))
    return out, info


def change_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """`audio` played `factor` times as fast: shorter and higher above 1, as sox's `speed`."""
    if factor == 1.0:
        return audio
    n_out = round(len(audio) / factor)
    return _to_int16(_resample(audio.astype(np.float64), n_out))


def apply_gain(audio: np.ndarray, db: float) -> np.ndarray:
    return _to_int16(audio.astype(np.float64) * 10 ** (db / 20))


def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """`noise` (as long as `audio`) added `snr_db` below the clip's speech level."""
    level = xtalk.active_level_db(audio)
    x = noise.astype(np.float64)
    power = np.mean(x**2) if len(x) else 0.0
    if not np.isfinite(level) or power <= 0:
        return audio
    scale = 10 ** ((level - snr_db - 10 * np.log10(power)) / 20)
    return _to_int16(audio.astype(np.float64) + x * scale)


class NoiseBank:
    """Named noise sources by category, as (name, category, int16 audio at 16 kHz)."""

    def __init__(self, sources: Sequence[tuple[str, str, np.ndarray]]):
        self.sources = [(n, c, np.asarray(a)) for n, c, a in sources if len(a)]
        if not self.sources:
            raise ValueError("a noise bank needs at least one non-empty source")

    def excerpt(
        self, rng: np.random.Generator, n: int, categories: Sequence[str] | None = None
    ) -> tuple[str, str, np.ndarray]:
        """A random `n`-sample stretch of a random source, looped when the source is shorter."""
        pool = [s for s in self.sources if categories is None or s[1] in categories]
        if not pool:
            raise ValueError(f"no noise source in categories {categories}")
        name, category, audio = pool[rng.integers(len(pool))]
        start = int(rng.integers(len(audio)))
        if len(audio) >= n + start:
            return name, category, audio[start : start + n]
        return name, category, np.resize(np.roll(audio, -start), n)


def synthetic_rir(rng: np.random.Generator, rt60: float, drr_db: float, sr: int = SR) -> np.ndarray:
    """A direct path of 1 and an exponentially decaying noise tail that loses 60 dB of energy
    in `rt60` seconds, carrying `drr_db` less energy than the direct path."""
    n = max(2, int(1.2 * rt60 * sr))
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-3 * np.log(10) * t / rt60)
    h[0] = 0.0
    h *= np.sqrt(10 ** (-drr_db / 10) / np.sum(h**2))
    h[0] = 1.0
    return h


def convolve(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """`audio` through `rir`, cut to its own length and put back at its own speech level."""
    x = audio.astype(np.float64)
    n = len(x) + len(rir) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(rir, nfft), nfft)[: len(x)]
    return _to_int16(_match_level(y, x))


def microphone_response(
    freqs: np.ndarray, low_cut: float, high_cut: float, peaks: Sequence[tuple[float, float]]
) -> np.ndarray:
    """Magnitude at `freqs`: 6th-order Butterworth-shaped band edges and (centre Hz, dB) bells a
    third of an octave wide."""
    f = np.maximum(freqs, 1e-3)
    gain = 1 / np.sqrt(1 + (low_cut / f) ** 12) / np.sqrt(1 + (f / high_cut) ** 12)
    for centre, db in peaks:
        octaves = np.log2(f / centre)
        gain = gain * 10 ** (db * np.exp(-0.5 * (octaves / (1 / 3)) ** 2) / 20)
    return gain


def channel(
    audio: np.ndarray, rng: np.random.Generator, cfg: ChannelConfig
) -> tuple[np.ndarray, dict]:
    """A drawn microphone applied to `audio`, which keeps its speech level."""
    low = float(rng.uniform(*cfg.low_cut_hz))
    high = float(rng.uniform(*cfg.high_cut_hz))
    peaks = [
        (
            float(np.exp(rng.uniform(np.log(200.0), np.log(6000.0)))),
            float(rng.uniform(*cfg.peak_db)),
        )
        for _ in range(cfg.peaks)
    ]
    x = audio.astype(np.float64)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    y = np.fft.irfft(np.fft.rfft(x) * microphone_response(freqs, low, high, peaks), len(x))
    info = {"low_cut_hz": low, "high_cut_hz": high, "peaks": peaks}
    if cfg.drive is not None:
        drive = float(rng.uniform(*cfg.drive))
        if drive > 0:
            peak = np.abs(y).max() or 1.0
            y = np.tanh(drive * y / peak) / np.tanh(drive) * peak
        info["drive"] = drive
    return _to_int16(_match_level(y, x)), info


_CODEC_ARGS = {
    "mp3": (["-c:a", "libmp3lame"], "mp3", True),
    "opus": (["-c:a", "libopus"], "ogg", True),
    "aac": (["-c:a", "aac"], "adts", True),
    "mulaw": (["-c:a", "pcm_mulaw", "-ar", "8000"], "wav", False),
}
_MAX_LAG = SR // 5  # encoder delay and priming stay well inside 0.2 s


def codec_round_trip(audio: np.ndarray, name: str, kbps: int = 32) -> np.ndarray:
    """`audio` encoded with `name` at `kbps` and decoded back to 16 kHz by ffmpeg, then shifted by
    the encoder's delay and cut or padded to its own length, so the words stay where they were."""
    args, fmt, lossy = _CODEC_ARGS[name]
    raw = ["-f", "s16le", "-ar", str(SR), "-ac", "1"]
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            *raw,
            "-i",
            "pipe:0",
            *args,
            *(["-b:a", f"{kbps}k"] if lossy else []),
            "-f",
            fmt,
            "pipe:1",
        ],
        input=audio.astype(np.int16).tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", *raw, "pipe:1"],
        input=encoded,
        capture_output=True,
        check=True,
    ).stdout
    y = np.frombuffer(decoded, dtype=np.int16).astype(np.float64)
    x = audio.astype(np.float64)
    n = len(x) + len(y)
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(y, nfft) * np.conj(np.fft.rfft(x, nfft)), nfft)
    lags = np.r_[corr[: _MAX_LAG + 1], corr[-_MAX_LAG:]]
    lag = int(np.argmax(lags))
    lag = lag if lag <= _MAX_LAG else lag - len(lags)
    y = y[lag:] if lag >= 0 else np.r_[np.zeros(-lag), y]
    y = np.r_[y[: len(x)], np.zeros(max(0, len(x) - len(y)))]
    return _to_int16(y)


# --- the chain ------------------------------------------------------------------------------------


class Augmenter:
    """The waveform chain for one training clip: `aug(row, clip, rng) -> (audio, info)`, where
    `info["stages"]` lists what was done, in order. Resources a stage needs are passed here:
    `noise` (a `NoiseBank`), `rirs` (impulse responses, for `source="bank"`), and `donors` plus
    `fetch(episode, start, end)` for crosstalk."""

    def __init__(
        self,
        config: AugmentConfig,
        noise: NoiseBank | None = None,
        rirs: Sequence[np.ndarray] | None = None,
        donors: xtalk.DonorPool | xtalk.ClipDonorPool | None = None,
        fetch: Callable[[str, float, float], np.ndarray] | None = None,
    ):
        self.config, self.noise, self.rirs = config, noise, list(rirs or [])
        if config.noise.p and noise is None:
            raise ValueError("noise.p > 0 needs a NoiseBank")
        for name in ("reverb", "donor_reverb"):
            r = getattr(config, name)
            if r.p and r.source == "bank" and not self.rirs:
                raise ValueError(f"{name}.source='bank' needs impulse responses")
        self.mixer = None
        if config.crosstalk.p:
            if donors is None or fetch is None:
                raise ValueError("crosstalk.p > 0 needs a donor pool and a fetch function")
            self.mixer = xtalk.Mixer(
                donors, fetch, config=config.crosstalk, donor_fx=self._donor_fx
            )

    def __call__(
        self, row: dict, clip: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        if row.get("split") in ("val", "test") or row.get("pot") == "gold":
            raise ValueError(f"{row.get('segment_id')}: val and gold audio are never augmented")
        cfg, audio, stages = self.config, clip, []
        if self.mixer is not None:
            audio, info = self.mixer(row, audio, rng)
            if info:
                stages.append(
                    {"stage": "crosstalk", **{k: v for k, v in info.items() if k != "segment_id"}}
                )
        if cfg.speed.p and rng.random() < cfg.speed.p:
            factor = float(cfg.speed.factors[rng.integers(len(cfg.speed.factors))])
            audio = change_speed(audio, factor)
            stages.append({"stage": "speed", "factor": factor})
        audio = self._reverb(audio, rng, cfg.reverb, stages)
        audio = self._channel(audio, rng, cfg.channel, stages)
        if cfg.noise.p and rng.random() < cfg.noise.p:
            name, category, noise = self.noise.excerpt(rng, len(audio), cfg.noise.categories)
            snr = float(rng.uniform(*cfg.noise.snr_db))
            audio = add_noise(audio, noise, snr)
            stages.append({"stage": "noise", "source": name, "category": category, "snr_db": snr})
        if cfg.gain.p and rng.random() < cfg.gain.p:
            db = float(rng.uniform(*cfg.gain.db))
            audio = apply_gain(audio, db)
            stages.append({"stage": "gain", "db": db})
        if cfg.codec.p and rng.random() < cfg.codec.p:
            name = str(cfg.codec.codecs[rng.integers(len(cfg.codec.codecs))])
            kbps = int(cfg.codec.kbps[rng.integers(len(cfg.codec.kbps))])
            audio = codec_round_trip(audio, name, kbps)
            stages.append({"stage": "codec", "codec": name, "kbps": kbps})
        return audio, {"segment_id": row.get("segment_id"), "stages": stages}

    def spec(self, feats: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return spec_augment(feats, rng, self.config.spec)[0]

    def _reverb(self, audio, rng, cfg: ReverbConfig, stages: list) -> np.ndarray:
        if not cfg.p or rng.random() >= cfg.p:
            return audio
        if cfg.source == "bank":
            i = int(rng.integers(len(self.rirs)))
            stages.append({"stage": "reverb", "rir": i})
            return convolve(audio, self.rirs[i])
        rt60, drr = float(rng.uniform(*cfg.rt60)), float(rng.uniform(*cfg.drr_db))
        stages.append({"stage": "reverb", "rt60": rt60, "drr_db": drr})
        return convolve(audio, synthetic_rir(rng, rt60, drr))

    @staticmethod
    def _channel(audio, rng, cfg: ChannelConfig, stages: list) -> np.ndarray:
        if not cfg.p or rng.random() >= cfg.p:
            return audio
        audio, info = channel(audio, rng, cfg)
        stages.append({"stage": "channel", **info})
        return audio

    def _donor_fx(self, audio: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        stages: list = []
        audio = self._reverb(audio, rng, self.config.donor_reverb, stages)
        audio = self._channel(audio, rng, self.config.donor_channel, stages)
        return audio, {"stages": stages}
