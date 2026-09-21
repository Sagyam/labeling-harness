"""Synthetic crosstalk for fine-tuning (roadmap item 3), mixed on the fly in the DataLoader workers.

A clean training clip gets short bursts of another person's speech laid over it. Its label is
unchanged: the model learns to transcribe the clip's own speaker and write nothing for the
interjection, which is what the references do in real overlap.

Every distribution here is a measured one (docs/findings.md, *Overlap windows: how long they are
and who is louder*, 2026-09-15), not a LibriMix-style guess:
  * windows are bursts: median 0.42 s, 83% under 1 s -- backchannels and interjections, not duets;
  * the two voices are within 3 dB of each other, median gap ~1.6 dB, and either can be louder.
    The model cannot learn "ignore the quieter voice"; it has to follow the clip's own speaker;
  * the interrupter comes from the **same episode** when one exists: the same room, microphone and
    level, so the other voice cannot be told apart by its channel. Another train episode only when
    the episode has no other voice with a long enough solo stretch.

What is deliberately not faithful: how much of a clip is overlapped. Real crosstalk is 3% of the
audio, and the errors live in clips more than 5% overlapped, so an augmented clip draws its
overlapped share with most of the weight on 5-15% and 15-40% (`SHARES`).

Donor audio is only ever solo speech from train clips (`DonorPool` refuses anything else), so no
val or gold audio is mixed into training (D76). Pure numpy; importable without torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

SR = 16_000

#: (quantile, seconds) of measured overlap-window durations over all 7,075 clips (findings.md).
#: The 0th point is a floor: the detector's shortest windows are a few frames long.
WINDOW_QUANTILES = (
    (0.0, 0.05),
    (0.1, 0.07),
    (0.2, 0.17),
    (0.3, 0.27),
    (0.4, 0.35),
    (0.5, 0.42),
    (0.6, 0.52),
    (0.7, 0.66),
    (0.8, 0.89),
    (0.9, 1.33),
    (0.95, 1.93),
    (0.99, 4.03),
    (1.0, 8.94),
)
#: |interrupter level - speaker level| in dB is triangular on [0, GAP_MAX_DB] with this mode:
#: median 1.60, IQR 1.13-2.01, p90 2.37 against the measured 1.59, 1.17-2.06, 2.51.
GAP_MODE_DB, GAP_MAX_DB = 1.7, 3.0
#: (weight, low, high) of an augmented clip's overlapped share. Most weight goes where the errors
#: are: the 5-15% and >15% buckets of the crosstalk study.
SHARES = ((0.4, 0.01, 0.05), (0.3, 0.05, 0.15), (0.3, 0.15, 0.40))
#: No clip is more than this share overlapped, whatever was drawn: the label must stay the
#: clip's own speaker's.
MAX_SHARE = 0.6
#: A window that would take the clip past this multiple of its drawn share is redrawn, so a light
#: target gets short bursts rather than landing in a heavier bucket.
OVERSHOOT = 1.5
MIN_WINDOW_S = 0.05
SPACING_S = 0.05  # between two windows in one clip
FADE_S = 0.01  # raised-cosine edges, so a burst never starts with a click
_FRAME, _HOP = 400, 160  # 25 ms / 10 ms, as the level measurement in findings.md


def sample_window_s(rng: np.random.Generator) -> float:
    """One overlap window's duration, from the measured deciles."""
    q, s = zip(*WINDOW_QUANTILES, strict=True)
    return float(np.interp(rng.random(), q, s))


def sample_gap_db(rng: np.random.Generator) -> float:
    """The interrupter's level minus the speaker's, in dB; positive means the interrupter is
    louder, which in this corpus is as common as the reverse."""
    gap = rng.triangular(0.0, GAP_MODE_DB, GAP_MAX_DB)
    return float(gap if rng.random() < 0.5 else -gap)


def sample_share(rng: np.random.Generator) -> float:
    """The share of an augmented clip to overlap."""
    weights = np.array([w for w, _, _ in SHARES])
    _, lo, hi = SHARES[rng.choice(len(SHARES), p=weights / weights.sum())]
    return float(rng.uniform(lo, hi))


def active_level_db(audio: np.ndarray) -> float:
    """Speech level in dB (re one int16 step): the mean power of the 25 ms frames within 30 dB of
    the loudest one, so pauses do not pull it down. -inf for silence."""
    x = np.asarray(audio, dtype=np.float64)
    if len(x) < _FRAME:
        power = np.array([np.mean(x**2)]) if len(x) else np.zeros(1)
    else:
        frames = np.lib.stride_tricks.sliding_window_view(x, _FRAME)[::_HOP]
        power = np.mean(frames**2, axis=1)
    top = power.max()
    if top <= 0:
        return -np.inf
    return float(10 * np.log10(power[power >= top * 1e-3].mean()))


# --- donors ---------------------------------------------------------------------------------------


def voice_key(row: dict, turn: dict) -> str:
    """A linked voice id is global across episodes; an unlinked diarizer label is only
    meaningful inside its episode."""
    return turn["voice"] or f"{row['episode_id']}:{turn['speaker']}"


def clip_voices(row: dict) -> set[str]:
    return {voice_key(row, t) for t in row.get("speaker_turns") or []}


def _subtract(spans: list[tuple[float, float]], cuts: Sequence[Sequence[float]]):
    for c0, c1 in cuts:
        spans = [
            piece
            for s, e in spans
            for piece in ((s, min(e, c0)), (max(s, c1), e))
            if piece[1] > piece[0]
        ]
    return spans


def solo_stretches(row: dict, min_s: float) -> list[tuple[float, float, str]]:
    """Where exactly one voice speaks in this clip, as episode-absolute (start, end, voice):
    each voice's turns minus every other voice's turns and the measured overlap."""
    turns = row.get("speaker_turns") or []
    out = []
    for voice in sorted({voice_key(row, t) for t in turns}):
        mine = sorted((t["start"], t["end"]) for t in turns if voice_key(row, t) == voice)
        others = [(t["start"], t["end"]) for t in turns if voice_key(row, t) != voice]
        spans = _subtract(mine, others + [tuple(o) for o in row.get("overlap_spans") or []])
        t0 = row["start_time"]
        out += [(t0 + s, t0 + e, voice) for s, e in spans if e - s >= min_s]
    return sorted(out)


@dataclass(frozen=True)
class Donor:
    episode: str
    voice: str
    start: float  # episode-absolute; the excerpt is [start, start + the requested duration]
    same_episode: bool


class DonorPool:
    """Solo stretches of every voice in the train clips, for interrupters to be cut from."""

    def __init__(self, rows: Sequence[dict], min_s: float = MIN_WINDOW_S):
        stretches = []
        for r in rows:
            if r.get("split") != "train" or r.get("pot") != "train":
                raise ValueError(f"{r['segment_id']}: donors come from train clips only (D76)")
            stretches += [(r["episode_id"], *s) for s in solo_stretches(r, min_s)]
        self.episodes = sorted({s[0] for s in stretches})
        self.voices = sorted({s[3] for s in stretches})
        ep_idx = {e: i for i, e in enumerate(self.episodes)}
        voice_idx = {v: i for i, v in enumerate(self.voices)}
        self.ep = np.array([ep_idx[s[0]] for s in stretches], dtype=np.int32)
        self.start = np.array([s[1] for s in stretches])
        self.end = np.array([s[2] for s in stretches])
        self.voice = np.array([voice_idx[s[3]] for s in stretches], dtype=np.int32)
        self._ep_idx, self._voice_idx = ep_idx, voice_idx

    def __len__(self) -> int:
        return len(self.start)

    def pick(
        self, rng: np.random.Generator, episode: str, exclude: set[str], seconds: float
    ) -> Donor | None:
        """A stretch of at least `seconds` by a voice not in `exclude`: from `episode` when it
        has one, else from any other train episode. Longer stretches are likelier, so every
        second of solo speech is about equally likely to be used."""
        ok = self.end - self.start >= seconds
        banned = [self._voice_idx[v] for v in exclude if v in self._voice_idx]
        if banned:
            ok &= ~np.isin(self.voice, banned)
        here = self.ep == self._ep_idx.get(episode, -1)
        same = bool((ok & here).any())
        ok &= here if same else ~here
        idx = np.flatnonzero(ok)
        if not len(idx):
            return None
        spare = self.end[idx] - self.start[idx] - seconds
        i = idx[rng.choice(len(idx), p=(spare + 0.1) / (spare + 0.1).sum())]
        start = self.start[i] + rng.uniform(0.0, self.end[i] - self.start[i] - seconds)
        return Donor(self.episodes[self.ep[i]], self.voices[self.voice[i]], float(start), same)


# --- placement and mixing -------------------------------------------------------------------------


def plan_windows(
    rng: np.random.Generator, clip_s: float, share: float
) -> list[tuple[float, float]]:
    """(offset, duration) windows in a clip, each drawn from the measured durations, apart from
    one another, until `share` of the clip is covered. A window that would pass OVERSHOOT x
    `share` (or MAX_SHARE) is redrawn; after 64 draws the clip keeps what it has, which may be
    nothing."""
    goal = share * clip_s
    cap = min(MAX_SHARE, OVERSHOOT * share) * clip_s
    wins: list[tuple[float, float]] = []
    total = 0.0
    for _ in range(64):
        if total >= goal:
            break
        d = sample_window_s(rng)
        if total + d > cap:
            continue
        s = float(rng.uniform(0.0, clip_s - d))
        if any(s < s2 + d2 + SPACING_S and s2 < s + d + SPACING_S for s2, d2 in wins):
            continue
        wins.append((s, d))
        total += d
    return wins


def mix(target: np.ndarray, pieces: Sequence[tuple[int, np.ndarray, float]]) -> np.ndarray:
    """Add each (offset in samples, audio, gap dB) piece to `target` at `gap` dB relative to the
    target's speech level, with faded edges. Audio outside the pieces is untouched unless the sum
    would clip, in which case the whole clip is scaled down."""
    out = target.astype(np.float64)
    level = active_level_db(target)
    for offset, audio, gap in pieces:
        piece = np.asarray(audio, dtype=np.float64)[: max(0, len(out) - offset)]
        piece_level = active_level_db(piece)
        if not len(piece) or not np.isfinite(piece_level) or not np.isfinite(level):
            continue
        fade = min(round(FADE_S * SR), len(piece) // 2)
        if fade:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
            piece[:fade] *= ramp
            piece[-fade:] *= ramp[::-1]
        out[offset : offset + len(piece)] += piece * 10 ** ((level + gap - piece_level) / 20)
    peak = np.abs(out).max() if len(out) else 0.0
    if peak > 32767:
        out *= 32767 / peak
    return np.round(out).astype(np.int16)


class Mixer:
    """Crosstalk for one training clip, called from `collate`.

    Only clips measured clean (`overlap_spans == []`) and diarized are candidates, each with
    probability `p`: a clip with real crosstalk keeps it as it is, and one never measured is not
    assumed clean. Returns the (possibly new) int16 audio and what was done, or None."""

    def __init__(
        self,
        pool: DonorPool,
        fetch: Callable[[str, float, float], np.ndarray],
        p: float,
    ):
        self.pool, self.fetch, self.p = pool, fetch, p

    def __call__(
        self, row: dict, clip: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict | None]:
        if row.get("overlap_spans") != [] or not row.get("speaker_turns"):
            return clip, None
        if rng.random() >= self.p:
            return clip, None
        clip_s = len(clip) / SR
        exclude = clip_voices(row)
        pieces, windows = [], []
        for offset, seconds in plan_windows(rng, clip_s, sample_share(rng)):
            donor = self.pool.pick(rng, row["episode_id"], exclude, seconds)
            if donor is None:
                continue
            gap = sample_gap_db(rng)
            audio = self.fetch(donor.episode, donor.start, donor.start + seconds)
            pieces.append((round(offset * SR), audio, gap))
            windows.append(
                {
                    "offset": offset,
                    "seconds": seconds,
                    "gap_db": gap,
                    "same_episode": donor.same_episode,
                }
            )
        if not pieces:
            return clip, None
        info = {
            "segment_id": row["segment_id"],
            "windows": windows,
            "share": sum(w["seconds"] for w in windows) / clip_s,
            "same_episode": sum(w["same_episode"] for w in windows),
        }
        return mix(clip, pieces), info
