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

Everything above is the default, and it is what the D96 sweep trained Flex on. `CrosstalkConfig`
overrides it for models built to handle overlap (roadmap C and D, 2026-09-26): windows of a chosen
length, at a chosen level gap, and optionally whole verified train clips as the second voice
(`ClipDonorPool`), whose transcripts a serialized-output or target-speaker label needs. A
`donor_fx` hook lets `augment.py` pass the second voice through a room and a microphone of its own
before it is mixed.

What the label says is `CrosstalkConfig.label`. "target" keeps the clip's own text: the model is
taught to write its speaker and nothing of the other voice, which is what a target-speaker model
wants and what the references did before D100. Gold now writes everything said, whoever said it
(D100), so a single-stream model is trained with "everything": the target's and the donor clip's
words merged in time order from the export's `label_words`, returned as `info["text"]`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

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
#: How the export splits `text` into `label_words`: app/services/normalize.py's WORD_TOKEN_RE,
#: copied because this module runs without the backend (a test keeps the two equal).
TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[ऀ-ॣ०-ॿ]+")


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
        self,
        rng: np.random.Generator,
        episode: str,
        exclude: set[str],
        seconds: float,
        other_episode: bool = False,
    ) -> Donor | None:
        """A stretch of at least `seconds` by a voice not in `exclude`: from `episode` when it
        has one, else from any other train episode. Longer stretches are likelier, so every
        second of solo speech is about equally likely to be used. `other_episode` rules the
        target's own episode out, for a target whose voices are unknown."""
        ok = self.end - self.start >= seconds
        banned = [self._voice_idx[v] for v in exclude if v in self._voice_idx]
        if banned:
            ok &= ~np.isin(self.voice, banned)
        here = self.ep == self._ep_idx.get(episode, -1)
        same = not other_episode and bool((ok & here).any())
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
    rng: np.random.Generator,
    clip_s: float,
    share: float,
    draw: Callable[[np.random.Generator], float] = sample_window_s,
    max_share: float = MAX_SHARE,
    overshoot: float | None = OVERSHOOT,
) -> list[tuple[float, float]]:
    """(offset, duration) windows in a clip, each drawn from `draw` (the measured durations by
    default), apart from one another, until `share` of the clip is covered. A window that would
    pass `overshoot` x `share` (or `max_share`; `max_share` alone when `overshoot` is None) is
    redrawn; after 64 draws the clip keeps what it has, which may be nothing."""
    goal = share * clip_s
    cap = (max_share if overshoot is None else min(max_share, overshoot * share)) * clip_s
    wins: list[tuple[float, float]] = []
    total = 0.0
    for _ in range(64):
        if total >= goal:
            break
        d = draw(rng)
        if total + d > cap or d > clip_s:
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


def _units(row: dict) -> list[tuple[float, str]] | None:
    """(start, word as written) for each word of `row["text"]`, punctuation kept with the word
    before it, timed by `row["label_words"]`. None when the words do not spell the text."""
    text, words = row.get("text") or "", row.get("label_words")
    tokens = list(TEXT_TOKEN_RE.finditer(text))
    if not words or [m.group(0) for m in tokens] != [w["word"] for w in words]:
        return None
    cuts = [0] + [m.start() for m in tokens[1:]] + [len(text)]
    return [
        (float(w["start"]), text[a:b].strip())
        for w, a, b in zip(words, cuts[:-1], cuts[1:], strict=True)
    ]


def merge_labels(target: dict, donors: Sequence[tuple[float, dict]]) -> str | None:
    """Everything said, in time order (D100): the target's words and each donor clip's, shifted by
    its offset in the target, interleaved by start time. A word keeps its punctuation; at the same
    start the target's word comes first. None when any side lacks words that spell its text."""
    mine = _units(target)
    if mine is None:
        return None
    timed = [(start, 0, i, w) for i, (start, w) in enumerate(mine)]
    for n, (offset, donor) in enumerate(donors, 1):
        theirs = _units(donor)
        if theirs is None:
            return None
        timed += [(offset + start, n, i, w) for i, (start, w) in enumerate(theirs)]
    return " ".join(w for *_, w in sorted(timed))


@dataclass(frozen=True)
class ClipDonor:
    segment_id: str
    episode: str
    start: float  # episode-absolute: the whole clip, so its verified text is the donor's label
    end: float
    same_episode: bool


class ClipDonorPool:
    """Whole verified train clips as the second voice (roadmap C item 1). A window is the donor's
    whole clip, so what the second voice said is known: `Mixer` names the donor in its info, which
    is what a serialized-output or target-speaker label is built from."""

    def __init__(self, rows: Sequence[dict]):
        for r in rows:
            if r.get("split") != "train" or r.get("pot") != "train":
                raise ValueError(f"{r['segment_id']}: donors come from train clips only (D76)")
        self.rows = list(rows)
        self.by_id = {r["segment_id"]: r for r in self.rows}
        self.worded = np.array([_units(r) is not None for r in self.rows], dtype=bool)
        self.duration = np.array([r["end_time"] - r["start_time"] for r in self.rows])
        self.episode = np.array([r["episode_id"] for r in self.rows])
        self.voices = [clip_voices(r) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def pick(
        self,
        rng: np.random.Generator,
        target: dict,
        seconds: tuple[float, float],
        used: frozenset[str] = frozenset(),
        worded: bool = False,
    ) -> ClipDonor | None:
        """A clip lasting `seconds` (lo, hi) by voices none of which is the target's, and not in
        `used`: from the target's episode when one qualifies, else from another train episode.
        Same-episode clips need both sides diarized; otherwise the same person could be talking
        over themself. `worded` keeps only clips whose `label_words` spell their text."""
        lo, hi = seconds
        mine = clip_voices(target)
        skip = used | {target["segment_id"]}
        ok = (self.duration >= lo) & (self.duration <= hi)
        if worded:
            ok &= self.worded
        ok &= np.array(
            [
                r["segment_id"] not in skip and not (v & mine)
                for r, v in zip(self.rows, self.voices, strict=True)
            ],
            dtype=bool,
        )
        here = self.episode == target["episode_id"]
        known = np.array([bool(v) and bool(mine) for v in self.voices], dtype=bool)
        same = bool((ok & here & known).any())
        ok &= (here & known) if same else ~here
        idx = np.flatnonzero(ok)
        if not len(idx):
            return None
        r = self.rows[idx[rng.integers(len(idx))]]
        return ClipDonor(r["segment_id"], r["episode_id"], r["start_time"], r["end_time"], same)


@dataclass(frozen=True)
class CrosstalkConfig:
    """How the second voice is laid over a clip. Every None is the measured distribution the D96
    sweep used; set a field to override it.

    - `p`: chance a candidate clip is mixed at all.
    - `seconds`: (lo, hi) length of each window, uniform. With `donor="clip"` it is the range of
      donor clip lengths, since a whole clip is laid down.
    - `gap_db`: (lo, hi) second voice's level minus the speaker's, uniform; positive is louder.
    - `share`: (lo, hi) share of the clip to overlap, uniform.
    - `max_share`: no clip is overlapped more than this.
    - `overshoot`: a window that would take the clip past `overshoot` x the drawn share is
      redrawn; None lets `max_share` alone cap it, which long windows need.
    - `donor`: "stretch" (solo speech cut from train clips; label unchanged) or "clip" (whole
      verified train clips, named in the info).
    - `clean_only`: mix only clips measured free of crosstalk and diarized. Off, any clip is a
      candidate, and one without speaker turns takes its donor from another episode.
    - `label`: "target" keeps the clip's text (the other voice goes unwritten); "everything" writes
      both voices' words in time order into `info["text"]`, as gold is labelled (D100). It needs
      `donor="clip"`, and target and donors with `label_words`.
    """

    p: float = 0.0
    seconds: tuple[float, float] | None = None
    gap_db: tuple[float, float] | None = None
    share: tuple[float, float] | None = None
    max_share: float = MAX_SHARE
    overshoot: float | None = OVERSHOOT
    donor: Literal["stretch", "clip"] = "stretch"
    clean_only: bool = True
    label: Literal["target", "everything"] = "target"

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        if self.seconds is not None and not 0.0 < self.seconds[0] <= self.seconds[1]:
            raise ValueError(f"seconds must be 0 < lo <= hi, got {self.seconds}")
        if self.gap_db is not None and not self.gap_db[0] <= self.gap_db[1]:
            raise ValueError(f"gap_db must be lo <= hi, got {self.gap_db}")
        if self.share is not None and not 0.0 < self.share[0] <= self.share[1] <= 1.0:
            raise ValueError(f"share must be 0 < lo <= hi <= 1, got {self.share}")
        if not 0.0 < self.max_share <= 1.0:
            raise ValueError(f"max_share must be in (0, 1], got {self.max_share}")
        if self.donor not in ("stretch", "clip"):
            raise ValueError(f"donor must be 'stretch' or 'clip', got {self.donor!r}")
        if self.label not in ("target", "everything"):
            raise ValueError(f"label must be 'target' or 'everything', got {self.label!r}")
        if self.label == "everything" and self.donor != "clip":
            raise ValueError("label='everything' needs donor='clip': a cut stretch has no text")

    def draw_seconds(self, rng: np.random.Generator) -> float:
        if self.seconds is None:
            return sample_window_s(rng)
        return float(rng.uniform(*self.seconds))

    def draw_gap_db(self, rng: np.random.Generator) -> float:
        if self.gap_db is None:
            return sample_gap_db(rng)
        return float(rng.uniform(*self.gap_db))

    def draw_share(self, rng: np.random.Generator) -> float:
        if self.share is None:
            return sample_share(rng)
        return float(rng.uniform(*self.share))


#: Processes a second voice before it is mixed (a room and a microphone of its own, from
#: augment.py): (int16 audio, rng) -> (int16 audio, what was done).
DonorFx = Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, dict]]


class Mixer:
    """Crosstalk for one training clip, called from `collate`.

    By default only clips measured clean (`overlap_spans == []`) and diarized are candidates, each
    with probability `p`: a clip with real crosstalk keeps it as it is, and one never measured is
    not assumed clean. Returns the (possibly new) int16 audio and what was done, or None.
    `p`, when given, overrides `config.p`."""

    def __init__(
        self,
        pool: DonorPool | ClipDonorPool,
        fetch: Callable[[str, float, float], np.ndarray],
        p: float | None = None,
        config: CrosstalkConfig | None = None,
        donor_fx: DonorFx | None = None,
    ):
        config = config or CrosstalkConfig()
        self.config = config if p is None else replace(config, p=p)
        want = ClipDonorPool if self.config.donor == "clip" else DonorPool
        if not isinstance(pool, want):
            raise TypeError(f"donor={self.config.donor!r} needs a {want.__name__}")
        self.pool, self.fetch, self.donor_fx = pool, fetch, donor_fx

    @property
    def p(self) -> float:
        return self.config.p

    def __call__(
        self, row: dict, clip: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict | None]:
        cfg = self.config
        if cfg.clean_only and (row.get("overlap_spans") != [] or not row.get("speaker_turns")):
            return clip, None
        if cfg.label == "everything" and _units(row) is None:
            return clip, None
        if rng.random() >= cfg.p:
            return clip, None
        clip_s = len(clip) / SR
        share = cfg.draw_share(rng)
        if cfg.donor == "clip":
            planned = self._plan_clips(rng, row, clip_s, share)
        else:
            planned = self._plan_stretches(rng, row, clip_s, share)
        pieces, windows = [], []
        for offset, seconds, donor in planned:
            audio = self.fetch(donor.episode, donor.start, donor.start + seconds)
            window = {
                "offset": offset,
                "seconds": seconds,
                "gap_db": cfg.draw_gap_db(rng),
                "same_episode": donor.same_episode,
            }
            if isinstance(donor, ClipDonor):
                window["donor_segment_id"] = donor.segment_id
            if self.donor_fx is not None:
                audio, window["fx"] = self.donor_fx(audio, rng)
            pieces.append((round(offset * SR), audio, window["gap_db"]))
            windows.append(window)
        if not pieces:
            return clip, None
        info = {
            "segment_id": row["segment_id"],
            "windows": windows,
            "share": sum(w["seconds"] for w in windows) / clip_s,
            "same_episode": sum(w["same_episode"] for w in windows),
        }
        if cfg.label == "everything":
            info["text"] = merge_labels(
                row,
                [(w["offset"], self.pool.by_id[w["donor_segment_id"]]) for w in windows],
            )
        return mix(clip, pieces), info

    def _plan_stretches(self, rng, row, clip_s, share):
        cfg = self.config
        exclude = clip_voices(row)
        out = []
        wins = plan_windows(rng, clip_s, share, cfg.draw_seconds, cfg.max_share, cfg.overshoot)
        for offset, seconds in wins:
            donor = self.pool.pick(
                rng, row["episode_id"], exclude, seconds, other_episode=not exclude
            )
            if donor is not None:
                out.append((offset, seconds, donor))
        return out

    def _plan_clips(self, rng, row, clip_s, share):
        """Whole donor clips placed like `plan_windows`' windows: apart, inside the clip, up to
        `share` of it and never past the cap."""
        cfg = self.config
        cap_share = (
            cfg.max_share if cfg.overshoot is None else min(cfg.max_share, cfg.overshoot * share)
        )
        goal, cap = share * clip_s, cap_share * clip_s
        lo, hi = cfg.seconds or (MIN_WINDOW_S, clip_s)
        out, total = [], 0.0
        for _ in range(64):
            if total >= goal or min(hi, cap - total, clip_s) < lo:
                break
            used = frozenset(d.segment_id for _, _, d in out)
            donor = self.pool.pick(
                rng, row, (lo, min(hi, cap - total, clip_s)), used, cfg.label == "everything"
            )
            if donor is None:
                break
            d = donor.end - donor.start
            s = float(rng.uniform(0.0, clip_s - d))
            if any(s < s2 + d2 + SPACING_S and s2 < s + d + SPACING_S for s2, d2, _ in out):
                continue
            out.append((s, d, donor))
            total += d
        return out
