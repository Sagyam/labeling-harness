"""Bucket every clip on every category, and total each category up (D91)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from app.llm.topic import TOPIC_LABELS
from app.services.inventory.constants import (
    ATTRIBUTION_SHARE,
    CATEGORIES,
    MIN_VOICE_WORDS,
    SPEAKING_RATE_BUCKETS,
    SPEAKING_RATE_EDGES,
    Category,
)
from app.services.inventory.facts import ClipRow, EpisodeRow, _round
from app.services.inventory.resolve import Resolution, VoiceIdentity


def speaking_rate_bucket(words_per_second: float | None) -> str:
    """The speaking-speed bucket for a rate in words per second of speech."""
    if words_per_second is None:
        return SPEAKING_RATE_BUCKETS[-1]
    for edge, bucket in zip(SPEAKING_RATE_EDGES, SPEAKING_RATE_BUCKETS, strict=False):
        if words_per_second < edge:
            return bucket
    return SPEAKING_RATE_BUCKETS[len(SPEAKING_RATE_EDGES)]


def attributed_words(clip: ClipRow) -> int:
    """The clip's reference words if its dominant voice holds enough of it to be credited."""
    if not clip.words or clip.voice is None:
        return 0
    total = sum(clip.voice_talk.values())
    if total <= 0:
        return 0
    return clip.words if clip.voice_talk.get(clip.voice, 0.0) / total >= ATTRIBUTION_SHARE else 0


def _person_bucket(
    clip: ClipRow,
    episode: EpisodeRow,
    key: str,
    identity: VoiceIdentity | None,
) -> str:
    """Gender or age for one clip: the voice's resolved value, else the whole episode's."""
    if identity is not None and identity.get(key) is not None:
        return identity.get(key)  # type: ignore[return-value]
    declared = clip.classes.get(key)
    if declared not in (None, "mixed", "undeclared"):
        return declared  # type: ignore[return-value]
    return "unresolved" if episode.declared else "undeclared"


def _role_bucket(clip: ClipRow, episode: EpisodeRow, resolution: Resolution | None) -> str:
    if resolution is not None and resolution.role:
        return resolution.role
    roles = {row.get("role") for row in episode.declared}
    if len(roles) == 1 and all(roles):
        return roles.pop()  # type: ignore[return-value]
    return "unresolved" if episode.declared else "undeclared"


def assign_buckets(
    clips: Sequence[ClipRow],
    episodes: Mapping[str, EpisodeRow],
    per_episode: Mapping[tuple[str, str], Resolution],
    per_voice: Mapping[str, VoiceIdentity],
) -> list[ClipRow]:
    """Give every clip its bucket on every category. Pure."""
    out: list[ClipRow] = []
    for clip in clips:
        episode = episodes[clip.episode]
        identity = per_voice.get(clip.voice) if clip.voice else None
        resolution = per_episode.get((clip.episode, clip.voice)) if clip.voice else None
        buckets = {
            "gender": _person_bucket(clip, episode, "gender", identity),
            "age_bracket": _person_bucket(clip, episode, "age_bracket", identity),
            "role": _role_bucket(clip, episode, resolution),
            "voice_exposure": clip.classes["voice_exposure"],
            "voice": clip.voice or "unlinked",
            "topic": episode.topic or "untagged",
            "genre": episode.genre or "untagged",
            "show": episode.show_id or "no show id",
            "cmi": clip.classes["cmi"],
            "speaking_rate": speaking_rate_bucket(clip.words_per_second),
            "duration": clip.classes["duration"],
            "crosstalk": clip.classes["overlap"],
            "speakers": clip.classes["speakers"],
            "noise": clip.classes["snr"],
            "reverb": clip.classes["reverb"],
            "bandwidth": clip.classes["bandwidth"],
        }
        out.append(replace(clip, buckets=buckets))
    return out


@dataclass
class BucketStats:
    """One bucket of one category, with the audio and the people in it."""

    bucket: str
    hours: float = 0.0
    clips: int = 0
    episodes: int = 0
    shows: int = 0
    #: Distinct linked voices with any clip here.
    voices: int = 0
    #: Voices with :data:`MIN_VOICE_WORDS` reference words attributed inside this bucket -- the
    #: n a comparison between people can use.
    usable_voices: int = 0
    verified_hours: float = 0.0
    screened_hours: float = 0.0
    gold_hours: float = 0.0
    val_hours: float = 0.0
    train_hours: float = 0.0
    #: Share of the category's measured hours (unknown buckets excluded), so shares sum to 1.
    share: float = 0.0
    #: True for a value the closed vocabulary does not contain -- dirt, not a stratum (D57).
    off_vocabulary: bool = False


@dataclass
class CategoryReport:
    """Every bucket of one category, in display order, plus what is missing and unmeasured."""

    key: str
    label: str
    group: str
    goals: tuple[str, ...]
    unit: str
    why: str
    defect: list[str] = field(default_factory=list)
    buckets: list[BucketStats] = field(default_factory=list)
    #: Closed-vocabulary buckets with no audio at all.
    absent: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    measured_hours: float = 0.0
    unknown_hours: float = 0.0
    top_bucket: str | None = None
    top_share: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["goals"] = list(self.goals)
        return data

    def stats(self, bucket: str) -> BucketStats | None:
        return next((b for b in self.buckets if b.bucket == bucket), None)


def build_category(
    category: Category, clips: Sequence[ClipRow], episodes: Mapping[str, EpisodeRow]
) -> CategoryReport:
    """Total one category over the clips."""
    stats: dict[str, BucketStats] = {}
    episode_sets: dict[str, set[str]] = defaultdict(set)
    show_sets: dict[str, set[str]] = defaultdict(set)
    voice_sets: dict[str, set[str]] = defaultdict(set)
    voice_words: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for clip in clips:
        bucket = clip.buckets[category.key]
        entry = stats.setdefault(bucket, BucketStats(bucket=bucket))
        hours = clip.duration / 3600
        entry.hours += hours
        entry.clips += 1
        if clip.tier == "verified":
            entry.verified_hours += hours
        elif clip.tier == "screened":
            entry.screened_hours += hours
        if clip.pot == "gold":
            entry.gold_hours += hours
        elif clip.pot == "val":
            entry.val_hours += hours
        elif clip.pot == "train":
            entry.train_hours += hours
        episode_sets[bucket].add(clip.episode)
        if (show := episodes[clip.episode].show_id) is not None:
            show_sets[bucket].add(show)
        if clip.voice is not None:
            voice_sets[bucket].add(clip.voice)
            voice_words[bucket][clip.voice] += attributed_words(clip)

    unknown = set(category.unknown)
    raw_hours = {bucket: entry.hours for bucket, entry in stats.items()}
    measured_hours = sum(h for b, h in raw_hours.items() if b not in unknown)
    top = max(
        ((b, h) for b, h in raw_hours.items() if b not in unknown and h > 0),
        key=lambda item: item[1],
        default=None,
    )
    for bucket, entry in stats.items():
        entry.episodes = len(episode_sets[bucket])
        entry.shows = len(show_sets[bucket])
        entry.voices = len(voice_sets[bucket])
        entry.usable_voices = sum(
            1 for words in voice_words[bucket].values() if words >= MIN_VOICE_WORDS
        )
        entry.share = (
            _round(entry.hours / measured_hours, 4)
            if measured_hours and bucket not in unknown
            else 0.0
        )
        entry.off_vocabulary = category.key == "topic" and bucket not in (
            *TOPIC_LABELS,
            "untagged",
        )
        for name in ("hours", "verified_hours", "screened_hours", "gold_hours", "val_hours",
                     "train_hours"):  # fmt: skip
            setattr(entry, name, _round(getattr(entry, name)))

    if category.buckets:
        order = list(category.buckets)
        extra = sorted(b for b in stats if b not in order)
        # An off-vocabulary topic sits after the taxonomy and before "untagged".
        known = [b for b in order if b not in unknown]
        order = known + extra + [b for b in order if b in unknown]
    else:
        order = sorted(
            (b for b in stats if b not in unknown), key=lambda b: (-stats[b].hours, b)
        ) + sorted(b for b in stats if b in unknown)

    buckets = [stats.get(b) or BucketStats(bucket=b) for b in order]
    return CategoryReport(
        key=category.key,
        label=category.label,
        group=category.group,
        goals=category.goals,
        unit=category.unit,
        why=category.why,
        defect=list(category.defect),
        buckets=buckets,
        absent=[b for b in category.buckets if b not in unknown and b not in stats],
        unknown=[b for b in order if b in unknown],
        measured_hours=_round(measured_hours),
        unknown_hours=_round(sum(h for b, h in raw_hours.items() if b in unknown)),
        top_bucket=top[0] if top else None,
        top_share=_round(top[1] / measured_hours, 4) if top and measured_hours else 0.0,
    )


def build_categories(
    clips: Sequence[ClipRow], episodes: Mapping[str, EpisodeRow]
) -> dict[str, CategoryReport]:
    """Every category, keyed and in :data:`CATEGORIES` order."""
    return {c.key: build_category(c, clips, episodes) for c in CATEGORIES}


def clip_table(
    clips: Sequence[ClipRow],
    categories: Mapping[str, CategoryReport],
    episode_index: Mapping[str, int],
    voice_index: Mapping[str, int],
) -> dict[str, Any]:
    """The clips as a compact columnar table, so the page can cut them any way it likes.

    Each row is ``[segment id, episode index, seconds, words, tier index, pot index, voice index,
    bucket index per category]``; bucket indexes point into the category's bucket list in the
    order the report lists them. A few hundred kilobytes for the whole corpus, and it is what
    makes a click on one bucket re-cut every other category without a round trip.
    """
    keys = list(categories)
    positions = {k: {b.bucket: i for i, b in enumerate(categories[k].buckets)} for k in keys}
    tiers = ["none", "screened", "verified"]
    pots = ["gold", "train", "val", "unassigned"]
    rows = [
        [
            clip.segment_id,
            episode_index[clip.episode],
            round(clip.duration, 2),
            clip.words if clip.words is not None else -1,
            tiers.index(clip.tier or "none"),
            pots.index(clip.pot) if clip.pot in pots else pots.index("unassigned"),
            voice_index.get(clip.voice, -1) if clip.voice else -1,
            *[positions[k][clip.buckets[k]] for k in keys],
        ]
        for clip in clips
    ]
    return {
        "columns": ["segment_id", "episode", "seconds", "words", "tier", "pot", "voice", *keys],
        "tiers": tiers,
        "pots": pots,
        "buckets": {k: [b.bucket for b in categories[k].buckets] for k in keys},
        "rows": rows,
    }


__all__ = [
    "BucketStats",
    "CategoryReport",
    "assign_buckets",
    "attributed_words",
    "build_categories",
    "build_category",
    "clip_table",
    "speaking_rate_bucket",
]
