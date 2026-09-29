"""What a pot is short of and what it has plenty of, bucket by bucket (D104).

Gold and train/val are different things and are never drawn together: gold is a benchmark sized
in minutes per stratum, train/val is training data sized in hours. Each is rated against its own
floor, from ``dataset.coverage``:

* **missing** -- a value of a closed list with no audio at all.
* **thin** -- under the pot's hour floor; or, for a people or content split, fewer voices than
  the floor, however many hours (four hours of one reviewer is one voice).
* **overdone** -- more than ``dominant_share`` of its category: it crowds the rest out, so the
  category says more about this one value than about the corpus.
* **plenty** -- at least ``plenty_factor`` times the hour floor: more of it adds little.
* **enough** -- everything between.

Buckets that are not strata are never rated as a gap: ``unknown`` (not measured or not
declared), ``defect`` (the diarizer's empty clip), ``off_list`` (a value off a closed
vocabulary) and ``unrated`` (a show, which is where audio comes from, not a stratum).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.inventory.categories import BucketStats, CategoryReport, build_categories
from app.services.inventory.constants import CATEGORY_BY_KEY
from app.services.inventory.facts import ClipRow, EpisodeRow, _round

#: The two views of the corpus. Train holds val (and any clip of an unassigned episode that is
#: not gold): the two are the same standard of data, one line redrawn freely between them.
POTS: tuple[tuple[str, str], ...] = (("train", "Train + val"), ("gold", "Gold (test)"))

STRATUM_STATUSES: tuple[str, ...] = ("missing", "thin", "enough", "plenty", "overdone")


@dataclass(frozen=True)
class Floor:
    """One pot's thresholds.

    Attributes:
        thin_hours: Under this, a bucket is thin.
        thin_voices: Under this many usable voices, a people or content bucket is thin.
        voice_words: Attributed reference words a voice needs inside a bucket to be usable.
        plenty_factor: At this many times ``thin_hours``, a bucket is plenty.
        dominant_share: Above this share of its category, a bucket is overdone.
    """

    thin_hours: float
    thin_voices: int
    voice_words: int
    plenty_factor: float
    dominant_share: float


def _duration(hours: float) -> str:
    if hours * 60 < 1:
        return f"{round(hours * 3600)} s"
    return f"{round(hours * 60)} min" if hours < 1 else f"{hours:.1f} h"


def _threshold(hours: float) -> str:
    return f"{round(hours * 60)} min" if hours < 1 else f"{hours:g} h"


def rate_bucket(entry: BucketStats, report: CategoryReport, floor: Floor) -> tuple[str, str]:
    """The status of one bucket, and the numbers behind it."""
    if not report.rated:
        return "unrated", ""
    if entry.bucket in report.unknown:
        return "unknown", ""
    if entry.bucket in report.defect:
        return "defect", ""
    if entry.off_vocabulary:
        return "off_list", ""
    if entry.clips == 0:
        return "missing", "no audio"
    if entry.hours < floor.thin_hours:
        return "thin", f"{_duration(entry.hours)}, under {_threshold(floor.thin_hours)}"
    if report.unit == "voices" and entry.usable_voices < floor.thin_voices:
        noun = "voice" if entry.usable_voices == 1 else "voices"
        return "thin", f"{entry.usable_voices} {noun}, under {floor.thin_voices}"
    if entry.share > floor.dominant_share:
        return "overdone", f"{entry.share * 100:.0f}% of the category"
    plenty = floor.plenty_factor * floor.thin_hours
    if entry.hours >= plenty:
        return "plenty", f"{_duration(entry.hours)}, over {_threshold(plenty)}"
    return "enough", ""


@dataclass
class PotReport:
    """One pot, cut by every category and rated against its own floor."""

    key: str
    label: str
    floor: Floor
    totals: dict[str, Any] = field(default_factory=dict)
    categories: list[CategoryReport] = field(default_factory=list)

    def category(self, key: str) -> CategoryReport:
        return next(c for c in self.categories if c.key == key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "floor": asdict(self.floor),
            "totals": self.totals,
            "categories": [c.as_dict() for c in self.categories],
        }


def in_pot(clip: ClipRow, pot: str) -> bool:
    """Whether a clip belongs to a view: ``gold``, or ``train`` for everything else."""
    return (clip.pot == "gold") == (pot == "gold")


def build_pot(
    pot: str,
    clips: Sequence[ClipRow],
    episodes: Mapping[str, EpisodeRow],
    floor: Floor,
    *,
    target_hours: float | None = None,
) -> PotReport:
    """Total and rate one pot's clips. ``clips`` must already carry their buckets. Pure."""
    mine = [c for c in clips if in_pot(c, pot)]
    categories = build_categories(mine, episodes, voice_words=floor.voice_words)
    for report in categories.values():
        counts = dict.fromkeys(STRATUM_STATUSES, 0)
        for entry in report.buckets:
            entry.status, entry.reason = rate_bucket(entry, report, floor)
            if entry.status in counts:
                counts[entry.status] += 1
        report.counts = counts
    hours = sum(c.duration for c in mine) / 3600
    verified = sum(c.duration for c in mine if c.tier == "verified") / 3600
    screened = sum(c.duration for c in mine if c.tier == "screened") / 3600
    episode_ids = {c.episode for c in mine}
    totals = {
        "hours": _round(hours),
        "speech_hours": _round(sum(c.speech_seconds or 0.0 for c in mine) / 3600),
        "clips": len(mine),
        "episodes": len(episode_ids),
        "shows": len({s for e in episode_ids if (s := episodes[e].show_id)}),
        "voices": len({c.voice for c in mine if c.voice}),
        "words": sum(c.words or 0 for c in mine),
        "verified_hours": _round(verified),
        "screened_hours": _round(screened),
        "unlabeled_hours": _round(max(0.0, hours - verified - screened)),
        "val_hours": _round(sum(c.duration for c in mine if c.pot == "val") / 3600),
        "target_hours": target_hours,
    }
    label = dict(POTS)[pot]
    return PotReport(
        key=pot, label=label, floor=floor, totals=totals, categories=list(categories.values())
    )


def category_meta() -> list[dict[str, Any]]:
    """What the page needs to know about each category, independent of any pot."""
    return [
        {
            "key": c.key,
            "label": c.label,
            "group": c.group,
            "unit": c.unit,
            "why": c.why,
            "rated": c.rated,
        }
        for c in CATEGORY_BY_KEY.values()
    ]


__all__ = [
    "POTS",
    "STRATUM_STATUSES",
    "Floor",
    "PotReport",
    "build_pot",
    "category_meta",
    "in_pot",
    "rate_bucket",
]
