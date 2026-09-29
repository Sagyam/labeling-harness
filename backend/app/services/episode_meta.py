"""What an episode is, and editing it after ingest: genre, topic and declared speakers (D102).

Genre and topic are two axes, not one. **Genre is how the recording was made** -- who is
talking to whom, scripted or not, in what room -- which is what drives crosstalk, register and
background noise. **Topic is what it is about** (D57), which drives vocabulary and English
share. A genre that names a subject (``finance``, ``cooking``) only repeats the topic, so the
genre list below holds formats alone, each with a test someone can apply without guessing.

Both are closed vocabularies for the reason every stratification variable here is: a value
spelled three ways is three strata. An edit off either list is refused, never stored.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.llm.topic import TOPIC_LABELS
from app.models import AuditLog, Episode
from app.services.speaker_meta import ALLOWED_VALUES, MAX_SPEAKERS


@dataclass(frozen=True)
class Genre:
    """One recording format, and the test that puts an episode in it."""

    value: str
    label: str
    description: str


#: The closed list. Ordered for the picker: conversation first, then one voice, then the rest.
#: Adding a genre is cheap; renaming one is not, since every episode keeps the old string.
GENRES: tuple[Genre, ...] = (
    Genre("podcast", "Podcast", "Unscripted conversation, seated, two or more voices."),
    Genre("interview", "Interview", "A host asks prepared questions; a guest answers."),
    Genre("talk_show", "Talk show", "Studio or TV production with several guests."),
    Genre("street_interview", "Street interview", "Outdoor vox pop with strangers."),
    Genre("news", "News", "Broadcast report or official statement; mostly read speech."),
    Genre("fm_radio", "FM radio", "A radio programme on air: host, phone-in callers, music beds."),
    Genre("commentary", "Commentary", "One host, unscripted, on a live subject: markets, a match."),
    Genre("explainer", "Explainer", "One person, scripted, teaching a how-to to camera or screen."),
    Genre("lecture", "Lecture", "A structured class; dense technical vocabulary."),
    Genre("review", "Review", "A product under test: electronics, cars, bikes."),
    Genre("vlog", "Vlog", "Personal, to camera, often outdoors: food, travel, daily life."),
    Genre("advert", "Advert", "Scripted sales copy, often a voiceover over music."),
    Genre("sketch", "Sketch", "A short scripted skit, acted by several voices."),
    Genre("standup", "Stand-up", "One performer to a live audience; comedy or a story."),
    Genre("streaming", "Streaming", "Casual talk over game audio."),
    Genre("speech", "Speech", "A rally or address: PA echo, crowd noise, cheering."),
    Genre("audiobook", "Audiobook", "Read narration, studio-recorded."),
)

GENRE_LABELS: tuple[str, ...] = tuple(g.value for g in GENRES)

#: :data:`ALLOWED_VALUES`' age brackets, youngest first, for pickers; a set has no order.
AGE_BRACKETS: tuple[str, ...] = ("under_20", "20_39", "40_59", "60_79", "80_plus")

#: Speaker fields a row carries, in the order the editor shows them.
ROW_FIELDS: tuple[str, ...] = ("role", "gender", "age_bracket")

#: Keys an edit may change; everything else in the metadata is carried through untouched.
EDITED_KEYS: tuple[str, ...] = (
    "genre",
    "topic",
    "topic_source",
    "topic_model",
    "speaker_count",
    "speakers",
)

_SPK_KEY = re.compile(r"^spk(\d+)$")


class MetadataEditError(ValueError):
    """An edit named a value the corpus does not record."""


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def check_genre(genre: str | None) -> str | None:
    """Return ``genre`` if it is on the list, ``None`` if blank; raise otherwise."""
    if _blank(genre):
        return None
    if genre not in GENRE_LABELS:
        raise MetadataEditError(f"genre {genre!r} is not one of {', '.join(GENRE_LABELS)}")
    return genre


def check_topic(topic: str | None) -> str | None:
    """Return ``topic`` if it is on the list, ``None`` if blank; raise otherwise."""
    if _blank(topic):
        return None
    if topic not in TOPIC_LABELS:
        raise MetadataEditError(f"topic {topic!r} is not one of {', '.join(TOPIC_LABELS)}")
    return topic


def speaker_rows(metadata: Mapping[str, Any] | None) -> list[dict[str, str | None]]:
    """The declared speakers as the editor's rows, one per person the diarizer was told about.

    ``speakers`` only holds rows someone filled in; ``speaker_count`` also counts the blank ones
    (D79). A ``spkN`` key lands at row ``N`` and blank rows fill the gaps; any other key follows
    in sorted order.
    """
    metadata = metadata or {}
    speakers = metadata.get("speakers")
    speakers = speakers if isinstance(speakers, Mapping) else {}
    placed: dict[int, Mapping[str, Any]] = {}
    others: list[Mapping[str, Any]] = []
    for key in sorted(speakers, key=str):
        fields = speakers[key]
        if not isinstance(fields, Mapping):
            continue
        match = _SPK_KEY.match(str(key))
        if match and int(match.group(1)) not in placed:
            placed[int(match.group(1))] = fields
        else:
            others.append(fields)

    count = metadata.get("speaker_count")
    count = count if isinstance(count, int) and not isinstance(count, bool) else 0
    size = max(count, max(placed, default=-1) + 1)
    rows = [placed.get(i, {}) for i in range(size)] + others
    return [{f: row.get(f) or None for f in ROW_FIELDS} for row in rows]


def _check_row(row: Mapping[str, Any]) -> dict[str, str]:
    unknown = set(row) - set(ROW_FIELDS)
    if unknown:
        raise MetadataEditError(f"speakers: unknown field {sorted(unknown)[0]!r}")
    kept: dict[str, str] = {}
    for field in ROW_FIELDS:
        value = row.get(field)
        if _blank(value):
            continue
        if not isinstance(value, str):
            raise MetadataEditError(f"speakers: {field} must be a string")
        vocabulary = ALLOWED_VALUES.get(field)
        if vocabulary is not None and value not in vocabulary:
            raise MetadataEditError(
                f"speakers: {field} {value!r} is not one of {sorted(vocabulary)}"
            )
        kept[field] = value.strip()
    return kept


def edited_metadata(
    current: Mapping[str, Any] | None,
    *,
    genre: str | None,
    topic: str | None,
    speakers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply one edit to an episode's metadata, returning a new dict.

    Speakers are stored in the ingest form's shape so every reader sees one format: a row is kept
    under ``spk<i>`` only when it declares a gender or an age bracket, and ``speaker_count`` is
    the number of rows. A topic set by hand is ``topic_source: manual``; an unchanged topic keeps
    whatever provenance it had.

    Raises:
        MetadataEditError: A genre, topic or speaker value off its closed list, an unknown
            speaker field, or more rows than :data:`MAX_SPEAKERS`.
    """
    current = dict(current or {})
    genre = check_genre(genre)
    topic = check_topic(topic)
    if len(speakers) > MAX_SPEAKERS:
        raise MetadataEditError(f"speakers: at most {MAX_SPEAKERS} rows")
    rows = [_check_row(row) for row in speakers]

    out = {k: v for k, v in current.items() if k not in ("genre", "speaker_count", "speakers")}
    if genre is not None:
        out["genre"] = genre

    if topic is None:
        for key in ("topic", "topic_source", "topic_model"):
            out.pop(key, None)
    elif topic != current.get("topic"):
        out.pop("topic_model", None)
        out |= {"topic": topic, "topic_source": "manual"}

    declared = {
        f"spk{i}": row for i, row in enumerate(rows) if "gender" in row or "age_bracket" in row
    }
    if rows:
        out["speaker_count"] = len(rows)
    if declared:
        out["speakers"] = declared
    return out


def update_episode_metadata(
    session: Session,
    episode: Episode,
    *,
    genre: str | None,
    topic: str | None,
    speakers: Sequence[Mapping[str, Any]],
    actor: str,
) -> dict[str, Any]:
    """Edit an episode's genre, topic and speakers, with one ``audit_logs`` row when it changed.

    The audit row holds the before and after of each changed key only. An edit that changes
    nothing writes nothing. The caller commits.
    """
    before = dict(episode.metadata_jsonb or {})
    after = edited_metadata(before, genre=genre, topic=topic, speakers=speakers)
    changed = [k for k in EDITED_KEYS if before.get(k) != after.get(k)]
    if changed:
        episode.metadata_jsonb = after
        session.add(
            AuditLog(
                entity_type="episodes",
                entity_id=str(episode.id),
                action="update_metadata",
                actor=actor,
                old_values_jsonb={k: before.get(k) for k in changed},
                new_values_jsonb={k: after.get(k) for k in changed},
            )
        )
        session.flush()
    return after
