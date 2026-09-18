"""Which declared speaker row is which diarized voice, where the episode forces the answer (D91).

The harness never links a row to a voice by listening (D78). But an episode's own structure
often leaves no choice, and the sociolinguistics notebook has been resolving those cases by hand
since 2026-09-18. The same three rules live here so the page and the notebook agree:

1. **only pair** -- one declared row and one voice: the row is the voice.
2. **recurring host** -- one row declares a host, and exactly one of the episode's voices also
   appears in other episodes: that voice is the host. A voice that recurs across a series is
   the series' host; nothing about the person is inferred.
3. **rows agree** -- every remaining row shares a value for a field: every remaining voice
   takes it. Applied per field, so two guests who share an age but not a gender resolve on age
   alone. Refused when more voices than rows remain -- the extra voices are people nobody
   declared, and giving them a declared value would be a guess.

Gender and age belong to the person, so a voice resolved in one episode carries the value to
every episode it appears in; two episodes that disagree make the voice a ``conflict``, which is
reported rather than averaged. Role belongs to the recording and stays per episode.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from app.services.inventory.facts import EpisodeRow

PERSON_FIELDS: tuple[str, ...] = ("gender", "age_bracket")
FIELDS: tuple[str, ...] = (*PERSON_FIELDS, "role")


@dataclass(frozen=True)
class Resolution:
    """One voice's declared values in one episode, and which rule produced them."""

    gender: str | None = None
    age_bracket: str | None = None
    role: str | None = None
    by: str = "unresolved"

    def get(self, key: str) -> str | None:
        return getattr(self, key)


def _agreed(rows: Sequence[Mapping[str, str]], key: str) -> str | None:
    values = {row.get(key) for row in rows}
    values.discard(None)
    values.discard("")
    return values.pop() if len(values) == 1 and all(row.get(key) for row in rows) else None


def resolve_episode(
    rows: Sequence[Mapping[str, str]],
    voices: Sequence[str],
    recurring: Iterable[str],
) -> dict[str, Resolution]:
    """Resolve one episode's voices against its declared rows, by the three rules above.

    Args:
        rows: The declared speaker rows.
        voices: The voices heard in the episode's clips.
        recurring: Voices that appear in more than one episode of the corpus.

    Returns:
        Voice to :class:`Resolution`, for the voices a rule reached. Others are absent.
    """
    rows = [dict(r) for r in rows]
    voices = list(voices)
    out: dict[str, Resolution] = {}
    if not rows or not voices:
        return out

    if len(rows) == 1 and len(voices) == 1:
        row = rows[0]
        out[voices[0]] = Resolution(
            row.get("gender"), row.get("age_bracket"), row.get("role"), "only_pair"
        )
        return out

    recurring = set(recurring)
    hosts = [r for r in rows if r.get("role") == "host"]
    recurring_here = [v for v in voices if v in recurring]
    if len(hosts) == 1 and len(recurring_here) == 1:
        host, voice = hosts[0], recurring_here[0]
        out[voice] = Resolution(
            host.get("gender"), host.get("age_bracket"), "host", "recurring_host"
        )
        rows = [r for r in rows if r is not host]
        voices = [v for v in voices if v != voice]

    if rows and voices and len(voices) <= len(rows):
        agreed = {key: _agreed(rows, key) for key in FIELDS}
        if any(agreed.values()):
            for voice in voices:
                out[voice] = Resolution(
                    agreed["gender"], agreed["age_bracket"], agreed["role"], "rows_agree"
                )
    return out


@dataclass(frozen=True)
class VoiceIdentity:
    """A voice's person-level values, pooled over every episode it was resolved in."""

    gender: str | None
    age_bracket: str | None
    #: ``resolved`` when every episode agrees, ``conflict`` when two disagree, ``unresolved``
    #: when no episode's rules reached this voice.
    status: str
    #: The rules that reached it, by episode count.
    by: Mapping[str, int]

    def get(self, key: str) -> str | None:
        return getattr(self, key)


def recurring_voices(episodes: Iterable[EpisodeRow]) -> set[str]:
    """Voices heard in more than one episode."""
    seen: Counter[str] = Counter()
    for episode in episodes:
        seen.update(set(episode.voices))
    return {voice for voice, count in seen.items() if count > 1}


def resolve_corpus(
    episodes: Sequence[EpisodeRow],
) -> tuple[dict[tuple[str, str], Resolution], dict[str, VoiceIdentity]]:
    """Resolve every episode, then pool each voice's person-level values across episodes.

    Returns:
        ``(per_episode, per_voice)``: ``(episode external id, voice) -> Resolution`` for the
        voices a rule reached, and ``voice -> VoiceIdentity`` for every voice heard anywhere.
    """
    recurring = recurring_voices(episodes)
    per_episode: dict[tuple[str, str], Resolution] = {}
    values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    rules: dict[str, Counter[str]] = defaultdict(Counter)
    heard: set[str] = set()
    for episode in episodes:
        heard.update(episode.voices)
        for voice, resolution in resolve_episode(
            episode.declared, episode.voices, recurring
        ).items():
            per_episode[(episode.external_id, voice)] = resolution
            rules[voice][resolution.by] += 1
            for key in PERSON_FIELDS:
                if (value := resolution.get(key)) is not None:
                    values[voice][key].add(value)

    per_voice: dict[str, VoiceIdentity] = {}
    for voice in sorted(heard):
        fields = values.get(voice, {})
        conflict = any(len(v) > 1 for v in fields.values())
        resolved = {k: next(iter(v)) for k, v in fields.items() if len(v) == 1}
        per_voice[voice] = VoiceIdentity(
            gender=None if conflict else resolved.get("gender"),
            age_bracket=None if conflict else resolved.get("age_bracket"),
            status="conflict" if conflict else "resolved" if resolved else "unresolved",
            by=dict(rules.get(voice, {})),
        )
    return per_episode, per_voice


__all__ = [
    "FIELDS",
    "PERSON_FIELDS",
    "Resolution",
    "VoiceIdentity",
    "recurring_voices",
    "resolve_corpus",
    "resolve_episode",
]
