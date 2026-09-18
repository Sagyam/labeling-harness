"""Voice profiles: one anonymous person followed across clips and episodes (D91).

A voice is an id and its measurements, never a name (D56). Its talk time comes from the
diarizer's turns cut to the corpus's clips, so it is time *in the corpus*, not time in the
source videos; its words come from the clips it dominates (:func:`attributed_words`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.inventory.categories import attributed_words
from app.services.inventory.constants import (
    MIN_VOICE_WORDS,
    RECURRING_HOST_EPISODES,
)
from app.services.inventory.facts import ClipRow, EpisodeRow, _round
from app.services.inventory.resolve import Resolution, VoiceIdentity


@dataclass
class VoiceEpisode:
    """One voice's appearance in one episode."""

    external_id: str
    title: str | None
    show_id: str | None
    genre: str | None
    published_at: str | None
    split: str
    minutes: float
    #: This voice's share of all linked talk in the episode's clips.
    share: float
    clips: int
    role: str | None
    resolved_by: str | None
    #: The other voices heard in the episode, most talk first.
    with_voices: list[str] = field(default_factory=list)


@dataclass
class VoiceProfile:
    voice: str
    talk_minutes: float
    clips: int
    words: int
    #: Whether the voice carries enough attributed words to count in a comparison.
    usable: bool
    episodes: list[VoiceEpisode]
    episode_count: int
    shows: list[str]
    genres: list[str]
    gender: str | None
    age_bracket: str | None
    #: ``resolved``, ``conflict`` or ``unresolved`` (see :class:`VoiceIdentity`).
    identity: str
    resolved_by: dict[str, int]
    #: Roles this voice has held, with episode counts.
    roles: dict[str, int]
    #: Hours of the clips this voice dominates, by pot.
    hours: dict[str, float]
    verified_minutes: float
    screened_minutes: float
    mean_cmi: float | None
    words_per_second: float | None
    exposure: str
    first_seen: str | None
    last_seen: str | None
    #: Other voices this one shares episodes with, by shared-episode count.
    co_voices: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_voices(
    clips: Sequence[ClipRow],
    episodes: Mapping[str, EpisodeRow],
    per_episode: Mapping[tuple[str, str], Resolution],
    per_voice: Mapping[str, VoiceIdentity],
) -> list[VoiceProfile]:
    """One profile per linked voice, most talk first."""
    talk: dict[str, float] = defaultdict(float)
    talk_by_episode: dict[tuple[str, str], float] = defaultdict(float)
    episode_talk: dict[str, float] = defaultdict(float)
    clips_by_episode: Counter[tuple[str, str]] = Counter()
    led: dict[str, list[ClipRow]] = defaultdict(list)
    for clip in clips:
        for voice, seconds in clip.voice_talk.items():
            talk[voice] += seconds
            talk_by_episode[(clip.episode, voice)] += seconds
            episode_talk[clip.episode] += seconds
        if clip.voice is not None:
            led[clip.voice].append(clip)
            clips_by_episode[(clip.episode, clip.voice)] += 1

    appearances: dict[str, list[str]] = defaultdict(list)
    for (episode, voice), seconds in talk_by_episode.items():
        if seconds > 0:
            appearances[voice].append(episode)

    profiles: list[VoiceProfile] = []
    for voice in sorted(talk, key=lambda v: (-talk[v], v)):
        mine = led.get(voice, [])
        words = sum(attributed_words(c) for c in mine)
        hours = {"gold": 0.0, "train": 0.0, "val": 0.0}
        verified = screened = 0.0
        cmi_weight = cmi_sum = 0.0
        speech = 0.0
        timed_words = 0
        for clip in mine:
            hours[clip.pot] = hours.get(clip.pot, 0.0) + clip.duration / 3600
            if clip.tier == "verified":
                verified += clip.duration / 60
            elif clip.tier == "screened":
                screened += clip.duration / 60
            if clip.cmi is not None:
                cmi_sum += clip.cmi * clip.duration
                cmi_weight += clip.duration
            if clip.words:
                timed_words += clip.words
                speech += clip.speech_seconds if clip.speech_seconds else clip.duration
        roles: Counter[str] = Counter()
        rows: list[VoiceEpisode] = []
        co: Counter[str] = Counter()
        for external_id in appearances[voice]:
            episode = episodes[external_id]
            resolution = per_episode.get((external_id, voice))
            if resolution is not None and resolution.role:
                roles[resolution.role] += 1
            others = [v for v in episode.voices if v != voice]
            co.update(others)
            rows.append(
                VoiceEpisode(
                    external_id=external_id,
                    title=episode.title,
                    show_id=episode.show_id,
                    genre=episode.genre,
                    published_at=episode.published_at,
                    split=episode.split,
                    minutes=_round(talk_by_episode[(external_id, voice)] / 60, 2),
                    share=_round(
                        talk_by_episode[(external_id, voice)] / episode_talk[external_id], 4
                    )
                    if episode_talk[external_id]
                    else 0.0,
                    clips=clips_by_episode[(external_id, voice)],
                    role=resolution.role if resolution else None,
                    resolved_by=resolution.by if resolution else None,
                    with_voices=others,
                )
            )
        rows.sort(key=lambda r: (r.published_at or "9999", r.external_id))
        dated = [r.published_at for r in rows if r.published_at]
        identity = per_voice.get(voice)
        profiles.append(
            VoiceProfile(
                voice=voice,
                talk_minutes=_round(talk[voice] / 60, 2),
                clips=len(mine),
                words=words,
                usable=words >= MIN_VOICE_WORDS,
                episodes=rows,
                episode_count=len(rows),
                shows=sorted({r.show_id for r in rows if r.show_id}),
                genres=sorted({r.genre for r in rows if r.genre}),
                gender=identity.gender if identity else None,
                age_bracket=identity.age_bracket if identity else None,
                identity=identity.status if identity else "unresolved",
                resolved_by=dict(identity.by) if identity else {},
                roles=dict(roles),
                hours={k: _round(v) for k, v in hours.items()},
                verified_minutes=_round(verified, 2),
                screened_minutes=_round(screened, 2),
                mean_cmi=_round(cmi_sum / cmi_weight, 2) if cmi_weight else None,
                words_per_second=_round(timed_words / speech, 2) if speech else None,
                exposure=mine[0].classes["voice_exposure"] if mine else "unlinked",
                first_seen=min(dated) if dated else None,
                last_seen=max(dated) if dated else None,
                co_voices=dict(co.most_common()),
            )
        )
    return profiles


def host_guest_episodes(profile: VoiceProfile) -> int:
    """Episodes this voice hosted with at least one other voice in the room."""
    return sum(1 for e in profile.episodes if e.role == "host" and e.with_voices)


def summarize_voices(profiles: Sequence[VoiceProfile]) -> dict[str, Any]:
    """The headline facts about the corpus's people, for the ledger and the recommender."""
    hosts = [p for p in profiles if host_guest_episodes(p) >= RECURRING_HOST_EPISODES]
    guests_on_two_shows = [p for p in profiles if p.roles.get("guest", 0) and len(p.shows) > 1]
    return {
        "voices": len(profiles),
        "usable": sum(1 for p in profiles if p.usable),
        "recurring": sum(1 for p in profiles if p.episode_count > 1),
        "single_episode": sum(1 for p in profiles if p.episode_count == 1),
        "recurring_hosts": len(hosts),
        "recurring_host_voices": [p.voice for p in hosts],
        "guests_on_two_shows": len(guests_on_two_shows),
        "gender_resolved": sum(1 for p in profiles if p.gender),
        "age_resolved": sum(1 for p in profiles if p.age_bracket),
        "conflicts": sum(1 for p in profiles if p.identity == "conflict"),
        "talk_hours": _round(sum(p.talk_minutes for p in profiles) / 60),
        "top_voice": profiles[0].voice if profiles else None,
        "top_voice_share": _round(
            profiles[0].talk_minutes / sum(p.talk_minutes for p in profiles), 4
        )
        if profiles and sum(p.talk_minutes for p in profiles)
        else 0.0,
    }


__all__ = [
    "VoiceEpisode",
    "VoiceProfile",
    "build_voices",
    "host_guest_episodes",
    "summarize_voices",
]
