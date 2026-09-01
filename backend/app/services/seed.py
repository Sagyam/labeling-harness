"""Synthetic development seed: rows straight into Postgres, no audio required.

This exists so schema work, queue building and the API can be exercised without a manifest or
clips. Realistic end-to-end data comes from :mod:`app.services.fixtures`, which writes an actual
manifest directory that the importer consumes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import random
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import AsrHypothesis, AsrSystem, Episode, Segment, SegmentScore
from app.services.corpus import SYSTEMS, perturb, sentence_for
from app.services.splits import assign_split


@dataclass(frozen=True)
class SeedSummary:
    """What a seed run inserted. Zeroes mean the data was already present."""

    episodes: int
    segments: int
    systems: int
    hypotheses: int


def seed_dev_data(
    session: Session,
    *,
    episodes: int = 1,
    segments_per_episode: int = 20,
    systems: int = 3,
    seed: int = 1234,
    settings: Settings | None = None,
) -> SeedSummary:
    """Insert a small synthetic dataset, skipping anything that already exists.

    Args:
        session: Open session; the caller commits.
        episodes: Number of episodes to create.
        segments_per_episode: Segments per episode.
        systems: Number of ASR systems (at most five are defined).
        seed: Random seed; the same seed always produces the same rows.
        settings: Override configuration, mainly for tests.

    Returns:
        Counts of the rows actually inserted.
    """
    settings = settings or get_settings()
    rng = random.Random(seed)
    now = dt.datetime.now(dt.UTC)

    system_rows: list[AsrSystem] = []
    systems_inserted = 0
    for system_id, model_id in SYSTEMS[:systems]:
        existing = session.scalar(sa.select(AsrSystem).where(AsrSystem.system_id == system_id))
        if existing is None:
            existing = AsrSystem(system_id=system_id, model_id=model_id, notes="synthetic seed")
            session.add(existing)
            session.flush()
            systems_inserted += 1
        system_rows.append(existing)

    episodes_inserted = segments_inserted = hypotheses_inserted = 0

    for episode_index in range(episodes):
        external_id = f"seed-show_ep{episode_index:03d}"
        episode = session.scalar(sa.select(Episode).where(Episode.external_id == external_id))
        if episode is None:
            split = assign_split(
                external_id,
                seed=settings.importer.split_seed,
                ratios=settings.importer.split_ratios,
            )
            episode = Episode(
                external_id=external_id,
                show_id="seed-show",
                title=f"Synthetic episode {episode_index}",
                source_uri=f"https://example.invalid/{external_id}",
                published_at=dt.date(2026, 1, 1) + dt.timedelta(days=episode_index),
                source_audio_checksum=f"sha256:{hashlib.sha256(external_id.encode()).hexdigest()}",
                duration_seconds=float(segments_per_episode * 12),
                split=split,
                split_seed=settings.importer.split_seed,
                split_assigned_at=now,
                metadata_jsonb={"synthetic": True},
            )
            session.add(episode)
            session.flush()
            episodes_inserted += 1

        # One lookup for the whole episode rather than one per segment.
        existing_segment_ids = set(
            session.scalars(
                sa.select(Segment.external_id).where(
                    Segment.external_id.like(f"{external_id}\\_%", escape="\\")
                )
            )
        )

        cursor = 0.0
        for segment_index in range(segments_per_episode):
            duration = round(rng.uniform(2.0, 18.0), 2)
            start, cursor = cursor, cursor + duration + round(rng.uniform(0.1, 1.5), 2)
            segment_external_id = f"{external_id}_{segment_index:04d}"
            if segment_external_id in existing_segment_ids:
                continue

            segment = Segment(
                episode_id=episode.id,
                external_id=segment_external_id,
                speaker_id=f"SPEAKER_{rng.randrange(2):02d}",
                start_time=round(start, 2),
                end_time=round(start + duration, 2),
                duration_seconds=duration,
                clip_object_key=f"clips/{external_id}/{segment_external_id}.flac",
                clip_checksum=f"sha256:{hashlib.sha256(segment_external_id.encode()).hexdigest()}",
                peaks_object_key=f"peaks/{external_id}/{segment_external_id}.json",
                p_en=round(rng.random(), 3),
                lid=rng.choice(["ne", "en", "mixed"]),
            )
            session.add(segment)
            session.flush()
            segments_inserted += 1

            reference = sentence_for(rng)
            #: How much the weaker systems disagree on this segment.
            difficulty = rng.random()
            for rank, system in enumerate(system_rows):
                text = perturb(reference, rng, strength=difficulty * min(rank, 2) / 2)
                session.add(
                    AsrHypothesis(
                        segment_id=segment.id,
                        asr_system_id=system.id,
                        text_raw=text,
                        text_normalized=text,
                        avg_logprob=round(-0.1 - difficulty * rng.uniform(0.2, 1.8), 3),
                        no_speech_prob=round(rng.random() ** 4, 4),
                    )
                )
                hypotheses_inserted += 1

            session.add(
                SegmentScore(
                    segment_id=segment.id,
                    cer_between_hypotheses=round(difficulty * rng.uniform(0.0, 0.5), 4),
                    word_disagreement_rate=round(difficulty * rng.uniform(0.0, 0.9), 4),
                    script_conflict_rate=round(difficulty * rng.uniform(0.0, 0.3), 4),
                    code_switch_density=round(rng.random(), 4),
                    flags_jsonb=["low_confidence"] if difficulty > 0.8 else [],
                )
            )
        session.flush()

    return SeedSummary(
        episodes=episodes_inserted,
        segments=segments_inserted,
        systems=systems_inserted,
        hypotheses=hypotheses_inserted,
    )
