"""Imported content: episodes, segments, ASR systems, hypotheses, words and scores.

Everything in this module is *provenance*: it arrives through the manifest and is never edited by
the harness. Hypotheses in particular are immutable -- a human correction becomes a new
``segment_labels`` row, it never overwrites what a model produced.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utc_now_column, utc_optional_column
from app.models.enums import PIPELINE_STATUSES, SPLITS, check_in

if TYPE_CHECKING:
    from app.models.provenance import ImportRun

JsonB = JSONB().with_variant(JSON(), "sqlite")


class Episode(Base):
    """One podcast episode. Carries the frozen train/val/test split."""

    __tablename__ = "episodes"
    __table_args__ = (
        CheckConstraint(check_in("split", SPLITS), name="split_allowed"),
        Index("ix_episodes_split", "split"),
        Index("ix_episodes_show_id", "show_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    show_id: Mapped[str | None] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(Text)
    source_uri: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[dt.date | None] = mapped_column(Date)
    source_audio_checksum: Mapped[str | None] = mapped_column(String(128))
    duration_seconds: Mapped[float | None] = mapped_column(Float)

    #: Assigned once at import from hash(external_id, split_seed); never recomputed.
    split: Mapped[str] = mapped_column(String(16), nullable=False, default="unassigned")
    split_seed: Mapped[int | None] = mapped_column(Integer)
    split_assigned_at: Mapped[dt.datetime | None] = utc_optional_column()

    metadata_jsonb: Mapped[dict[str, Any] | None] = mapped_column(JsonB)
    created_at: Mapped[dt.datetime] = utc_now_column()
    updated_at: Mapped[dt.datetime] = utc_now_column(onupdate=lambda: dt.datetime.now(dt.UTC))

    segments: Mapped[list[Segment]] = relationship(
        back_populates="episode", cascade="all, delete-orphan", order_by="Segment.start_time"
    )


class Segment(Base):
    """One diarized, segmented span of audio with a stored clip."""

    __tablename__ = "segments"
    __table_args__ = (
        CheckConstraint(check_in("pipeline_status", PIPELINE_STATUSES), name="status_allowed"),
        CheckConstraint("end_time > start_time", name="end_after_start"),
        Index("ix_segments_episode_id", "episode_id"),
        Index("ix_segments_pipeline_status", "pipeline_status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    episode_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False
    )
    #: ``segment_id`` from the manifest; the idempotency key for import.
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    speaker_id: Mapped[str | None] = mapped_column(String(64))

    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    clip_object_key: Mapped[str] = mapped_column(Text, nullable=False)
    clip_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    peaks_object_key: Mapped[str | None] = mapped_column(Text)

    p_en: Mapped[float | None] = mapped_column(Float)
    lid: Mapped[str | None] = mapped_column(String(16))

    pipeline_status: Mapped[str] = mapped_column(String(16), nullable=False, default="imported")
    import_run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("import_runs.id", ondelete="SET NULL")
    )
    created_at: Mapped[dt.datetime] = utc_now_column()
    updated_at: Mapped[dt.datetime] = utc_now_column(onupdate=lambda: dt.datetime.now(dt.UTC))

    episode: Mapped[Episode] = relationship(back_populates="segments")
    import_run: Mapped[ImportRun | None] = relationship(back_populates="segments")
    hypotheses: Mapped[list[AsrHypothesis]] = relationship(
        back_populates="segment", cascade="all, delete-orphan"
    )
    scores: Mapped[SegmentScore | None] = relationship(
        back_populates="segment", cascade="all, delete-orphan", uselist=False
    )


class AsrSystem(Base):
    """One upstream ASR system, identified by the manifest's ``system_id``."""

    __tablename__ = "asr_systems"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    system_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(255))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = utc_now_column()


class AsrHypothesis(Base):
    """An immutable imported transcript for one (segment, system) pair."""

    __tablename__ = "asr_hypotheses"
    __table_args__ = (
        UniqueConstraint("segment_id", "asr_system_id", name="uq_asr_hypotheses_segment_system"),
        Index("ix_asr_hypotheses_segment_id", "segment_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    segment_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("segments.id", ondelete="CASCADE"), nullable=False
    )
    asr_system_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("asr_systems.id", ondelete="RESTRICT"), nullable=False
    )
    text_raw: Mapped[str] = mapped_column(Text, nullable=False)
    text_normalized: Mapped[str | None] = mapped_column(Text)
    avg_logprob: Mapped[float | None] = mapped_column(Float)
    no_speech_prob: Mapped[float | None] = mapped_column(Float)
    metadata_jsonb: Mapped[dict[str, Any] | None] = mapped_column(JsonB)
    created_at: Mapped[dt.datetime] = utc_now_column()

    segment: Mapped[Segment] = relationship(back_populates="hypotheses")
    system: Mapped[AsrSystem] = relationship()
    words: Mapped[list[HypothesisWord]] = relationship(
        back_populates="hypothesis",
        cascade="all, delete-orphan",
        order_by="HypothesisWord.position",
    )


class HypothesisWord(Base):
    """Optional word-level timing, language and script for one hypothesis token.

    Kept batched and with clean foreign keys so this table can be partitioned later without a
    schema redesign.
    """

    __tablename__ = "hypothesis_words"
    __table_args__ = (Index("ix_hypothesis_words_hypothesis_id", "hypothesis_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    hypothesis_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("asr_hypotheses.id", ondelete="CASCADE"), nullable=False
    )
    #: Token order within the hypothesis; word timings may be absent, order never is.
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    word_raw: Mapped[str] = mapped_column(Text, nullable=False)
    start_time: Mapped[float | None] = mapped_column(Float)
    end_time: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    predicted_language: Mapped[str | None] = mapped_column(String(16))
    predicted_script: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[dt.datetime] = utc_now_column()

    hypothesis: Mapped[AsrHypothesis] = relationship(back_populates="words")


class SegmentScore(Base):
    """Agreement scores and rule flags as received from the pipeline.

    The harness recomputes nothing here; a missing score stays null.
    """

    __tablename__ = "segment_scores"

    segment_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("segments.id", ondelete="CASCADE"), primary_key=True
    )
    cer_between_hypotheses: Mapped[float | None] = mapped_column(Float)
    word_disagreement_rate: Mapped[float | None] = mapped_column(Float)
    script_conflict_rate: Mapped[float | None] = mapped_column(Float)
    code_switch_density: Mapped[float | None] = mapped_column(Float)
    #: Rule flags: imported flags plus those computed at import time.
    flags_jsonb: Mapped[list[str] | None] = mapped_column(JsonB)
    imported_at: Mapped[dt.datetime] = utc_now_column()

    segment: Mapped[Segment] = relationship(back_populates="scores")
