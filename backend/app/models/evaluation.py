"""Fine-tuned models and their per-clip scores, imported from the training notebook (D83).

A model's transcripts are produced on a GPU by the notebook and imported here; the harness never
runs the model. They are kept apart from ``asr_hypotheses`` on purpose: a fine-tuned model is
not a recogniser of the ingest pipeline, and its output must never reach the disagreement
scores, the queue or an export.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utc_now_column, utc_optional_column
from app.models.content import JsonB
from app.models.enums import EVAL_SPLITS, check_in


class AsrModel(Base):
    """One fine-tuned model, described by the ``model_card.json`` its notebook wrote."""

    __tablename__ = "asr_models"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    #: The model folder's name under ``data/models/asr/``; the idempotency key for import.
    slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    architecture: Mapped[str | None] = mapped_column(Text)
    #: When the notebook finished training it, from the card; null when the card does not say.
    trained_at: Mapped[dt.datetime | None] = utc_optional_column()
    #: The whole card as written, so a field the harness does not model yet is not lost.
    card_jsonb: Mapped[dict[str, Any]] = mapped_column(JsonB, nullable=False)
    created_at: Mapped[dt.datetime] = utc_now_column()
    updated_at: Mapped[dt.datetime] = utc_now_column(onupdate=lambda: dt.datetime.now(dt.UTC))

    runs: Mapped[list[ModelEvalRun]] = relationship(
        back_populates="model", cascade="all, delete-orphan", order_by="ModelEvalRun.id"
    )


class ModelEvalRun(Base):
    """One scoring of one model's transcripts of one split, against the labels of that moment."""

    __tablename__ = "model_eval_runs"
    __table_args__ = (
        CheckConstraint(check_in("split", EVAL_SPLITS), name="split_allowed"),
        UniqueConstraint("model_id", "source_sha256"),
        Index("ix_model_eval_runs_model_id", "model_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("asr_models.id", ondelete="CASCADE"), nullable=False
    )
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    #: How the transcripts were decoded, e.g. ``greedy+cap+retry`` (roadmap Phase 1).
    decoder: Mapped[str | None] = mapped_column(String(128))
    #: ``fold_version()`` the clips were scored under; runs compare only under the same one.
    fold_version: Mapped[str] = mapped_column(String(64), nullable=False)
    clip_count: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Folded/raw WER, CER, loops, the episode-bootstrap interval and the breakdowns.
    metrics_jsonb: Mapped[dict[str, Any]] = mapped_column(JsonB, nullable=False)
    #: The hypothesis file's name, for provenance only.
    source: Mapped[str | None] = mapped_column(Text)
    #: SHA-256 of the hypothesis file; importing the same file again is a no-op.
    source_sha256: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[dt.datetime] = utc_now_column()

    model: Mapped[AsrModel] = relationship(back_populates="runs")
    clips: Mapped[list[ModelEvalClip]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ModelEvalClip(Base):
    """One clip of a run: what the model wrote, the reference it was scored against, the counts.

    The reference is a snapshot. Labels are append-only and gold is relabeled and replaced over
    time, so the run keeps the text it was actually scored against; ``ref_label_id`` says which
    label that was.
    """

    __tablename__ = "model_eval_clips"
    __table_args__ = (
        UniqueConstraint("run_id", "segment_id"),
        Index("ix_model_eval_clips_run_id_errors", "run_id", "errors"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("model_eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    segment_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("segments.id", ondelete="CASCADE"), nullable=False
    )
    ref_label_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("segment_labels.id", ondelete="SET NULL")
    )
    ref_text: Mapped[str] = mapped_column(Text, nullable=False)
    hyp_text: Mapped[str] = mapped_column(Text, nullable=False)

    #: Folded word counts (``fold.word_errors``), the headline numbers.
    ref_words: Mapped[int] = mapped_column(Integer, nullable=False)
    errors: Mapped[int] = mapped_column(Integer, nullable=False)
    substitutions: Mapped[int] = mapped_column(Integer, nullable=False)
    deletions: Mapped[int] = mapped_column(Integer, nullable=False)
    insertions: Mapped[int] = mapped_column(Integer, nullable=False)
    #: The raw lexical rate, reported beside the folded one.
    raw_ref_words: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_errors: Mapped[int] = mapped_column(Integer, nullable=False)
    ref_chars: Mapped[int] = mapped_column(Integer, nullable=False)
    char_errors: Mapped[int] = mapped_column(Integer, nullable=False)
    #: A 3-word sequence repeated 5+ times: the decoder was stuck, not transcribing.
    is_loop: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: GPU seconds the notebook spent on this clip, when it recorded them.
    compute_s: Mapped[float | None] = mapped_column(Float)

    run: Mapped[ModelEvalRun] = relationship(back_populates="clips")
