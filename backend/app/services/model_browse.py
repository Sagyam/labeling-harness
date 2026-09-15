"""Read side of the Models page: a run's clips, sorted and filtered, and one clip's alignment.

Everything here reads what :mod:`app.services.model_import` stored; nothing is rescored except
the word alignment of the one clip on screen, which is recomputed from the stored texts because
it is not worth a table. When the fold rules have changed since the run was imported that
alignment can disagree with the stored counts, and the caller is told so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import Episode, ModelEvalClip, Segment
from app.services.clip_classes import overlap_bucket
from app.services.fold import fold_version, word_errors

ClipSort = Literal[
    "errors", "wer", "deletions", "insertions", "substitutions", "duration", "overlap"
]


@dataclass(frozen=True)
class ClipFilter:
    """Which of a run's clips to show. ``genre`` ``unknown`` matches an episode without one.

    ``class_axis`` and ``class_bucket`` together keep the clips in one bucket of one axis of the
    run's class breakdowns (D87), read from the classes stored with the run.
    """

    genre: str | None = None
    overlap: str | None = None
    loops_only: bool = False
    min_errors: int = 0
    class_axis: str | None = None
    class_bucket: str | None = None


def _genre_sql() -> sa.ColumnElement[str]:
    return sa.func.coalesce(Episode.metadata_jsonb["genre"].astext, "unknown")


def _overlap_bucket_sql() -> sa.ColumnElement[str]:
    """:func:`~app.services.clip_classes.overlap_bucket` in SQL: a filter matches the breakdown."""
    share = ModelEvalClip.overlap_share
    return sa.case(
        (share.is_(None), "unmeasured"),
        (share <= 0, "none"),
        (share < 0.05, "0-5%"),
        (share <= 0.15, "5-15%"),
        else_=">15%",
    )


def _wer_sql() -> sa.ColumnElement[float]:
    return ModelEvalClip.errors * 100.0 / sa.func.nullif(ModelEvalClip.ref_words, 0)


_SORT_COLUMNS: dict[str, Any] = {
    "errors": ModelEvalClip.errors,
    "wer": _wer_sql(),
    "deletions": ModelEvalClip.deletions,
    "insertions": ModelEvalClip.insertions,
    "substitutions": ModelEvalClip.substitutions,
    "duration": Segment.duration_seconds,
    "overlap": ModelEvalClip.overlap_share,
}


def _row(clip: ModelEvalClip, segment: Segment, episode_id: str, genre: str) -> dict[str, Any]:
    return {
        "segment_id": segment.id,
        "external_id": segment.external_id,
        "episode_external_id": episode_id,
        "genre": genre,
        "duration_seconds": segment.duration_seconds,
        "overlap_share": clip.overlap_share,
        "overlap_bucket": overlap_bucket(clip.overlap_share),
        "ref_words": clip.ref_words,
        "errors": clip.errors,
        "substitutions": clip.substitutions,
        "deletions": clip.deletions,
        "insertions": clip.insertions,
        "wer": 100 * clip.errors / clip.ref_words if clip.ref_words else 0.0,
        "raw_errors": clip.raw_errors,
        "raw_ref_words": clip.raw_ref_words,
        "char_errors": clip.char_errors,
        "ref_chars": clip.ref_chars,
        "is_loop": clip.is_loop,
        "classes": clip.classes_jsonb or {},
        "ref_text": clip.ref_text,
        "hyp_text": clip.hyp_text,
    }


def _base(run_id: int) -> sa.Select:
    return (
        sa.select(ModelEvalClip, Segment, Episode.external_id, _genre_sql())
        .join(Segment, Segment.id == ModelEvalClip.segment_id)
        .join(Episode, Episode.id == Segment.episode_id)
        .where(ModelEvalClip.run_id == run_id)
    )


def list_run_clips(
    session: Session,
    run_id: int,
    *,
    sort: ClipSort = "errors",
    descending: bool = True,
    where: ClipFilter = ClipFilter(),
    offset: int = 0,
    limit: int = 50,
) -> tuple[int, list[dict[str, Any]]]:
    """One page of a run's clips and how many match the filter. Ties break on the clip id."""
    query = _base(run_id)
    if where.genre is not None:
        query = query.where(_genre_sql() == where.genre)
    if where.overlap is not None:
        query = query.where(_overlap_bucket_sql() == where.overlap)
    if where.loops_only:
        query = query.where(ModelEvalClip.is_loop.is_(True))
    if where.min_errors > 0:
        query = query.where(ModelEvalClip.errors >= where.min_errors)
    if where.class_axis is not None and where.class_bucket is not None:
        query = query.where(
            ModelEvalClip.classes_jsonb[where.class_axis].astext == where.class_bucket
        )

    total = session.scalar(sa.select(sa.func.count()).select_from(query.subquery())) or 0
    column = _SORT_COLUMNS[sort]
    ordered = column.desc().nulls_last() if descending else column.asc().nulls_last()
    page = session.execute(
        query.order_by(ordered, Segment.external_id).offset(offset).limit(limit)
    ).all()
    return total, [_row(clip, segment, ep, genre) for clip, segment, ep, genre in page]


def clip_detail(
    session: Session, run_id: int, segment_id: int, *, run_fold_version: str
) -> dict[str, Any] | None:
    """One clip's row plus the folded word alignment of its texts, or ``None`` if not in the run."""
    found = session.execute(_base(run_id).where(ModelEvalClip.segment_id == segment_id)).first()
    if found is None:
        return None
    clip, segment, episode_id, genre = found
    alignment = word_errors(clip.ref_text, clip.hyp_text)
    return _row(clip, segment, episode_id, genre) | {
        "ops": [
            {"kind": op.kind, "ref": list(op.ref), "hyp": list(op.hyp), "similarity": op.similarity}
            for op in alignment.ops
        ],
        "fold_version": run_fold_version,
        "fold_version_changed": run_fold_version != fold_version(),
    }
