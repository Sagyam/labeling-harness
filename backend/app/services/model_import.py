"""Import a fine-tuned model's card and its gold/val transcripts from its folder (D83).

The training notebook writes, and the owner copies from Drive, one folder per model::

    data/models/asr/<slug>/
        model_card.json   {"name": ..., "created_at": ..., "description": ..., "architecture": ...,
                           "decoder": ..., ...anything else, kept verbatim}
        gold.jsonl        {"segment_id": ..., "text": ..., "compute_s": ...} per line
        val.jsonl         the same, for the val split; either file may be absent

The folder name is the model's identity. Each transcript file becomes one run, scored against
the current label of every clip at import time and keeping that reference as a snapshot. The
same file imported twice is a no-op; an edited card updates the model.

A file naming a clip the harness does not have is refused whole: it was made from another
dataset. A clip that has *left* the split since the notebook ran (gold is chosen by hand, D71),
or whose current label no longer carries a transcript, is skipped and counted instead, so the run
says what it left out rather than scoring a train clip as gold.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.models import (
    AsrModel,
    AuditLog,
    LabelVersion,
    ModelEvalClip,
    ModelEvalRun,
    Segment,
    SegmentLabel,
)
from app.models.enums import APPROVED_DISPOSITIONS, EVAL_SPLITS
from app.services.clip_classes import overlap_bucket, overlap_share
from app.services.fold import fold_version
from app.services.model_eval import ScoredClip, score_clip, summarize
from app.services.normalize import load_ruleset, normalize_text
from app.services.stats import latest_labels_subquery

CARD_NAME = "model_card.json"


class ModelImportError(ValueError):
    """The model folder or one of its files is malformed; nothing was written for it."""


@dataclass(frozen=True)
class Card:
    """The fields of a model card the harness reads, and the card as written."""

    name: str
    description: str | None
    architecture: str | None
    decoder: str | None
    trained_at: dt.datetime | None
    raw: dict[str, Any]


@dataclass
class ImportReport:
    models: list[str] = field(default_factory=list)
    runs_created: int = 0
    runs_unchanged: int = 0
    #: Clips in a file that were not scored: no longer in the split, or no transcript to score.
    skipped: int = 0


def _text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def read_card(folder: Path) -> Card:
    """Read and check ``model_card.json``.

    Raises:
        ModelImportError: The card is missing, is not a JSON object, has no ``name``, or has a
            ``created_at`` that is not an ISO date.
    """
    path = folder / CARD_NAME
    if not path.is_file():
        raise ModelImportError(f"{folder.name}: no {CARD_NAME}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelImportError(f"{folder.name}: {CARD_NAME} is not JSON: {exc}") from exc
    if not isinstance(raw, dict) or not _text(raw.get("name")):
        raise ModelImportError(f"{folder.name}: {CARD_NAME} needs a 'name'")
    trained_at = None
    if raw.get("created_at"):
        try:
            trained_at = dt.datetime.fromisoformat(str(raw["created_at"]))
        except ValueError as exc:
            raise ModelImportError(f"{folder.name}: created_at is not an ISO date") from exc
        if trained_at.tzinfo is None:  # UTC everywhere
            trained_at = trained_at.replace(tzinfo=dt.UTC)
    return Card(
        name=str(raw["name"]),
        description=_text(raw.get("description")),
        architecture=_text(raw.get("architecture")),
        decoder=_text(raw.get("decoder")),
        trained_at=trained_at,
        raw=raw,
    )


def _read_hyps(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            segment_id, text = str(row["segment_id"]), row["text"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelImportError(f"{path.name}:{number}: not a segment_id/text row") from exc
        if segment_id in seen:
            raise ModelImportError(f"{path.name}: {segment_id} appears twice")
        seen.add(segment_id)
        compute = row.get("compute_s")
        rows.append(
            {
                "segment_id": segment_id,
                "text": str(text or "").strip(),
                "compute_s": float(compute) if compute is not None else None,
            }
        )
    if not rows:
        raise ModelImportError(f"{path.name}: no clips")
    return rows


def _in_split(segment: Segment, split: str) -> bool:
    if split == "gold":
        return segment.pot == "gold"
    return segment.pot == "train" and segment.episode.split == "val"


def _upsert_model(session: Session, slug: str, card: Card) -> AsrModel:
    model = session.scalars(sa.select(AsrModel).where(AsrModel.slug == slug)).first()
    if model is None:
        model = AsrModel(slug=slug, name=card.name, card_jsonb=card.raw)
        session.add(model)
    model.name = card.name
    model.description = card.description
    model.architecture = card.architecture
    model.trained_at = card.trained_at
    model.card_jsonb = card.raw
    session.flush()
    return model


def _import_run(
    session: Session,
    model: AsrModel,
    card: Card,
    split: str,
    path: Path,
    *,
    version: LabelVersion,
    actor: str,
    report: ImportReport,
) -> None:
    checksum = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    exists = session.scalar(
        sa.select(ModelEvalRun.id).where(
            ModelEvalRun.model_id == model.id, ModelEvalRun.source_sha256 == checksum
        )
    )
    if exists is not None:
        report.runs_unchanged += 1
        return

    rows = _read_hyps(path)
    ids = [r["segment_id"] for r in rows]
    segments = {
        s.external_id: s
        for s in session.scalars(
            sa.select(Segment)
            .options(selectinload(Segment.episode))
            .where(Segment.external_id.in_(ids))
        )
    }
    unknown = [sid for sid in ids if sid not in segments]
    if unknown:
        raise ModelImportError(
            f"{path.name}: {len(unknown)} clip(s) the harness does not have, first {unknown[0]!r};"
            " was it made from another dataset?"
        )

    current = latest_labels_subquery()
    labels: dict[int, SegmentLabel] = dict(
        session.execute(
            sa.select(current.c.segment_id, SegmentLabel)
            .join(SegmentLabel, SegmentLabel.id == current.c.id)
            .where(
                current.c.label_version_id == version.id,
                current.c.segment_id.in_([s.id for s in segments.values()]),
            )
        )
        .tuples()
        .all()
    )

    # The export normalizes every label it writes, so the notebook's references are normalized;
    # scoring against the raw label would count the ruleset's own rewrites as model errors.
    ruleset = load_ruleset()
    skipped = {"not_in_split": 0, "no_reference": 0}
    clips: list[ModelEvalClip] = []
    scored: list[ScoredClip] = []
    for row in rows:
        segment = segments[row["segment_id"]]
        if not _in_split(segment, split):
            skipped["not_in_split"] += 1
            continue
        label = labels.get(segment.id)
        if label is None or label.disposition not in APPROVED_DISPOSITIONS:
            skipped["no_reference"] += 1
            continue
        reference = normalize_text(label.final_text, ruleset) or ""
        score = score_clip(reference, row["text"])
        share = overlap_share(segment.overlap_spans_jsonb, segment.duration_seconds)
        genre = (segment.episode.metadata_jsonb or {}).get("genre")
        scored.append(
            ScoredClip(
                episode=segment.episode.external_id,
                genre=genre,
                overlap=overlap_bucket(share),
                score=score,
            )
        )
        clips.append(
            ModelEvalClip(
                segment_id=segment.id,
                ref_label_id=label.id,
                ref_text=reference,
                hyp_text=row["text"],
                compute_s=row["compute_s"],
                overlap_share=share,
                **vars(score),
            )
        )

    metrics = summarize(scored) | {"skipped": skipped}
    run = ModelEvalRun(
        model_id=model.id,
        split=split,
        decoder=card.decoder,
        fold_version=fold_version(),
        clip_count=len(clips),
        metrics_jsonb=metrics,
        source=path.name,
        source_sha256=checksum,
        clips=clips,
    )
    session.add(run)
    session.flush()
    session.add(
        AuditLog(
            entity_type="asr_model",
            entity_id=model.slug,
            action="model_eval_import",
            actor=actor,
            new_values_jsonb={
                "run_id": run.id,
                "split": split,
                "clips": len(clips),
                "skipped": skipped,
                "wer": metrics["wer"],
                "source": path.name,
            },
        )
    )
    report.runs_created += 1
    report.skipped += sum(skipped.values())


def import_model_dir(
    session: Session,
    folder: Path,
    *,
    settings: Settings | None = None,
    actor: str,
    report: ImportReport | None = None,
) -> ImportReport:
    """Upsert one model from its card and import each of its transcript files as a run.

    Args:
        session: Open session; the caller commits.
        folder: ``<root>/<slug>``, holding ``model_card.json`` and ``gold.jsonl``/``val.jsonl``.
        settings: Configuration override; the label version scored against comes from it.
        actor: Recorded on each run's ``audit_logs`` row.
        report: Accumulates across folders when given.

    Raises:
        ModelImportError: The card or a transcript file is malformed, or names unknown clips.
    """
    settings = settings or get_settings()
    report = report if report is not None else ImportReport()
    card = read_card(folder)
    version_name = settings.labels.default_label_version
    version = session.scalar(sa.select(LabelVersion).where(LabelVersion.name == version_name))

    model = _upsert_model(session, folder.name, card)
    report.models.append(folder.name)
    for split in EVAL_SPLITS:
        path = folder / f"{split}.jsonl"
        if not path.is_file():
            continue
        if version is None:
            raise ModelImportError(
                f"label version {version_name!r} does not exist; nothing to score"
            )
        _import_run(session, model, card, split, path, version=version, actor=actor, report=report)
    session.flush()
    return report


def scan_models(
    session: Session, root: Path, *, settings: Settings | None = None, actor: str
) -> ImportReport:
    """Import every folder under ``root`` that holds a model card, in name order."""
    report = ImportReport()
    if not root.is_dir():
        return report
    for folder in sorted(p for p in root.iterdir() if (p / CARD_NAME).is_file()):
        import_model_dir(session, folder, settings=settings, actor=actor, report=report)
    return report
