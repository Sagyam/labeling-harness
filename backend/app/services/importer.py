"""Manifest importer.

Import runs in two phases:

1. **Plan.** Read and validate the manifest, probe every clip, compare checksums against what is
   already stored, and decide what would change. Nothing is written. Any problem raises here, so a
   malformed export leaves the database exactly as it was.
2. **Apply.** Write the episode (assigning its frozen split on first sight), segments, hypotheses,
   words and scores, upload clips, and store precomputed peaks.

A dry run stops after phase 1 and reports the plan.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import (
    AsrHypothesis,
    AsrSystem,
    Episode,
    HypothesisWord,
    ImportRun,
    Segment,
    SegmentScore,
)
from app.services.audio import ClipFormatError, compute_peaks, validate_clip
from app.services.manifest import Manifest, ManifestError, read_manifest
from app.services.splits import assign_split
from app.storage.base import ObjectStorage
from app.utils.hashing import checksums_match, sha256_file
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ImportError_(RuntimeError):
    """An import could not proceed. Named with a trailing underscore to avoid shadowing the
    builtin ``ImportError``, which means something entirely different."""


class ClipChangedError(ImportError_):
    """A segment's clip differs from the one already imported."""


@dataclass
class ImportReport:
    """What an import did, or -- in a dry run -- what it would have done."""

    source_path: str
    episode_id: str
    split: str
    episode_created: bool = False
    segments_inserted: int = 0
    segments_skipped: int = 0
    hypotheses_inserted: int = 0
    hypotheses_skipped: int = 0
    words_inserted: int = 0
    clips_uploaded: int = 0
    clips_replaced: int = 0
    peaks_written: int = 0
    peaks_reused: int = 0
    systems_created: int = 0
    dry_run: bool = False
    import_run_id: int | None = None
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        """A short human-readable summary, for the CLI."""
        header = f"{'DRY RUN -- ' if self.dry_run else ''}import of {self.episode_id}"
        lines = [
            header,
            f"  source            {self.source_path}",
            f"  split             {self.split}"
            + (" (newly assigned)" if self.episode_created else " (already frozen)"),
            f"  segments          {self.segments_inserted} inserted, "
            f"{self.segments_skipped} unchanged",
            f"  hypotheses        {self.hypotheses_inserted} inserted, "
            f"{self.hypotheses_skipped} unchanged",
            f"  words             {self.words_inserted} inserted",
            f"  clips             {self.clips_uploaded} uploaded, {self.clips_replaced} replaced",
            f"  peaks             {self.peaks_written} computed, {self.peaks_reused} reused",
            f"  asr systems       {self.systems_created} created",
        ]
        if self.import_run_id is not None:
            lines.append(f"  import_run_id     {self.import_run_id}")
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


@dataclass
class _SegmentPlan:
    """Per-segment decision reached in the planning phase."""

    record: dict[str, Any]
    clip_path: Path
    clip_checksum: str
    existing: Segment | None
    clip_changed: bool
    supplied_peaks: Path | None
    missing_system_ids: list[str]


def clip_object_key(episode_id: str, segment_id: str) -> str:
    """Object key for a segment's clip."""
    return f"clips/{episode_id}/{segment_id}.flac"


def peaks_object_key(episode_id: str, segment_id: str) -> str:
    """Object key for a segment's precomputed peaks."""
    return f"peaks/{episode_id}/{segment_id}.json"


def _plan(
    session: Session, manifest: Manifest, settings: Settings, *, allow_clip_change: bool
) -> list[_SegmentPlan]:
    """Validate every clip and work out what each segment needs. Writes nothing."""
    plans: list[_SegmentPlan] = []
    for record in manifest.segments:
        segment_id = str(record["segment_id"])
        clip_path = manifest.clip_path(record)
        try:
            validate_clip(clip_path, settings)
        except ClipFormatError as exc:
            raise ImportError_(f"{segment_id}: {exc}") from exc

        actual = sha256_file(clip_path)
        declared = record.get("clip_checksum")
        if declared and not checksums_match(declared, actual):
            raise ImportError_(
                f"{segment_id}: clip checksum mismatch -- manifest declares {declared}, "
                f"the file on disk is {actual}. The export is corrupt or was modified."
            )

        existing = session.scalar(sa.select(Segment).where(Segment.external_id == segment_id))
        clip_changed = existing is not None and not checksums_match(existing.clip_checksum, actual)
        if clip_changed and not allow_clip_change:
            raise ClipChangedError(
                f"{segment_id}: clip checksum changed since it was imported "
                f"({existing.clip_checksum} -> {actual}). Re-run with --allow-clip-change if the "
                "upstream audio was intentionally regenerated."
            )

        missing_system_ids: list[str] = []
        if existing is not None:
            present = {
                system_id
                for (system_id,) in session.execute(
                    sa.select(AsrSystem.system_id)
                    .join(AsrHypothesis, AsrHypothesis.asr_system_id == AsrSystem.id)
                    .where(AsrHypothesis.segment_id == existing.id)
                )
            }
            missing_system_ids = [
                str(h["system_id"]) for h in record["hypotheses"] if h["system_id"] not in present
            ]

        plans.append(
            _SegmentPlan(
                record=record,
                clip_path=clip_path,
                clip_checksum=actual,
                existing=existing,
                clip_changed=clip_changed,
                supplied_peaks=manifest.peaks_path(record),
                missing_system_ids=missing_system_ids,
            )
        )
    return plans


def _dry_run_report(
    manifest: Manifest, plans: list[_SegmentPlan], split: str, episode_exists: bool
) -> ImportReport:
    report = ImportReport(
        source_path=str(manifest.root),
        episode_id=manifest.episode_id,
        split=split,
        episode_created=not episode_exists,
        dry_run=True,
    )
    for plan in plans:
        if plan.existing is None:
            report.segments_inserted += 1
            report.hypotheses_inserted += len(plan.record["hypotheses"])
            report.words_inserted += sum(
                len(h.get("words") or []) for h in plan.record["hypotheses"]
            )
            report.clips_uploaded += 1
            report.peaks_reused += 1 if plan.supplied_peaks else 0
            report.peaks_written += 0 if plan.supplied_peaks else 1
        else:
            report.segments_skipped += 1
            report.hypotheses_inserted += len(plan.missing_system_ids)
            report.hypotheses_skipped += len(plan.record["hypotheses"]) - len(
                plan.missing_system_ids
            )
            if plan.clip_changed:
                report.clips_replaced += 1
    return report


def _upsert_systems(
    session: Session, manifest: Manifest, report: ImportReport
) -> dict[str, AsrSystem]:
    systems: dict[str, AsrSystem] = {}
    for record in manifest.segments:
        for hypothesis in record["hypotheses"]:
            system_id = str(hypothesis["system_id"])
            if system_id in systems:
                continue
            system = session.scalar(sa.select(AsrSystem).where(AsrSystem.system_id == system_id))
            if system is None:
                system = AsrSystem(system_id=system_id, model_id=hypothesis.get("model_id"))
                session.add(system)
                session.flush()
                report.systems_created += 1
            systems[system_id] = system
    return systems


def _upsert_episode(
    session: Session, manifest: Manifest, settings: Settings, report: ImportReport
) -> Episode:
    """Fetch or create the episode. The split is assigned once and never recomputed."""
    external_id = manifest.episode_id
    episode = session.scalar(sa.select(Episode).where(Episode.external_id == external_id))
    known = {
        "episode_id",
        "show_id",
        "title",
        "source_uri",
        "published_at",
        "source_audio_checksum",
        "duration_seconds",
        "pipeline_version",
        "pipeline_commit",
    }
    extra = {k: v for k, v in manifest.episode.items() if k not in known}
    published_at = manifest.episode.get("published_at")

    if episode is None:
        episode = Episode(
            external_id=external_id,
            show_id=manifest.episode.get("show_id"),
            title=manifest.episode.get("title"),
            source_uri=manifest.episode.get("source_uri"),
            published_at=dt.date.fromisoformat(published_at) if published_at else None,
            source_audio_checksum=manifest.episode.get("source_audio_checksum"),
            duration_seconds=manifest.episode.get("duration_seconds"),
            split=report.split,
            split_seed=settings.importer.split_seed,
            split_assigned_at=dt.datetime.now(dt.UTC),
            metadata_jsonb=extra or None,
        )
        session.add(episode)
        session.flush()
        report.episode_created = True
    return episode


def _import_hypotheses(
    session: Session,
    plan: _SegmentPlan,
    segment: Segment,
    systems: dict[str, AsrSystem],
    report: ImportReport,
    *,
    only_missing: bool,
) -> None:
    wanted = set(plan.missing_system_ids) if only_missing else None
    for record in plan.record["hypotheses"]:
        system_id = str(record["system_id"])
        if wanted is not None and system_id not in wanted:
            report.hypotheses_skipped += 1
            continue
        hypothesis = AsrHypothesis(
            segment_id=segment.id,
            asr_system_id=systems[system_id].id,
            text_raw=record["text"],
            text_normalized=record["text"],
            avg_logprob=record.get("avg_logprob"),
            no_speech_prob=record.get("no_speech_prob"),
            metadata_jsonb={
                k: v
                for k, v in record.items()
                if k
                not in {"system_id", "model_id", "text", "avg_logprob", "no_speech_prob", "words"}
            }
            or None,
        )
        session.add(hypothesis)
        session.flush()
        report.hypotheses_inserted += 1

        words = record.get("words") or []
        if words:
            session.add_all(
                HypothesisWord(
                    hypothesis_id=hypothesis.id,
                    position=position,
                    word_raw=str(word["word"]),
                    start_time=word.get("start"),
                    end_time=word.get("end"),
                    confidence=word.get("confidence"),
                    predicted_language=word.get("predicted_language"),
                    predicted_script=word.get("predicted_script"),
                )
                for position, word in enumerate(words)
            )
            report.words_inserted += len(words)


def _store_clip_and_peaks(
    plan: _SegmentPlan,
    manifest: Manifest,
    storage: ObjectStorage,
    settings: Settings,
    report: ImportReport,
    *,
    replacing: bool,
) -> tuple[str, str]:
    segment_id = str(plan.record["segment_id"])
    clip_key = clip_object_key(manifest.episode_id, segment_id)
    storage.put_file(clip_key, plan.clip_path, content_type="audio/flac")
    if replacing:
        report.clips_replaced += 1
    else:
        report.clips_uploaded += 1

    peaks_key = peaks_object_key(manifest.episode_id, segment_id)
    if plan.supplied_peaks is not None:
        storage.put_file(peaks_key, plan.supplied_peaks, content_type="application/json")
        report.peaks_reused += 1
    else:
        payload = compute_peaks(plan.clip_path, settings.importer.peaks_buckets)
        storage.put_bytes(
            peaks_key,
            json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )
        report.peaks_written += 1
    return clip_key, peaks_key


def import_manifest(
    session: Session,
    root: Path | str,
    *,
    storage: ObjectStorage,
    settings: Settings | None = None,
    dry_run: bool = False,
    allow_clip_change: bool = False,
) -> ImportReport:
    """Import an upstream export directory.

    Args:
        session: Open session; the caller commits. On failure the caller must roll back.
        root: The ``export_<episode_id>/`` directory.
        storage: Object storage for clips and peaks.
        settings: Configuration override.
        dry_run: Plan and report without writing anything.
        allow_clip_change: Accept a clip whose checksum differs from the imported one.

    Returns:
        An :class:`ImportReport`. In a dry run the counters describe the planned changes.

    Raises:
        ImportError_: The manifest is malformed, a clip is not 16 kHz mono FLAC, a checksum does
            not match, or a clip changed without the override.
    """
    settings = settings or get_settings()
    try:
        manifest = read_manifest(root)
    except ManifestError as exc:
        raise ImportError_(str(exc)) from exc

    episode_exists = (
        session.scalar(
            sa.select(sa.func.count())
            .select_from(Episode)
            .where(Episode.external_id == manifest.episode_id)
        )
        > 0
    )
    existing_episode = session.scalar(
        sa.select(Episode).where(Episode.external_id == manifest.episode_id)
    )
    split = (
        existing_episode.split
        if existing_episode is not None
        else assign_split(
            manifest.episode_id,
            seed=settings.importer.split_seed,
            ratios=settings.importer.split_ratios,
        )
    )

    plans = _plan(session, manifest, settings, allow_clip_change=allow_clip_change)

    if dry_run:
        report = _dry_run_report(manifest, plans, split, episode_exists)
        logger.info("import_dry_run", episode_id=manifest.episode_id, source=str(manifest.root))
        return report

    report = ImportReport(
        source_path=str(manifest.root), episode_id=manifest.episode_id, split=split
    )
    run = ImportRun(
        source_path=str(manifest.root),
        pipeline_version=manifest.episode.get("pipeline_version"),
        pipeline_commit=manifest.episode.get("pipeline_commit"),
        status="running",
    )
    session.add(run)
    session.flush()
    report.import_run_id = run.id

    episode = _upsert_episode(session, manifest, settings, report)
    systems = _upsert_systems(session, manifest, report)

    for plan in plans:
        record = plan.record
        segment_id = str(record["segment_id"])

        if plan.existing is None:
            clip_key, peaks_key = _store_clip_and_peaks(
                plan, manifest, storage, settings, report, replacing=False
            )
            segment = Segment(
                episode_id=episode.id,
                external_id=segment_id,
                speaker_id=record.get("speaker_id"),
                start_time=float(record["start_time"]),
                end_time=float(record["end_time"]),
                duration_seconds=float(record["end_time"]) - float(record["start_time"]),
                clip_object_key=clip_key,
                clip_checksum=plan.clip_checksum,
                peaks_object_key=peaks_key,
                p_en=record.get("p_en"),
                lid=record.get("lid"),
                pipeline_status="imported",
                import_run_id=run.id,
            )
            session.add(segment)
            session.flush()
            report.segments_inserted += 1

            _import_hypotheses(session, plan, segment, systems, report, only_missing=False)

            scores = record.get("scores") or {}
            session.add(
                SegmentScore(
                    segment_id=segment.id,
                    cer_between_hypotheses=scores.get("cer_between_hypotheses"),
                    word_disagreement_rate=scores.get("word_disagreement_rate"),
                    script_conflict_rate=scores.get("script_conflict_rate"),
                    code_switch_density=scores.get("code_switch_density"),
                    flags_jsonb=list(record.get("flags") or []),
                )
            )
        else:
            segment = plan.existing
            report.segments_skipped += 1
            if plan.clip_changed:
                clip_key, peaks_key = _store_clip_and_peaks(
                    plan, manifest, storage, settings, report, replacing=True
                )
                segment.clip_object_key = clip_key
                segment.clip_checksum = plan.clip_checksum
                segment.peaks_object_key = peaks_key
                report.warnings.append(f"{segment_id}: clip replaced")
            if plan.missing_system_ids:
                _import_hypotheses(session, plan, segment, systems, report, only_missing=True)
            else:
                report.hypotheses_skipped += len(record["hypotheses"])

    session.flush()
    run.segments_inserted = report.segments_inserted
    run.segments_skipped = report.segments_skipped
    run.hypotheses_inserted = report.hypotheses_inserted
    run.finished_at = dt.datetime.now(dt.UTC)
    run.status = "succeeded"
    session.flush()

    logger.info(
        "import_complete",
        episode_id=manifest.episode_id,
        segments_inserted=report.segments_inserted,
        segments_skipped=report.segments_skipped,
        hypotheses_inserted=report.hypotheses_inserted,
        import_run_id=run.id,
    )
    return report
