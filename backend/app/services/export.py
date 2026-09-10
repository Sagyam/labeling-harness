"""Dataset export.

Four export kinds, each writing a manifest alongside the data:

1. ``training`` -- train and val splits, approved labels only.
2. ``gold`` -- test split only, retaining the seed system per segment.
3. ``analytics`` -- everything labeled, including word-level fields where they were imported.
4. ``error_mining`` -- ``uncertain`` and ``unusable_audio``, for pipeline debugging.

Every export that has an object store to read from also writes out the **whole audio** of each
episode it drew rows from, under ``episodes/``. That is what makes a serious diarizer possible:
speaker identity is not something the ingest pipeline guesses at any more (D58), so pyannote 3.1
or its successor runs here instead, against a recording rather than a bag of clips, with no time
budget and no need to re-ingest anything when a better model appears (D62).

Exports are deterministic: the same inputs and filters produce byte-identical output, so two exports
of "the same" dataset really are the same dataset. ``disposition``, ``verification_tier``, ``pot``
and ``seed_system_id`` are retained in every record because they are what make the dataset
defensible to a reviewer.

``verification_tier`` is the one a consumer must not ignore (D63). A ``verified`` row was played
and read; a ``screened`` row was accepted on the cross-ASR disagreement signal without anyone
listening. Both are legitimate ways to build a corpus and they are not the same claim, so the
manifest reports the mix per split and the ``gold`` export refuses to write a screened row at all.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.models import (
    AsrHypothesis,
    AsrSystem,
    Episode,
    ImportRun,
    LabelVersion,
    Segment,
    SegmentLabel,
)
from app.services.normalize import Ruleset, load_ruleset, normalize_text
from app.services.pots import effective_split, effective_split_sql
from app.services.stats import latest_labels_subquery
from app.storage.base import ObjectStorage
from app.utils.hashing import sha256_file
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ExportError(RuntimeError):
    """The export could not be produced."""


class GoldPurityError(ExportError):
    """A gold-pot row is not what the gold pot promises."""


@dataclass(frozen=True)
class ExportKind:
    """Declarative definition of one export kind."""

    name: str
    splits: tuple[str, ...]
    dispositions: tuple[str, ...]
    include_words: bool = False
    include_hypotheses: bool = False


EXPORT_KINDS: dict[str, ExportKind] = {
    "training": ExportKind(
        name="training",
        splits=("train", "val"),
        dispositions=("accepted_unchanged", "edited"),
    ),
    "gold": ExportKind(
        name="gold",
        splits=("test",),
        dispositions=("accepted_unchanged", "edited"),
    ),
    "analytics": ExportKind(
        name="analytics",
        splits=("train", "val", "test", "unassigned"),
        dispositions=("accepted_unchanged", "edited", "uncertain", "unusable_audio"),
        include_words=True,
        include_hypotheses=True,
    ),
    "error_mining": ExportKind(
        name="error_mining",
        splits=("train", "val", "test", "unassigned"),
        dispositions=("uncertain", "unusable_audio"),
        include_hypotheses=True,
    ),
}


@dataclass
class ExportResult:
    """Where the export landed and what it contains."""

    kind: str
    output_dir: Path
    data_path: Path
    manifest_path: Path
    row_count: int = 0
    row_counts_by_split: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        """A short human-readable summary, for the CLI."""
        by_split = ", ".join(f"{k}={v}" for k, v in sorted(self.row_counts_by_split.items()))
        return "\n".join(
            [
                f"{self.kind} export",
                f"  directory   {self.output_dir}",
                f"  rows        {self.row_count} ({by_split or 'none'})",
                f"  data        {self.data_path.name}",
                f"  manifest    {self.manifest_path.name}",
            ]
        )


def git_commit() -> str | None:
    """The current git commit, so an export can be traced back to the code that made it."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[3],
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    return result.stdout.strip() or None


def _hypothesis_payload(hypothesis: AsrHypothesis, *, include_words: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "system_id": hypothesis.system.system_id,
        "model_id": hypothesis.system.model_id,
        "text": hypothesis.text_raw,
        "avg_logprob": hypothesis.avg_logprob,
        "no_speech_prob": hypothesis.no_speech_prob,
    }
    if include_words:
        payload["words"] = [
            {
                "word": word.word_raw,
                "start": word.start_time,
                "end": word.end_time,
                "confidence": word.confidence,
                "predicted_language": word.predicted_language,
                "predicted_script": word.predicted_script,
                "speaker": word.speaker,
            }
            for word in hypothesis.words
        ]
    return payload


def _record(
    segment: Segment,
    label: SegmentLabel,
    version: LabelVersion,
    kind: ExportKind,
    seed_system_id: str | None,
    ruleset: Ruleset,
    spanning_episode_ids: set[int],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "segment_id": segment.external_id,
        "episode_id": segment.episode.external_id,
        "audio_path": f"clips/{segment.external_id}.flac",
        "start_time": segment.start_time,
        "end_time": segment.end_time,
        "text": normalize_text(label.final_text, ruleset),
        "disposition": label.disposition,
        #: Whether a human actually listened to this clip, or waved it through on the cross-ASR
        #: disagreement signal. Carried on every row of every kind, because a consumer that cannot
        #: tell the two apart is reading a claim the corpus never made (D63).
        "verification_tier": label.verification_tier,
        "pot": segment.pot,
        #: True when this clip's episode has clips in both pots -- the same speaker and topic on
        #: both sides of the train/test line, which per-clip gold allows (D71). Lets a gold result
        #: be reported with and without those clips.
        "episode_spans_pots": segment.episode_id in spanning_episode_ids,
        "seed_system_id": seed_system_id,
        "label_version": version.name,
        "policy_version": version.policy_version,
        "split": effective_split(segment.pot, segment.episode.split),
        "code_switch_density": segment.scores.code_switch_density if segment.scores else None,
    }
    if kind.include_hypotheses:
        record["speaker_id"] = segment.speaker_id
        record["duration_seconds"] = segment.duration_seconds
        record["p_en"] = segment.p_en
        record["lid"] = segment.lid
        record["notes"] = label.notes
        record["episode_metadata"] = segment.episode.metadata_jsonb
        record["scores"] = (
            {
                "cer_between_hypotheses": segment.scores.cer_between_hypotheses,
                "word_disagreement_rate": segment.scores.word_disagreement_rate,
                "script_conflict_rate": segment.scores.script_conflict_rate,
                "code_switch_density": segment.scores.code_switch_density,
                "flags": list(segment.scores.flags_jsonb or []),
            }
            if segment.scores
            else None
        )
        record["hypotheses"] = [
            _hypothesis_payload(h, include_words=kind.include_words)
            for h in sorted(segment.hypotheses, key=lambda h: h.system.system_id)
        ]
    return record


def _write_episode_audio(
    output_dir: Path,
    keys_by_episode: dict[str, str],
    storage: ObjectStorage | None,
) -> list[dict[str, Any]]:
    """Copy each episode's whole recording into ``episodes/`` beside the exported rows.

    This is what a post-export diarizer runs on (D58): a recording, not the clips cut out of it.
    Written here rather than referenced by object key so an export directory is self-contained --
    it can be moved to a machine with no access to this deployment's object store at all.

    A missing or unreadable object is a warning, never a failure. The metadata export is complete
    and correct without the audio; refusing to write it because one episode predates the column
    would be a poor trade.

    Args:
        output_dir: The export directory.
        keys_by_episode: ``{episode external id: object key}``, for episodes with audio.
        storage: Object store to read from, or None to skip audio entirely.

    Returns:
        One manifest file entry per recording written, sorted by name so the manifest stays
        deterministic.
    """
    if storage is None or not keys_by_episode:
        return []

    audio_dir = output_dir / "episodes"
    audio_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict[str, Any]] = []
    for external_id, key in sorted(keys_by_episode.items()):
        destination = audio_dir / f"{external_id}.flac"
        try:
            destination.write_bytes(storage.get_bytes(key))
        except Exception as exc:
            logger.warning("episode_audio_missing", episode=external_id, key=key, error=str(exc))
            destination.unlink(missing_ok=True)
            continue
        written.append(
            {
                "name": f"episodes/{destination.name}",
                "sha256": sha256_file(destination).removeprefix("sha256:"),
                "bytes": destination.stat().st_size,
            }
        )
    return written


def export_dataset(
    session: Session,
    *,
    kind: str,
    output_root: Path | str | None = None,
    label_version: str | None = None,
    episode: str | None = None,
    settings: Settings | None = None,
    storage: ObjectStorage | None = None,
) -> ExportResult:
    """Write one export kind to disk.

    Args:
        session: Open session.
        kind: One of ``training``, ``gold``, ``analytics``, ``error_mining``.
        output_root: Directory to write beneath. Defaults to the configured export root.
        label_version: Which label version to export. Defaults to the configured one.
        episode: Restrict to a single episode external id.
        settings: Configuration override.
        storage: Object store to copy episode audio out of. Omitted means no audio is written --
            the metadata export is still complete and correct, it just cannot be diarized.

    Returns:
        Where the files landed and how many rows they hold.

    Raises:
        ExportError: The export kind is unknown.
    """
    settings = settings or get_settings()
    definition = EXPORT_KINDS.get(kind)
    if definition is None:
        raise ExportError(f"unknown export kind {kind!r}; choose one of {sorted(EXPORT_KINDS)}")
    # Loaded once per export rather than per row, so every record in a file is written through
    # exactly one table even if the file on disk changes underneath a long export (D64).
    ruleset = load_ruleset()

    version_name = label_version or settings.labels.default_label_version
    version = session.scalar(sa.select(LabelVersion).where(LabelVersion.name == version_name))

    output_dir = Path(output_root or settings.export.output_root) / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"{kind}.jsonl"
    manifest_path = output_dir / "manifest.json"

    records: list[dict[str, Any]] = []
    import_run_ids: set[int] = set()
    #: Episodes the exported rows came from, so their audio can be written out beside the data.
    #: Keyed by external id, which is also the filename, so a duplicate is a no-op rather than a
    #: second copy of the same recording.
    episode_audio: dict[str, str] = {}

    if version is not None:
        current = latest_labels_subquery()
        query = (
            sa.select(Segment, SegmentLabel)
            .join(Episode, Episode.id == Segment.episode_id)
            .join(current, current.c.segment_id == Segment.id)
            .join(SegmentLabel, SegmentLabel.id == current.c.id)
            .options(
                selectinload(Segment.episode),
                selectinload(Segment.scores),
                selectinload(Segment.hypotheses).selectinload(AsrHypothesis.system),
                selectinload(Segment.hypotheses).selectinload(AsrHypothesis.words),
            )
            .where(
                current.c.label_version_id == version.id,
                current.c.disposition.in_(definition.dispositions),
                effective_split_sql().in_(definition.splits),
            )
            # Deterministic order: the same inputs must produce byte-identical output.
            .order_by(Segment.external_id)
        )
        if episode:
            query = query.where(Episode.external_id == episode)

        rows = list(session.execute(query))

        # Resolve every seed system in one query. Fetching them per row cost two round trips each
        # -- the hypothesis, then its lazy-loaded system -- on the one path built for bulk output.
        seed_ids = {label.seed_hypothesis_id for _, label in rows if label.seed_hypothesis_id}
        seed_systems: dict[int, str] = {}
        if seed_ids:
            seed_systems = dict(
                session.execute(
                    sa.select(AsrHypothesis.id, AsrSystem.system_id)
                    .join(AsrSystem, AsrSystem.id == AsrHypothesis.asr_system_id)
                    .where(AsrHypothesis.id.in_(seed_ids))
                ).all()
            )

        spanning_episode_ids = set(
            session.scalars(
                sa.select(Segment.episode_id)
                .group_by(Segment.episode_id)
                .having(sa.func.count(sa.distinct(Segment.pot)) > 1)
            )
        )

        for segment, label in rows:
            records.append(
                _record(
                    segment,
                    label,
                    version,
                    definition,
                    seed_systems.get(label.seed_hypothesis_id),
                    ruleset,
                    spanning_episode_ids,
                )
            )
            if segment.import_run_id:
                import_run_ids.add(segment.import_run_id)
            if segment.episode.audio_object_key:
                episode_audio[segment.episode.external_id] = segment.episode.audio_object_key

    # The gold pot's whole value is that every row in it was checked by a human. `record_decision`
    # refuses to write a screened gold label in the first place and `set_segment_pot` refuses to
    # move a screened clip into gold, so reaching here means something got in another way -- a
    # direct insert or update that went round both.
    # Refusing the export is the only response that cannot end with a screened row being cited as
    # a benchmark result (D63).
    if kind == "gold":
        screened = [r["segment_id"] for r in records if r["verification_tier"] != "verified"]
        if screened:
            raise GoldPurityError(
                f"{len(screened)} gold row(s) are not verified, first {screened[0]!r};"
                " the gold export is a claim that every row was listened to, so it will not"
                " be written until these are re-verified or taken out of the gold pot"
            )

    with data_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    audio_files = _write_episode_audio(output_dir, episode_audio, storage)

    counts_by_split: dict[str, int] = {}
    #: ``{split: {tier: rows}}``. Reported rather than summarised to a single number because the
    #: interesting question is not "how much was screened" but "was the *test* split screened".
    tiers_by_split: dict[str, dict[str, int]] = {}
    for record in records:
        counts_by_split[record["split"]] = counts_by_split.get(record["split"], 0) + 1
        tiers = tiers_by_split.setdefault(record["split"], {})
        tier = record["verification_tier"]
        tiers[tier] = tiers.get(tier, 0) + 1

    runs = (
        session.scalars(sa.select(ImportRun).where(ImportRun.id.in_(import_run_ids))).all()
        if import_run_ids
        else []
    )
    manifest = {
        "kind": kind,
        "label_version": version_name,
        "policy_version": version.policy_version if version else settings.labels.policy_version,
        #: Which orthographic table the `text` field was written through (D64). A consumer
        #: evaluating against this corpus must apply the same one to its own references, or
        #: spelling disagreement is counted as word error.
        "normalization_version": ruleset.version,
        "filters": {
            "splits": list(definition.splits),
            "dispositions": list(definition.dispositions),
            "episode": episode,
        },
        "row_count": len(records),
        "row_counts_by_split": dict(sorted(counts_by_split.items())),
        "verification_tiers_by_split": {
            split: dict(sorted(tiers.items())) for split, tiers in sorted(tiers_by_split.items())
        },
        "files": [
            {
                "name": data_path.name,
                "sha256": sha256_file(data_path).removeprefix("sha256:"),
                "bytes": data_path.stat().st_size,
            },
            *audio_files,
        ],
        "exported_at": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": git_commit(),
        "import_runs": [
            {
                "id": run.id,
                "source_path": run.source_path,
                "pipeline_version": run.pipeline_version,
                "pipeline_commit": run.pipeline_commit,
            }
            for run in sorted(runs, key=lambda r: r.id)
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if kind == "analytics" and records:
        try:
            from app.services.alignment import run_cross_verification_on_records

            verif_summary = run_cross_verification_on_records(records)
            verif_path = output_dir / "timestamp_verification_report.json"
            verif_path.write_text(
                json.dumps(asdict(verif_summary), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            logger.info("timestamp_verification_report_written", path=str(verif_path))
        except Exception as exc:
            logger.warning("timestamp_verification_failed", error=str(exc))

    logger.info("export_complete", kind=kind, rows=len(records), directory=str(output_dir))
    return ExportResult(
        kind=kind,
        output_dir=output_dir,
        data_path=data_path,
        manifest_path=manifest_path,
        row_count=len(records),
        row_counts_by_split=dict(sorted(counts_by_split.items())),
    )
