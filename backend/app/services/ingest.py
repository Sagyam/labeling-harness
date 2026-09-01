"""Backend ingestion service for podcast episodes.

Coordinates the 5-stage ingestion pipeline:
1. Audio normalization via FFmpeg with loudnorm (16 kHz mono FLAC)
2. Utterance segmentation via Silero VAD (2.0s - 20.0s boundaries)
3. Cloud ASR inference via OpenRouter (logged to llm_requests)
4. Orthography-aware token tagging, CMI, and rule flags
5. Manifest generation and direct database import + queue building
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.llm.openrouter import OpenRouterClient
from app.services.analysis import analyze_transcript
from app.services.importer import import_manifest
from app.services.queue_builder import build_queue
from app.services.silero_vad import (
    SileroVAD,
    extract_clips,
    segment_audio_to_slices,
)
from app.storage import get_storage
from app.storage.base import ObjectStorage
from app.utils.hashing import sha256_file
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestLog:
    timestamp: str
    level: str
    message: str


@dataclass
class IngestJob:
    job_id: str
    episode_id: str
    show_id: str
    title: str
    audio_path: Path
    work_dir: Path
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    stage: str = "upload"
    progress: float = 0.0
    active_segments: int = 0
    total_segments: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    logs: list[IngestLog] = field(default_factory=list)
    listeners: list[asyncio.Queue] = field(default_factory=list)

    def log(self, message: str, level: str = "info") -> None:
        ts = dt.datetime.now(dt.UTC).strftime("%H:%M:%S")
        entry = IngestLog(timestamp=ts, level=level, message=message)
        self.logs.append(entry)
        self._emit({"type": "log", "timestamp": ts, "level": level, "message": message})

    def set_progress(
        self, stage: str, progress: float, active_segments: int = 0, total_segments: int = 0
    ) -> None:
        self.stage = stage
        self.progress = round(progress, 1)
        if active_segments:
            self.active_segments = active_segments
        if total_segments:
            self.total_segments = total_segments

        self._emit(
            {
                "type": "progress",
                "stage": self.stage,
                "progress": self.progress,
                "active_segments": self.active_segments,
                "total_segments": self.total_segments,
            }
        )

    def complete(self, summary: dict[str, Any]) -> None:
        self.status = "completed"
        self.stage = "complete"
        self.progress = 100.0
        self.log("Ingestion pipeline finished successfully.", "success")
        self._emit({"type": "complete", "summary": summary, "episode_id": self.episode_id})

    def fail(self, error_message: str) -> None:
        self.status = "failed"
        self.stage = "failed"
        self.error = error_message
        self.log(f"Pipeline error: {error_message}", "error")
        self._emit({"type": "error", "error": error_message})

    def _emit(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event)
        for q in list(self.listeners):
            with contextlib.suppress(Exception):
                q.put_nowait(payload)


class IngestionManager:
    """In-memory manager tracking active and historical ingestion jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}

    def create_job(
        self,
        *,
        episode_id: str,
        show_id: str,
        title: str,
        audio_path: Path,
        work_dir: Path,
    ) -> IngestJob:
        job_id = str(uuid.uuid4())
        job = IngestJob(
            job_id=job_id,
            episode_id=episode_id,
            show_id=show_id,
            title=title,
            audio_path=audio_path,
            work_dir=work_dir,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)


manager = IngestionManager()


def normalize_audio(input_path: Path, output_path: Path) -> float:
    """Stage 1: Normalize audio using FFmpeg with loudnorm filter.

    Converts to 16 kHz mono FLAC. Returns duration in seconds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-23:LRA=7:tp=-2,aresample=16000",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "flac",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err_msg = proc.stderr.decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"FFmpeg normalization failed (exit code {proc.returncode}): {err_msg}")

    info = sf.info(str(output_path))
    return float(info.duration)


def run_pipeline(
    job: IngestJob,
    session_factory: Callable[[], Session],
    storage: ObjectStorage | None = None,
    settings: Settings | None = None,
) -> None:
    """Run all 5 stages synchronously inside background worker thread."""
    settings = settings or get_settings()
    storage = storage or get_storage(settings)

    job.status = "processing"
    job.log(f"Starting ingestion for '{job.title}' ({job.episode_id})")

    norm_flac = job.work_dir / f"{job.episode_id}_normalized.flac"

    # Stage 1: Normalize Audio
    try:
        job.set_progress("normalizing", 5.0)
        job.log("Stage 1/5: Normalizing audio (FFmpeg loudnorm, 16 kHz mono FLAC)...")
        duration = normalize_audio(job.audio_path, norm_flac)
        source_checksum = sha256_file(job.audio_path)
        job.log(f"Audio normalized: {duration:.1f}s ({duration / 60:.1f} min)")
        job.set_progress("normalizing", 20.0)
    except Exception as exc:
        job.fail(f"Stage 1 Audio Normalization failed: {exc}")
        return

    # Stage 2: Silero VAD Segmentation
    try:
        job.set_progress("segmenting", 22.0)
        job.log("Stage 2/5: Detecting speech turns via Silero VAD (2.0s - 20.0s bounds)...")
        vad = SileroVAD()
        audio_data, sr = sf.read(str(norm_flac), dtype="float32")
        turns = vad.detect_turns(audio_data, sample_rate=sr)
        job.log(f"Detected {len(turns)} raw speech turns")

        slices = segment_audio_to_slices(turns, duration)
        job.log(f"Partitioned into {len(slices)} bounded utterances (2.0s - 20.0s)")

        clips_dir = job.work_dir / "clips"
        segments = extract_clips(norm_flac, slices, job.episode_id, clips_dir)
        job.total_segments = len(segments)
        job.active_segments = len(segments)
        job.log(f"Extracted {len(segments)} audio clips to disk")
        job.set_progress(
            "segmenting", 40.0, total_segments=len(segments), active_segments=len(segments)
        )
    except Exception as exc:
        job.fail(f"Stage 2 Silero VAD Segmentation failed: {exc}")
        return

    # Stage 3 & 4: Cloud ASR & Token Analysis
    segment_records: list[dict[str, Any]] = []
    try:
        job.set_progress("transcribing", 42.0)
        job.log(f"Stage 3/5: Cloud ASR inference via OpenRouter for {len(segments)} segments...")

        with session_factory() as session:
            client = OpenRouterClient(session)

            asr_routes = [r for r in client.config.routes if r.startswith("asr")]
            if not asr_routes:
                asr_routes = ["asr"]

            for idx, seg in enumerate(segments):
                step_progress = 40.0 + (35.0 * (idx + 1) / len(segments))
                job.set_progress("transcribing", step_progress, active_segments=idx + 1)

                hypotheses: list[dict[str, Any]] = []
                for route_name in asr_routes:
                    sys_name = route_name.removeprefix("asr_").replace("_", "-") or "openrouter-asr"
                    asr_res = client.transcribe(
                        seg.clip_path,
                        route=route_name,
                        language="ne",
                        prompt=(
                            "यो नेपाली र अंग्रेजी भाषाको कुराकानी हो। "
                            "Transcribe strictly in authentic spoken Nepali (Devanagari) and "
                            "English (Latin). Do not transcribe Nepali words into Hindi."
                        ),
                    )
                    hypotheses.append(
                        {
                            "system_id": sys_name,
                            "model_id": asr_res.model,
                            "text": asr_res.text,
                            "avg_logprob": asr_res.avg_logprob,
                            "no_speech_prob": asr_res.no_speech_prob,
                            "words": asr_res.words,
                        }
                    )

                # Cross-system disagreement calculation
                word_disagreement_rate = 0.0
                cer_between_hyps = 0.0
                if len(hypotheses) >= 2:
                    import difflib

                    w1 = hypotheses[0]["text"].split()
                    w2 = hypotheses[1]["text"].split()
                    matcher = difflib.SequenceMatcher(None, w1, w2)
                    word_disagreement_rate = round(1.0 - matcher.ratio(), 4)

                    c_matcher = difflib.SequenceMatcher(
                        None, hypotheses[0]["text"], hypotheses[1]["text"]
                    )
                    cer_between_hyps = round(1.0 - c_matcher.ratio(), 4)

                primary_hyp = hypotheses[0]
                analysis = analyze_transcript(
                    primary_hyp["text"],
                    duration_seconds=seg.duration,
                    word_disagreement_rate=word_disagreement_rate,
                    avg_logprob=primary_hyp["avg_logprob"],
                    no_speech_prob=primary_hyp["no_speech_prob"],
                    settings=settings,
                )

                if (idx + 1) % 5 == 0 or idx == len(segments) - 1:
                    snippet = primary_hyp["text"][:30]
                    models_str = ", ".join(h["system_id"] for h in hypotheses)
                    msg = (
                        f"[{idx + 1}/{len(segments)}] {seg.segment_id} ({models_str}): "
                        f"'{snippet}...' (CMI={analysis.cmi}%, Disagree={word_disagreement_rate})"
                    )
                    job.log(msg)

                segment_records.append(
                    {
                        "segment_id": seg.segment_id,
                        "episode_id": job.episode_id,
                        "speaker_id": "spk0",
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "clip_path": seg.clip_rel_path,
                        "clip_checksum": seg.clip_checksum,
                        "hypotheses": hypotheses,
                        "scores": {
                            "cmi": analysis.cmi,
                            "code_switch_density": analysis.code_switch_density,
                            "word_disagreement_rate": word_disagreement_rate,
                            "cer_between_hypotheses": cer_between_hyps,
                            "avg_logprob": primary_hyp["avg_logprob"],
                            "flags": analysis.flags,
                        },
                    }
                )
            session.commit()
        job.set_progress("analyzing", 80.0)
        job.log("Stage 4/5: Orthography analysis, CMI and rule flags completed")
    except Exception as exc:
        job.fail(f"Stage 3/4 ASR Inference / Analysis failed: {exc}")
        return

    # Stage 5: Manifest Generation, Direct Import & Queue Building
    try:
        job.set_progress("importing", 85.0)
        job.log("Stage 5/5: Generating manifest and importing directly into database...")

        episode_meta = {
            "episode_id": job.episode_id,
            "show_id": job.show_id,
            "title": job.title,
            "source_uri": f"file://{job.audio_path.name}",
            "published_at": dt.date.today().isoformat(),
            "duration_seconds": duration,
            "source_audio_checksum": source_checksum,
            "pipeline_version": "web_v1",
            "pipeline_commit": "web",
        }

        # Write episode.json
        with open(job.work_dir / "episode.json", "w", encoding="utf-8") as f:
            json.dump(episode_meta, f, indent=2, ensure_ascii=False)

        # Write segments.jsonl
        with open(job.work_dir / "segments.jsonl", "w", encoding="utf-8") as f:
            for rec in segment_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with session_factory() as session:
            import_report = import_manifest(
                session,
                job.work_dir,
                storage=storage,
                settings=settings,
            )
            job.log(
                f"Database import: {import_report.segments_inserted} segments, "
                f"{import_report.clips_uploaded} clips uploaded to storage"
            )

            job.log("Building prioritized annotation queues...")
            queue_report = build_queue(
                session, episode_external_id=job.episode_id, settings=settings
            )
            session.commit()

            msg = (
                f"Queue built: {queue_report.tasks_created} tasks created "
                f"({queue_report.review_tasks} review, {queue_report.audit_tasks} audit, "
                f"{queue_report.error_tasks} error)"
            )
            job.log(msg)

        job.complete(
            {
                "episode_id": job.episode_id,
                "duration_seconds": round(duration, 1),
                "segments": len(segment_records),
                "tasks_created": queue_report.tasks_created,
                "review_tasks": queue_report.review_tasks,
            }
        )
    except Exception as exc:
        job.fail(f"Stage 5 Database Import & Queue Building failed: {exc}")
        return
