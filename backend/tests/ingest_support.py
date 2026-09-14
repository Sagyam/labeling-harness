"""Helpers shared by the ingest test modules."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.services.ingest import IngestJob, manager


def make_test_audio(path: Path, duration_seconds: float = 6.0, sample_rate: int = 16000) -> Path:
    """Generate a test audio file with alternating bursts of tone and silence."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    # 440 Hz tone
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Insert 0.5s silence gap in the middle
    mid_start = int(2.5 * sample_rate)
    mid_end = int(3.0 * sample_rate)
    audio[mid_start:mid_end] = 0.0

    sf.write(str(path), audio, sample_rate, format="WAV")
    return path


def _pipeline_job(tmp_path: Path, name: str, *, seconds: float = 45.0) -> IngestJob:
    """A job whose audio is long enough to cut into several segments.

    45 s against ``MAX_SEG_SECONDS = 20`` gives three, which is the minimum that can show a run
    surviving a discard -- with one segment there is nothing left to keep.
    """
    return IngestJob(
        job_id=f"discard-{name}",
        episode_id=f"web_{name}",
        show_id="podcast",
        title=f"Discard Test {name}",
        audio_path=make_test_audio(tmp_path / f"{name}_raw.wav", duration_seconds=seconds),
        work_dir=tmp_path / f"work_{name}",
    )


def _drain(timeout: float = 5.0) -> None:
    """Block until the queue is empty and nothing is running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with manager._lock:
            if manager._pending.empty() and manager._running_id is None and not manager._waiting:
                return
        time.sleep(0.01)
    raise AssertionError("ingestion queue did not drain")


def _queued_job(episode_id: str, tmp_path: Path) -> IngestJob:
    work_dir = tmp_path / episode_id
    work_dir.mkdir(parents=True, exist_ok=True)
    return manager.create_job(
        episode_id=episode_id,
        show_id="demo",
        title=episode_id,
        work_dir=work_dir,
        audio_path=work_dir / "source.wav",
    )
