"""Cut one downloaded recording into the distillation corpus (D101), as ingest cuts an episode.

Stage 1's loudness normalisation to 16 kHz mono FLAC, then Silero VAD and the same 2-20 s
slicing, so a student trains on clips shaped like the labelled ones. A source is written to
``<id>.partial`` and renamed when complete: an interrupted run redoes at most the source it was
on, and a finished source is never cut again. Nothing here touches Postgres.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from app.services.distill_corpus import (
    ClipSpan,
    ScreenResult,
    Source,
    SourceInput,
    clip_rows,
    read_source,
    refusal,
    screen_source,
    source_dir,
    window_starts,
)
from app.services.ingest.audio import normalize_audio
from app.services.silero_vad import (
    MAX_SEG_SECONDS,
    MIN_SEG_SECONDS,
    SileroVAD,
    extract_clips,
    segment_audio_to_slices,
)
from app.utils.hashing import sha256_file


@dataclass
class PrepareResult:
    """What happened to one source: ``prepared``, ``done`` (already cut) or ``refused``."""

    status: str
    source: Source
    clips: int = 0
    speech_seconds: float = 0.0
    reason: str | None = None
    seconds_taken: float = 0.0


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]], *, append: bool = False) -> None:
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_source(
    inp: SourceInput,
    *,
    root: Path,
    known_video_ids: set[str],
    blocked_channels: Sequence[str],
    vad: SileroVAD,
    workers: int,
) -> PrepareResult:
    """Cut one recording into ``root/sources/<source_id>/``, unless it is done or refused.

    Args:
        inp: The audio and its info JSON.
        root: The corpus root (``distill.root``).
        known_video_ids: Video ids already episodes in the harness (D101's first layer).
        blocked_channels: ``distill.blocked_channels`` (the second layer).
        vad: The VAD, loaded once for the whole run.
        workers: Threads for writing clips.
    """
    started = time.perf_counter()
    info = json.loads(inp.info.read_text(encoding="utf-8")) if inp.info else None
    source = read_source(inp.audio, info, audio_sha256=sha256_file(inp.audio))
    final = root / source_dir(source.source_id)
    if final.is_dir():
        return PrepareResult("done", source)

    reason = refusal(source, known_video_ids=known_video_ids, blocked_channels=blocked_channels)
    if reason:
        root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            root / "refused.jsonl",
            [{**source.to_json(), "reason": reason, "at": _now()}],
            append=True,
        )
        return PrepareResult("refused", source, reason=reason)

    partial = final.with_name(final.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    normalized = partial / "normalized.flac"
    duration = normalize_audio(inp.audio, normalized)
    audio, sample_rate = sf.read(str(normalized), dtype="float32")
    turns = vad.detect_turns(audio, sample_rate=sample_rate)
    slices = segment_audio_to_slices(turns, duration, audio=audio, sample_rate=sample_rate)
    segments = extract_clips(
        normalized, slices, source.source_id, partial / "clips", max_workers=workers
    )
    normalized.unlink()

    spans = [ClipSpan(s.segment_id, s.start_time, s.end_time, s.clip_checksum) for s in segments]
    rows = clip_rows(source, spans)
    speech = round(sum(r["duration"] for r in rows), 3)
    _write_jsonl(partial / "clips.jsonl", rows)
    meta = {
        **source.to_json(),
        "normalized_duration": round(duration, 3),
        "clips": len(rows),
        "speech_seconds": speech,
        "cut": {
            "min_seg": MIN_SEG_SECONDS,
            "max_seg": MAX_SEG_SECONDS,
            "vad": "silero" if vad.available else "energy",
        },
        "prepared_at": _now(),
    }
    (partial / "source.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    partial.rename(final)
    return PrepareResult(
        "prepared", source, len(rows), speech, seconds_taken=time.perf_counter() - started
    )


# --- the voiceprint screen and the corpus manifest ----------------------------------------------


class Embedder(Protocol):
    """What the screen needs of the voiceprint model (``voiceprint.VoiceEmbedder``)."""

    def embed(self, chunks: Sequence[np.ndarray]) -> list[np.ndarray | None]: ...


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def screen_folder(
    folder: Path,
    gold: Mapping[str, np.ndarray],
    embedder: Embedder,
    *,
    threshold: float,
    window_seconds: float,
    min_seconds: float,
    batch: int = 256,
) -> ScreenResult:
    """Screen one prepared source against gold's voices and write its ``screen.json``.

    Every clip is cut into non-overlapping windows of ``window_seconds`` and embedded; nothing is
    diarized, so a window can hold more than one voice, which the threshold was measured with
    (D101, findings.md).
    """
    keys: list[tuple[str, float]] = []
    chunks: list[np.ndarray] = []
    for row in _read_jsonl(folder / "clips.jsonl"):
        audio, sample_rate = sf.read(
            str(folder / "clips" / f"{row['segment_id']}.flac"), dtype="float32"
        )
        width = round(window_seconds * sample_rate)
        for start in window_starts(len(audio) / sample_rate, window_seconds):
            i = round(start * sample_rate)
            keys.append((row["segment_id"], start))
            chunks.append(audio[i : i + width])
    vectors: list[np.ndarray | None] = []
    for i in range(0, len(chunks), batch):
        vectors += embedder.embed(chunks[i : i + batch])
    windows = [(k[0], k[1], v) for k, v in zip(keys, vectors, strict=True)]
    result = screen_source(
        windows, gold, threshold=threshold, window_seconds=window_seconds, min_seconds=min_seconds
    )
    record = {
        **asdict(result),
        "threshold": threshold,
        "window_seconds": window_seconds,
        "min_seconds": min_seconds,
        "windows_screened": len(windows),
        "gold_voices": len(gold),
        "screened_at": _now(),
    }
    (folder / "screen.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return result


@dataclass
class ManifestReport:
    """What ``build_manifest`` wrote: cleared clips and hours, and the sources held back."""

    clips: int = 0
    hours: float = 0.0
    quarantined: list[str] = field(default_factory=list)
    unscreened: list[str] = field(default_factory=list)


def build_manifest(root: Path) -> ManifestReport:
    """Write ``root/clips.jsonl`` from every source the screen cleared, and only those.

    A source not yet screened, or quarantined, stays out of the corpus (D101).
    """
    report = ManifestReport()
    rows: list[dict[str, Any]] = []
    for folder in sorted(p for p in (root / "sources").glob("*") if p.is_dir()):
        if folder.name.endswith(".partial"):
            continue
        screen = folder / "screen.json"
        if not screen.is_file():
            report.unscreened.append(folder.name)
            continue
        if json.loads(screen.read_text(encoding="utf-8"))["verdict"] != "clear":
            report.quarantined.append(folder.name)
            continue
        rows += _read_jsonl(folder / "clips.jsonl")
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "clips.jsonl", rows)
    report.clips = len(rows)
    report.hours = round(sum(r["duration"] for r in rows) / 3600, 3)
    return report
