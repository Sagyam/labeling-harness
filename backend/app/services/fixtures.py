"""Builder for synthetic upstream export directories.

The GPU pipeline is out of scope for this repository, so development and tests need something that
produces exactly what it produces: an ``export_<episode_id>/`` directory with a valid manifest and
real 16 kHz mono FLAC clips. Everything is derived from a seed, so a fixture is byte-reproducible.

The knobs that produce *invalid* exports (a WAV clip, the wrong sample rate, a segment with no
hypotheses) exist so the importer's rejection paths can be tested against real files rather than
mocks.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.services.audio import compute_peaks
from app.services.corpus import SYSTEMS, perturb, sentence_for
from app.utils.hashing import sha256_bytes, sha256_file

DEFAULT_EPISODE_ID = "fixture-show_ep000"


def _clip_rng(seed: int, segment_id: str) -> np.random.Generator:
    """A per-segment random source.

    Clip audio must depend only on ``(seed, segment_id)``. If it drew from the shared stream, an
    unrelated knob such as the number of systems would shift the stream and change every clip's
    bytes -- which the importer would then correctly reject as a changed checksum, making
    incremental fixtures impossible to write.
    """
    digest = hashlib.blake2b(f"{seed}:{segment_id}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _write_clip(
    path: Path,
    *,
    seconds: float,
    rng: np.random.Generator,
    sample_rate: int,
    clip_format: str,
    channels: int,
) -> None:
    """Write a deterministic speech-ish waveform: a wandering tone plus shaped noise."""
    frames = max(1, int(seconds * sample_rate))
    t = np.arange(frames, dtype=np.float64) / sample_rate
    base = float(rng.uniform(90.0, 240.0))
    wobble = np.sin(2 * np.pi * float(rng.uniform(1.5, 4.0)) * t)
    tone = np.sin(2 * np.pi * (base + 20 * wobble) * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * float(rng.uniform(0.5, 2.0)) * t)
    noise = rng.uniform(-1.0, 1.0, size=frames)
    data = 0.55 * tone * envelope + 0.05 * noise
    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate, format=clip_format)


def build_export_fixture(
    root: Path | str,
    *,
    episode_id: str = DEFAULT_EPISODE_ID,
    segments: int = 5,
    systems: int = 3,
    seed: int = 1234,
    with_peaks: bool = False,
    with_words: bool = True,
    with_scores: bool = True,
    clip_format: str = "FLAC",
    sample_rate: int = 16000,
    channels: int = 1,
    empty_hypothesis_segments: Sequence[int] = (),
) -> Path:
    """Write a synthetic export directory and return its path.

    Every segment is derived from ``(seed, segment_id)`` alone, so growing an export -- more
    segments, more systems -- leaves the existing segments untouched.

    Args:
        root: Directory to create. Existing contents are left alone; files are overwritten.
        episode_id: Manifest episode id, also the prefix of every segment id.
        segments: Number of segments to emit.
        systems: Number of ASR systems per segment (at most five are defined).
        seed: Random seed. The same seed yields byte-identical output.
        with_peaks: Also write ``peaks/<segment_id>.json``, as an upstream that precomputes them.
        with_words: Include word-level timings on hypotheses.
        with_scores: Include the ``scores`` block.
        clip_format: ``FLAC`` normally; ``WAV`` to exercise the importer's rejection path.
        sample_rate: Clip sample rate; anything but 16000 must be rejected at import.
        channels: Clip channel count; anything but 1 must be rejected at import.
        empty_hypothesis_segments: Indices that get zero hypotheses, for the error queue.

    Returns:
        The export directory path.
    """
    root = Path(root)
    (root / "clips").mkdir(parents=True, exist_ok=True)
    extension = "flac" if clip_format.upper() == "FLAC" else clip_format.lower()

    records: list[dict[str, Any]] = []
    cursor = 0.0
    for index in range(segments):
        segment_id = f"{episode_id}_{index:04d}"
        # Everything about a segment is derived from (seed, segment_id) alone, so adding systems
        # or dropping word timings leaves the other segments byte-identical.
        rng = random.Random(f"{seed}:{segment_id}")
        duration = round(rng.uniform(1.5, 16.0), 2)
        start, cursor = cursor, cursor + duration + round(rng.uniform(0.1, 1.2), 2)

        clip_relative = f"clips/{segment_id}.{extension}"
        clip_path = root / clip_relative
        _write_clip(
            clip_path,
            seconds=duration,
            rng=_clip_rng(seed, segment_id),
            sample_rate=sample_rate,
            clip_format=clip_format.upper(),
            channels=channels,
        )

        reference = sentence_for(rng)
        difficulty = rng.random()
        hypotheses: list[dict[str, Any]] = []
        if index not in set(empty_hypothesis_segments):
            for rank, (system_id, model_id) in enumerate(SYSTEMS[:systems]):
                text = perturb(reference, rng, strength=difficulty * min(rank, 2) / 2)
                hypothesis: dict[str, Any] = {
                    "system_id": system_id,
                    "model_id": model_id,
                    "text": text,
                    "avg_logprob": round(-0.1 - difficulty * rng.uniform(0.2, 1.8), 4),
                    "no_speech_prob": round(rng.random() ** 4, 4),
                }
                if with_words:
                    tokens = text.split()
                    step = duration / max(len(tokens), 1)
                    hypothesis["words"] = [
                        {
                            "word": token,
                            "start": round(start + position * step, 3),
                            "end": round(start + (position + 1) * step, 3),
                            "confidence": round(rng.uniform(0.5, 1.0), 3),
                            "predicted_language": "en" if token.isascii() else "ne",
                            "predicted_script": "latin" if token.isascii() else "devanagari",
                        }
                        for position, token in enumerate(tokens)
                    ]
                hypotheses.append(hypothesis)

        record: dict[str, Any] = {
            "segment_id": segment_id,
            "episode_id": episode_id,
            "speaker_id": f"SPEAKER_{rng.randrange(2):02d}",
            "start_time": round(start, 3),
            "end_time": round(start + duration, 3),
            "clip_path": clip_relative,
            "clip_checksum": sha256_file(clip_path),
            "p_en": round(rng.random(), 3),
            "lid": rng.choice(["ne", "en", "mixed"]),
            "hypotheses": hypotheses,
        }
        if with_scores:
            record["scores"] = {
                "cer_between_hypotheses": round(difficulty * rng.uniform(0.0, 0.5), 4),
                "word_disagreement_rate": round(difficulty * rng.uniform(0.0, 0.9), 4),
                "script_conflict_rate": round(difficulty * rng.uniform(0.0, 0.3), 4),
                "code_switch_density": round(rng.random(), 4),
            }
            record["flags"] = ["low_confidence"] if difficulty > 0.8 else []

        if with_peaks:
            peaks_path = root / "peaks" / f"{segment_id}.json"
            peaks_path.parent.mkdir(parents=True, exist_ok=True)
            peaks_path.write_text(json.dumps(compute_peaks(clip_path, 200)), encoding="utf-8")

        records.append(record)

    episode = {
        "episode_id": episode_id,
        "show_id": episode_id.split("_")[0],
        "title": f"Synthetic fixture {episode_id}",
        "source_uri": f"https://example.invalid/{episode_id}",
        "published_at": dt.date(2026, 1, 1).isoformat(),
        "source_audio_checksum": sha256_bytes(episode_id.encode()),
        "duration_seconds": round(cursor, 3),
        "pipeline_version": "fixture-v1",
        "pipeline_commit": "0000000",
    }
    (root / "episode.json").write_text(json.dumps(episode, indent=2), encoding="utf-8")
    (root / "segments.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return root
