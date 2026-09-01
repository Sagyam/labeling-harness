"""Reading and validating an upstream export directory.

The manifest is the entire boundary between the GPU pipeline and this harness: everything the
harness knows arrives through it. A malformed manifest must therefore fail loudly *before* any row
is written, which is why validation happens here, over the whole directory, rather than row by row
inside the importer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
EPISODE_FILENAME = "episode.json"
SEGMENTS_FILENAME = "segments.jsonl"
CLIPS_DIRNAME = "clips"
PEAKS_DIRNAME = "peaks"


class ManifestError(ValueError):
    """The export directory is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Manifest:
    """A validated export directory, held in memory before any database write."""

    root: Path
    episode: dict[str, Any]
    segments: list[dict[str, Any]]

    @property
    def episode_id(self) -> str:
        """The manifest's episode id."""
        return str(self.episode["episode_id"])

    def clip_path(self, segment: dict[str, Any]) -> Path:
        """Absolute path to a segment's clip."""
        return self.root / str(segment["clip_path"])

    def peaks_path(self, segment: dict[str, Any]) -> Path | None:
        """Absolute path to a segment's precomputed peaks, if the pipeline supplied them."""
        explicit = segment.get("peaks_path")
        if explicit:
            candidate = self.root / str(explicit)
        else:
            candidate = self.root / PEAKS_DIRNAME / f"{segment['segment_id']}.json"
        return candidate if candidate.is_file() else None


@lru_cache(maxsize=4)
def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _validate(instance: dict[str, Any], schema_name: str, source: str) -> None:
    errors = sorted(_validator(schema_name).iter_errors(instance), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.path) or "<root>"
        raise ManifestError(f"{source}: schema violation at {location}: {first.message}")


def validate_episode(episode: dict[str, Any], *, source: str) -> None:
    """Validate an ``episode.json`` object.

    Raises:
        ManifestError: The object violates ``episode.schema.json``.
    """
    _validate(episode, "episode.schema.json", source)


def validate_segment(segment: dict[str, Any], *, source: str) -> None:
    """Validate one ``segments.jsonl`` record, including cross-field rules.

    Raises:
        ManifestError: The record violates ``segment.schema.json`` or its own time bounds.
    """
    _validate(segment, "segment.schema.json", source)
    if float(segment["end_time"]) <= float(segment["start_time"]):
        raise ManifestError(
            f"{source}: end_time ({segment['end_time']}) must be greater than "
            f"start_time ({segment['start_time']})"
        )
    system_ids = [h["system_id"] for h in segment["hypotheses"]]
    duplicates = {s for s in system_ids if system_ids.count(s) > 1}
    if duplicates:
        raise ManifestError(
            f"{source}: duplicate system_id in hypotheses: {', '.join(sorted(duplicates))}"
        )


def read_manifest(root: Path | str) -> Manifest:
    """Read and fully validate an export directory.

    Args:
        root: The ``export_<episode_id>/`` directory.

    Returns:
        The validated manifest. Nothing has been written anywhere.

    Raises:
        ManifestError: Any structural, schema or consistency problem, naming the offending file
            and line.
    """
    root = Path(root)
    if not root.is_dir():
        raise ManifestError(f"{root} is not a directory")

    episode_path = root / EPISODE_FILENAME
    segments_path = root / SEGMENTS_FILENAME
    if not episode_path.is_file():
        raise ManifestError(f"{root}: missing {EPISODE_FILENAME}")
    if not segments_path.is_file():
        raise ManifestError(f"{root}: missing {SEGMENTS_FILENAME}")

    try:
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{episode_path}: invalid JSON ({exc})") from exc
    if not isinstance(episode, dict):
        raise ManifestError(f"{episode_path}: must contain a JSON object")
    validate_episode(episode, source=str(episode_path))
    episode_id = str(episode["episode_id"])

    segments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, raw in enumerate(segments_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        source = f"{segments_path}: line {number}"
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{source}: invalid JSON ({exc})") from exc
        if not isinstance(record, dict):
            raise ManifestError(f"{source}: must be a JSON object")
        validate_segment(record, source=source)

        segment_id = str(record["segment_id"])
        if segment_id in seen:
            raise ManifestError(f"{source}: duplicate segment_id {segment_id!r}")
        seen.add(segment_id)
        if str(record["episode_id"]) != episode_id:
            raise ManifestError(
                f"{source}: segment belongs to episode {record['episode_id']!r}, "
                f"but {EPISODE_FILENAME} declares {episode_id!r}"
            )
        segments.append(record)

    if not segments:
        raise ManifestError(f"{segments_path}: contains no segments")
    return Manifest(root=root, episode=episode, segments=segments)
