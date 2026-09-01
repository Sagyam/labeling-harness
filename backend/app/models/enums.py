"""Allowed values for the schema's three status fields and their neighbours.

These are enforced as database CHECK constraints rather than native Postgres enums: a CHECK is
equally strict and far cheaper to extend in a migration.
"""

from __future__ import annotations

from typing import Final, Literal

Split = Literal["train", "val", "test", "unassigned"]
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test", "unassigned")

PipelineStatus = Literal["imported", "queued", "labeled", "excluded"]
PIPELINE_STATUSES: Final[tuple[str, ...]] = ("imported", "queued", "labeled", "excluded")

QueueName = Literal["review", "audit", "error"]
QUEUE_NAMES: Final[tuple[str, ...]] = ("review", "audit", "error")

TaskStatus = Literal["pending", "in_progress", "done", "skipped"]
TASK_STATUSES: Final[tuple[str, ...]] = ("pending", "in_progress", "done", "skipped")
ACTIVE_TASK_STATUSES: Final[tuple[str, ...]] = ("pending", "in_progress")

Disposition = Literal["accepted_unchanged", "edited", "unusable_audio", "uncertain"]
DISPOSITIONS: Final[tuple[str, ...]] = (
    "accepted_unchanged",
    "edited",
    "unusable_audio",
    "uncertain",
)
#: Dispositions that carry a usable human transcript.
APPROVED_DISPOSITIONS: Final[tuple[str, ...]] = ("accepted_unchanged", "edited")

EventAction = Literal["accept", "edit", "skip", "flag", "reopen"]
EVENT_ACTIONS: Final[tuple[str, ...]] = ("accept", "edit", "skip", "flag", "reopen")

ImportStatus = Literal["running", "succeeded", "failed", "dry_run"]
IMPORT_STATUSES: Final[tuple[str, ...]] = ("running", "succeeded", "failed", "dry_run")


def check_in(column: str, values: tuple[str, ...]) -> str:
    """Render a SQL ``IN`` predicate for a CHECK constraint."""
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({rendered})"
