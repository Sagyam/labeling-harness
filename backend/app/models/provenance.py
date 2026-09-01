"""Import provenance: one row per import invocation."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utc_now_column, utc_optional_column
from app.models.enums import IMPORT_STATUSES, check_in

if TYPE_CHECKING:
    from app.models.content import Segment


class ImportRun(Base):
    """A single run of the manifest importer, successful or not."""

    __tablename__ = "import_runs"
    __table_args__ = (CheckConstraint(check_in("status", IMPORT_STATUSES), name="status_allowed"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_version: Mapped[str | None] = mapped_column(String(64))
    pipeline_commit: Mapped[str | None] = mapped_column(String(64))

    segments_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    segments_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hypotheses_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[dt.datetime] = utc_now_column()
    finished_at: Mapped[dt.datetime | None] = utc_optional_column()
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    notes: Mapped[str | None] = mapped_column(Text)

    segments: Mapped[list[Segment]] = relationship(back_populates="import_run")
