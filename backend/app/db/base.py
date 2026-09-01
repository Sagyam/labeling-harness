"""Declarative base and shared column helpers."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def utc_now_column(**kwargs: Any) -> MappedColumn[dt.datetime]:
    """A non-null ``timestamptz`` defaulting to the database's current time.

    Every timestamp in this schema is timezone-aware and stored in UTC.
    """
    return mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, **kwargs
    )


def utc_optional_column(**kwargs: Any) -> MappedColumn[dt.datetime | None]:
    """A nullable ``timestamptz``."""
    return mapped_column(DateTime(timezone=True), nullable=True, **kwargs)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
