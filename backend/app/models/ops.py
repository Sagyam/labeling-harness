"""Operational tables: audit log, transliteration cache and the OpenRouter request log."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from sqlalchemy import BigInteger, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utc_now_column
from app.models.content import JsonB


class AuditLog(Base):
    """Append-only record of every write the API performs."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_entity_type_entity_id", "entity_type", "entity_id"),
        Index("ix_audit_logs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    old_values_jsonb: Mapped[dict[str, Any] | None] = mapped_column(JsonB)
    new_values_jsonb: Mapped[dict[str, Any] | None] = mapped_column(JsonB)
    created_at: Mapped[dt.datetime] = utc_now_column()


class TranslitCacheEntry(Base):
    """Cached Latin -> Devanagari candidates.

    Also the by-product worth keeping: a romanization lexicon for this speaker community.
    """

    __tablename__ = "translit_cache"

    latin_token: Mapped[str] = mapped_column(String(128), primary_key=True)
    candidates_jsonb: Mapped[list[str]] = mapped_column(JsonB, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[dt.datetime] = utc_now_column()
    updated_at: Mapped[dt.datetime] = utc_now_column(onupdate=lambda: dt.datetime.now(dt.UTC))


class LlmRequest(Base):
    """One OpenRouter request, logged for billing control. No route is wired at MVP."""

    __tablename__ = "llm_requests"
    __table_args__ = (
        Index("ix_llm_requests_route_created_at", "route", "created_at"),
        Index("ix_llm_requests_request_hash", "request_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    route: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str | None] = mapped_column(String(128))
    request_hash: Mapped[str | None] = mapped_column(String(64))
    input_summary: Mapped[str | None] = mapped_column(Text)
    output_json: Mapped[dict[str, Any] | None] = mapped_column(JsonB)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = utc_now_column()
