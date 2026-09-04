"""Cost aggregation and analytics service for external AI providers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.llm.cost import (
    MODEL_PRICING_CATALOG,
    estimate_request_cost,
    vendor_for_route_or_model,
)
from app.models import LlmRequest


def _row_cost(row: LlmRequest) -> float:
    """Return the row's cost in float USD, estimating if None."""
    if row.estimated_cost_usd is not None:
        return float(row.estimated_cost_usd)
    est = estimate_request_cost(
        route=row.route,
        model=row.model,
        status=row.status,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
    )
    return float(est)


def collect_cost_report(session: Session) -> dict[str, Any]:
    """Compile comprehensive cost analytics across ElevenLabs, OpenRouter, and Vertex AI."""
    # Query all requests ordered by time
    rows = list(session.scalars(sa.select(LlmRequest).order_by(LlmRequest.created_at.asc())))

    total_cost = 0.0
    total_requests = len(rows)
    successful_requests = 0
    failed_requests = 0
    dry_run_requests = 0
    latencies: list[int] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Grouping structures
    vendor_stats: dict[str, dict[str, Any]] = {
        "ElevenLabs": {
            "vendor": "ElevenLabs",
            "cost_usd": 0.0,
            "percentage": 0.0,
            "requests": 0,
            "successful": 0,
            "failed": 0,
            "dry_run": 0,
            "latencies": [],
        },
        "OpenRouter": {
            "vendor": "OpenRouter",
            "cost_usd": 0.0,
            "percentage": 0.0,
            "requests": 0,
            "successful": 0,
            "failed": 0,
            "dry_run": 0,
            "latencies": [],
        },
        "Google Cloud Vertex AI": {
            "vendor": "Google Cloud Vertex AI",
            "cost_usd": 0.0,
            "percentage": 0.0,
            "requests": 0,
            "successful": 0,
            "failed": 0,
            "dry_run": 0,
            "latencies": [],
        },
    }

    model_stats: dict[str, dict[str, Any]] = {}
    daily_spend: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"cost_usd": 0.0, "requests": 0, "by_vendor": defaultdict(float)}
    )

    for row in rows:
        cost = _row_cost(row)
        total_cost += cost
        vendor = vendor_for_route_or_model(row.route, row.model)

        if vendor not in vendor_stats:
            vendor_stats[vendor] = {
                "vendor": vendor,
                "cost_usd": 0.0,
                "percentage": 0.0,
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "dry_run": 0,
                "latencies": [],
            }

        v_data = vendor_stats[vendor]
        v_data["requests"] += 1
        v_data["cost_usd"] += cost

        if row.status == "succeeded":
            successful_requests += 1
            v_data["successful"] += 1
            if row.latency_ms is not None:
                latencies.append(row.latency_ms)
                v_data["latencies"].append(row.latency_ms)
        elif row.status == "failed":
            failed_requests += 1
            v_data["failed"] += 1
        elif row.status == "dry_run":
            dry_run_requests += 1
            v_data["dry_run"] += 1

        p_tok = row.prompt_tokens or 0
        c_tok = row.completion_tokens or 0
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

        # Model / Route key
        route_key = row.route
        if route_key not in model_stats:
            # Look up catalog rate display if available
            catalog_match = next(
                (
                    c
                    for c in MODEL_PRICING_CATALOG
                    if c["route"] == route_key or c["model"] == row.model
                ),
                None,
            )
            rate_display = catalog_match["effective_rate_display"] if catalog_match else None
            model_stats[route_key] = {
                "route": route_key,
                "model": row.model or route_key,
                "vendor": vendor,
                "cost_usd": 0.0,
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "dry_run": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latencies": [],
                "effective_rate_display": rate_display,
            }

        m_data = model_stats[route_key]
        m_data["requests"] += 1
        m_data["cost_usd"] += cost
        m_data["prompt_tokens"] += p_tok
        m_data["completion_tokens"] += c_tok
        if row.status == "succeeded":
            m_data["successful"] += 1
            if row.latency_ms is not None:
                m_data["latencies"].append(row.latency_ms)
        elif row.status == "failed":
            m_data["failed"] += 1
        elif row.status == "dry_run":
            m_data["dry_run"] += 1

        # Daily timeline
        day_str = row.created_at.strftime("%Y-%m-%d")
        d_point = daily_spend[day_str]
        d_point["cost_usd"] += cost
        d_point["requests"] += 1
        d_point["by_vendor"][vendor] += cost

    # Calculate averages and percentages
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None

    vendor_breakdown = []
    for _v_name, v_data in vendor_stats.items():
        v_lats = v_data.pop("latencies")
        v_data["average_latency_ms"] = round(sum(v_lats) / len(v_lats), 1) if v_lats else None
        v_data["cost_usd"] = round(v_data["cost_usd"], 6)
        v_data["percentage"] = (v_data["cost_usd"] / total_cost * 100) if total_cost > 0 else 0.0
        v_data["percentage"] = round(v_data["percentage"], 1)
        vendor_breakdown.append(v_data)

    model_breakdown = []
    for _m_key, m_data in model_stats.items():
        m_lats = m_data.pop("latencies")
        m_data["average_latency_ms"] = round(sum(m_lats) / len(m_lats), 1) if m_lats else None
        m_data["cost_usd"] = round(m_data["cost_usd"], 6)
        model_breakdown.append(m_data)

    timeline_points = [
        {
            "date": d,
            "cost_usd": round(vals["cost_usd"], 6),
            "requests": vals["requests"],
            "by_vendor": {k: round(v, 6) for k, v in vals["by_vendor"].items()},
        }
        for d, vals in sorted(daily_spend.items())
    ]

    return {
        "summary": {
            "total_cost_usd": round(total_cost, 6),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "dry_run_requests": dry_run_requests,
            "average_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        },
        "vendor_breakdown": vendor_breakdown,
        "model_breakdown": model_breakdown,
        "daily_timeline": timeline_points,
        "pricing_catalog": MODEL_PRICING_CATALOG,
    }


def query_cost_requests(
    session: Session,
    *,
    vendor: str | None = None,
    route: str | None = None,
    status: str | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    """Query paginated request records with optional search and filters."""
    query = sa.select(LlmRequest)

    if route:
        query = query.where(LlmRequest.route == route)
    if status:
        query = query.where(LlmRequest.status == status)
    if search:
        term = f"%{search.strip()}%"
        query = query.where(
            sa.or_(
                LlmRequest.route.ilike(term),
                LlmRequest.model.ilike(term),
                LlmRequest.input_summary.ilike(term),
                LlmRequest.error_message.ilike(term),
            )
        )

    # Fetch rows to filter by vendor if needed and count
    all_rows = list(session.scalars(query.order_by(LlmRequest.created_at.desc())))

    if vendor:
        all_rows = [r for r in all_rows if vendor_for_route_or_model(r.route, r.model) == vendor]

    total = len(all_rows)
    page = all_rows[offset : offset + limit]

    items = []
    for r in page:
        cost = _row_cost(r)
        items.append(
            {
                "id": r.id,
                "route": r.route,
                "model": r.model,
                "vendor": vendor_for_route_or_model(r.route, r.model),
                "status": r.status,
                "estimated_cost_usd": round(cost, 6),
                "latency_ms": r.latency_ms,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "input_summary": r.input_summary,
                "error_message": r.error_message,
                "created_at": r.created_at,
            }
        )

    return total, items
