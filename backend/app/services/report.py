"""Status report: one command, no SQL knowledge required.

Answers the questions the owner actually asks between sessions -- how much is done, how fast is it
going, when will it finish, and is the upstream pipeline still healthy (accept rate over time).
"""

from __future__ import annotations

import datetime as dt
import html
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import (
    AsrHypothesis,
    Episode,
    HypothesisWord,
    Segment,
    SegmentScore,
)
from app.models.enums import PIPELINE_STATUSES, SPLITS
from app.services.stats import collect_stats, latest_labels_subquery


def collect_report(session: Session) -> dict[str, Any]:
    """Gather every figure the report shows, as plain JSON-serializable data.

    The corpus, label, throughput and queue counters come from :func:`collect_stats`, which is the
    single definition of each of them -- the report adds only the figures the live progress display
    has no use for: the accept rate over time, split balance, score means and word coverage.
    """
    stats = collect_stats(session)

    episodes = stats["episodes"] or 0
    segments = stats["segments"]["total"]
    segments_by_status = {name: stats["segments"][name] for name in PIPELINE_STATUSES}
    labels = {name: count for name, count in stats["labels"].items() if name != "total"}
    labeled_total = stats["labels"]["total"]
    accept_rate = stats["accept_rate"]

    median_seconds = stats["throughput"]["median_seconds_per_segment"]
    annotator_hours = stats["throughput"]["annotator_hours"]
    events = stats["throughput"]["events"]
    segments_per_hour = 3600 / median_seconds if median_seconds else None
    backlog = stats["throughput"]["backlog"]
    queues = stats["queues"]

    # Accept rate over time: the health check on the upstream pipeline. This reads the *current*
    # label per segment, exactly as the headline accept rate above does -- counting superseded
    # labels here would let the trend line disagree with the summary it is meant to explain.
    current = latest_labels_subquery()
    day = sa.func.date_trunc("day", current.c.created_at).label("day")
    accept_rate_by_day = [
        {
            "day": row.day.date().isoformat(),
            "labeled": row.labeled,
            "accepted": row.accepted,
            "accept_rate": round(row.accepted / row.labeled, 4) if row.labeled else None,
        }
        for row in session.execute(
            sa.select(
                day,
                sa.func.count().label("labeled"),
                sa.func.count()
                .filter(current.c.disposition == "accepted_unchanged")
                .label("accepted"),
            )
            .group_by(day)
            .order_by(day)
        )
    ]

    score_means = session.execute(
        sa.select(
            sa.func.avg(SegmentScore.word_disagreement_rate),
            sa.func.avg(SegmentScore.script_conflict_rate),
            sa.func.avg(SegmentScore.code_switch_density),
            sa.func.avg(SegmentScore.cer_between_hypotheses),
        )
    ).one()

    split_balance = {name: {"episodes": 0, "segments": 0, "hours": 0.0} for name in SPLITS}
    for split, episode_count, segment_count, seconds in session.execute(
        sa.select(
            Episode.split,
            sa.func.count(sa.distinct(Episode.id)),
            sa.func.count(Segment.id),
            sa.func.coalesce(sa.func.sum(Segment.duration_seconds), 0.0),
        )
        .select_from(Episode)
        .outerjoin(Segment, Segment.episode_id == Episode.id)
        .group_by(Episode.split)
    ):
        split_balance[split] = {
            "episodes": episode_count,
            "segments": segment_count,
            "hours": round(float(seconds) / 3600, 3),
        }

    hypotheses_total = session.scalar(sa.select(sa.func.count()).select_from(AsrHypothesis)) or 0
    hypotheses_with_words = (
        session.scalar(
            sa.select(sa.func.count(sa.distinct(HypothesisWord.hypothesis_id))).select_from(
                HypothesisWord
            )
        )
        or 0
    )

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "corpus": {
            "episodes": episodes,
            "segments": segments,
            "audio_hours": stats["audio_hours"],
            "segments_by_status": segments_by_status,
            "episode_titles": [
                {"external_id": external_id, "title": title, "split": split}
                for external_id, title, split in session.execute(
                    sa.select(Episode.external_id, Episode.title, Episode.split).order_by(
                        Episode.external_id
                    )
                )
            ],
        },
        "labels": {"total": labeled_total, **labels},
        "accept_rate": round(accept_rate, 4) if accept_rate is not None else None,
        "accept_rate_by_day": accept_rate_by_day,
        "throughput": {
            "median_seconds_per_segment": median_seconds,
            "segments_per_hour": round(segments_per_hour, 1) if segments_per_hour else None,
            "annotator_hours": annotator_hours,
            "events": events,
        },
        "queue": {
            "backlog": backlog,
            "by_queue": queues,
            "projected_hours_to_finish": (
                round(backlog * median_seconds / 3600, 2) if median_seconds and backlog else None
            ),
        },
        "scores": {
            "mean_word_disagreement_rate": _round(score_means[0]),
            "mean_script_conflict_rate": _round(score_means[1]),
            "mean_code_switch_density": _round(score_means[2]),
            "mean_cer_between_hypotheses": _round(score_means[3]),
        },
        "split_balance": split_balance,
        "word_timestamp_coverage": {
            "hypotheses_total": hypotheses_total,
            "hypotheses_with_words": hypotheses_with_words,
            "fraction": round(hypotheses_with_words / hypotheses_total, 4)
            if hypotheses_total
            else 0.0,
        },
    }


def _round(value: Any, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None


def _fmt(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def render_text(report: dict[str, Any]) -> str:
    """Render the report as plain text."""
    corpus = report["corpus"]
    throughput = report["throughput"]
    queue = report["queue"]
    lines = [
        "Nepanglish annotation harness -- status report",
        f"generated {report['generated_at']}",
        "",
        "CORPUS",
        f"  episodes                {corpus['episodes']}",
        f"  segments                {corpus['segments']}",
        f"  audio hours             {corpus['audio_hours']}",
        "  segments by status      "
        + ", ".join(f"{k}={v}" for k, v in corpus["segments_by_status"].items()),
        "",
        "LABELS",
        f"  total                   {report['labels']['total']}",
        *(
            f"  {name:<22}  {report['labels'][name]}"
            for name in ("accepted_unchanged", "edited", "unusable_audio", "uncertain")
        ),
        f"  Accept rate             {_fmt(report['accept_rate'])}",
    ]
    if report["accept_rate_by_day"]:
        lines.append("  accept rate by day")
        lines.extend(
            f"    {row['day']}  {row['accepted']}/{row['labeled']}  {_fmt(row['accept_rate'])}"
            for row in report["accept_rate_by_day"]
        )
    lines += [
        "",
        "THROUGHPUT",
        f"  median seconds/segment  {_fmt(throughput['median_seconds_per_segment'])}",
        f"  segments/hour           {_fmt(throughput['segments_per_hour'])}",
        f"  annotator hours         {throughput['annotator_hours']}",
        "",
        "QUEUE",
        f"  backlog                 {queue['backlog']}",
        "  by queue                " + ", ".join(f"{k}={v}" for k, v in queue["by_queue"].items()),
        f"  Projected hours left    {_fmt(queue['projected_hours_to_finish'])}",
        "",
        "SCORES (means over imported segments)",
        *(f"  {name:<22}  {_fmt(value)}" for name, value in report["scores"].items()),
        "",
        "SPLIT BALANCE",
        *(
            f"  {name:<22}  {entry['episodes']} episodes, {entry['segments']} segments, "
            f"{entry['hours']} h"
            for name, entry in report["split_balance"].items()
        ),
        "",
        "WORD TIMESTAMP COVERAGE",
        f"  hypotheses with words   {report['word_timestamp_coverage']['hypotheses_with_words']}"
        f" / {report['word_timestamp_coverage']['hypotheses_total']}"
        f" ({report['word_timestamp_coverage']['fraction']})",
    ]
    return "\n".join(lines) + "\n"


def render_html(report: dict[str, Any]) -> str:
    """Render the report as a self-contained HTML page.

    Every value is escaped: episode titles arrive from an upstream manifest, so they are data,
    not markup.
    """

    def esc(value: Any) -> str:
        return html.escape(str(value))

    def table(rows: list[tuple[str, Any]]) -> str:
        body = "".join(
            f"<tr><th>{esc(name)}</th><td>{esc(_fmt(value))}</td></tr>" for name, value in rows
        )
        return f"<table>{body}</table>"

    corpus = report["corpus"]
    sections = [
        (
            "Corpus",
            table(
                [
                    ("Episodes", corpus["episodes"]),
                    ("Segments", corpus["segments"]),
                    ("Audio hours", corpus["audio_hours"]),
                    *[(f"Segments {k}", v) for k, v in corpus["segments_by_status"].items()],
                ]
            ),
        ),
        (
            "Labels",
            table(
                [
                    ("Total", report["labels"]["total"]),
                    ("Accepted unchanged", report["labels"]["accepted_unchanged"]),
                    ("Edited", report["labels"]["edited"]),
                    ("Unusable audio", report["labels"]["unusable_audio"]),
                    ("Uncertain", report["labels"]["uncertain"]),
                    ("Accept rate", report["accept_rate"]),
                ]
            ),
        ),
        (
            "Throughput",
            table(
                [
                    ("Median seconds/segment", report["throughput"]["median_seconds_per_segment"]),
                    ("Segments/hour", report["throughput"]["segments_per_hour"]),
                    ("Annotator hours", report["throughput"]["annotator_hours"]),
                ]
            ),
        ),
        (
            "Queue",
            table(
                [
                    ("Backlog", report["queue"]["backlog"]),
                    *[(f"Queue {k}", v) for k, v in report["queue"]["by_queue"].items()],
                    ("Projected hours to finish", report["queue"]["projected_hours_to_finish"]),
                ]
            ),
        ),
        ("Scores", table(list(report["scores"].items()))),
        (
            "Split balance",
            table(
                [
                    (
                        name,
                        f"{entry['episodes']} episodes, {entry['segments']} segments,"
                        f" {entry['hours']} h",
                    )
                    for name, entry in report["split_balance"].items()
                ]
            ),
        ),
        (
            "Word timestamp coverage",
            table(
                [
                    (
                        "Hypotheses with words",
                        report["word_timestamp_coverage"]["hypotheses_with_words"],
                    ),
                    ("Hypotheses total", report["word_timestamp_coverage"]["hypotheses_total"]),
                    ("Fraction", report["word_timestamp_coverage"]["fraction"]),
                ]
            ),
        ),
        (
            "Episodes",
            "<table>"
            + "".join(
                f"<tr><th>{esc(e['external_id'])}</th><td>{esc(e['title'] or '')}</td>"
                f"<td>{esc(e['split'])}</td></tr>"
                for e in corpus["episode_titles"]
            )
            + "</table>",
        ),
    ]
    body = "".join(
        f"<section><h2>{esc(title)}</h2>{content}</section>" for title, content in sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Annotation harness status</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem auto;
         max-width: 60rem; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .meta {{ color: #666; margin-bottom: 2rem; }}
  section {{ margin-bottom: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 0.35rem 0.75rem; border-bottom: 1px solid #e5e5e5; }}
  th {{ width: 16rem; font-weight: 600; color: #444; }}
</style>
</head>
<body>
<h1>Nepanglish annotation harness</h1>
<p class="meta">Generated {esc(report["generated_at"])}</p>
{body}
</body>
</html>
"""
