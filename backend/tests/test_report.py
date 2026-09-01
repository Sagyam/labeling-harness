"""Tests for the status report."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import AnnotationTask
from app.services.fixtures import build_export_fixture
from app.services.importer import import_manifest
from app.services.labeling import Decision, record_decision
from app.services.queue_builder import build_queue
from app.services.report import collect_report, render_html, render_text
from app.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.db


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path / "objects")


@pytest.fixture
def worked_corpus(db_session: Session, tmp_path: Path, storage, settings: Settings) -> None:
    root = build_export_fixture(
        tmp_path / "export_rep", episode_id="rep_ep001", segments=8, systems=2
    )
    import_manifest(db_session, root, storage=storage, settings=settings)
    build_queue(db_session, settings=settings, audit_sample_rate=0.0)
    tasks = list(db_session.scalars(sa.select(AnnotationTask).order_by(AnnotationTask.id)))
    for index, task in enumerate(tasks[:5]):
        record_decision(
            db_session,
            task,
            Decision(
                disposition="accepted_unchanged" if index < 4 else "uncertain",
                duration_ms=4000 + index * 1000,
            ),
            settings=settings,
        )
    db_session.flush()


def test_report_counts_the_corpus(db_session: Session, worked_corpus: None) -> None:
    report = collect_report(db_session)
    assert report["corpus"]["episodes"] == 1
    assert report["corpus"]["segments"] == 8
    assert report["corpus"]["audio_hours"] > 0
    assert report["corpus"]["segments_by_status"]["labeled"] == 4


def test_report_breaks_down_dispositions_and_accept_rate(
    db_session: Session, worked_corpus: None
) -> None:
    report = collect_report(db_session)
    assert report["labels"]["accepted_unchanged"] == 4
    assert report["labels"]["uncertain"] == 1
    assert report["accept_rate"] == pytest.approx(0.8)


def test_report_includes_accept_rate_over_time(db_session: Session, worked_corpus: None) -> None:
    """If this number collapses, the upstream ASR stage regressed."""
    report = collect_report(db_session)
    assert report["accept_rate_by_day"]
    day = report["accept_rate_by_day"][0]
    assert {"day", "labeled", "accepted", "accept_rate"} <= set(day)


def test_report_includes_throughput(db_session: Session, worked_corpus: None) -> None:
    report = collect_report(db_session)
    throughput = report["throughput"]
    assert throughput["median_seconds_per_segment"] == pytest.approx(6.0)
    assert throughput["segments_per_hour"] > 0
    assert throughput["annotator_hours"] > 0


def test_report_projects_completion(db_session: Session, worked_corpus: None) -> None:
    report = collect_report(db_session)
    assert report["queue"]["backlog"] == 3
    assert report["queue"]["projected_hours_to_finish"] is not None


def test_report_includes_score_distributions(db_session: Session, worked_corpus: None) -> None:
    scores = collect_report(db_session)["scores"]
    assert scores["mean_word_disagreement_rate"] is not None
    assert scores["mean_script_conflict_rate"] is not None
    assert scores["mean_code_switch_density"] is not None


def test_report_includes_split_balance_in_hours(db_session: Session, worked_corpus: None) -> None:
    balance = collect_report(db_session)["split_balance"]
    assert set(balance) == {"train", "val", "test", "unassigned"}
    assert sum(entry["hours"] for entry in balance.values()) > 0


def test_report_includes_word_timestamp_coverage(db_session: Session, worked_corpus: None) -> None:
    coverage = collect_report(db_session)["word_timestamp_coverage"]
    assert coverage["hypotheses_total"] > 0
    assert coverage["hypotheses_with_words"] > 0
    assert 0.0 <= coverage["fraction"] <= 1.0


def test_text_report_is_readable(db_session: Session, worked_corpus: None) -> None:
    text = render_text(collect_report(db_session))
    assert "Nepanglish annotation harness" in text
    assert "Accept rate" in text
    assert "THROUGHPUT" in text
    assert "Projected hours left" in text


def test_html_report_is_well_formed(db_session: Session, worked_corpus: None) -> None:
    html = render_html(collect_report(db_session))
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "Accept rate" in html


def test_report_runs_against_an_empty_database(db_session: Session) -> None:
    report = collect_report(db_session)
    assert report["corpus"]["episodes"] == 0
    assert report["accept_rate"] is None
    assert report["throughput"]["median_seconds_per_segment"] is None
    assert render_text(report)
    assert render_html(report)


def test_html_escapes_untrusted_text(db_session: Session, worked_corpus: None) -> None:
    """Episode titles come from an upstream manifest; they are data, not markup."""
    from app.models import Episode

    episode = db_session.scalars(sa.select(Episode)).one()
    episode.title = "<script>alert('x')</script>"
    db_session.flush()
    html = render_html(collect_report(db_session))
    assert "<script>alert" not in html
