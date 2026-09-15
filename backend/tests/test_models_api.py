"""The Models page's endpoints: models, runs, and a run's clips worst first (D83)."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import ModelEvalRun, Segment
from app.services.model_import import import_model_dir
from tests.model_support import CARD, write_model

pytestmark = pytest.mark.db


def _gold(session: Session) -> list[str]:
    return sorted(session.scalars(sa.select(Segment.external_id).where(Segment.pot == "gold")))


@pytest.fixture
def run_id(db_session: Session, model_corpus: dict[str, str], settings: Settings) -> int:
    """A gold run of three clips: one loses two words, one gains one, one is perfect."""
    a, b, c = _gold(db_session)
    rows = [
        {"segment_id": a, "text": "एक दुई तीन"},
        {"segment_id": b, "text": model_corpus[b] + " थप"},
        {"segment_id": c, "text": model_corpus[c]},
    ]
    folder = write_model(settings.models.root, "flex-ft", gold=rows)
    import_model_dir(db_session, folder, settings=settings, actor="test")
    return db_session.scalars(sa.select(ModelEvalRun.id)).one()


def test_models_are_listed_with_their_runs(client, run_id: int) -> None:
    body = client.get("/models").json()
    assert [m["slug"] for m in body] == ["flex-ft"]
    model = body[0]
    assert model["name"] == CARD["name"]
    assert model["architecture"] == CARD["architecture"]
    run = model["runs"][0]
    assert (run["id"], run["split"], run["clip_count"]) == (run_id, "gold", 3)
    assert run["metrics"]["errors"] == 3
    assert run["fold_version"]


def test_one_model_carries_its_card(client, run_id: int) -> None:
    body = client.get("/models/flex-ft").json()
    assert body["card"]["decoder"] == "greedy+cap+retry"
    assert client.get("/models/nope").status_code == 404


def test_rescan_imports_the_model_folders(
    client, model_corpus: dict[str, str], settings: Settings
) -> None:
    write_model(settings.models.root, "fresh", card=CARD | {"name": "Fresh"})
    body = client.post("/models/rescan").json()
    assert body["models"] == ["fresh"]
    assert [m["slug"] for m in client.get("/models").json()] == ["fresh"]


def test_rescan_refuses_a_bad_folder_with_422(
    client, model_corpus: dict[str, str], settings: Settings
) -> None:
    write_model(settings.models.root, "bad", gold=[{"segment_id": "not_a_clip", "text": "x"}])
    response = client.post("/models/rescan")
    assert response.status_code == 422
    assert "not_a_clip" in response.json()["detail"]


def test_clips_come_worst_first_with_rates(client, run_id: int) -> None:
    body = client.get(f"/model-runs/{run_id}/clips").json()
    assert body["total"] == 3
    errors = [row["errors"] for row in body["rows"]]
    assert errors == [2, 1, 0]
    worst = body["rows"][0]
    assert worst["deletions"] == 2
    assert worst["wer"] == pytest.approx(100 * 2 / worst["ref_words"])
    assert worst["genre"] == "podcast"
    assert worst["overlap_bucket"] == "0-5%"
    assert worst["classes"]["overlap"] == "0-5%"
    assert worst["hyp_text"] == "एक दुई तीन"


def test_clips_sort_and_page(client, run_id: int) -> None:
    body = client.get(f"/model-runs/{run_id}/clips?sort=insertions&limit=1").json()
    assert body["total"] == 3 and len(body["rows"]) == 1
    assert body["rows"][0]["insertions"] == 1
    second = client.get(f"/model-runs/{run_id}/clips?offset=1&limit=1").json()
    assert second["rows"][0]["errors"] == 1
    ascending = client.get(f"/model-runs/{run_id}/clips?order=asc").json()
    assert ascending["rows"][0]["errors"] == 0
    assert client.get(f"/model-runs/{run_id}/clips?sort=bogus").status_code == 422


@pytest.mark.parametrize(
    ("query", "count"),
    [
        ("overlap=0-5%25", 1),
        ("overlap=unmeasured", 2),
        ("genre=podcast", 3),
        ("genre=tech_review", 0),
        ("min_errors=1", 2),
        ("loops_only=true", 0),
        ("class_axis=overlap&class_bucket=0-5%25", 1),
        ("class_axis=speakers&class_bucket=undiarized", 3),
        ("class_axis=speakers&class_bucket=2", 0),
    ],
)
def test_clips_filter(client, run_id: int, query: str, count: int) -> None:
    assert client.get(f"/model-runs/{run_id}/clips?{query}").json()["total"] == count


def test_a_clip_carries_the_alignment_that_was_counted(
    client, run_id: int, db_session: Session
) -> None:
    worst = client.get(f"/model-runs/{run_id}/clips?limit=1").json()["rows"][0]
    body = client.get(f"/model-runs/{run_id}/clips/{worst['segment_id']}").json()
    kinds = [op["kind"] for op in body["ops"]]
    assert kinds.count("del") == body["deletions"] == 2
    assert sum(k in {"sub", "del", "ins"} for k in kinds) == body["errors"]
    assert body["fold_version_changed"] is False
    assert body["ops"][0]["ref"] == ["एक"] and body["ops"][0]["hyp"] == ["एक"]


def test_unknown_runs_and_clips_are_404(client, run_id: int) -> None:
    assert client.get("/model-runs/999999/clips").status_code == 404
    assert client.get(f"/model-runs/{run_id}/clips/999999").status_code == 404


def test_an_unknown_class_axis_is_422(client, run_id: int) -> None:
    response = client.get(f"/model-runs/{run_id}/clips?class_axis=bogus&class_bucket=x")
    assert response.status_code == 422


def test_the_class_axes_come_in_display_order(client) -> None:
    axes = client.get("/model-classes").json()
    assert axes[0]["name"] == "overlap"
    assert axes[0]["buckets"][0] == "none" and axes[0]["baseline"] == "none"
    assert [a["name"] for a in axes if a["descriptive"]] == ["cmi"]
