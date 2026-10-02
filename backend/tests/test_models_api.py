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


# --- error files (docs/WER-Breakdown.md, step 4) -------------------------------------------------


def _error_file(tmp_path, run: str, set_name: str) -> bytes:
    from app.services import error_mining

    path = tmp_path / f"{run}-{set_name}.parquet"
    clips = [{"clip_id": "c1", "group": "g", "ref": "म घर", "hyp": "म घर", "overlap_share": 0.0}]
    error_mining.write(error_mining.rows(run, set_name, clips), path)
    return path.read_bytes()


def test_error_files_are_uploaded_beside_the_model(
    client, run_id: int, settings: Settings, tmp_path, db_session: Session
) -> None:
    from app.models import AuditLog

    files = [
        ("files", ("anything.parquet", _error_file(tmp_path, "vanilla-s1", "fleurs"))),
        ("files", ("other.parquet", _error_file(tmp_path, "vanilla-s1", "gold"))),
    ]
    response = client.post("/models/flex-ft/errors", files=files)
    assert response.status_code == 200, response.text
    assert [(f["set"], f["run"]) for f in response.json()] == [
        ("gold", "vanilla-s1"),
        ("fleurs", "vanilla-s1"),
    ]
    stored = sorted(p.name for p in (settings.models.root / "flex-ft" / "errors").iterdir())
    assert stored == ["fleurs.parquet", "gold.parquet"]
    audit = db_session.scalars(
        sa.select(AuditLog).where(AuditLog.action == "model_errors_upload")
    ).one()
    assert audit.entity_id == "flex-ft"
    assert [f["set"] for f in audit.new_values_jsonb["files"]] == ["gold", "fleurs"]


def test_an_error_upload_is_refused_whole_with_422(
    client, run_id: int, settings: Settings, tmp_path
) -> None:
    files = [
        ("files", ("a.parquet", _error_file(tmp_path, "vanilla-s1", "fleurs"))),
        ("files", ("b.parquet", _error_file(tmp_path, "vanilla-s0", "gold"))),
    ]
    response = client.post("/models/flex-ft/errors", files=files)
    assert response.status_code == 422
    assert "one run" in response.json()["detail"]
    assert not list((settings.models.root / "flex-ft" / "errors").glob("*.parquet"))


def test_errors_for_an_unknown_model_are_404(client, model_corpus, tmp_path) -> None:
    files = [("files", ("a.parquet", _error_file(tmp_path, "vanilla-s1", "fleurs")))]
    assert client.post("/models/nope/errors", files=files).status_code == 404


def test_rescan_lists_the_error_files_a_folder_brought(
    client, model_corpus: dict[str, str], settings: Settings, tmp_path
) -> None:
    folder = write_model(settings.models.root, "fresh", card=CARD | {"name": "Fresh"})
    (folder / "errors").mkdir()
    (folder / "errors" / "fleurs.parquet").write_bytes(_error_file(tmp_path, "r", "fleurs"))
    (folder / "errors" / "val.parquet").write_bytes(b"broken")
    body = client.post("/models/rescan").json()
    assert body["error_files"] == ["fresh/fleurs"]
    assert [r.split(":")[0] for r in body["error_files_refused"]] == ["fresh/val.parquet"]


# --- reading error files (docs/WER-Breakdown.md, step 5) -----------------------------------------


@pytest.fixture
def mined(client, run_id: int, settings: Settings, db_session: Session, tmp_path) -> dict:
    """flex-ft with its gold run mined and a FLEURS file; a base model with FLEURS only."""
    from app.services import error_mining
    from app.services.error_backfill import derive_error_files

    derive_error_files(db_session, settings.models.root / "flex-ft")

    def fleurs(run: str, hyp: str) -> bytes:
        clips = [
            {"clip_id": "f1", "group": "s1", "ref": "पाँच वटा", "hyp": hyp, "overlap_share": 0.0},
            {"clip_id": "f2", "group": "s2", "ref": "राम्रो छ", "hyp": "राम्रो छ",
             "overlap_share": None},
        ]  # fmt: skip
        path = tmp_path / f"{run}.parquet"
        error_mining.write(error_mining.rows(run, "fleurs", clips), path)
        return path.read_bytes()

    upload = [("files", ("f.parquet", fleurs("flex-ft", "5 वटा थप")))]  # the card's run name
    assert client.post("/models/flex-ft/errors", files=upload).status_code == 200
    write_model(settings.models.root, "base", card=CARD | {"name": "Base"})
    client.post("/models/rescan")
    upload = [("files", ("f.parquet", fleurs("base", "पाँच वटा")))]
    assert client.post("/models/base/errors", files=upload).status_code == 200
    return {"run_id": run_id}


def test_the_error_files_are_listed_with_their_scores(client, mined: dict) -> None:
    body = client.get("/models/flex-ft/errors").json()
    files = {f["set"]: f for f in body["files"]}
    assert list(files) == ["gold", "fleurs"]
    assert files["fleurs"]["errors"] == 1 and files["fleurs"]["ref_words"] == 4
    assert files["fleurs"]["fold_current"] is True
    # gold was mined from the imported run, so the two agree.
    assert files["gold"]["imported_wer"] == pytest.approx(files["gold"]["wer"])
    assert body["refused"] == []


def test_a_breakdown_against_a_base(client, mined: dict) -> None:
    body = client.get("/models/flex-ft/errors/fleurs/breakdown", params={"base": "base"}).json()
    assert body["wer"] == pytest.approx(25.0)
    assert body["vs_base"]["wer"][0] == pytest.approx(25.0)
    assert [b["bucket"] for b in body["overlap"]] == ["none", "unmeasured"]
    assert body["numbers"]["ref_clips"] == 1
    assert body["base"] == "base"
    gold = client.get("/models/flex-ft/errors/gold/breakdown").json()
    assert gold["imported_wer"] == pytest.approx(gold["wer"])


def test_breakdown_refusals(client, mined: dict) -> None:
    assert client.get("/models/flex-ft/errors/val/breakdown").status_code == 404  # no file
    assert client.get("/models/flex-ft/errors/librispeech/breakdown").status_code == 422
    assert client.get("/models/flex-ft/errors/..%2Fgold/breakdown").status_code in (404, 422)
    missing = client.get("/models/flex-ft/errors/gold/breakdown", params={"base": "base"})
    assert missing.status_code == 404 and "base" in missing.json()["detail"]
    assert client.get("/models/nope/errors").status_code == 404


def test_the_confusion_table_and_its_occurrences(client, mined: dict) -> None:
    url = "/models/flex-ft/errors/fleurs"
    body = client.get(f"{url}/confusion", params={"kind": ["ins"], "base": "base"}).json()
    assert body["total"] == 1
    row = body["rows"][0]
    assert (row["hyp"], row["count"], row["base_count"], row["change"]) == ("थप", 1, 0, 1)
    forgiven = client.get(f"{url}/confusion", params={"forgiven": "number"}).json()
    assert [(r["ref"], r["hyp"]) for r in forgiven["rows"]] == [("पाँच", "5")]
    pairs = client.get(f"{url}/pairs", params={"kind": ["ins"], "hyp": "थप"}).json()
    assert [p["clip_id"] for p in pairs["rows"]] == ["f1"]
    assert [op["ref"] for op in pairs["rows"][0]["before"]] == [["पाँच"], ["वटा"]]
    sampled = client.get(f"{url}/pairs", params={"sample": 1, "seed": 3}).json()
    assert sampled["total"] == 5 and len(sampled["rows"]) == 1  # every row, matches included
    assert client.get(f"{url}/confusion", params={"kind": ["oops"]}).status_code == 422
    unmeasured = client.get(f"{url}/pairs", params={"snr_bucket": "unmeasured"}).json()
    assert unmeasured["total"] == 5 and unmeasured["rows"][0]["snr_bucket"] == "unmeasured"
    assert client.get(f"{url}/confusion", params={"snr_bucket": "<15 dB"}).json()["total"] == 0
    assert client.get(f"{url}/confusion", params={"snr_bucket": "loud"}).status_code == 422


def test_one_clip_and_where_its_audio_is(client, mined: dict, db_session: Session) -> None:
    fleurs = client.get("/models/flex-ft/errors/fleurs/clips/f1").json()
    assert [op["kind"] for op in fleurs["ops"]] == ["fold", "match", "ins"]
    assert fleurs["segment_id"] is None and fleurs["run_id"] is None
    gold_id = _gold(db_session)[0]
    gold = client.get(f"/models/flex-ft/errors/gold/clips/{gold_id}").json()
    segment = db_session.scalars(sa.select(Segment).where(Segment.external_id == gold_id)).one()
    assert (gold["segment_id"], gold["run_id"]) == (segment.id, mined["run_id"])
    assert client.get("/models/flex-ft/errors/fleurs/clips/nope").status_code == 404


# --- the rulebook (fold-v4) ----------------------------------------------------------------------


def test_the_rulebook_lists_every_rule_and_tag_with_its_evidence(client, mined: dict) -> None:
    from app.services.fold import RULEBOOK, TAGS, fold_version

    body = client.get("/fold/rulebook").json()
    assert body["fold_version"] == fold_version()
    assert [r["id"] for r in body["rules"]] == [r.id for r in (*RULEBOOK, *TAGS)]
    number = next(r for r in body["rules"] if r["id"] == "number")
    assert number["action"] == "fold" and number["examples"][0] == ["15", "पन्ध्र"]
    assert {t["tier"] for t in body["tiers"]} == {1, 2, 3, 4}
    fleurs = next(e for e in body["evidence"] if (e["model"], e["set"]) == ("flex-ft", "fleurs"))
    assert fleurs["current"] is True and fleurs["words"] == 4
    assert fleurs["rules"]["number"]["top"] == [{"ref": "पाँच", "hyp": "5", "count": 1}]


def test_occurrences_filter_by_the_rule_that_forgave_them(client, mined: dict) -> None:
    url = "/models/flex-ft/errors/fleurs/pairs"
    rows = client.get(url, params={"fold_rule": "number"}).json()["rows"]
    assert [(r["ref"], r["hyp"]) for r in rows] == [(["पाँच"], ["5"])]
    assert client.get(url, params={"variant": "ba-va"}).json()["total"] == 0
