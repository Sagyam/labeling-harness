"""Error files derived for a model already imported (D110)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import ModelEvalClip, ModelEvalRun, Segment
from app.services import error_mining
from app.services.error_backfill import derive_error_files
from app.services.model_import import import_model_dir
from tests.model_support import CARD, write_model

pytestmark = pytest.mark.db


@pytest.fixture
def folder(db_session: Session, model_corpus: dict[str, str], settings: Settings) -> Path:
    gold = sorted(db_session.scalars(sa.select(Segment.external_id).where(Segment.pot == "gold")))
    rows = [
        {"segment_id": gold[0], "text": "एक दुई तीन"},
        {"segment_id": gold[1], "text": model_corpus[gold[1]] + " थप"},
        {"segment_id": gold[2], "text": model_corpus[gold[2]]},
    ]
    path = write_model(settings.models.root, "flex-ft", card=CARD | {"run_name": "vanilla-s1"},
                       gold=rows)  # fmt: skip
    import_model_dir(db_session, path, settings=settings, actor="test")
    return path


def test_gold_rows_reproduce_the_imported_run(db_session: Session, folder: Path) -> None:
    files = derive_error_files(db_session, folder)
    assert [(f.set, f.run) for f in files] == [("gold", "vanilla-s1")]
    meta, found = error_mining.read(folder / "errors" / "gold.parquet")
    assert meta["run"] == "vanilla-s1"
    clips = db_session.scalars(
        sa.select(ModelEvalClip).join(ModelEvalRun).where(ModelEvalRun.split == "gold")
    ).all()
    assert sum(r["kind"] in ("sub", "del", "ins") for r in found) == sum(c.errors for c in clips)
    assert sum(len(r["ref"]) for r in found) == sum(c.ref_words for c in clips)
    by_clip = {r["clip_id"]: r for r in found}
    shares = {c.segment.external_id: c.overlap_share for c in clips}
    assert {k: by_clip[k]["overlap_share"] for k in shares} == shares
    assert {r["group"] for r in found} == {"mx_ep"}
    snr = {c.segment.external_id: (c.segment.acoustics_jsonb or {}).get("snr_db") for c in clips}
    assert {k: by_clip[k]["snr_db"] for k in snr} == snr


def test_gold_rows_carry_the_segments_measured_snr(db_session: Session, folder: Path) -> None:
    segment = db_session.scalars(
        sa.select(Segment).where(Segment.pot == "gold").order_by(Segment.external_id).limit(1)
    ).one()
    segment.acoustics_jsonb = {"version": "acoustics-v2", "snr_db": 18.0, "c50_db": 50.0}
    db_session.flush()
    derive_error_files(db_session, folder)
    _, found = error_mining.read(folder / "errors" / "gold.parquet")
    by_clip = {r["clip_id"]: r for r in found}
    assert (by_clip[segment.external_id]["snr_db"], by_clip[segment.external_id]["snr_bucket"]) == (
        18.0,
        "15-25 dB",
    )


def test_public_sets_come_from_the_folders_benchmark_lines(
    db_session: Session, folder: Path, tmp_path: Path
) -> None:
    lines = [
        {"id": "-None", "group": "", "ref": "म घर", "hyp": "म घर", "errors": 0, "words": 2,
         "speech_type": "cs"},
        {"id": "-None", "group": "", "ref": "राम्रो", "hyp": "नराम्रो", "errors": 1, "words": 1,
         "speech_type": "en"},
    ]  # fmt: skip
    (folder / "benchmarks").mkdir()
    (folder / "benchmarks" / "nepali_cs.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8"
    )
    measured = tmp_path / "benchmarks"
    (measured / "overlap").mkdir(parents=True)
    (measured / "acoustics").mkdir()
    partial = measured / "overlap" / "nepali_cs.jsonl"
    partial.write_text(
        json.dumps({"clip_id": "-None#2", "duration": 2.0, "overlap_share": 0.5, "spans": []})
        + "\n"
    )
    loud = measured / "acoustics" / "nepali_cs.jsonl"
    loud.write_text(
        json.dumps({"clip_id": "-None", "duration": 2.0, "snr_db": 8.0, "c50_db": 1.0,
                    "bandwidth_hz": 8000.0}) + "\n"
    )  # fmt: skip
    from app.services.benchmark_overlap import write_acoustics, write_overlap

    write_overlap(partial, measured / "overlap" / "nepali_cs.parquet")
    write_acoustics(loud, measured / "acoustics" / "nepali_cs.parquet")
    files = derive_error_files(db_session, folder, conditions_dirs=[measured])
    assert [f.set for f in files] == ["gold", "nepali_cs"]
    _, found = error_mining.read(folder / "errors" / "nepali_cs.parquet")
    assert [(r["clip_id"], r["by"], r["overlap_bucket"], r["snr_bucket"]) for r in found] == [
        ("-None", "cs", "unmeasured", "<15 dB"),
        ("-None", "cs", "unmeasured", "<15 dB"),
        ("-None#2", "en", ">15%", "unmeasured"),
    ]


def test_an_unimported_folder_is_refused(db_session: Session, settings: Settings) -> None:
    path = write_model(settings.models.root, "never-imported")
    with pytest.raises(ValueError, match="never-imported"):
        derive_error_files(db_session, path)
