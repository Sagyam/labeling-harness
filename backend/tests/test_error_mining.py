"""Error mining's rows (docs/WER-Breakdown.md, step 1): one per aligned word pair, classified.

Everything the breakdown and the Errors panel report is a sum over these rows, so the test the
rest rests on is the first one: the rows give back the score, clip by clip, on real transcripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from app.services import error_mining
from app.services.clip_classes import overlap_bucket
from app.services.error_mining import MinedFileError, pairs, read, rows, write
from app.services.fold import fold_tokens, fold_version, word_errors

_CLIPS = json.loads(
    (Path(__file__).parent / "reference" / "error_mining_clips.json").read_text(encoding="utf-8")
)


def _by_kind(found: list[dict], kind: str) -> list[dict]:
    return [r for r in found if r["kind"] == kind]


@pytest.mark.parametrize("clip", _CLIPS, ids=[c["clip_id"] for c in _CLIPS])
def test_the_rows_reproduce_the_score(clip: dict) -> None:
    """Real gold and FLEURS pairs: S, D, I and reference words equal word_errors', clip by clip."""
    scored = word_errors(clip["ref"], clip["hyp"])
    found = pairs(clip["ref"], clip["hyp"])
    assert len(_by_kind(found, "sub")) == scored.substitutions
    assert len(_by_kind(found, "del")) == scored.deletions
    assert len(_by_kind(found, "ins")) == scored.insertions
    assert sum(len(r["ref"]) for r in found) == scored.ref_words


@pytest.mark.parametrize("clip", _CLIPS, ids=[c["clip_id"] for c in _CLIPS])
def test_a_clips_rows_in_order_give_back_its_folded_words(clip: dict) -> None:
    found = sorted(pairs(clip["ref"], clip["hyp"]), key=lambda r: r["pos"])
    assert [r["pos"] for r in found] == list(range(len(found)))
    assert [w for r in found for w in r["ref"]] == fold_tokens(clip["ref"])
    assert [w for r in found for w in r["hyp"]] == fold_tokens(clip["hyp"])


def test_an_existing_alignment_is_used_rather_than_aligned_again() -> None:
    alignment = word_errors("टिम राम्रो", "team नराम्रो")
    found = pairs("ignored", "ignored", alignment=alignment)
    assert [r["kind"] for r in found] == ["fold", "sub"]


@pytest.mark.parametrize(
    ("ref", "hyp", "kind", "forgiven"),
    [
        ("तीन", "तिन", "match", "spelling"),  # vowel length
        ("टिम", "team", "fold", "script"),
        ("45", "पैँतालीस", "fold", "number"),
        ("6", "छ", "fold", "number"),  # छ is no number to the tag, but rule 2 matched it
        ("गर्नुभयो", "गर्नु भयो", "merge", "merge"),
        ("राम्रो", "राम्रो", "match", None),
        ("राम्रो", "नराम्रो", "sub", None),
    ],
)
def test_how_a_pair_was_forgiven(ref: str, hyp: str, kind: str, forgiven: str | None) -> None:
    (row,) = pairs(ref, hyp)
    assert row["kind"] == kind
    assert row["forgiven"] == forgiven
    assert row["identical"] is (kind == "match" and forgiven is None)


@pytest.mark.parametrize(
    ("word", "script"),
    [("राम्रो", "dev"), ("phone", "lat"), ("queriesहरू", "mix"), ("45", "none")],
)
def test_the_script_of_a_side(word: str, script: str) -> None:
    (row,) = pairs(word, word)
    assert (row["ref_script"], row["hyp_script"]) == (script, script)


def test_a_side_of_two_scripts_is_mixed_and_an_absent_side_has_none() -> None:
    (merged,) = _by_kind(pairs("गर्नुभयो", "गर्नु भयो"), "merge")
    assert (merged["ref_script"], merged["hyp_script"]) == ("dev", "dev")
    assert _by_kind(pairs("phone मा", "फोनमा"), "merge")[0]["ref_script"] == "mix"
    (deleted,) = _by_kind(pairs("म घर जान्छु", "म जान्छु"), "del")
    assert deleted["hyp_script"] is None
    assert deleted["hyp"] == []
    (inserted,) = _by_kind(pairs("म जान्छु", "म घर जान्छु"), "ins")
    assert inserted["ref_script"] is None
    assert inserted["ref"] == []


def test_number_similarity_and_romanization() -> None:
    (row,) = pairs("छत्तीस", "06")
    assert row["kind"] == "sub"
    assert (row["number"], row["ref_number"]) == (True, True)
    (row,) = pairs("राम्रो", "5")
    assert (row["number"], row["ref_number"]) == (True, False)  # only the model wrote a number
    (row,) = pairs("cache", "cage")
    assert row["number"] is False
    assert 0 < row["similarity"] < 1
    assert (row["ref_roman"], row["hyp_roman"]) == ("cache", "cage")
    (row,) = pairs("राम्रो", "राम्रो")
    assert row["similarity"] == 1.0
    assert row["ref_roman"] == row["hyp_roman"] != ""


def test_rows_add_the_clip_columns() -> None:
    clips = [
        {"clip_id": "c1", "group": "ep1", "ref": "म घर", "hyp": "म घर", "overlap_share": 0.2,
         "snr_db": 12.5},
        {"clip_id": "c2", "group": "ep2", "ref": "म", "hyp": "म", "overlap_share": None,
         "by": "read"},
    ]  # fmt: skip
    found = rows("vanilla-s1", "fleurs", clips)
    assert [(r["clip_id"], r["pos"]) for r in found] == [("c1", 0), ("c1", 1), ("c2", 0)]
    first, last = found[0], found[-1]
    assert (first["run"], first["set"], first["group"]) == ("vanilla-s1", "fleurs", "ep1")
    assert (first["overlap_share"], first["overlap_bucket"], first["by"]) == (0.2, ">15%", None)
    assert (first["snr_db"], first["snr_bucket"]) == (12.5, "<15 dB")
    assert (last["overlap_share"], last["overlap_bucket"]) == (None, "unmeasured")
    assert (last["snr_db"], last["snr_bucket"]) == (None, "unmeasured")
    assert last["by"] == "read"
    assert list(first) == list(error_mining.COLUMNS)


@pytest.mark.parametrize("share", [None, 0.0, 0.01, 0.049, 0.05, 0.15, 0.151, 0.9])
def test_the_overlap_bucket_is_the_corpus_one(share: float | None) -> None:
    assert error_mining.overlap_bucket(share) == overlap_bucket(share)


@pytest.mark.parametrize("db", [None, -4.5, 14.9, 15.0, 24.9, 25.0, 35.0, 44.99, 45.0, 72.0])
def test_the_snr_bucket_is_the_corpus_one(db: float | None) -> None:
    from app.services.clip_classes import AXES, _snr_bucket

    acoustics = None if db is None else {"snr_db": db}
    assert error_mining.snr_bucket(db) == _snr_bucket(acoustics)
    (axis,) = [a for a in AXES if a.name == "snr"]
    assert axis.buckets == error_mining.SNR_BUCKETS
    from typing import get_args

    from app.api.model_errors import SnrBucket

    assert get_args(SnrBucket) == error_mining.SNR_BUCKETS


def test_the_sets_are_gold_val_and_the_public_benchmarks() -> None:
    assert error_mining.SETS == (
        "gold", "val", "fleurs", "slr54", "common_voice", "indicvoices", "nepali_cs",
    )  # fmt: skip


# --- the file ------------------------------------------------------------------------------------


def _sample() -> list[dict]:
    clips = [
        {"clip_id": c["clip_id"], "group": c["group"], "ref": c["ref"], "hyp": c["hyp"],
         "overlap_share": 0.03}
        for c in _CLIPS
        if c["set"] == "fleurs"
    ]  # fmt: skip
    return rows("vanilla-s1", "fleurs", clips)


def test_write_then_read_round_trips_rows_and_metadata(tmp_path: Path) -> None:
    written = _sample()
    path = tmp_path / "fleurs.parquet"
    write(written, path)
    meta, back = read(path)
    assert back == written
    assert meta["run"] == "vanilla-s1"
    assert meta["set"] == "fleurs"
    assert meta["fold_version"] == fold_version()
    assert meta["miner_version"] == error_mining.MINER_VERSION
    assert meta["created_at"].endswith("+00:00")
    assert error_mining.metadata(path) == meta
    assert not list(tmp_path.glob("*.tmp*")), "no temporary file is left behind"


def test_one_file_holds_one_run_and_one_set(tmp_path: Path) -> None:
    mixed = _sample()
    mixed[-1] = {**mixed[-1], "set": "gold"}
    with pytest.raises(ValueError, match="one run and one set"):
        write(mixed, tmp_path / "x.parquet")
    with pytest.raises(ValueError, match="no rows"):
        write([], tmp_path / "x.parquet")


def test_a_file_from_another_miner_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "fleurs.parquet"
    write(_sample(), path)
    old = tmp_path / "old.parquet"
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet('{path}')) TO '{old}' (FORMAT parquet, KV_METADATA "
        "{run: 'vanilla-s1', set: 'fleurs', fold_version: 'fold-v3+norm-v3', "
        "miner_version: 'mine-v0', created_at: '2026-10-01T00:00:00+00:00'})"
    )
    with pytest.raises(MinedFileError, match="mine-v0"):
        read(old)


def test_a_file_missing_a_column_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "fleurs.parquet"
    write(_sample(), path)
    short = tmp_path / "short.parquet"
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (similarity) FROM read_parquet('{path}')) TO '{short}' "
        "(FORMAT parquet, KV_METADATA {run: 'vanilla-s1', set: 'fleurs', fold_version: 'x', "
        f"miner_version: '{error_mining.MINER_VERSION}', created_at: 'x'}})"
    )
    with pytest.raises(MinedFileError, match="similarity"):
        error_mining.metadata(short)


def test_a_file_that_is_not_parquet_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises(MinedFileError):
        error_mining.metadata(path)


def test_a_repeated_clip_id_is_numbered_in_file_order() -> None:
    """nepali_cs has three clips with no video id or start ("-None"): pos must not collide, and
    two runs on the same set must still name each clip alike."""
    clips = [{"clip_id": "-None", "group": "", "ref": "म", "hyp": "म"} for _ in range(3)]
    found = rows("vanilla-s1", "nepali_cs", clips)
    assert [r["clip_id"] for r in found] == ["-None", "-None#2", "-None#3"]


def test_the_dataset_copy_carries_error_mining_beside_fold() -> None:
    """ftkit.harness_scorer lays these out flat; error mining needs fold.py and its normalizer."""
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "upload_harness_copy", root / "scripts" / "upload_harness_copy.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    files = script.harness_files()
    assert sorted(files) == [
        "harness/attribution.py", "harness/error_mining.py", "harness/error_store.py",
        "harness/fold.py", "harness/normalization.yaml", "harness/normalize.py",
    ]  # fmt: skip
    assert all(path.is_file() for path in files.values())


@pytest.mark.parametrize(
    ("ref", "hyp", "rule"),
    [
        ("तीन", "तिन", "vowel-length"),
        ("टिम", "team", "sound-skeleton"),
        ("45", "पैँतालीस", "number"),
        ("B", "बी", "letter-names"),
        ("होइन", "हैन", "contracted-verb"),
        ("गर्नुभयो", "गर्नु भयो", "spacing"),
        ("राम्रो", "राम्रो", None),
        ("राम्रो", "नराम्रो", None),
    ],
)
def test_a_forgiven_pair_names_the_rule_that_forgave_it(
    ref: str, hyp: str, rule: str | None
) -> None:
    (row,) = pairs(ref, hyp)
    assert row["fold_rule"] == rule


@pytest.mark.parametrize(
    ("ref", "hyp", "tag"),
    [("दावी", "दाबी", "ba-va"), ("गरेँ", "गरे", "nasal-dropped"), ("हौ", "हौँ", "nasal-added"),
     ("घर", "वन", None), ("टिम", "team", None)],
)  # fmt: skip
def test_a_charged_substitution_carries_its_variant_tag(
    ref: str, hyp: str, tag: str | None
) -> None:
    (row,) = pairs(ref, hyp)
    assert row["variant"] == tag


def test_a_deleted_or_inserted_repeat_is_tagged_and_other_gaps_are_not() -> None:
    (deleted,) = _by_kind(pairs("म म जान्छु", "म जान्छु"), "del")
    assert deleted["variant"] == "repetition"
    (inserted,) = _by_kind(pairs("म जान्छु", "म जान्छु जान्छु"), "ins")
    assert inserted["variant"] == "repetition"
    (other,) = _by_kind(pairs("म घर जान्छु", "म जान्छु"), "del")
    assert other["variant"] is None
