"""Error files beside a model, and the breakdown read from them (D110)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from app.services import error_mining
from app.services.error_store import (
    MAX_FILE_BYTES,
    ErrorFileError,
    ErrorFilter,
    accept_uploads,
    list_files,
)


def _clip(clip_id: str, ref: str, hyp: str, group: str = "g1", share: float | None = 0.0):
    return {"clip_id": clip_id, "group": group, "ref": ref, "hyp": hyp, "overlap_share": share}


def _file(tmp_path: Path, run: str, set_name: str, clips=None, name: str | None = None) -> bytes:
    clips = clips or [_clip("c1", "म घर जान्छु", "म घर जान्छु")]
    path = tmp_path / (name or f"{run}-{set_name}.parquet")
    error_mining.write(error_mining.rows(run, set_name, clips), path)
    return path.read_bytes()


# --- arrival (step 4) ----------------------------------------------------------------------------


def test_an_upload_is_stored_under_its_set_whatever_the_file_was_called(tmp_path: Path) -> None:
    model = tmp_path / "model"
    data = _file(tmp_path, "vanilla-s1", "fleurs")
    stored = accept_uploads(model, [("../../evil name.parquet", data)])
    assert [(f.set, f.run) for f in stored] == [("fleurs", "vanilla-s1")]
    assert sorted(p.name for p in (model / "errors").iterdir()) == ["fleurs.parquet"]


def test_files_of_one_upload_must_name_one_run(tmp_path: Path) -> None:
    model = tmp_path / "model"
    uploads = [
        ("a.parquet", _file(tmp_path, "vanilla-s1", "fleurs")),
        ("b.parquet", _file(tmp_path, "vanilla-s0", "gold")),
    ]
    with pytest.raises(ErrorFileError, match="one run"):
        accept_uploads(model, uploads)
    assert not (model / "errors").exists() or not list((model / "errors").iterdir())


def test_a_file_from_another_run_than_the_folders_other_files_is_refused(tmp_path: Path) -> None:
    model = tmp_path / "model"
    accept_uploads(model, [("a.parquet", _file(tmp_path, "vanilla-s1", "fleurs"))])
    with pytest.raises(ErrorFileError, match="vanilla-s1"):
        accept_uploads(model, [("b.parquet", _file(tmp_path, "vanilla-s0", "gold"))])
    # Replacing the only file it disagrees with is allowed.
    accept_uploads(model, [("c.parquet", _file(tmp_path, "vanilla-s0", "fleurs"))])
    assert [f.run for f in list_files(model)[0]] == ["vanilla-s0"]


def test_two_files_for_one_set_in_one_upload_are_refused(tmp_path: Path) -> None:
    data = _file(tmp_path, "vanilla-s1", "fleurs")
    with pytest.raises(ErrorFileError, match="fleurs"):
        accept_uploads(tmp_path / "model", [("a.parquet", data), ("b.parquet", data)])


def test_an_oversized_or_malformed_file_refuses_the_whole_upload(tmp_path: Path) -> None:
    model = tmp_path / "model"
    good = ("a.parquet", _file(tmp_path, "vanilla-s1", "fleurs"))
    with pytest.raises(ErrorFileError, match="50 MB"):
        accept_uploads(model, [good, ("big.parquet", b"0" * (MAX_FILE_BYTES + 1))])
    with pytest.raises(ErrorFileError, match="not a Parquet"):
        accept_uploads(model, [good, ("bad.parquet", b"not parquet")])
    with pytest.raises(ErrorFileError, match="no file"):
        accept_uploads(model, [])
    assert not list((model / "errors").glob("*"))


def test_an_unknown_set_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    source = tmp_path / "src.parquet"
    source.write_bytes(_file(tmp_path, "vanilla-s1", "fleurs"))
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet('{source}')) TO '{path}' (FORMAT parquet, KV_METADATA "
        "{run: 'vanilla-s1', set: 'librispeech', fold_version: 'x', "
        f"miner_version: '{error_mining.MINER_VERSION}', created_at: 'x'}})"
    )
    with pytest.raises(ErrorFileError, match="librispeech"):
        accept_uploads(tmp_path / "model", [("x.parquet", path.read_bytes())])


def test_listing_names_each_file_and_sets_aside_what_it_cannot_read(tmp_path: Path) -> None:
    model = tmp_path / "model"
    accept_uploads(model, [("a.parquet", _file(tmp_path, "vanilla-s1", "gold"))])
    (model / "errors" / "val.parquet").write_bytes(b"broken")
    (model / "errors" / "notes.txt").write_text("ignored")
    files, refused = list_files(model)
    assert [f.set for f in files] == ["gold"]
    assert files[0].path == model / "errors" / "gold.parquet"
    assert files[0].miner_version == error_mining.MINER_VERSION
    assert [name for name, _ in refused] == ["val.parquet"]
    assert list_files(tmp_path / "no-such-model") == ([], [])


def test_a_file_whose_name_disagrees_with_its_set_is_set_aside(tmp_path: Path) -> None:
    model = tmp_path / "model"
    (model / "errors").mkdir(parents=True)
    (model / "errors" / "val.parquet").write_bytes(_file(tmp_path, "vanilla-s1", "gold"))
    files, refused = list_files(model)
    assert files == []
    assert "gold" in refused[0][1]


# --- reading (step 5) ----------------------------------------------------------------------------
#
# Six clips, numbers worked by hand: 11 reference words, 5 errors (S 3, D 1, I 1).
#   c1 g1 none   म घर जान्छु / same        3 words, 0 errors
#   c2 g1 >15%   घर 45 दिन / घर दिन        3 words, D 45 (a number)
#   c3 g2 0-5%   म आयो / म आयो थप          2 words, I थप
#   c4 g2 unmeasured  पाँच / 6             1 word,  S पाँच->6 (a number)
#   c5 g3 none   राम्रो / नराम्रो            1 word,  S
#   c6 g3 none   नराम्रो / राम्रो            1 word,  S
# The base got everything right but c3's insertion: 1 error in 11 words.

_REFS = {
    "c1": ("g1", 0.0, "म घर जान्छु", "म घर जान्छु"),
    "c2": ("g1", 0.2, "घर 45 दिन", "घर दिन"),
    "c3": ("g2", 0.03, "म आयो", "म आयो थप"),
    "c4": ("g2", None, "पाँच", "6"),
    "c5": ("g3", 0.0, "राम्रो", "नराम्रो"),
    "c6": ("g3", 0.0, "नराम्रो", "राम्रो"),
}


def _write(tmp_path: Path, run: str, hyps: dict[str, str] | None = None) -> Path:
    clips = [
        _clip(cid, ref, (hyps or {}).get(cid, hyp), group, share)
        for cid, (group, share, ref, hyp) in _REFS.items()
    ]
    path = tmp_path / f"{run}.parquet"
    error_mining.write(error_mining.rows(run, "fleurs", clips), path)
    return path


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return _write(tmp_path, "run")


@pytest.fixture
def base(tmp_path: Path) -> Path:
    right = {cid: ref for cid, (_, _, ref, _) in _REFS.items() if cid != "c3"}
    return _write(tmp_path, "base", right)


def test_a_files_summary(run: Path) -> None:
    from app.services.error_store import summary

    got = summary(run)
    assert (got["clips"], got["ref_words"], got["errors"]) == (6, 11, 5)
    assert got["wer"] == pytest.approx(100 * 5 / 11)
    assert (got["sub"], got["del"], got["ins"]) == pytest.approx((100 * 3 / 11, 100 / 11, 100 / 11))


def test_the_overlap_block_by_hand(run: Path) -> None:
    from app.services.error_store import breakdown

    got = breakdown(run)
    assert got["wer"] == pytest.approx(100 * 5 / 11)
    lo, hi = got["wer_ci"]
    assert lo <= got["wer"] <= hi
    buckets = {b["bucket"]: b for b in got["overlap"]}
    assert list(buckets) == ["none", "0-5%", ">15%", "unmeasured"]  # no 5-15% clip
    none = buckets["none"]
    assert (none["clips"], none["ref_words"], none["errors"]) == (3, 5, 2)
    assert none["share_of_words"] == pytest.approx(5 / 11)
    assert none["share_of_errors"] == pytest.approx(2 / 5)
    assert none["wer"] == pytest.approx(40.0)
    assert (none["sub"], none["del"], none["ins"]) == pytest.approx((40.0, 0.0, 0.0))
    assert buckets[">15%"]["del"] == pytest.approx(100 / 3)
    assert buckets["0-5%"]["ins"] == pytest.approx(50.0)
    assert "vs_base" not in none
    assert got["by_values"] == []  # FLEURS has no split column of its own


def test_a_sets_own_split_values_are_offered(tmp_path: Path) -> None:
    from app.services.error_store import breakdown

    clips = [_clip("a", "म", "म") | {"by": "read"}, _clip("b", "म", "म") | {"by": "conversation"}]
    path = tmp_path / "iv.parquet"
    error_mining.write(error_mining.rows("run", "indicvoices", clips), path)
    assert breakdown(path)["by_values"] == ["conversation", "read"]


def test_the_numbers_block_by_hand(run: Path) -> None:
    from app.services.error_store import breakdown

    numbers = breakdown(run)["numbers"]
    assert (numbers["errors"], numbers["sub"], numbers["del"], numbers["ins"]) == (2, 1, 1, 0)
    assert numbers["share_of_errors"] == pytest.approx(0.4)
    # Every row that involves a number left out: 9 words, 3 errors.
    assert numbers["wer_without"] == pytest.approx(100 * 3 / 9)
    assert (numbers["ref_clips"], numbers["ref_clip_words"]) == (2, 4)
    assert numbers["ref_clip_wer"] == pytest.approx(50.0)


def test_against_a_base_each_block_carries_the_difference(run: Path, base: Path) -> None:
    from app.services.error_store import breakdown

    got = breakdown(run, base=base)
    diff, lo, hi = got["vs_base"]["wer"]
    assert diff == pytest.approx(100 * 4 / 11)
    assert lo <= diff <= hi
    assert got["vs_base"]["clips"] == 6
    buckets = {b["bucket"]: b for b in got["overlap"]}
    assert buckets["none"]["vs_base"]["wer"][0] == pytest.approx(40.0)
    assert buckets["0-5%"]["vs_base"]["wer"][0] == pytest.approx(0.0)
    assert got["numbers"]["vs_base"]["errors"] == 2
    assert got["numbers"]["vs_base"]["wer_without"] == pytest.approx(100 * 2 / 9)


def test_the_confusion_table(run: Path, base: Path) -> None:
    from app.services.error_store import ErrorFilter, confusion

    subs = confusion(run, ErrorFilter(kind=["sub"]))
    assert subs["total"] == 3
    assert {(r["ref"], r["hyp"], r["count"]) for r in subs["rows"]} == {
        ("पाँच", "6", 1), ("राम्रो", "नराम्रो", 1), ("नराम्रो", "राम्रो", 1),
    }  # fmt: skip
    assert all(r["share"] == pytest.approx(1 / 3) for r in subs["rows"])
    folded = confusion(run, ErrorFilter(kind=["sub"]), both_ways=True)
    top = folded["rows"][0]
    assert (top["count"], top["forward"], top["backward"]) == (2, 1, 1)
    assert {top["ref"], top["hyp"]} == {"राम्रो", "नराम्रो"}
    errors = confusion(run, ErrorFilter(kind=["sub", "del", "ins"]), base=base, sort="change")
    ins = next(r for r in errors["rows"] if r["kind"] == "ins")
    assert (ins["hyp"], ins["count"], ins["base_count"], ins["change"]) == ("थप", 1, 1, 0)
    assert errors["rows"][-1]["change"] == 0  # the one row that did not grow sorts last
    numbers = confusion(run, ErrorFilter(kind=["sub", "del"], number=True))
    assert {(r["kind"], r["ref"]) for r in numbers["rows"]} == {("sub", "पाँच"), ("del", "45")}
    page = confusion(run, ErrorFilter(kind=["sub"]), limit=1, offset=1)
    assert (page["total"], len(page["rows"])) == (3, 1)


@pytest.mark.parametrize(
    ("flt", "clips"),
    [
        ({"overlap_bucket": ">15%"}, {"c2"}),
        ({"ref_script": "none"}, {"c2"}),  # 45 is digits; पाँच is Devanagari
        ({"hyp_script": "none"}, {"c4"}),
        ({"similarity_min": 0.5, "kind": ["sub"]}, {"c5", "c6"}),
        ({"forgiven": "spelling"}, set()),
    ],
)
def test_filters(run: Path, flt: dict, clips: set[str]) -> None:
    from app.services.error_store import ErrorFilter, occurrences

    found = occurrences(run, ErrorFilter(**({"kind": ["sub", "del", "ins"]} | flt)))
    assert {r["clip_id"] for r in found["rows"]} == clips


def test_rows_filter_by_their_clips_snr_bucket(tmp_path: Path) -> None:
    from app.services.error_store import confusion, occurrences

    clips = [
        _clip("loud", "म घर", "म") | {"snr_db": 50.0},
        _clip("noisy", "म घर", "म") | {"snr_db": 9.0},
        _clip("unknown", "म घर", "म"),
    ]
    path = tmp_path / "gold.parquet"
    error_mining.write(error_mining.rows("run", "gold", clips), path)
    noisy = occurrences(path, ErrorFilter(kind=["del"], snr_bucket="<15 dB"))
    assert [(r["clip_id"], r["snr_bucket"]) for r in noisy["rows"]] == [("noisy", "<15 dB")]
    table = confusion(path, ErrorFilter(kind=["del"], snr_bucket="45+ dB"))
    assert (table["total"], table["rows"][0]["count"]) == (1, 1)
    assert occurrences(path, ErrorFilter(snr_bucket="unmeasured"))["total"] == 2


def test_occurrences_carry_their_context_and_sample_by_seed(run: Path) -> None:
    from app.services.error_store import ErrorFilter, occurrences

    (one,) = occurrences(run, ErrorFilter(kind=["del"]))["rows"]
    assert (one["clip_id"], one["ref"], one["hyp"]) == ("c2", ["45"], [])
    assert [op["ref"] for op in one["before"]] == [["घर"]]
    assert [op["ref"] for op in one["after"]] == [["दिन"]]
    pair = occurrences(run, ErrorFilter(kind=["sub"]), ref="पाँच", hyp="6")
    assert [r["clip_id"] for r in pair["rows"]] == ["c4"]
    a = occurrences(run, ErrorFilter(kind=["sub", "del", "ins"]), sample=3, seed=7)
    b = occurrences(run, ErrorFilter(kind=["sub", "del", "ins"]), sample=3, seed=7)
    assert [r["clip_id"] for r in a["rows"]] == [r["clip_id"] for r in b["rows"]]
    assert (a["total"], len(a["rows"])) == (5, 3)


def test_one_clip_in_order(run: Path) -> None:
    from app.services.error_store import clip_ops

    assert clip_ops(run, "c2") == [
        {"kind": "match", "ref": ["घर"], "hyp": ["घर"], "similarity": 1.0},
        {"kind": "del", "ref": ["45"], "hyp": [], "similarity": 1.0},
        {"kind": "match", "ref": ["दिन"], "hyp": ["दिन"], "similarity": 1.0},
    ]
    assert clip_ops(run, "nope") == []


def test_it_resamples_as_model_eval_does_without_importing_it() -> None:
    """The notebooks carry this module without the database layer model_eval pulls in."""
    import subprocess
    import sys

    from app.services import error_store, model_eval

    assert (error_store.BOOTSTRAP_ROUNDS, error_store.BOOTSTRAP_SEED) == (
        model_eval.BOOTSTRAP_ROUNDS,
        model_eval.BOOTSTRAP_SEED,
    )
    code = (
        "import sys, app.services.error_store; "
        "print(sorted(m for m in sys.modules if m.startswith(('sqlalchemy', 'app.'))))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "sqlalchemy" not in loaded.stdout
    assert "app.services.model_eval" not in loaded.stdout


def test_a_report_is_the_breakdown_and_the_top_rows_of_each_kind(run: Path) -> None:
    """What a notebook stores beside a WER: the page's own queries, written down."""
    from app.services.error_store import breakdown, report

    got = report(run, top=2)
    assert {k: got[k] for k in ("wer", "overlap", "numbers")} == {
        k: breakdown(run)[k] for k in ("wer", "overlap", "numbers")
    }
    assert [len(got["top"][k]) for k in ("sub", "del", "ins")] == [2, 1, 1]
    assert got["top"]["del"][0]["ref"] == "45"
    assert got["top"]["sub"][0]["share"] == pytest.approx(1 / 3)


# --- the attribution card (D111) -----------------------------------------------------------------


def test_each_error_is_one_kind_and_the_card_reads_both_conditions(tmp_path: Path) -> None:
    from app.services.error_store import breakdown

    def clip(cid, ref, hyp, share=0.0, snr=50.0):
        return _clip(cid, ref, hyp, share=share) | {"snr_db": snr}

    clips = [
        clip("en", "cache", "cage"),
        clip("similar", "रहेको", "रहेका"),
        clip("other", "घर", "कुकुर"),
        clip("script", "Captain", "क्याट्रिना"),
        clip("number", "45", "46"),
        clip("del", "म घर", "म"),
        clip("ins", "म", "म थप"),
        clip("noisy", "म घर", "म घर", snr=10.0),
        clip("talk", "म घर", "म", share=0.5, snr=10.0),  # crosstalk wins over SNR
    ]
    path = tmp_path / "gold.parquet"
    error_mining.write(error_mining.rows("run", "gold", clips), path)
    got = breakdown(path)["attribution"]
    assert got["wer"] == pytest.approx(breakdown(path)["wer"])
    assert [(c["factor"], c["bucket"], c["clips"]) for c in got["conditions"]] == [
        ("crosstalk", ">15%", 1),
        ("snr", "<15 dB", 1),
    ]
    # Every clip is in group g1, so the noisy clip's 0 errors give SNR a ratio of 0: no cost.
    noisy = got["conditions"][1]
    assert noisy["within"]["points"] == 0.0
    # The crosstalk clip's deletion is attributed by its ratio; the rest, one error per kind.
    xt = got["conditions"][0]
    ratio = xt["within"]["ratio"]
    kinds = {k["kind"]: k["points"] for k in got["rest"]["within"]["kinds"]}
    words = got["ref_words"]
    expected = dict.fromkeys(kinds, 100 / words) | {"deletion": 100 * (1 + 1 / ratio) / words}
    assert kinds == pytest.approx(expected)


def test_against_a_base_the_card_says_which_rows_moved(run: Path, base: Path) -> None:
    from app.services.error_store import breakdown

    got = breakdown(run, base=base)
    vs = got["attribution"]["vs_base"]
    assert vs["clips"] == 6
    assert vs["wer"][0] == pytest.approx(got["vs_base"]["wer"][0])
    kinds = sum(v[0] for v in vs["kinds"].values())
    conditions = sum(v[0] or 0.0 for v in vs["factors"].values() if v[0] is not None)
    assert kinds + conditions == pytest.approx(vs["wer"][0])
    assert "vs_base" not in breakdown(run)["attribution"]


# --- the rulebook's evidence (fold-v4) -----------------------------------------------------------


def test_each_rule_and_tag_counts_the_pairs_it_forgave_or_tagged(tmp_path: Path) -> None:
    from app.services.error_store import rule_evidence

    clips = [
        _clip("c1", "होइन B दावी अनि म म जान्छु", "हैन बी दाबी अनि म जान्छु"),
        _clip("c2", "होइन राम्रो", "हैन नराम्रो"),
    ]
    _file(tmp_path, "r", "gold", clips, name="gold.parquet")
    found = rule_evidence(tmp_path / "gold.parquet")
    assert found["words"] == 9
    rules = found["rules"]
    assert rules["contracted-verb"]["pairs"] == 2
    assert rules["contracted-verb"]["per_100"] == pytest.approx(200 / 9)
    assert rules["contracted-verb"]["top"] == [{"ref": "होइन", "hyp": "हैन", "count": 2}]
    assert rules["letter-names"]["pairs"] == 1
    assert rules["ba-va"]["pairs"] == 1 and rules["repetition"]["pairs"] == 1
    assert set(rules) == {"contracted-verb", "letter-names", "ba-va", "repetition"}


def test_occurrences_filter_by_rule_and_tag(tmp_path: Path) -> None:
    from app.services.error_store import occurrences

    clips = [_clip("c1", "होइन B दावी", "हैन बी दाबी")]
    _file(tmp_path, "r", "gold", clips, name="gold.parquet")
    path = tmp_path / "gold.parquet"
    by_rule = occurrences(path, ErrorFilter(fold_rule="letter-names"))
    assert [(r["ref"], r["hyp"]) for r in by_rule["rows"]] == [(["B"], ["बी"])]
    by_tag = occurrences(path, ErrorFilter(variant="ba-va"))
    assert [(r["ref"], r["hyp"]) for r in by_tag["rows"]] == [(["दावी"], ["दाबी"])]
