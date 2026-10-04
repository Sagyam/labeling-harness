"""The one evaluation every model goes through (D105): notebooks/src/evalkit.py.

It runs in Colab, not in the app, but it is what makes two notebooks' numbers comparable and what
the paper's tables are built from, so its rules are tested here. The scorer is a stand-in with the
shape of ``ftkit.harness_scorer``: the real one needs the dataset's own fold.py at run time."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_SRC = Path(__file__).resolve().parents[2] / "notebooks" / "src"
sys.path.insert(0, str(_SRC))  # evalkit imports the distill and sweep kits beside it
_spec = importlib.util.spec_from_file_location("evalkit", _SRC / "evalkit.py")
evalkit = importlib.util.module_from_spec(_spec)
sys.modules["evalkit"] = evalkit
_spec.loader.exec_module(evalkit)


class _Score:
    """Word-by-position errors: enough to tell a right transcript from a wrong one."""

    @staticmethod
    def per_clip(refs, hyps):
        out = []
        for ref, hyp in zip(refs, hyps, strict=True):
            r, h = ref.split(), hyp.split()
            sub = sum(a != b for a, b in zip(r, h, strict=False))
            dele, ins = max(0, len(r) - len(h)), max(0, len(h) - len(r))
            errors = sub + dele + ins
            out.append(
                {"words": len(r), "errors": errors, "sub": sub, "del": dele, "ins": ins,
                 "raw_words": len(r), "raw_errors": errors, "chars": len(ref),
                 "char_errors": errors, "loop": 0}
            )  # fmt: skip
        return out

    @staticmethod
    def summarize(clips):
        t = {k: sum(c[k] for c in clips) for k in clips[0]}
        words = max(t["words"], 1)
        return {
            "wer": 100 * t["errors"] / words,
            "raw_wer": 100 * t["raw_errors"] / max(t["raw_words"], 1),
            "cer": 100 * t["char_errors"] / max(t["chars"], 1),
            "sub": 100 * t["sub"] / words,
            "del": 100 * t["del"] / words,
            "ins": 100 * t["ins"] / words,
            "loops": t["loop"],
            "clips": len(clips),
        }


class _Forgiving(_Score):
    """A newer fold: an ``x`` in the transcript is forgiven."""

    @staticmethod
    def per_clip(refs, hyps):
        fixed = []
        for ref, hyp in zip(refs, hyps, strict=True):
            r = ref.split()
            words = [r[i] if w == "x" and i < len(r) else w for i, w in enumerate(hyp.split())]
            fixed.append(" ".join(words))
        return _Score.per_clip(refs, fixed)


def _row(sid: str, text: str, episode: str = "e1", **classes: str) -> dict:
    return {
        "segment_id": sid,
        "episode_id": episode,
        "text": text,
        "start_time": 0.0,
        "end_time": 4.0,
        "classes": classes,
    }


def _splits() -> dict[str, list[dict]]:
    return {
        "gold": [
            _row("g1", "एक दुई तीन चार", "e1", overlap="none"),
            _row("g2", "one two three four", "e2", overlap=">15%"),
            _row("g3", "पाँच छ सात आठ", "e3", overlap="none"),
        ],
        "val": [_row("v1", "क ख ग घ", "e4", overlap="none"), _row("v2", "a b c d", "e5")],
    }


def _decoder(hyps: dict[str, str], log=(), calls: list | None = None):
    def decode(rows):
        if calls is not None:
            calls.append([r["segment_id"] for r in rows])
        texts = [hyps.get(r["segment_id"], r["text"]) for r in rows]
        return (
            texts,
            [0.4] * len(rows),
            [entry for entry in log if entry[0] in {r["segment_id"] for r in rows}],
        )

    return decode


# --- the export every model shares ---------------------------------------------------------------


def test_the_export_is_recognised_by_its_date():
    evalkit.check_export({"exported_at": "2026-09-30T11:42:29.894123+00:00"}, "2026-09-30")


@pytest.mark.parametrize("manifest", [{"exported_at": "2026-09-26T08:00:00+00:00"}, {}])
def test_another_export_is_refused(manifest):
    with pytest.raises(ValueError, match="2026-09-30"):
        evalkit.check_export(manifest, "2026-09-30")


def test_an_empty_expectation_is_refused_rather_than_matching_everything():
    with pytest.raises(ValueError):
        evalkit.check_export({"exported_at": "2026-09-30T11:42:29+00:00"}, "")


def test_kit_digests_name_the_code_that_ran(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("x = 2\n")
    (tmp_path / "notes.txt").write_text("not a kit")
    digests = evalkit.kit_digests(tmp_path)
    assert list(digests) == ["a.py", "b.py"]
    assert digests["a.py"] != digests["b.py"] and len(digests["a.py"]) == 16
    assert evalkit.kit_digests(tmp_path) == digests


# --- transcripts and pairing ---------------------------------------------------------------------


def test_transcripts_round_trip_in_the_models_page_format(tmp_path: Path):
    rows = _splits()["gold"]
    evalkit.write_hyps(tmp_path / "harness" / "gold.jsonl", rows, ["क", "b", "ग"], [0.1, 0.2, 0.3])
    assert evalkit.read_hyps(tmp_path / "harness" / "gold.jsonl") == {
        "g1": "क",
        "g2": "b",
        "g3": "ग",
    }
    first = json.loads((tmp_path / "harness" / "gold.jsonl").read_text("utf-8").splitlines()[0])
    assert first == {"segment_id": "g1", "text": "क", "compute_s": 0.1}


def test_a_reference_is_scored_on_the_clips_it_has():
    rows = _splits()["gold"]
    ref = evalkit.reference_counts(rows, {"g1": "एक दुई तीन चार", "g3": "पाँच छ x y"}, _Score)
    assert ref == {"g1": [0, 4], "g3": [2, 4]}


def test_a_pair_is_taken_on_the_clips_both_runs_scored():
    rows = _splits()["gold"]
    a = {"g1": [0, 4], "g2": [0, 4]}
    b = {"g1": [2, 4], "g2": [0, 4], "g3": [4, 4]}
    out = evalkit.pair(rows, a, b, keys=("overlap",), n=200)
    assert out["clips"] == 2
    assert out["all"][0] == pytest.approx(25.0)  # 2 errors more over 8 words; g3 is not shared
    assert out["overlap=none"][0] == pytest.approx(50.0)
    assert out["overlap=>15%"][0] == pytest.approx(0.0)
    assert all(lo <= d <= hi for d, lo, hi in (out["all"], out["overlap=none"]))


def test_runs_that_share_no_clip_cannot_be_paired():
    with pytest.raises(ValueError):
        evalkit.pair(_splits()["gold"], {"g1": [0, 4]}, {"g2": [0, 4]})


def test_a_pair_resamples_the_group_it_is_told_to():
    rows = [{**_row(f"c{i}", "a b"), "speaker": "only"} for i in range(6)]
    a = {r["segment_id"]: [0, 2] for r in rows}
    b = {r["segment_id"]: [i % 2, 2] for i, r in enumerate(rows)}
    d, lo, hi = evalkit.pair(rows, a, b, keys=(), group="speaker", n=200)["all"]
    assert lo == d == hi  # one speaker: every resample is the same clips


# --- a split --------------------------------------------------------------------------------------


def test_a_split_is_scored_overall_and_per_clip_class():
    rows = _splits()["gold"]
    m, per = evalkit.score_split(
        rows, ["एक दुई तीन चार", "one two x y", "पाँच छ सात आठ"], [1.0, 2.0, 3.0], [], _Score
    )
    assert m["wer"] == pytest.approx(100 * 2 / 12)
    assert m["sub"] + m["del"] + m["ins"] == pytest.approx(m["wer"])
    assert m["rtf"] == pytest.approx(6.0 / 12.0)
    assert m["by_class"]["overlap"]["none"]["wer"] == 0.0
    assert m["by_class"]["overlap"][">15%"]["wer"] == pytest.approx(50.0)
    assert per == {"g1": [0, 4], "g2": [2, 4], "g3": [0, 4]}
    assert "greedy_only" not in m and m["retried"] == []


def test_the_greedy_only_score_is_the_same_decode_before_its_retries():
    rows = _splits()["gold"]
    log = [("g2", "x x x x", "one two three four")]
    m, _ = evalkit.score_split(rows, [r["text"] for r in rows], [1.0] * 3, log, _Score)
    assert m["wer"] == 0.0
    assert m["greedy_only"]["wer"] == pytest.approx(100 * 4 / 12)
    assert m["retried"] == [{"segment_id": "g2", "first": "x x x x", "retry": "one two three four"}]


# --- a run ----------------------------------------------------------------------------------------


def test_a_run_writes_what_the_models_page_and_the_next_notebook_read(tmp_path: Path):
    calls: list = []
    row = evalkit.evaluate_run(
        tmp_path,
        "vanilla-s0",
        splits=_splits(),
        decode=_decoder({"g2": "one two x y"}, calls=calls),
        score=_Score,
        card={"name": "Flex FT", "decoder": "greedy+retry"},
        meta={"recipe": "vanilla", "seed": 0, "best_epoch": 4},
        keys=("overlap",),
    )
    assert calls == [["g1", "g2", "g3"], ["v1", "v2"]]  # gold first
    assert (row["run_name"], row["recipe"], row["seed"], row["best_epoch"]) == (
        "vanilla-s0",
        "vanilla",
        0,
        4,
    )
    assert row["val_wer"] == 0.0 and row["gold_wer"] == pytest.approx(100 * 2 / 12)
    assert row["gold"]["vs"] == {} and row["gold"]["rtf"] == pytest.approx(0.1)
    assert row["gold_by_class"]["overlap"][">15%"]["wer"] == pytest.approx(50.0)
    assert json.loads((tmp_path / "result.json").read_text("utf-8")) == row
    card = json.loads((tmp_path / "harness" / "model_card.json").read_text("utf-8"))
    assert (card["name"], card["run_name"], card["recipe"]) == ("Flex FT", "vanilla-s0", "vanilla")
    assert card["gold_wer"] == row["gold_wer"] and set(card["gold_sid"]) == {"sub", "del", "ins"}
    assert card["created_at"].endswith("+00:00")
    assert evalkit.read_hyps(tmp_path / "harness" / "gold.jsonl")["g2"] == "one two x y"
    assert json.loads((tmp_path / "per_clip.json").read_text())["gold"]["g2"] == [2, 4]
    assert "by_class" in json.loads((tmp_path / "gold_metrics.json").read_text("utf-8"))


def test_a_run_is_paired_against_each_reference_that_scored_its_clips(tmp_path: Path):
    references = {
        "teacher": {
            "gold": {"g1": [0, 4], "g2": [0, 4], "g3": [0, 4]},
            "val": {"v1": [1, 4], "v2": [1, 4]},
        },
        "old-gold-only": {"gold": {"g1": [4, 4]}},  # an earlier run: one gold clip, no val
        "stranger": {"gold": {"zz": [0, 4]}},
    }
    row = evalkit.evaluate_run(
        tmp_path, "student", splits=_splits(), decode=_decoder({"g2": "one two x y"}),
        score=_Score, card={}, references=references, keys=(),
    )  # fmt: skip
    assert set(row["gold"]["vs"]) == {"teacher", "old-gold-only"}
    assert row["gold"]["vs"]["teacher"]["all"][0] == pytest.approx(100 * 2 / 12)
    assert row["gold"]["vs"]["old-gold-only"]["clips"] == 1
    assert set(row["val"]["vs"]) == {"teacher"}
    assert row["val"]["vs"]["teacher"]["all"][0] == pytest.approx(-25.0)


# --- scoring a finished run again under newer rules -----------------------------------------------


def _finished_run(tmp_path: Path) -> None:
    evalkit.evaluate_run(
        tmp_path, "vanilla-s0", splits=_splits(),
        decode=_decoder({"g2": "one two x y"}, log=[("g2", "x x x x", "one two x y")]),
        score=_Score, card={"name": "Flex FT", "fold_version": "fold-v3"},
        meta={"recipe": "vanilla", "seed": 0}, keys=("overlap",),
    )  # fmt: skip


def test_a_finished_run_is_scored_again_from_its_transcripts_without_decoding(tmp_path: Path):
    _finished_run(tmp_path)
    before = json.loads((tmp_path / "harness" / "model_card.json").read_text("utf-8"))
    assert before["gold_wer"] == pytest.approx(100 * 2 / 12)

    row = evalkit.rescore_run(
        tmp_path, "vanilla-s0", splits=_splits(), score=_Forgiving, fold_version="fold-v4",
        keys=("overlap",),
    )  # fmt: skip

    assert row["gold_wer"] == pytest.approx(100 * 1 / 12)  # the x is forgiven, the y is not
    assert (row["run_name"], row["recipe"], row["seed"]) == ("vanilla-s0", "vanilla", 0)
    card = json.loads((tmp_path / "harness" / "model_card.json").read_text("utf-8"))
    assert (card["name"], card["fold_version"], card["gold_wer"]) == (
        "Flex FT", "fold-v4", row["gold_wer"],
    )  # fmt: skip
    assert card["created_at"] == before["created_at"] and card["rescored_at"].endswith("+00:00")
    gold = json.loads((tmp_path / "gold_metrics.json").read_text("utf-8"))
    assert gold["retried"] == [{"segment_id": "g2", "first": "x x x x", "retry": "one two x y"}]
    assert gold["rtf"] == pytest.approx(0.1)  # the compute the decode took, kept
    assert evalkit.read_hyps(tmp_path / "harness" / "gold.jsonl")["g2"] == "one two x y"
    assert json.loads((tmp_path / "per_clip.json").read_text())["gold"]["g2"] == [1, 4]
    assert json.loads((tmp_path / "result.json").read_text("utf-8")) == row


def test_a_finished_run_is_paired_again_against_its_references(tmp_path: Path):
    _finished_run(tmp_path)
    row = evalkit.rescore_run(
        tmp_path, "vanilla-s0", splits=_splits(), score=_Score, fold_version="fold-v4",
        references={"teacher": {"gold": {"g1": [0, 4], "g2": [0, 4], "g3": [0, 4]}}}, keys=(),
    )  # fmt: skip
    assert row["gold"]["vs"]["teacher"]["all"][0] == pytest.approx(100 * 2 / 12)


def test_a_clip_the_run_never_transcribed_is_refused(tmp_path: Path):
    _finished_run(tmp_path)
    splits = _splits()
    splits["gold"].append(_row("g4", "a b"))
    with pytest.raises(ValueError, match="g4"):
        evalkit.rescore_run(tmp_path, "vanilla-s0", splits=splits, score=_Score, fold_version="v")


def test_a_public_set_is_scored_again_from_its_lines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        evalkit, "load_benchmark", lambda name, work, token, limit=None: _bench_rows()
    )
    evalkit.run_benchmarks(
        tmp_path, decode=_decoder({"iv-00001": "a b x y"}), score=_Score, work=tmp_path / "w",
        token="t", names=("indicvoices",), limit=3,
    )  # fmt: skip
    before = json.loads((tmp_path / "benchmarks" / "indicvoices.json").read_text("utf-8"))
    assert before["wer"] == pytest.approx(100 * 2 / 11)

    summary = evalkit.rescore_benchmark(tmp_path, "indicvoices", score=_Forgiving)

    assert summary["wer"] == pytest.approx(100 * 1 / 11)
    assert summary["by"]["Conversation"]["wer"] == pytest.approx(25.0)
    assert {k: summary[k] for k in ("hours", "x_realtime", "retried", "limit")} == {
        k: before[k] for k in ("hours", "x_realtime", "retried", "limit")
    }
    assert json.loads((tmp_path / "benchmarks" / "indicvoices.json").read_text("utf-8")) == summary
    lines = (tmp_path / "benchmarks" / "indicvoices.jsonl").read_text("utf-8").splitlines()
    assert json.loads(lines[1]) == {"id": "iv-00001", "group": "s1", "ref": "a b c d",
                                    "hyp": "a b x y", "errors": 1, "words": 4,
                                    "scenario": "Conversation"}  # fmt: skip


# --- plain WER ------------------------------------------------------------------------------------


def test_plain_tokens_drop_punctuation_and_lowercase_latin_and_nothing_else():
    assert evalkit.plain_tokens("सेनाको आगमन हुनुअघि, Haiti -ले 1800 को।") == [
        "सेनाको", "आगमन", "हुनुअघि", "haiti", "ले", "1800", "को",
    ]  # fmt: skip
    assert evalkit.plain_tokens("टिम") != evalkit.plain_tokens("team")  # no fold
    assert evalkit.plain_tokens(None) == evalkit.plain_tokens("  ") == []


def test_plain_tokens_compose_a_decomposed_letter():
    # क + nukta (two code points) is NFC-composed the same way whichever form the text used
    assert evalkit.plain_tokens("क़लम") == evalkit.plain_tokens("क़लम")


@pytest.mark.parametrize(
    ("ref", "hyp", "errors"),
    [("a b c", "a b c", 0), ("a b c", "a x c", 1), ("a b c", "a c", 1), ("a b", "a x b y", 2),
     ("", "a b", 2), ("a b", "", 2)],
)  # fmt: skip
def test_edit_distance_counts_substitutions_deletions_and_insertions(ref, hyp, errors):
    assert evalkit.edit_distance(ref.split(), hyp.split()) == errors


def test_plain_wer_is_pooled_over_the_clips():
    assert evalkit.plain_wer(["a b c d", "e f"], ["a b c d", "x y"]) == pytest.approx(100 * 2 / 6)
    assert evalkit.plain_wer(["Team, गयो।"], ["team गयो"]) == 0.0


def test_plain_cer_counts_characters_of_the_plain_text_with_its_spaces():
    # "ab cd" is 5 characters; "abcd" drops the space: one deletion, where plain WER charges 2
    assert evalkit.plain_cer(["ab cd"], ["abcd"]) == pytest.approx(100 * 1 / 5)
    assert evalkit.plain_wer(["ab cd"], ["abcd"]) == pytest.approx(100 * 2 / 2)
    assert evalkit.plain_cer(["Team, गयो।"], ["team गयो"]) == 0.0


def test_plain_cer_is_pooled_over_the_clips():
    assert evalkit.plain_cer(["abc", "d"], ["abc", "x"]) == pytest.approx(100 * 1 / 4)
    assert evalkit.plain_cer([""], ["a"]) == 100.0  # no reference characters: never a division by 0


# --- the public sets ------------------------------------------------------------------------------


def test_the_registry_is_the_five_sets_of_2026_09_27():
    assert list(evalkit.BENCHMARKS) == [
        "fleurs",
        "slr54",
        "common_voice",
        "indicvoices",
        "nepali_cs",
    ]
    assert {b.name: b.base_wer for b in evalkit.BENCHMARKS.values()} == {
        "fleurs": 11.10,
        "slr54": 8.17,
        "common_voice": 8.78,
        "indicvoices": 12.70,
        "nepali_cs": 10.98,
    }
    assert [b.name for b in evalkit.BENCHMARKS.values() if b.layout == "tar"] == ["common_voice"]
    # FLEURS repeats sentences and nepali_cs cuts videos: neither is resampled by speaker
    assert (evalkit.BENCHMARKS["fleurs"].group, evalkit.BENCHMARKS["nepali_cs"].group) == (
        "id",
        "video_id",
    )


def test_a_benchmark_clip_is_named_as_the_2026_09_27_files_named_it():
    audio = np.zeros(32_000, dtype=np.int16)
    cs = evalkit.benchmark_row(
        evalkit.BENCHMARKS["nepali_cs"],
        {
            "video_id": "-3PhO-CA2iw",
            "start_ms": 108368,
            "transcription": "MCQ हरू",
            "speech_type": "cs",
        },
        3,
        audio,
    )
    assert (cs["segment_id"], cs["episode_id"], cs["speech_type"]) == (
        "-3PhO-CA2iw-108368",
        "-3PhO-CA2iw",
        "cs",
    )
    assert (cs["start_time"], cs["end_time"], cs["text"]) == (0.0, 2.0, "MCQ हरू")
    fleurs = evalkit.benchmark_row(
        evalkit.BENCHMARKS["fleurs"],
        {"path": "test/10022685377830883165.wav", "id": 1947, "raw_transcription": "क"},
        0,
        audio,
    )
    assert (fleurs["segment_id"], fleurs["episode_id"]) == ("10022685377830883165.wav", "1947")
    iv = evalkit.benchmark_row(
        evalkit.BENCHMARKS["indicvoices"],
        {"speaker_id": "S42", "text": None, "scenario": "Read"},
        7,
        audio,
    )
    assert (iv["segment_id"], iv["text"], iv["scenario"]) == ("iv-00007", "", "Read")


def _bench_rows() -> list[dict]:
    bench = evalkit.BENCHMARKS["indicvoices"]
    audio = np.zeros(16_000, dtype=np.int16)
    records = [
        {"speaker_id": "s1", "text": "क ख ग घ", "scenario": "Read"},
        {"speaker_id": "s1", "text": "a b c d", "scenario": "Conversation"},
        {"speaker_id": "s2", "text": "Team, गयो। अब", "scenario": "Read"},
    ]
    return [evalkit.benchmark_row(bench, r, i, audio) for i, r in enumerate(records)]


def test_a_public_set_is_scored_folded_plain_and_by_its_own_column():
    rows = _bench_rows()
    summary, lines = evalkit.score_benchmark(
        "indicvoices", rows, ["क ख ग घ", "a b x y", "team गयो अब"], _Score
    )
    assert summary["clips"] == 3 and summary["hours"] == pytest.approx(3 / 3600)
    assert summary["plain_wer"] == pytest.approx(100 * 2 / 11)
    assert summary["plain_cer"] == pytest.approx(100 * 2 / 25)  # 7 + 7 + 11 characters
    assert summary["by"]["Conversation"]["wer"] == pytest.approx(50.0)
    assert set(summary["by"]) == {"Read", "Conversation"}
    assert lines[1] == {"id": "iv-00001", "group": "s1", "ref": "a b c d", "hyp": "a b x y",
                        "errors": 2, "words": 4, "scenario": "Conversation"}  # fmt: skip


def test_two_models_are_paired_on_a_public_set_by_its_group():
    rows = _bench_rows()
    _, base = evalkit.score_benchmark("indicvoices", rows, [r["text"] for r in rows], _Score)
    _, tuned = evalkit.score_benchmark(
        "indicvoices", rows, ["क ख ग घ", "a b x y", "Team, गयो। अब"], _Score
    )
    out = evalkit.pair_benchmark(base, tuned, n=200)
    assert out["clips"] == 3 and out["all"][0] == pytest.approx(100 * 2 / 11)
    assert list(out) == ["clips", "all"]


def test_public_sets_are_written_one_at_a_time_and_a_finished_one_is_skipped(
    tmp_path: Path, monkeypatch
):
    loaded = []

    def fake_load(name, work, token, limit=None):
        loaded.append((name, limit))
        return _bench_rows()

    monkeypatch.setattr(evalkit, "load_benchmark", fake_load)
    monkeypatch.setitem(evalkit.BENCHMARKS, "fleurs", evalkit.BENCHMARKS["indicvoices"])
    summaries = evalkit.run_benchmarks(
        tmp_path, decode=_decoder({}), score=_Score, work=tmp_path / "w", token="t",
        names=("fleurs", "indicvoices"), limit=3, done={"fleurs"},
    )  # fmt: skip
    assert loaded == [("indicvoices", 3)] and list(summaries) == ["indicvoices"]
    saved = json.loads((tmp_path / "benchmarks" / "indicvoices.json").read_text("utf-8"))
    assert (
        saved["wer"] == 0.0
        and saved["limit"] == 3
        and saved["x_realtime"] == pytest.approx(3 / 1.2)
    )
    assert len((tmp_path / "benchmarks" / "indicvoices.jsonl").read_text("utf-8").splitlines()) == 3
    assert not (tmp_path / "benchmarks" / "fleurs.json").exists()


def test_encoded_audio_becomes_sixteen_kilohertz_mono_int16():
    stereo = np.stack(
        [np.full(1600, 0.5, dtype=np.float32), np.full(1600, -0.25, dtype=np.float32)], axis=1
    )
    buffer = io.BytesIO()
    sf.write(buffer, stereo, 16_000, format="WAV", subtype="FLOAT")
    audio = evalkit._audio(buffer.getvalue())
    assert audio.dtype == np.int16 and audio.shape == (1600,)
    assert int(audio[0]) == round(0.125 * 32767)  # the two channels averaged


def test_a_public_set_streams_one_clip_at_a_time(tmp_path: Path, monkeypatch):
    """iter_benchmark decodes a record only when it is asked for: the overlap pass holds one
    clip in RAM. load_benchmark is that iterator collected, cut at `limit`."""
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16_000, format="WAV")
    pulled = []

    def records(bench, work, token):
        for k in range(5):
            pulled.append(k)
            yield {"path": f"a/{k}.wav", "raw_transcription": "x", "id": k}, buffer.getvalue()

    monkeypatch.setattr(evalkit, "_parquet_records", records)
    clips = evalkit.iter_benchmark("fleurs", tmp_path, "t")
    first = next(clips)
    assert (first["segment_id"], first["episode_id"], pulled) == ("0.wav", "0", [0])
    assert [r["segment_id"] for r in evalkit.load_benchmark("fleurs", tmp_path, "t", limit=2)] == [
        "0.wav",
        "1.wav",
    ]


def test_error_mining_knows_every_public_set_and_its_split_column():
    from app.services import error_mining

    assert ("gold", "val", *evalkit.BENCHMARKS) == error_mining.SETS
    assert {b.name: b.by for b in evalkit.BENCHMARKS.values() if b.by} == error_mining.SET_BY


# --- error mining (D110) ------------------------------------------------------------------------


class _FoldScore(_Score):
    """The shape of ftkit.harness_scorer, on the harness's own fold: each clip keeps the alignment
    it was counted from, so mining aligns nothing again."""

    @staticmethod
    def per_clip(refs, hyps):
        from app.services.fold import word_errors

        out = []
        for ref, hyp in zip(refs, hyps, strict=True):
            a = word_errors(ref, hyp)
            out.append(
                {"words": a.ref_words, "errors": a.errors, "sub": a.substitutions,
                 "del": a.deletions, "ins": a.insertions, "raw_words": a.ref_words,
                 "raw_errors": a.errors, "chars": len(ref), "char_errors": a.errors, "loop": 0,
                 "alignment": a}
            )  # fmt: skip
        return out

    @staticmethod
    def summarize(clips):
        return _Score.summarize([{k: v for k, v in c.items() if k != "alignment"} for c in clips])


def _sid(found):
    errors = [r for r in found if r["kind"] in ("sub", "del", "ins")]
    return len(errors), sum(len(r["ref"]) for r in found)


def test_a_run_writes_error_files_whose_rows_reproduce_its_scores(tmp_path: Path):
    from app.services import error_mining

    splits = _splits()
    splits["gold"][0]["overlap_share"] = 0.2
    splits["gold"][0]["acoustics"] = {"version": "acoustics-v2", "snr_db": 30.0}
    row = evalkit.evaluate_run(
        tmp_path, "vanilla-s0", splits=splits, decode=_decoder({"g2": "one two x y"}),
        score=_FoldScore, card={"name": "Flex FT"},
    )  # fmt: skip
    for split in ("gold", "val"):
        meta, found = error_mining.read(tmp_path / "harness" / "errors" / f"{split}.parquet")
        assert (meta["run"], meta["set"]) == ("vanilla-s0", split)
        errors, words = _sid(found)
        assert 100 * errors / words == pytest.approx(row[f"{split}_wer"])
    metrics = json.loads((tmp_path / "gold_metrics.json").read_text("utf-8"))
    breakdown = metrics["breakdown"]
    assert breakdown["wer"] == pytest.approx(row["gold_wer"])
    assert {b["bucket"] for b in breakdown["overlap"]} == {">15%", "unmeasured"}
    _, gold = error_mining.read(tmp_path / "harness" / "errors" / "gold.parquet")
    snr = {r["clip_id"]: r["snr_bucket"] for r in gold}
    assert snr[splits["gold"][0]["segment_id"]] == "25-35 dB"
    assert set(snr.values()) == {"25-35 dB", "unmeasured"}
    assert breakdown["top"]["sub"][0]["count"] == 1
    assert "alignment" not in json.dumps(metrics)


def test_a_split_s_crosstalk_is_measured_from_the_export_s_clip_spans(tmp_path: Path):
    """The export carries each clip's ``overlap_spans`` (clip-relative), not a share."""
    from app.services import error_mining

    splits = _splits()
    splits["gold"][0]["overlap_spans"] = [[0.0, 1.0]]  # a quarter of a 4 s clip
    splits["gold"][1]["overlap_spans"] = []
    evalkit.evaluate_run(
        tmp_path, "vanilla-s0", splits=splits, decode=_decoder({}), score=_FoldScore, card={},
    )  # fmt: skip
    _, gold = error_mining.read(tmp_path / "harness" / "errors" / "gold.parquet")
    share = {r["clip_id"]: r["overlap_share"] for r in gold}
    assert share == {"g1": pytest.approx(0.25), "g2": 0.0, "g3": None}


def test_a_public_set_writes_its_error_file_with_its_measured_overlap(tmp_path: Path, monkeypatch):
    from app.services import error_mining
    from app.services.benchmark_overlap import write_acoustics, write_overlap

    monkeypatch.setattr(evalkit, "load_benchmark", lambda *a, **k: _bench_rows())
    measured = tmp_path / "conditions"
    overlap, acoustics = measured / "overlap", measured / "acoustics"
    overlap.mkdir(parents=True)
    acoustics.mkdir()
    (overlap / "indicvoices.jsonl").write_text(
        json.dumps({"clip_id": "iv-00001", "duration": 1.0, "overlap_share": 0.5, "spans": []})
        + "\n"
    )
    write_overlap(overlap / "indicvoices.jsonl", overlap / "indicvoices.parquet")
    (acoustics / "indicvoices.jsonl").write_text(
        json.dumps({"clip_id": "iv-00000", "duration": 1.0, "snr_db": 50.0, "c50_db": 60.0,
                    "bandwidth_hz": 8000.0}) + "\n"
    )  # fmt: skip
    write_acoustics(acoustics / "indicvoices.jsonl", acoustics / "indicvoices.parquet")
    hyps = {"iv-00001": "a b x y"}
    summaries = evalkit.run_benchmarks(
        tmp_path / "vanilla-s1", decode=_decoder(hyps), score=_FoldScore, work=tmp_path / "w",
        token="t", names=("indicvoices",), conditions_dir=measured,
    )  # fmt: skip
    path = tmp_path / "vanilla-s1" / "harness" / "errors" / "indicvoices.parquet"
    meta, found = error_mining.read(path)
    assert (meta["run"], meta["set"]) == ("vanilla-s1", "indicvoices")
    errors, words = _sid(found)
    assert 100 * errors / words == pytest.approx(summaries["indicvoices"]["wer"])
    by_clip = {r["clip_id"]: (r["overlap_bucket"], r["snr_bucket"], r["by"]) for r in found}
    assert by_clip["iv-00001"] == (">15%", "unmeasured", "Conversation")
    assert by_clip["iv-00000"] == ("unmeasured", "45+ dB", "Read")
    assert summaries["indicvoices"]["breakdown"]["by_values"] == ["Conversation", "Read"]


def test_without_error_mining_in_the_harness_copy_a_run_still_scores(tmp_path: Path, monkeypatch):
    """A dataset uploaded before error mining: the run is scored and says what it skipped."""
    monkeypatch.setattr(evalkit, "_miner", lambda: None)
    row = evalkit.evaluate_run(
        tmp_path, "r", splits=_splits(), decode=_decoder({}), score=_FoldScore, card={"name": "x"}
    )
    assert row["gold_wer"] == 0.0
    assert not (tmp_path / "harness" / "errors").exists()
    assert json.loads((tmp_path / "gold_metrics.json").read_text("utf-8"))["breakdown"] is None
