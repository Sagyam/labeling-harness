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
