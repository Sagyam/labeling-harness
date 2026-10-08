"""Score cards (scripts/score_cards.py): every number read back from a run's files, none made up."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "score_cards.py"
_spec = importlib.util.spec_from_file_location("score_cards", _PATH)
cards = importlib.util.module_from_spec(_spec)
sys.modules["score_cards"] = cards  # dataclasses look their module up while building
_spec.loader.exec_module(cards)


def _rarity(wers):
    names = ("never", "1-9", "10-49", "50-99", "100+")
    return {"buckets": [{"bucket": n, "wer": w} for n, w in zip(names, wers, strict=True)]}


def _metrics(wer: float, **extra) -> dict:
    return {
        "wer": wer, "sub": wer - 2, "del": 1.0, "ins": 1.0, "raw_wer": wer + 3, "cer": wer / 2,
        "clips": 10, "loops": 0, "rtf": 0.01,
        "by_class": {"overlap": {"none": {"wer": wer - 1, "clips": 8},
                                 ">15%": {"wer": wer + 9, "clips": 2}}},
        "breakdown": {
            "numbers": {"wer_without": wer - 0.5, "share_of_errors": 0.04},
            "overlap": [{"bucket": "unmeasured", "wer": wer, "clips": 10}],
            "attribution": {
                "conditions": [
                    {"factor": "crosstalk", "bucket": ">15%", "within": {"points": 1.5}}
                ],
                "rest": {"within": {"kinds": [{"kind": "nepali_other", "points": wer - 1.5}]}},
            },
            "rarity": _rarity([40.0, 20.0, 10.0, None, 5.0]),
        },
        **extra,
    }  # fmt: skip


def _read(files: dict):
    return lambda repo, path: files.get(path)


_MODEL = cards.Model(
    "toy",
    "Toy",
    (
        cards.Stage("Before training", missing="random weights: nothing to score"),
        cards.Stage("Stage 1", "repo", "p/toy-human"),
        cards.Stage("Stage 2", "repo", "p/toy-distill"),
    ),
)
_FILES = {
    "p/toy-human/result.json": {"train_clips": 100, "best_epoch": 3},
    "p/toy-human/harness/model_card.json": {"architecture": "toy net", "fold_version": "f"},
    "p/toy-human/gold_metrics.json": _metrics(20.0),
    "p/toy-human/val_metrics.json": _metrics(12.0),
    "p/toy-distill/gold_metrics.json": _metrics(15.0),
    "p/toy-distill/val_metrics.json": _metrics(9.0),
    "p/toy-distill/benchmarks/fleurs.json": _metrics(18.0, plain_wer=30.0, plain_cer=11.0),
}


def _render() -> str:
    runs = [cards.fetch(s, _read(_FILES)) for s in _MODEL.stages]
    return cards.render(_MODEL, runs)


def test_a_stage_without_a_run_says_why_and_its_cells_are_dashes() -> None:
    text = _render()
    assert "random weights: nothing to score" in text
    assert "| gold | — | 20.00 | 15.00 |" in text
    assert "| fleurs | — | — | 18.00 |" in text


def test_each_set_carries_sdi_raw_wer_and_cer() -> None:
    text = _render()
    gold = text.split("### gold", 1)[1]
    assert "| substitutions | — | 18.00 | 13.00 |" in gold
    assert "| raw WER | — | 23.00 | 18.00 |" in gold
    assert "| CER | — | 10.00 | 7.50 |" in gold


def test_gold_and_val_are_split_by_clip_class_with_their_clip_counts() -> None:
    text = _render()
    assert "| crosstalk: none | — | 19.00 (8) | 14.00 (8) |" in text
    assert "| crosstalk: >15% | — | 29.00 (2) | 24.00 (2) |" in text


def test_the_attribution_card_adds_up_to_the_wer() -> None:
    section = _render().split("## Where the WER comes from", 1)[1]
    gold = section.split("### gold", 1)[1].split("###", 1)[0]
    assert "| crosstalk: >15% | — | 1.50 | 1.50 |" in gold
    assert "| error: nepali other | — | 18.50 | 13.50 |" in gold
    assert "| WER | — | 20.00 | 15.00 |" in gold


def test_word_rarity_is_one_table_per_stage_buckets_down_sets_across() -> None:
    section = _render().split("## WER by word rarity", 1)[1]
    stage2 = section.split("### Stage 2", 1)[1]
    assert "| Bucket | val | gold | fleurs |" in stage2
    assert "| never | 40.00 | 40.00 | 40.00 |" in stage2
    assert "| 50-99 | — | — | — |" in stage2


def test_regenerating_a_card_keeps_its_notes(tmp_path: Path) -> None:
    path = tmp_path / "toy-score-card.md"
    cards.write_card(path, "# old scores\n")
    text = path.read_text("utf-8")
    path.write_text(text + "Stage 1 stopped after epoch 7.\n", "utf-8")
    cards.write_card(path, "# new scores\n")
    text = path.read_text("utf-8")
    assert text.startswith("# new scores\n")
    assert "old scores" not in text
    assert text.endswith("## Notes\n\nStage 1 stopped after epoch 7.\n")
