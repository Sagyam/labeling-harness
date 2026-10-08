#!/usr/bin/env python
"""Write every model's score card, docs/score-card/<model>-score-card.md, from the hub.

  uv run --no-sync --project backend --with huggingface_hub python scripts/score_cards.py
  ... score_cards.py --model whisper

A card is one model at every point of its training -- before training, stage 1, stage 2 -- as
each run's notebook scored it (evalkit): WER with S/D/I, raw WER and CER on val, gold and the
public sets, gold and val by every clip class (crosstalk, SNR, ...), each public set by crosstalk,
the attribution card (D111), and the WER by word rarity (D123). Nothing is scored here; every
number is read from the run's files.

Everything above the notes line is rewritten on each run; the notes below it are hand-written
(training logs, what a run does not show) and kept.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = REPO_ROOT / "docs" / "score-card"
FLEX_REPO = "Sagyam/nepanglish-asr-flex-ft"
STUDENTS_REPO = "Sagyam/nepanglish-asr-students"
FLEX = "flex-2026-09-30"
STUDENTS = "students-2026-09-30"
BAKEOFF = "speech-llm-bakeoff-2026-10-08"

#: The sets, in the order every table lists them.
SETS = ("val", "gold", "fleurs", "slr54", "common_voice", "indicvoices", "nepali_cs")
PUBLIC = SETS[2:]
#: The clip classes gold and val are split by, and the order of each one's values.
CLASSES: dict[str, tuple[str, ...]] = {
    "overlap": ("none", "0-5%", "5-15%", ">15%"),
    "snr": ("<15 dB", "15-25 dB", "25-35 dB", "35-45 dB", "45+ dB", "unmeasured"),
    "reverb": ("<40 dB", "40-50 dB", "50-55 dB", "55+ dB", "unmeasured"),
    "bandwidth": ("<4.5 kHz", "4.5-6.5 kHz", "6.5+ kHz"),
    "speakers": ("1", "2", "3+"),
    "turn_changes": ("0", "1", "2+"),
    "pause": ("<2%", "2-10%", ">10%"),
    "cmi": ("0", "<15", "15-30", "30+"),
    "duration": ("<5 s", "5-15 s", "15+ s"),
    "voice_exposure": ("unseen", "<10 min", "10-60 min", "1 h+"),
    "gender": (),
    "age_bracket": (),
}
CLASS_NAMES = {"overlap": "crosstalk", "snr": "SNR", "cmi": "CMI", "turn_changes": "turn changes"}
NOTES_LINE = "<!-- notes: everything below this line is hand-written and kept on regeneration -->"
DASH = "—"


@dataclass(frozen=True)
class Stage:
    """One column of a card: a run, or a point with no run and the reason."""

    label: str
    repo: str | None = None
    folder: str | None = None
    missing: str = "not scored on this export"


@dataclass(frozen=True)
class Model:
    slug: str
    title: str
    stages: tuple[Stage, ...]


def _student(slug: str, title: str, before: Stage) -> Model:
    return Model(
        slug,
        title,
        (
            before,
            Stage("Stage 1: human labels", STUDENTS_REPO, f"{STUDENTS}/{slug}-human"),
            Stage("Stage 2: + pseudo-labels", STUDENTS_REPO, f"{STUDENTS}/{slug}-distill"),
        ),
    )


MODELS: tuple[Model, ...] = (
    Model(
        "flex",
        "Indic-Transcribe-Flex (the teacher)",
        (
            Stage("Before training: base", FLEX_REPO, f"{FLEX}/base"),
            Stage("Stage 1: human labels (vanilla-s1)", FLEX_REPO, f"{FLEX}/vanilla-s1"),
            Stage("Shipped: blend-075, the teacher", FLEX_REPO, f"{FLEX}/blend-075"),
        ),
    ),
    _student("whisper", "Whisper-large-v3-turbo", Stage("Before training")),
    _student("indicconformer", "IndicConformer", Stage("Before training")),
    _student("parakeet", "Parakeet", Stage("Before training")),
    _student(
        "gemma-e2b",
        "Gemma 4 E2B",
        Stage("Before training: zero-shot", STUDENTS_REPO, f"{BAKEOFF}/gemma-4-e2b"),
    ),
    _student(
        "conformer",
        "Conformer from scratch",
        Stage("Before training", missing="random weights: nothing to score"),
    ),
)


# --- reading a run -------------------------------------------------------------------------------


def fetch(stage: Stage, read: Callable[[str, str], dict | None]) -> dict[str, Any] | None:
    """A run's files as one dict: `result`, `card`, and `sets` {set: metrics or summary}.

    `read(repo, path)` returns a JSON file of the hub, or None when it is not there."""
    if stage.repo is None or stage.folder is None:
        return None
    base = stage.folder

    def get(path: str) -> dict | None:
        return read(stage.repo, f"{base}/{path}")

    sets = {}
    for name in SETS:
        path = f"{name}_metrics.json" if name in ("val", "gold") else f"benchmarks/{name}.json"
        found = get(path)
        if found is not None:
            sets[name] = found
    return {
        "result": get("result.json") or {},
        "card": get("harness/model_card.json") or {},
        "sets": sets,
    }


# --- rendering -----------------------------------------------------------------------------------


def _num(value: Any, digits: int = 2) -> str:
    return DASH if value is None else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return [*out, ""]


def _stage_header(model: Model) -> list[str]:
    return ["", *(s.label for s in model.stages)]


def _cells(runs: Sequence[dict | None], value: Callable[[dict], str]) -> list[str]:
    out = []
    for run in runs:
        try:
            out.append(DASH if run is None else value(run))
        except (KeyError, TypeError):
            out.append(DASH)
    return out


def _set(run: dict, name: str) -> dict:
    return run["sets"][name]


def _wer_clips(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return DASH
    return f"{entry['wer']:.2f} ({entry['clips']})"


def _ordered(values: set[str], order: Sequence[str]) -> list[str]:
    return [v for v in order if v in values] + sorted(values - set(order))


def _overview(model: Model, runs: Sequence[dict | None]) -> list[str]:
    def card(key: str) -> Callable[[dict], str]:
        return lambda run: str(run["card"][key]) if run["card"].get(key) is not None else DASH

    def result(key: str, fmt: Callable[[Any], str] = str) -> Callable[[dict], str]:
        return lambda run: fmt(run["result"][key]) if run["result"].get(key) is not None else DASH

    rows = [
        ["Run", *(f"`{s.repo}/{s.folder}`" if s.folder else s.missing for s in model.stages)],
        ["Architecture", *_cells(runs, card("architecture"))],
        ["What it is", *_cells(runs, card("description"))],
        ["Train clips", *_cells(runs, result("train_clips"))],
        ["Pseudo-label hours", *_cells(runs, result("pseudo_hours"))],
        ["Human share of draws", *_cells(runs, result("human_share"))],
        ["Epochs (cap)", *_cells(runs, result("epochs"))],
        ["Best epoch", *_cells(runs, result("best_epoch"))],
        ["Learning rate", *_cells(runs, card("lr"))],
        ["Decoder", *_cells(runs, card("decoder"))],
        ["Fold", *_cells(runs, card("fold_version"))],
        ["Scored", *_cells(runs, lambda r: str(r["card"]["created_at"])[:10])],
    ]
    return ["## The runs", "", *_table(_stage_header(model), rows)]


def _headline(model: Model, runs: Sequence[dict | None]) -> list[str]:
    rows = [[name, *_cells(runs, lambda r, n=name: _num(_set(r, n)["wer"]))] for name in SETS]

    def mean(run: dict) -> str:
        if not all(n in run["sets"] for n in PUBLIC):
            return DASH
        return _num(sum(run["sets"][n]["wer"] for n in PUBLIC) / len(PUBLIC))

    rows.append(["public mean", *_cells(runs, mean)])
    return [
        "## WER on every set",
        "",
        "Folded WER (fold-v4), per 100 reference words of each set's own labels.",
        "",
        *_table(_stage_header(model), rows),
    ]


_DETAIL = (
    ("WER", "wer"),
    ("substitutions", "sub"),
    ("deletions", "del"),
    ("insertions", "ins"),
    ("raw WER", "raw_wer"),
    ("CER", "cer"),
    ("plain WER", "plain_wer"),
    ("plain CER", "plain_cer"),
)


def _set_rows(runs: Sequence[dict | None], name: str) -> list[list[str]]:
    """One set's metrics, a row each, a cell per stage."""

    def cell(value: Callable[[dict], str]) -> list[str]:
        return _cells(runs, lambda r: value(_set(r, name)))

    def numbers(key: str) -> Callable[[dict], Any]:
        return lambda m: m["breakdown"]["numbers"][key]

    rows = [
        [label, *cell(lambda m, k=key: _num(m.get(k)))]
        for label, key in _DETAIL
        if name in PUBLIC or not key.startswith("plain")
    ]
    rows += [
        ["clips", *cell(lambda m: str(m["clips"]))],
        ["loops", *cell(lambda m: str(m["loops"]))],
        ["WER without number errors", *cell(lambda m: _num(numbers("wer_without")(m)))],
        [
            "number errors, share of errors",
            *cell(lambda m: f"{100 * numbers('share_of_errors')(m):.1f}%"),
        ],
    ]
    if name in ("val", "gold"):
        rows.append(["real-time factor", *cell(lambda m: _num(m["rtf"], 4))])
    else:
        rows.append(["times real time", *cell(lambda m: _num(m.get("x_realtime"), 0))])
    return rows


def _details(model: Model, runs: Sequence[dict | None]) -> list[str]:
    out = [
        "## Each set",
        "",
        "Substitutions, deletions and insertions are per 100 reference words and add up to the "
        "WER. Raw WER compares without the fold; plain WER and CER are the normalisation "
        "published Nepali results use (public sets only).",
        "",
    ]
    for name in SETS:
        if any(run and name in run["sets"] for run in runs):
            out += [f"### {name}", "", *_table(_stage_header(model), _set_rows(runs, name))]
    return out


def _class_rows(runs: Sequence[dict | None], split: str) -> list[list[str]]:
    """Gold's or val's WER (clips) per value of every clip class, a cell per stage."""

    def classes(run: dict) -> dict:
        return _set(run, split).get("by_class") or {}

    rows = []
    for key, order in CLASSES.items():
        values: set[str] = set()
        for run in runs:
            if run and split in run["sets"]:
                values |= set(classes(run).get(key) or {})
        for value in _ordered(values, order):
            label = f"{CLASS_NAMES.get(key, key.replace('_', ' '))}: {value}"
            cells = _cells(runs, lambda r, k=key, v=value: _wer_clips(classes(r).get(k, {}).get(v)))
            rows.append([label, *cells])
    return rows


def _by_class(model: Model, runs: Sequence[dict | None]) -> list[str]:
    out = [
        "## Gold and val by clip class",
        "",
        "WER (clips) of the clips in each class (D87): crosstalk share, speech-to-noise ratio, "
        "reverberation, bandwidth, speakers, turn changes, pauses, code-mixing index, duration, "
        "how long the voice was heard in train, and the voice's gender and age.",
        "",
    ]
    for split in ("gold", "val"):
        rows = _class_rows(runs, split)
        if rows:
            out += [f"### {split}", "", *_table(_stage_header(model), rows)]
    return out


def _public_crosstalk(model: Model, runs: Sequence[dict | None]) -> list[str]:
    out = [
        "## Public sets by crosstalk",
        "",
        "WER (clips) per crosstalk bucket, measured on each clip "
        "(scripts/measure_benchmark_overlap.py); a set never measured is left out.",
        "",
    ]
    found = False
    for name in PUBLIC:
        buckets: set[str] = set()
        for run in runs:
            if run and name in run["sets"]:
                overlap = (run["sets"][name].get("breakdown") or {}).get("overlap") or []
                buckets |= {b["bucket"] for b in overlap}
        if not buckets - {"unmeasured"}:
            continue
        found = True

        def bucket(run: dict, b: str, n: str = name) -> str:
            got = {x["bucket"]: x for x in _set(run, n)["breakdown"]["overlap"]}
            return _wer_clips(got.get(b))

        rows = [
            [b, *_cells(runs, lambda r, b=b: bucket(r, b))]
            for b in _ordered(buckets, CLASSES["overlap"])
        ]
        out += [f"### {name}", "", *_table(_stage_header(model), rows)]
    return out if found else []


def _attribution(model: Model, runs: Sequence[dict | None]) -> list[str]:
    out = [
        "## Where the WER comes from (attribution card, D111)",
        "",
        "Points of WER, adding up to the set's WER: what each crosstalk and SNR bucket costs "
        "within episode, then the rest by kind of error.",
        "",
    ]
    for name in SETS:
        labels: list[tuple[str, Callable[[dict], float | None]]] = []
        seen: set[str] = set()
        for run in runs:
            card = ((run or {}).get("sets", {}).get(name) or {}).get("breakdown", {}) or {}
            card = card.get("attribution") or {}
            for c in card.get("conditions", []):
                label = f"{CLASS_NAMES.get(c['factor'], c['factor'])}: {c['bucket']}"
                if label not in seen:
                    seen.add(label)
                    labels.append((label, _condition(c["factor"], c["bucket"], name)))
            for k in (card.get("rest") or {}).get("within", {}).get("kinds", []):
                label = f"error: {k['kind'].replace('_', ' ')}"
                if label not in seen:
                    seen.add(label)
                    labels.append((label, _kind(k["kind"], name)))
        if not labels:
            continue
        rows = [[label, *_cells(runs, lambda r, f=f: _num(f(r)))] for label, f in labels]
        rows.append(["WER", *_cells(runs, lambda r, n=name: _num(_set(r, n)["wer"]))])
        out += [f"### {name}", "", *_table(_stage_header(model), rows)]
    return out


def _condition(factor: str, bucket: str, name: str) -> Callable[[dict], float | None]:
    def get(run: dict) -> float | None:
        for c in _set(run, name)["breakdown"]["attribution"]["conditions"]:
            if (c["factor"], c["bucket"]) == (factor, bucket):
                return (c.get("within") or {}).get("points")
        return None

    return get


def _kind(kind: str, name: str) -> Callable[[dict], float | None]:
    def get(run: dict) -> float | None:
        rest = _set(run, name)["breakdown"]["attribution"]["rest"]["within"]
        return next((k["points"] for k in rest["kinds"] if k["kind"] == kind), None)

    return get


def _rarity(model: Model, runs: Sequence[dict | None]) -> list[str]:
    out = [
        "## WER by word rarity (D123)",
        "",
        "WER of each set's words, by how often the word occurs in what the students train on "
        "(the human train labels and the teacher's pseudo-labels, `harness/word_counts.json`). "
        "A substitution and a deletion are the reference word's, an insertion the inserted word's.",
        "",
    ]
    found = False
    for stage, run in zip(model.stages, runs, strict=True):
        if run is None:
            continue
        blocks = {
            n: run["sets"][n]["breakdown"]["rarity"]
            for n in SETS
            if n in run["sets"] and (run["sets"][n].get("breakdown") or {}).get("rarity")
        }
        if not blocks:
            continue
        found = True
        names = [b["bucket"] for b in next(iter(blocks.values()))["buckets"]]
        columns = list(blocks)
        rows = [
            [bucket, *(_num(blocks[n]["buckets"][i]["wer"]) for n in columns)]
            for i, bucket in enumerate(names)
        ]
        out += [f"### {stage.label}", "", *_table(["Bucket", *columns], rows)]
    return out if found else []


def render(model: Model, runs: Sequence[dict | None]) -> str:
    """The generated part of a card: everything above the notes line."""
    lines = [
        f"# {model.title}: score card",
        "",
        "<!-- generated by scripts/score_cards.py from the hub; edit only the notes -->",
        "",
        "Every number was written by the run's notebook (`notebooks/src/evalkit.py`) on the "
        "2026-09-30 export and is read back here, not recomputed. A dash is a point with no run "
        "or a number that run did not record.",
        "",
    ]
    for section in (
        _overview,
        _headline,
        _details,
        _by_class,
        _public_crosstalk,
        _attribution,
        _rarity,
    ):
        lines += section(model, runs)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_card(path: Path, generated: str) -> None:
    """Write `generated` above the notes line, keeping whatever notes `path` already holds."""
    notes = "\n## Notes\n\n"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if NOTES_LINE in old:
            notes = old.split(NOTES_LINE, 1)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{generated}\n{NOTES_LINE}\n{notes.lstrip(chr(10))}", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", choices=[m.slug for m in MODELS])
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = {repo: set(api.list_repo_files(repo)) for repo in (FLEX_REPO, STUDENTS_REPO)}

    def read(repo: str, path: str) -> dict | None:
        if path not in files[repo]:
            return None
        return json.loads(Path(hf_hub_download(repo, path)).read_text(encoding="utf-8"))

    for model in MODELS:
        if args.model and model.slug not in args.model:
            continue
        runs = [fetch(stage, read) for stage in model.stages]
        path = CARD_DIR / f"{model.slug}-score-card.md"
        write_card(path, render(model, runs))
        print(f"{path.relative_to(REPO_ROOT)}: {sum(r is not None for r in runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
