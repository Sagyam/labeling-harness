#!/usr/bin/env python
"""Split finished runs' WER by word rarity (D123), from the error files they left on the hub.

  uv run --no-sync --project backend --with huggingface_hub python scripts/backfill_rarity.py
  ... backfill_rarity.py --run Sagyam/nepanglish-asr-flex-ft:flex-2026-09-30/base

First the training corpus's word counts: the export's human train labels plus the teacher's
filtered pseudo-labels (`teacher.json` names the teacher), counted with this checkout's fold and
uploaded to the dataset as `harness/word_counts.json` and beside the labels, where 05_Teacher
would have written them. Then, for each run, every `harness/errors/<set>.parquet` is read and the
run's `<split>_metrics.json` and `benchmarks/<set>.json` gain `breakdown.rarity`
(`evalkit.add_rarity`): nothing is decoded or scored again. One commit per repo.

The default runs are the best weights of each model: Flex (blend-075, the teacher) and the four
students' stage 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "notebooks" / "src"))

import evalkit
from app.services.error_mining import metadata
from app.services.fold import fold_version

DATASET_REPO = "Sagyam/nepanglish-asr"
FLEX_REPO = "Sagyam/nepanglish-asr-flex-ft"
STUDENTS = "Sagyam/nepanglish-asr-students:students-2026-09-30"
RUNS = (
    f"{FLEX_REPO}:flex-2026-09-30/blend-075",
    f"{STUDENTS}/whisper-distill",
    f"{STUDENTS}/indicconformer-distill",
    f"{STUDENTS}/parakeet-distill",
    f"{STUDENTS}/gemma-e2b-distill",
)


def _jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def count_words(api, work: Path, *, upload: bool) -> Path:
    """The training corpus's counts, written under `work` and, with `upload`, to the dataset."""
    from huggingface_hub import CommitOperationAdd, hf_hub_download

    teacher = json.loads(Path(hf_hub_download(FLEX_REPO, "teacher.json")).read_text("utf-8"))
    labels = f"distill/pseudo/{teacher['run'].replace('/', '--')}"
    get = lambda name: hf_hub_download(DATASET_REPO, name, repo_type="dataset")  # noqa: E731
    manifest = json.loads(Path(get("training/manifest.json")).read_text("utf-8"))
    train = [r for r in _jsonl(get("training/training.jsonl")) if r["split"] == "train"]
    pseudo = _jsonl(get(f"{labels}/labels.jsonl"))
    path = evalkit.count_training_words(
        work / "word_counts.json", train, pseudo, export=manifest["exported_at"], labels=labels
    )
    data = json.loads(path.read_text("utf-8"))
    print(
        f"word counts: {len(train)} human + {len(pseudo)} pseudo clips, {data['tokens']} words, "
        f"{data['types']} spelling keys, {data['fold_version']}"
    )
    if upload:
        remote = (evalkit.WORD_COUNTS_FILE, f"{labels}/word_counts.json")
        commit = api.create_commit(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            operations=[CommitOperationAdd(r, str(path)) for r in remote],
            commit_message=f"{labels}: the training corpus's word counts (D123)",
        )
        print("counts:", commit.commit_url)
    return path


def backfill(api, spec: str, counts: Path, work: Path) -> tuple[str, list[tuple[str, Path]]]:
    """One run's breakdowns with rarity added; returns its repo and (remote, local) pairs."""
    from huggingface_hub import hf_hub_download

    repo, folder = spec.split(":", 1)
    out = work / folder
    wanted = [
        f
        for f in api.list_repo_files(repo)
        if f.startswith(folder + "/")
        and (
            f.startswith(f"{folder}/harness/errors/")
            or f.removeprefix(folder + "/") in {f"{s}_metrics.json" for s in evalkit.SPLITS}
            or (f.startswith(f"{folder}/benchmarks/") and f.endswith(".json"))
        )
    ]
    for remote in wanted:
        local = out / remote.removeprefix(folder + "/")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(hf_hub_download(repo, remote)).read_bytes())
    for errors in sorted((out / "harness" / "errors").glob("*.parquet")):
        got = metadata(errors)["fold_version"]
        if got != fold_version():
            raise SystemExit(f"{spec} {errors.name}: mined under {got}, not {fold_version()}")
    print(f"\n{spec}")
    added = evalkit.add_rarity(out, counts)
    summaries = [
        out / (f"{name}_metrics.json" if name in evalkit.SPLITS else f"benchmarks/{name}.json")
        for name in added
    ]
    return repo, [(f"{folder}/{p.relative_to(out).as_posix()}", p) for p in summaries]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", help="repo:folder; default: the best five")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, upload nothing")
    parser.add_argument("--work", type=Path, help="where to keep the files (default: a temp dir)")
    args = parser.parse_args(argv)
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        work = args.work or Path(tmp)
        counts = count_words(api, work, upload=not args.dry_run)
        by_repo: dict[str, list[tuple[str, Path]]] = {}
        for spec in args.run or RUNS:
            repo, files = backfill(api, spec, counts, work / "runs")
            by_repo.setdefault(repo, []).extend(files)
        if args.dry_run:
            return 0
        for repo, files in by_repo.items():
            commit = api.create_commit(
                repo_id=repo,
                operations=[CommitOperationAdd(r, str(p)) for r, p in files],
                commit_message=f"breakdowns split by word rarity (D123), {len(files)} files",
            )
            print(f"{repo}: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
