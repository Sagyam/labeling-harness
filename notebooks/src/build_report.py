"""Build notebooks/07_Report.ipynb: `python notebooks/src/build_report.py`. Step 7 of the
protocol (D105): every model's scores, read back from the hub, in one table. No GPU and no model:
the numbers were written by evalkit when each run was scored, and the pairings are taken here
from the per-clip counts each run left. Shared code lives in evalkit.py, sweep.py and distill.py,
written out by %%writefile cells."""

import sys
from pathlib import Path

import nbkit
from nbkit import code, md

HERE = Path(__file__).parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent

INTRO = """
# 07 — Report: every model, one table

Step 7 of the protocol (D105). No GPU, and no model is loaded. Each run of notebooks 03 and 06
left its scores, its transcripts and its per-clip error counts on the hub; this notebook reads
them back and lays them side by side: base Flex, every Flex run and blend, the frozen teacher, and
every student at every stage.

**Why the numbers can sit in one table.** Every run was scored by the same code (`evalkit`), on
the same export (`DATASET_EXPORT`, checked by each notebook before it trained or scored), with the
same fold, and each model card records the export, the fold version and the kits' digests. A run
scored on another export is left out and named.

**Pairings are taken here**, from the per-clip counts, on the clips both runs scored, with
episodes resampled (`evalkit.pair`). So any two models can be compared, not only the pairs a
training notebook thought of.

**The success bar for a student** (roadmap §B): its interval against the teacher contains zero, or
lies below it, on gold and val, class by class. Read the crosstalk classes with care: clips
overlapped beyond `MAX_OVERLAP_SHARE` were kept out of the pseudo-labels in step 5.

**Gold is held out.** Nothing in the protocol is chosen on gold or on the public sets: recipes,
blends and the teacher are chosen on val. The gold and public columns are scores, not selections.

**Outputs**: `FLEX_REPO/<flex prefix>/report.json` and `report.md`.
"""

CONFIG = r"""
NOTEBOOK = "07_Report"
DATASET_REPO = "Sagyam/nepanglish-asr"
DATASET_EXPORT = "2026-09-30"        # runs scored on another export are left out
FLEX_REPO = "Sagyam/nepanglish-asr-flex-ft"      # teacher.json names the Flex runs' folder
STUDENTS_REPO = "Sagyam/nepanglish-asr-students"
STUDENTS_PREFIX = "students-2026-09-30"
SMOKE = False             # True reads the -smoke folders, and teacher-smoke.json if there is one
CLASSES = ("overlap", "cmi")   # the clip classes every pairing is also split by
"""

SETUP_TAIL = r"""
if SMOKE:
    STUDENTS_PREFIX += "-smoke"
"""

LOAD = r"""
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

import distill
import evalkit

TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)


def fetch_json(repo, remote):
    # A JSON file of a model repo, or None when it is not there.
    if not api.file_exists(repo, remote):
        return None
    return json.loads(Path(hf_hub_download(repo, remote, token=TOKEN)).read_text("utf-8"))


teacher = fetch_json(FLEX_REPO, distill.smoke_source(SMOKE, "teacher-smoke.json", "teacher.json",
                                                     lambda path: api.file_exists(FLEX_REPO, path)))
if teacher is None:
    raise RuntimeError(f"{FLEX_REPO} has no teacher.json: run 03e_Flex_Ship.ipynb first")
FLEX_PREFIX, TEACHER = teacher["run"].rsplit("/", 1)

# The labels, for each clip's episode and classes; no audio.
root = Path(snapshot_download(DATASET_REPO, repo_type="dataset", token=TOKEN,
                              allow_patterns=["training/training.jsonl", "training/manifest.json", "gold/gold.jsonl"]))
export = json.loads((root / "training" / "manifest.json").read_text())
evalkit.check_export(export, DATASET_EXPORT)
training = [json.loads(line) for line in (root / "training" / "training.jsonl").open(encoding="utf-8")]
rows = {"gold": [json.loads(line) for line in (root / "gold" / "gold.jsonl").open(encoding="utf-8")],
        "val": [r for r in training if r["split"] == "val"]}


def runs_of(repo, prefix):
    # The run folders under `prefix` that hold a result.
    return sorted({f.split("/")[1] for f in api.list_repo_files(repo)
                   if f.startswith(prefix + "/") and f.endswith("/result.json") and f.count("/") == 2})


models, left_out = [], []
for repo, prefix in ((FLEX_REPO, FLEX_PREFIX), (STUDENTS_REPO, STUDENTS_PREFIX)):
    for run in runs_of(repo, prefix):
        card = fetch_json(repo, f"{prefix}/{run}/harness/model_card.json") or {}
        if card.get("dataset_export") != export["exported_at"]:
            left_out.append((run, card.get("dataset_export")))
            continue
        models.append({"repo": repo, "prefix": prefix, "run": run, "card": card,
                       "row": fetch_json(repo, f"{prefix}/{run}/result.json"),
                       "counts": fetch_json(repo, f"{prefix}/{run}/per_clip.json") or {}})
by_run = {m["run"]: m for m in models}
if TEACHER not in by_run:
    raise RuntimeError(f"the teacher, {TEACHER}, has no result on this export under {FLEX_REPO}/{FLEX_PREFIX}")


def order(m):
    # base, the Flex runs, the teacher last among them, then the students by name and stage
    row = m["row"]
    stage = {"human": 0, "distill": 1, "distill-aug": 2}.get(row.get("stage"), 0)
    return (m["repo"] != FLEX_REPO, row.get("student") or "", stage,
            m["run"] != "base", m["run"] == TEACHER, m["run"])


models.sort(key=order)
print(f"export {export['exported_at'][:10]} | teacher {TEACHER} | {len(models)} runs:",
      ", ".join(m["run"] for m in models))
for run, other in left_out:
    print(f"left out: {run}, scored on the {other} export")


def line(m):
    return f"{m['wer']:6.2f} (S {m['sub']:.2f}  D {m['del']:.2f}  I {m['ins']:.2f})"


def label(m):
    return m["run"] + (" *" if m["run"] == TEACHER else "")
"""

GOLD_NOTE = """
## Gold and val

Folded WER with S/D/I, raw WER, and the real-time factor of the decode on the GPU it was scored
on. `*` marks the teacher.
"""

GOLD = r"""
print(f"{'model':<28}{'epoch':>6}  {'val':<34}{'gold':<34}{'gold raw':>9}{'gold RTF':>10}")
for m in models:
    r = m["row"]
    print(f"{label(m):<28}{str(r.get('best_epoch') or '-'):>6}  {line(r['val']):<34}{line(r['gold']):<34}"
          f"{r['gold']['raw_wer']:9.2f}{r['gold']['rtf']:10.4f}")
print("\ngold by crosstalk bucket, folded WER:")
print(f"{'model':<28}" + "".join(f"{b:>10}" for b in evalkit.BUCKETS))
for m in models:
    by = (m["row"].get("gold_by_class") or {}).get("overlap") or {}
    print(f"{label(m):<28}" + "".join(f"{by[b]['wer']:>10.2f}" if b in by else f"{'-':>10}" for b in evalkit.BUCKETS))
"""

PAIRED_NOTE = """
## Every model against the teacher

Model minus teacher in WER points, with a 95% interval from resampling episodes, on the clips both
scored: overall and per value of each class in `CLASSES`. An interval that contains zero, or lies
below it, meets the bar on that class.
"""

PAIRED = r"""
paired = {}
for m in models:
    if m["run"] == TEACHER:
        continue
    paired[m["run"]] = {}
    for split in ("val", "gold"):
        a, b = by_run[TEACHER]["counts"].get(split), m["counts"].get(split)
        if not a or not b:
            continue
        try:
            paired[m["run"]][split] = evalkit.pair(rows[split], a, b, keys=CLASSES)
        except ValueError:
            continue
for split in ("gold", "val"):
    keys = sorted({k for p in paired.values() for k in p.get(split, {}) if k != "clips"}, key=lambda k: (k != "all", k))
    print(f"\n{split}: model minus {TEACHER} [95% CI]")
    for run, p in paired.items():
        if split not in p:
            continue
        cells = "  ".join(f"{k} {p[split][k][0]:+.2f} [{p[split][k][1]:+.2f},{p[split][k][2]:+.2f}]"
                          for k in keys if k in p[split])
        print(f"  {run:<26} {cells}")
"""

STAGES_NOTE = """
## What each stage added to a student

Stage 2 minus stage 1 is the pseudo-labels together with the extra training (D106: stage 2
continues stage 1's weights, so the two cannot be told apart here). Stage 3 minus stage 2 is the
augmentation alone: both start from stage 1's weights.
"""

STAGES = r"""
steps = {}
for student in sorted({m["row"]["student"] for m in models if m["row"].get("student")}):
    have = {m["row"]["stage"]: m for m in models if m["row"].get("student") == student}
    for before, after in (("human", "distill"), ("distill", "distill-aug")):
        if before not in have or after not in have:
            continue
        name = f"{student}: {after} minus {before}"
        steps[name] = {}
        for split in ("val", "gold"):
            a, b = have[before]["counts"].get(split), have[after]["counts"].get(split)
            if a and b:
                steps[name][split] = evalkit.pair(rows[split], a, b, keys=())
        cells = "  ".join(f"{split} {v['all'][0]:+.2f} [{v['all'][1]:+.2f},{v['all'][2]:+.2f}]"
                          for split, v in steps[name].items())
        print(f"{name:<44} {cells}")
if not steps:
    print("no student has two stages scored yet")
"""

PUBLIC_NOTE = """
## The public sets

Folded WER per set and the mean over the sets, then each model minus base Flex with the set's own
unit resampled. For Flex, the difference to base is what fine-tuning cost outside our domain, and
what blending recovered. For a student that never knew Nepali there was nothing to forget: its
columns say how far the training generalises.
"""

PUBLIC = r"""
SETS = tuple(evalkit.BENCHMARKS)


def set_files(m, name):
    # (summary, per-clip lines) of a model on a public set; None when it was not scored.
    remote = f"{m['prefix']}/{m['run']}/benchmarks/{name}.json"
    summary = fetch_json(m["repo"], remote)
    if summary is None:
        return None
    lines = Path(hf_hub_download(m["repo"], remote + "l", token=TOKEN)).read_text("utf-8").splitlines()
    return summary, [json.loads(x) for x in lines]


public = {m["run"]: {name: got for name in SETS if (got := set_files(m, name)) is not None} for m in models}
print(f"{'model':<28}" + "".join(f"{name:>14}" for name in SETS) + f"{'mean':>8}")
means = {}
for m in models:
    got = public[m["run"]]
    if not got:
        continue
    if len(got) == len(SETS):
        means[m["run"]] = sum(got[name][0]["wer"] for name in SETS) / len(SETS)
    print(f"{label(m):<28}" + "".join(f"{got[name][0]['wer']:>14.2f}" if name in got else f"{'-':>14}" for name in SETS)
          + (f"{means[m['run']]:>8.2f}" if m["run"] in means else f"{'-':>8}"))
vs_base = {}
if public.get("base"):
    print("\nmodel minus base [95% CI]")
    for m in models:
        if m["run"] == "base" or not public[m["run"]]:
            continue
        vs_base[m["run"]] = {name: evalkit.pair_benchmark(public["base"][name][1], public[m["run"]][name][1])["all"]
                             for name in SETS if name in public[m["run"]] and name in public["base"]}
        cells = "  ".join(f"{name} {d:+.2f} [{lo:+.2f},{hi:+.2f}]" for name, (d, lo, hi) in vs_base[m["run"]].items())
        print(f"  {label(m):<26} {cells}")
else:
    print("\nbase Flex has no public-set scores under this prefix: run 03b_Flex_Benchmarks.ipynb")
"""

BREAKDOWN_NOTE = """
## Where the errors are

Blocks 1 and 2 of the breakdown (docs/architecture.md), per run and set, as each run's notebook wrote them beside
its WER (nothing is recomputed here): the WER of each crosstalk bucket with its share of the
errors, then the number errors' share and the WER over everything that is not a number. A run
scored before error mining has no breakdown and is left out of these tables. The rows behind them
are in each run's `harness/errors/`, which the Models page reads.
"""

BREAKDOWN = r"""
BUCKET_COLUMNS = (*evalkit.BUCKETS, "unmeasured")
breakdowns = {}
for m in models:
    got = {}
    for split in ("gold", "val"):
        metrics = fetch_json(m["repo"], f"{m['prefix']}/{m['run']}/{split}_metrics.json") or {}
        if metrics.get("breakdown"):
            got[split] = metrics["breakdown"]
    for name, (summary, _) in public[m["run"]].items():
        if summary.get("breakdown"):
            got[name] = summary["breakdown"]
    breakdowns[m["run"]] = got
for set_name in ("gold", "val", *SETS):
    have = {run: b[set_name] for run, b in breakdowns.items() if set_name in b}
    if not have:
        print(f"\n{set_name}: no run has a breakdown (scored before error mining)")
        continue
    print(f"\n{set_name}: WER by crosstalk bucket (share of errors) | numbers: share of errors, WER without them")
    print(f"{'model':<28}{'WER':>8}" + "".join(f"{b:>16}" for b in BUCKET_COLUMNS) + f"{'numbers':>9}{'without':>9}")
    for run, b in have.items():
        buckets = {x["bucket"]: x for x in b["overlap"]}
        cells = "".join(f"{buckets[k]['wer']:>9.2f} ({100 * buckets[k]['share_of_errors']:>3.0f}%)" if k in buckets
                        else f"{'-':>16}" for k in BUCKET_COLUMNS)
        n = b["numbers"]
        print(f"{label(by_run[run]):<28}{b['wer']:>8.2f}{cells}{100 * n['share_of_errors']:>8.1f}%{n['wer_without']:>9.2f}")
"""

WRITE_NOTE = """
## Write the report

`report.json` holds every number above, the breakdowns included; `report.md` is the three main tables as Markdown, to paste
into `docs/findings.md`.
"""

WRITE = r"""
def cell(v):
    return f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"


md_lines = [f"Export {export['exported_at'][:10]}, {models[0]['card'].get('fold_version', '')}. "
            f"Teacher: `{TEACHER}` (marked `*`). Folded WER (S / D / I).", "",
            "| Model | Val | Gold | Gold raw | Gold minus teacher | Public mean |", "|---|---|---|---|---|---|"]
for m in models:
    r = m["row"]
    vs = paired.get(m["run"], {}).get("gold")
    md_lines.append(
        f"| {label(m)} | {r['val']['wer']:.2f} ({r['val']['sub']:.2f} / {r['val']['del']:.2f} / {r['val']['ins']:.2f}) "
        f"| {r['gold']['wer']:.2f} ({r['gold']['sub']:.2f} / {r['gold']['del']:.2f} / {r['gold']['ins']:.2f}) "
        f"| {r['gold']['raw_wer']:.2f} | {cell(vs['all']) if vs else '—'} "
        f"| {format(means[m['run']], '.2f') if m['run'] in means else '—'} |")
md_lines += ["", "| Model | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
for m in models:
    got = public[m["run"]]
    if got:
        md_lines.append(f"| {label(m)} | " + " | ".join(f"{got[n][0]['wer']:.2f}" if n in got else "—" for n in SETS) + " |")
if steps:
    md_lines += ["", "| Step | Val | Gold |", "|---|---|---|"]
    md_lines += [f"| {name} | " + " | ".join(cell(v[s]["all"]) if s in v else "—" for s in ("val", "gold")) + " |"
                 for name, v in steps.items()]
report = {
    "export": export["exported_at"], "teacher": teacher,
    "models": [{"run": m["run"], "repo": m["repo"], "prefix": m["prefix"],
                "architecture": m["card"].get("architecture"), "row": m["row"]} for m in models],
    "left_out": left_out, "vs_teacher": paired, "stage_steps": steps,
    "public": {run: {name: got[name][0] for name in got} for run, got in public.items()},
    "public_mean": means, "public_vs_base": vs_base, "breakdowns": breakdowns,
}
out = FT / "report"
out.mkdir(exist_ok=True)
(out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
(out / "report.md").write_text("\n".join(md_lines) + "\n", "utf-8")
api.upload_folder(repo_id=FLEX_REPO, folder_path=str(out), path_in_repo=FLEX_PREFIX,
                  commit_message=f"{FLEX_PREFIX}: report, {len(models)} runs")
print("\n".join(md_lines))
print("\nuploaded", f"https://huggingface.co/{FLEX_REPO}/tree/main/{FLEX_PREFIX}")
"""

cells = nbkit.cpu(
    [
        md(INTRO),
        md("## Config"),
        code(CONFIG),
        md("## Setup"),
        nbkit.setup("numpy", tail=SETUP_TAIL),
        *nbkit.kits("evalkit", "sweep", "distill"),
        code(LOAD),
        md(GOLD_NOTE),
        code(GOLD),
        md(PAIRED_NOTE),
        code(PAIRED),
        md(STAGES_NOTE),
        code(STAGES),
        md(PUBLIC_NOTE),
        code(PUBLIC),
        md(BREAKDOWN_NOTE),
        code(BREAKDOWN),
        md(WRITE_NOTE),
        code(WRITE),
    ]
)

NOTEBOOKS = {"07_Report.ipynb": cells}

if __name__ == "__main__":
    nbkit.write(NOTEBOOKS, OUT_DIR)
