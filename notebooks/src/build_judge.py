"""Build notebooks/08_Judge_Headroom.ipynb: `python notebooks/src/build_judge.py`. Step 8 of the
protocol (D105, D117; roadmap G1): before any LLM judge is chosen, how much could one gain by
picking among the teacher's beam candidates, on val, gold and the five public sets? The rules are
in judgekit.py, tested in backend/tests/test_judge_kit.py; the model loading and decoder come
from the Flex notebooks."""

import sys
from pathlib import Path

import nbkit
from build_flex import DECODE, config, load
from nbkit import code, md

HERE = Path(__file__).parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent

INTRO = """
# 08 — The judge's ceiling: how much is there to pick?

Roadmap G1's pilot asks whether an LLM that *picks* one of the beam's candidates, and never
rewrites one, can fix words the teacher gets wrong. This notebook asks the cheaper question first:
how much could any picker gain, and where? It chooses no judge.

**What runs.** The teacher `teacher.json` names decodes every clip of val, gold and the five
public sets with beam search, `K` candidates each, under the standard decoder's length cap (13
tokens a second, at most 300). On the public sets it also decodes greedily with the standard
decoder; on val and gold its stored transcripts from 03e are that decode. Every candidate is
scored with fold.py.

**The rows, on every set**, each against the standard greedy decode of the same model:
- **greedy**: greedy, cap and loop retry, the decoder every number in 03-07 comes from.
- **top@1**: the beam's own first choice.
- **random**: one of the `K` drawn at random, the floor a judge has to clear.
- **MBR**: the candidate closest to the other `K - 1` (minimum Bayes risk). It never reads the
  reference, so it is a judge anyone could run without an LLM.
- **oracle@2, 4, 8**: the candidate with the fewest folded errors against the reference, among the
  top 2, 4 or 8. It reads the answer key, so it is the ceiling no judge reaches.

**The numbers**: folded WER split into S/D/I, raw WER, CER and plain WER and CER; per clip class
on val and gold (crosstalk, SNR, speakers, CMI, length) and per the public sets' own column; every
difference paired with greedy on the same clips, with a 95% interval from resampling episodes
(speakers, sentences or videos on the public sets).

**Is it a fluke?** Each gain carries its interval; the same rows on seven sets; the gain in raw WER
and CER as well as folded WER; and how it spreads over the episodes: how many improve, and the
share of the net gain the five most improved carry.

**The rule** (D117), fixed before any candidate was decoded and applied on val alone: if oracle@`K`
is less than `GAIN_TO_CONTINUE` points of folded val WER below the top candidate, G1 stops. Gold and
the public sets are reported and never choose.

**Small blast radius.** Each set's candidates go to `OUT_REPO/<RUN_PREFIX>/judge-headroom/<set>/
shards/` every `SHARD_CLIPS` clips, and a rerun skips every clip already in a shard; each set's
report is uploaded as it is scored, and a rerun skips a set already reported. The shards are also
the judge's input if G1 continues.
"""

CONFIG = r"""
TEACHER_FILE_REAL = "teacher.json"   # names the model in OUT_REPO (03e_Flex_Ship.ipynb)
RUN = "judge-headroom"               # OUT_REPO/<RUN_PREFIX>/<RUN>/
SETS = ("val", "gold", "fleurs", "slr54", "common_voice", "indicvoices", "nepali_cs")
K = 8                                # beam width and candidates kept per clip
KS = (2, 4, 8)                       # oracle@k reported for each
GAIN_TO_CONTINUE = 1.0               # D117: oracle@K must be this many points of val WER below top@1
BEAM_BUDGET_S, BEAM_ITEMS = 150.0, 12  # an eighth of the greedy batch: beam K holds K× the decoder state
SHARD_CLIPS = 300                    # clips per uploaded shard: a lost runtime loses at most one
BENCH_LIMIT = None                   # the first N clips of each public set; None for all (a smoke run takes 40)
RESCORE = False                      # True scores a set again even when its report is on the hub
WORKERS = None                       # processes aligning candidates; None for every core
"""

TEACHER_NOTE = """
## The teacher, and its greedy decode of val and gold

The model `teacher.json` names, loaded as `05_Teacher.ipynb` and the students load it. 03e decoded
val and gold with it as shipped; those transcripts are the greedy row there.
"""

TEACHER = r"""
import distill
import judgekit

TEACHER_FILE = distill.smoke_source(SMOKE, "teacher-smoke.json", TEACHER_FILE_REAL,
                                    lambda path: api.file_exists(OUT_REPO, path))
teacher = fetch_json(TEACHER_FILE)
TEACHER = teacher["run"]
OUT = OUT_ROOT / RUN
DEST = f"{RUN_PREFIX}/{RUN}"  # Setup already put "-smoke" on RUN_PREFIX for a smoke run
(OUT / "greedy" / "harness").mkdir(parents=True, exist_ok=True)
for split in ("val", "gold"):
    for name in (f"harness/{split}.jsonl", f"{split}_metrics.json"):
        remote = f"{TEACHER}/{name}"
        if api.file_exists(OUT_REPO, remote):
            shutil.copy(hf_hub_download(OUT_REPO, remote, token=TOKEN), OUT / "greedy" / name)
stored_greedy = evalkit.stored_decode(OUT / "greedy")
with ftkit.timed(f"downloading the weights of {TEACHER}"):
    root = snapshot_download(OUT_REPO, allow_patterns=[f"{TEACHER}/best/*"], token=TOKEN)
free_model()
load_model(Path(root) / TEACHER / "best").eval()
LIMIT = BENCH_LIMIT or (40 if SMOKE else None)
WORKERS = WORKERS or os.cpu_count() or 1
print("teacher:", TEACHER, f"(val {teacher['val_wer']:.2f}, gold {teacher['gold_wer']:.2f}) | sets:",
      ", ".join(SETS), "| outputs:", DEST)
"""

DECODE_NOTE = """
## Beam search, K candidates per clip, set by set and shard by shard

`num_beams = num_return_sequences = K`, so each clip's candidates come back best first by the
beam's own score (length-normalised log-prob). The cap is a logits processor: once a row has
written its clip's cap, every token but end-of-text is ruled out. Under beam search row `r` belongs
to clip `r // K` (`judgekit.rows_at_cap`). No loop retry on the candidates: a looping candidate
scores badly and the oracle passes it by. On a public set each shard also holds the greedy decode
(with its loop retry) and what scoring needs, so no set's audio is loaded twice.
"""

NBEST = r'''
import time

from transformers import LogitsProcessor, LogitsProcessorList


class CapPerClip(LogitsProcessor):
    """Ends every row that has written its clip's cap: all scores but end-of-text go to -inf."""

    def __init__(self, clip_caps, start, num_beams):
        self.clip_caps, self.start, self.num_beams = clip_caps, start, num_beams

    def __call__(self, input_ids, scores):
        rows = judgekit.rows_at_cap(input_ids.shape[1] - self.start, self.clip_caps, self.num_beams)
        if rows:
            idx = torch.tensor(rows, device=scores.device)
            eos = scores[idx, EOS].clone()
            scores[idx] = -float("inf")
            scores[idx, EOS] = eos
        return scores


def nbest(rows):
    """K candidates per clip, best first, each with the beam's score."""
    audio = [store.clip_f32(r) for r in rows]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(a) for a in audio), PAD_TO_S))
    for i, a in enumerate(audio):
        wav[i, :len(a)] = a
    feats, _, mask = features(wav.cuda(), torch.tensor([len(a) for a in audio], device="cuda"))
    prompt = torch.tensor([PROMPT] * len(rows), device="cuda")
    cap = CapPerClip(judgekit.caps([ftkit.duration(r) for r in rows]), prompt.shape[1], K)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(input_features=feats, attention_mask=mask, decoder_input_ids=prompt,
                             do_sample=False, num_beams=K, num_return_sequences=K,
                             max_new_tokens=judgekit.MAX_TOKENS, eos_token_id=EOS, pad_token_id=PAD,
                             return_dict_in_generate=True, output_scores=True,
                             logits_processor=LogitsProcessorList([cap]))
    texts = [tk.decode(tk.strip_prompt_and_trim(o.tolist(), PROMPT)).strip() for o in out.sequences]
    scores = [round(float(s), 5) for s in out.sequences_scores.float().tolist()]
    return [[{"text": t, "score": s} for t, s in zip(ts, ss, strict=True)]
            for ts, ss in zip(judgekit.group(texts, K), judgekit.group(scores, K), strict=True)]


def shard_names(name):
    prefix = f"{DEST}/{name}/shards/"
    return sorted(f for f in api.list_repo_files(OUT_REPO) if f.startswith(prefix))


def set_rows(name):
    """A set's clips: val and gold from the export, a public set loaded with its audio."""
    if name in ("val", "gold"):
        return splits[name]
    with ftkit.timed(f"{name}: loading"):
        return evalkit.load_benchmark(name, FT / "bench", TOKEN, LIMIT)


def decode_set(name):
    done = set()
    for shard in shard_names(name):
        for line in Path(hf_hub_download(OUT_REPO, shard, token=TOKEN)).read_text("utf-8").splitlines():
            done.add(json.loads(line)["segment_id"])
    rows = set_rows(name)
    todo = [r for r in rows if r["segment_id"] not in done]
    print(f"{name}: {len(done)} clips already decoded, {len(todo)} to decode "
          f"({sum(ftkit.duration(r) for r in todo) / 3600:.2f} h)")
    by = evalkit.BENCHMARKS[name].by if name in evalkit.BENCHMARKS else None
    first = len(shard_names(name))
    for k in range(0, len(todo), SHARD_CLIPS):
        part, number = todo[k:k + SHARD_CLIPS], first + k // SHARD_CLIPS + 1
        out = []
        with ftkit.timed(f"{name} shard {number}: beam {K} over {len(part)} clips"):
            for batch in ftkit.bucket_batches(part, budget_s=BEAM_BUDGET_S, max_items=BEAM_ITEMS,
                                              pad_to_s=PAD_TO_S, shuffle=False):
                chunk = [part[i] for i in batch]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                cands = nbest(chunk)
                torch.cuda.synchronize()
                took = time.perf_counter() - t0
                audio_s = sum(ftkit.duration(r) for r in chunk)
                for r, cs in zip(chunk, cands, strict=True):
                    out.append({"segment_id": r["segment_id"], "duration": round(ftkit.duration(r), 3),
                                "compute_s": round(took * ftkit.duration(r) / audio_s, 4), "candidates": cs})
        if name not in ("val", "gold"):
            with ftkit.timed(f"{name} shard {number}: greedy over {len(part)} clips"):
                texts, compute, log = decode(part)
            retried = {s for s, _, _ in log}
            by_id = {o["segment_id"]: o for o in out}  # the beam ran in batches sorted by length
            for r, t, c in zip(part, texts, compute, strict=True):
                by_id[r["segment_id"]].update({
                    "greedy": t, "greedy_compute_s": round(c, 4), "retried": r["segment_id"] in retried,
                    "ref": r["text"], "group": r["episode_id"], "by": r.get(by) if by else None})
        beam_s, audio_s = sum(o["compute_s"] for o in out), sum(o["duration"] for o in out)
        print(f"{name} shard {number}: beam RTF {beam_s / audio_s:.4f}")
        local = OUT / name / "shards" / f"part-{number:05d}.jsonl"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), "utf-8")
        with ftkit.timed(f"{name} shard {number}: uploading"):
            api.upload_file(path_or_fileobj=str(local), path_in_repo=f"{DEST}/{name}/shards/{local.name}",
                            repo_id=OUT_REPO, commit_message=f"{DEST}/{name}: shard {number} ({len(out)} clips)")
    del rows


model.eval()
for name in SETS:
    decode_set(name)
'''

SCORE_NOTE = """
## Every row on every set

Each set's shards are read back from the repo, so this cell works on whatever has been decoded.
Every candidate and the greedy transcript are aligned once, on every core. A set's report goes to
`<set>/report.json` as soon as it is scored; on val the D117 decision is written to
`headroom.json` as well.
"""

SCORE = r'''
from collections import Counter

from rapidfuzz.distance import Levenshtein

from app.services.fold import fold_tokens  # harness_scorer put the dataset's fold.py on the path

ROWS = ("greedy", "top1", "random", "mbr", *(f"oracle@{k}" for k in KS))


def align_one(pair):
    """One transcript's per-clip counts (without the alignment) and its folded tokens."""
    ref, hyp = pair
    c = score.per_clip([ref], [hyp])[0]
    c.pop("alignment", None)
    return c, fold_tokens(hyp)


def as_units(clips, errors, units):
    return [{"errors": c[errors], "words": c[units]} for c in clips]


def diff(base, pick, groups):
    """Folded WER, raw WER and CER of `pick` minus `base`, each with its interval."""
    return {metric: list(sweep.paired_bootstrap(as_units(base, e, u), as_units(pick, e, u), groups))
            for metric, e, u in (("wer", "errors", "words"), ("raw_wer", "raw_errors", "raw_words"),
                                 ("cer", "char_errors", "chars"))}


def score_set(name):
    records = {}
    for shard in shard_names(name):
        for line in Path(hf_hub_download(OUT_REPO, shard, token=TOKEN)).read_text("utf-8").splitlines():
            r = json.loads(line)
            records[r["segment_id"]] = r
    if name in ("val", "gold"):
        rows = [r for r in splits[name] if r["segment_id"] in records]
        greedy_texts = stored_greedy(rows)[0]
        refs, groups = [r["text"] for r in rows], [r["episode_id"] for r in rows]
    else:
        rows = list(records.values())
        greedy_texts = [r["greedy"] for r in rows]
        refs, groups = [r["ref"] for r in rows], [r["group"] for r in rows]
    cands = [[c["text"] for c in records[r["segment_id"]]["candidates"]] for r in rows]
    pairs = [(ref, t) for ref, cs in zip(refs, cands, strict=True) for t in cs]
    pairs += list(zip(refs, greedy_texts, strict=True))
    with ftkit.timed(f"{name}: aligning {len(pairs)} transcripts on {WORKERS} processes"):
        aligned = judgekit.parallel_map(align_one, pairs, workers=WORKERS)
    per, toks, at = [], [], 0
    for cs in cands:
        per.append([a[0] for a in aligned[at:at + len(cs)]])
        toks.append([a[1] for a in aligned[at:at + len(cs)]])
        at += len(cs)
    greedy = [a[0] for a in aligned[at:]]

    index = {"top1": [0] * len(rows), "random": judgekit.random_index([len(c) for c in cands]),
             "mbr": judgekit.mbr_index(toks, Levenshtein.distance)}
    index.update({f"oracle@{k}": judgekit.oracle_index(per, k) for k in KS})
    picks = {"greedy": greedy, **{row: [p[i] for p, i in zip(per, idx, strict=True)] for row, idx in index.items()}}
    texts = {"greedy": greedy_texts,
             **{row: [c[i] for c, i in zip(cands, idx, strict=True)] for row, idx in index.items()}}
    report = {"set": name, "clips": len(rows), "groups": len(set(groups)), "rows": {}}
    for row in ROWS:
        m = score.summarize(picks[row])
        m["plain_wer"] = evalkit.plain_wer(refs, texts[row])
        m["plain_cer"] = evalkit.plain_cer(refs, texts[row])
        if row != "greedy":
            m["vs_greedy"] = diff(greedy, picks[row], groups)
            m["spread"] = judgekit.spread(greedy, picks[row], groups)
        report["rows"][row] = m
    report["ranks"] = {str(r): c for r, c in sorted(Counter(index[f"oracle@{max(KS)}"]).items())}
    report["mbr_ranks"] = {str(r): c for r, c in sorted(Counter(index["mbr"]).items())}
    spread = judgekit.distinct(cands, key=lambda t: " ".join(fold_tokens(t)))
    report["distinct"] = {k: v for k, v in spread.items() if k != "per_clip"}
    report["beam_rtf"] = (sum(records[r["segment_id"]]["compute_s"] for r in rows)
                          / max(sum(records[r["segment_id"]]["duration"] for r in rows), 1e-9))
    if name in ("val", "gold"):
        report["by_class"] = {row: distill.by_class(rows, picks[row], score.summarize, keys=evalkit.REPORT_KEYS)
                              for row in ("greedy", "top1", "mbr", f"oracle@{max(KS)}")}
    elif evalkit.BENCHMARKS[name].by:
        by = evalkit.BENCHMARKS[name].by
        by_rows = [{"classes": {by: r["by"]}} for r in rows]
        report["by_class"] = {row: distill.by_class(by_rows, picks[row], score.summarize, keys=[by])
                              for row in ("greedy", "top1", "mbr", f"oracle@{max(KS)}")}
    if name == "val":
        decision = judgekit.headroom(per, groups, score.summarize, ks=KS, threshold=GAIN_TO_CONTINUE)
        report["decision"] = decision["decision"]
        report["gain_over_top1"] = [decision["gain"], *decision["gain_ci"]]
    report.update({"teacher": TEACHER, "dataset_export": export["exported_at"],
                   "fold_version": score.fold_version, "kits": evalkit.kit_digests(FT), "notebook": NOTEBOOK})
    local = OUT / name / "report.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    api.upload_file(path_or_fileobj=str(local), path_in_repo=f"{DEST}/{name}/report.json", repo_id=OUT_REPO,
                    commit_message=f"{DEST}/{name}: report")
    if name == "val":
        api.upload_file(path_or_fileobj=str(local), path_in_repo=f"{DEST}/headroom.json", repo_id=OUT_REPO,
                        commit_message=f"{DEST}: headroom, {report['decision']}")
    return report


reports = {}
for name in SETS:
    remote = f"{DEST}/{name}/report.json"
    if not RESCORE and api.file_exists(OUT_REPO, remote):
        reports[name] = fetch_json(remote)
        print(f"{name}: already scored, read back")
        continue
    if not shard_names(name):
        print(f"{name}: nothing decoded yet, skipped")
        continue
    reports[name] = score_set(name)
    m = reports[name]["rows"]
    print(f"{name}: greedy {m['greedy']['wer']:.2f} | top@1 {m['top1']['wer']:.2f} | MBR {m['mbr']['wer']:.2f} "
          f"| oracle@{max(KS)} {m[f'oracle@{max(KS)}']['wer']:.2f}")
'''

REPORT_NOTE = """
## The report

One table per question, every set in each. Differences are against greedy on the same clips, in
points, with 95% intervals; a negative number is fewer errors.
"""

REPORT = r"""
def ci(d):
    return f"{d[0]:+6.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"


TOP = f"oracle@{max(KS)}"
names = [n for n in SETS if n in reports]
print(f"Folded WER (S / D / I)\n{'set':<13} {'clips':>6}" + "".join(f"{r:>22}" for r in ROWS))
for n in names:
    rr = reports[n]["rows"]
    print(f"{n:<13} {reports[n]['clips']:>6}" + "".join(
        f"{rr[r]['wer']:>7.2f} ({rr[r]['sub']:.1f}/{rr[r]['del']:.1f}/{rr[r]['ins']:.1f})" for r in ROWS))

for metric, label in (("wer", "folded WER"), ("raw_wer", "raw WER"), ("cer", "CER")):
    print(f"\n{label}, minus greedy\n{'set':<13}" + "".join(f"{r:>25}" for r in ROWS[1:]))
    for n in names:
        rr = reports[n]["rows"]
        print(f"{n:<13}" + "".join(f"{ci(rr[r]['vs_greedy'][metric]):>25}" for r in ROWS[1:]))

print(f"\nPlain WER / CER (NFC, punctuation out, Latin lowercased)\n{'set':<13}" + "".join(f"{r:>16}" for r in ROWS))
for n in names:
    rr = reports[n]["rows"]
    print(f"{n:<13}" + "".join(f"{rr[r]['plain_wer']:>8.2f}/{rr[r]['plain_cer']:<7.2f}" for r in ROWS))

print(f"\nIs it a fluke? How the gain against greedy spreads over each set's groups\n"
      f"{'set':<13} {'groups':>6} | {'MBR: better/same/worse, top-5 share':>38} | {TOP + ': better/same/worse, top-5 share':>42}")
for n in names:
    rr, line = reports[n]["rows"], f"{n:<13} {reports[n]['groups']:>6}"
    for r, w in (("mbr", 38), (TOP, 42)):
        s = rr[r]["spread"]
        share = "n/a" if s["top_share"] is None else f"{s['top_share']:.0%}"
        cell = f"{s['improved']}/{s['unchanged']}/{s['worse']}, {share}"
        line += f" | {cell:>{w}}"
    print(line)

print(f"\nWhat there is to pick from\n{'set':<13} {'distinct of ' + str(K):>14} {'one only':>9} {'beam RTF':>9}  rank the oracle picked")
for n in names:
    r = reports[n]
    print(f"{n:<13} {r['distinct']['mean']:>14.1f} {r['distinct']['single']:>9} {r['beam_rtf']:>9.4f}  {r['ranks']}")

for n in names:
    by = reports[n].get("by_class")
    if not by:
        continue
    for key, values in by["greedy"].items():
        print(f"\n{n} by {key}\n  {'class':<14} {'clips':>6} {'greedy':>7} {'top@1':>7} {'MBR':>7} {TOP:>9} {'MBR gain':>9} {'ceiling':>8}")
        for v, g in values.items():
            t, b, o = by["top1"][key][v], by["mbr"][key][v], by[TOP][key][v]
            print(f"  {v:<14} {g['clips']:>6} {g['wer']:>7.2f} {t['wer']:>7.2f} {b['wer']:>7.2f} {o['wer']:>9.2f} "
                  f"{g['wer'] - b['wer']:>9.2f} {g['wer'] - o['wer']:>8.2f}")

if "val" in reports:
    lo_hi = reports["val"]["gain_over_top1"]
    print(f"\nDECISION (D117, val, threshold {GAIN_TO_CONTINUE}): {reports['val']['decision'].upper()} "
          f"(oracle@{max(KS)} {lo_hi[0]:.2f} [{lo_hi[1]:.2f}, {lo_hi[2]:.2f}] below top@1)")
"""

cells = [
    md(INTRO + nbkit.SMOKE_NOTE),
    md("## Config"),
    config("08_Judge_Headroom", CONFIG),
    md("## Setup"),
    nbkit.setup("rapidfuzz duckdb"),
    *nbkit.kits("ftkit", "evalkit", "sweep", "distill", "judgekit"),
    load(("val", "gold")),
    code(DECODE),
    md(TEACHER_NOTE),
    code(TEACHER),
    md(DECODE_NOTE),
    code(NBEST),
    md(SCORE_NOTE),
    code(SCORE),
    md(REPORT_NOTE),
    code(REPORT),
]

NOTEBOOKS = {"08_Judge_Headroom.ipynb": cells}

if __name__ == "__main__":
    nbkit.write(NOTEBOOKS, OUT_DIR)
