"""Build notebooks/08b_Judge_Pilot.ipynb: `python notebooks/src/build_judge_pilot.py`. Roadmap G1's
LLM-judge pilot (D118): does a local LLM that picks one of the teacher's 8 beam candidates beat the
cross-model vote V3 on 300 val clips? The rules are in judgekit.py, tested in
backend/tests/test_judge_kit.py; the candidates are the ones 08_Judge_Headroom decoded."""

import sys
from pathlib import Path

import nbkit
from build_flex import config, load
from nbkit import code, md

HERE = Path(__file__).parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent

INTRO = """
# 08b — The judge pilot: does an LLM pick better than a vote of smaller models?

`08_Judge_Headroom` found room among the teacher's 8 beam candidates (oracle@8 1.36 points of
folded val WER under the beam's own first choice), and a vote by the teacher's greedy decode and
four students (V3) took a quarter of it: val 6.50, gold 10.16. This notebook asks whether a local
LLM that reads the candidates (and in one case hears the clip) takes more. Everything below is
fixed in D118, before any judge was run.

**The sample.** 300 val clips drawn at random with a fixed seed; nothing preselects them.

**The judges**, each with thinking off and on, so four judge rows:
- **text only:** `Qwen/Qwen3.8-27B` reads the candidates;
- **audio-aware:** `google/gemma-4-12B-it` hears the clip the teacher transcribed, then reads them.

Each sees the clip's distinct candidates, shuffled per clip (LLMs favour the first one shown) and
numbered from 1, and answers with a number. It picks; it never rewrites. A clip with one distinct
candidate is not sent. An answer that is not exactly one number in range is a parse failure and
keeps the top candidate. Thinking has a budget (`THINK_BUDGET` tokens); a judge still thinking
there has its thinking closed for it, followed by `Answer: `, and writes the number.

**The rows, on the same clips:** greedy (the standard decoder), top@1 (the beam's first choice),
V3, an n-gram picker (an interpolated Kneser-Ney trigram on the train labels, its weight chosen on
the val clips outside the sample), the four judges, and oracle@8 (reads the reference: the
ceiling). Folded WER with S/D/I, raw WER, CER, plain WER and CER, per clip class; every judge
paired against V3, top@1 and greedy with episodes resampled; how its change against V3 spreads
over the episodes; how often it agrees with V3; the beam rank it picked; parse failures, forced
answers, and its time per clip.

**The rule** (D118): a judge row passes if its folded val WER minus V3's has its whole interval
below zero, episodes resampled, at Bonferroni's 98.75% (four rows, four chances). If none passes,
G1 stops and the negative is the result. A row that passes is then run on gold, reported and never
chosen on.

**Small blast radius.** Each judge's picks go to `OUT_REPO/<RUN_PREFIX>/judge-pilot/<split>/<judge>/`
every `SHARD_CLIPS` clips, and a rerun skips every clip already there; the report and the decision
are uploaded as soon as they are scored.
"""

CONFIG = r"""
TEACHER_FILE_REAL = "teacher.json"   # names the model in OUT_REPO (03e_Flex_Ship.ipynb)
HEADROOM = "judge-headroom"          # the candidates: OUT_REPO/<RUN_PREFIX>/<HEADROOM>/<split>/shards/ (08)
RUN = "judge-pilot"                  # OUT_REPO/<RUN_PREFIX>/<RUN>/
STUDENTS_REPO = "Sagyam/nepanglish-asr-students"
STUDENTS_PREFIX = "students-2026-09-30"
VOTERS = ("whisper-distill", "indicconformer-distill", "parakeet-distill", "conformer-distill")  # V3: stage 2
V3_VAL = 6.50                        # findings.md, A cross-model vote: the recomputed V3 must match it
SAMPLE_N, SAMPLE_SEED, SHOW_SEED = 300, 0, 0  # D118
QWEN, GEMMA = "Qwen/Qwen3.8-27B", "google/gemma-4-12B-it"
JUDGES = {  # consecutive rows of one model share one load
    "qwen-text": {"model": QWEN, "audio": False, "think": False},
    "qwen-text-think": {"model": QWEN, "audio": False, "think": True, "close": "</think>",
                        "force": "\n</think>\n\nAnswer: ", "chat": {"reasoning_effort": "medium"}},
    "gemma-audio": {"model": GEMMA, "audio": True, "think": False},
    "gemma-audio-think": {"model": GEMMA, "audio": True, "think": True, "close": "<channel|>",
                          "force": "\n<channel|>Answer: "},
}
THINK_BUDGET = 2048                  # thinking tokens before the answer is forced
ANSWER_TOKENS = 16                   # tokens for the number itself
LEVEL = 1 - 0.05 / len(JUDGES)       # D118: Bonferroni over the judge rows
NGRAM_ORDER = 3
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)  # n-gram weight grid, chosen outside the sample
BATCH = 16                           # clips per generate call; halved on an out-of-memory error
SHARD_CLIPS = 100                    # clips per uploaded shard: a lost runtime loses at most one
GOLD_FOR_PASSED = True               # run a passing judge on gold (reported, never chosen on)
RESCORE = False                      # True scores a split again even when its report is on the hub
WORKERS = None                       # processes aligning candidates; None for every core
"""

INPUTS_NOTE = """
## The candidates, the vote and the n-gram picker

The teacher's beam candidates come from 08's shards, its greedy transcripts from 03e, the voters'
from the students repo; inputs are always the real runs' (a smoke run only writes under its own
prefix). Every candidate is aligned once. V3 is recomputed on the whole of val and must match its
published score before any judge runs.
"""

INPUTS = r'''
import hashlib
import time
from collections import Counter

import soundfile as sf
from rapidfuzz.distance import Levenshtein

import distill
import judgekit
from app.services.fold import fold_tokens  # harness_scorer put the dataset's fold.py on the path

if SMOKE:
    SAMPLE_N, THINK_BUDGET = 12, 256
BASE = RUN_PREFIX.removesuffix("-smoke")  # where the inputs are; Setup put "-smoke" on RUN_PREFIX
DEST = f"{RUN_PREFIX}/{RUN}"
OUT = OUT_ROOT / RUN
WORKERS = WORKERS or os.cpu_count() or 1


def fetch_json(remote, repo=OUT_REPO):
    """A JSON file of `repo`, or None when it is not there."""
    if not api.file_exists(repo, remote):
        return None
    return json.loads(Path(hf_hub_download(repo, remote, token=TOKEN)).read_text("utf-8"))


def fetch_lines(remote, repo=OUT_REPO):
    path = Path(hf_hub_download(repo, remote, token=TOKEN))
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def files_under(prefix, repo=OUT_REPO):
    return sorted(f for f in api.list_repo_files(repo) if f.startswith(prefix))


TEACHER_FILE = distill.smoke_source(SMOKE, "teacher-smoke.json", TEACHER_FILE_REAL,
                                    lambda path: api.file_exists(OUT_REPO, path))
TEACHER = fetch_json(TEACHER_FILE)["run"]


def align_one(pair):
    ref, hyp = pair
    c = score.per_clip([ref], [hyp])[0]
    c.pop("alignment", None)
    return c


def prepare(split):
    """A split's clips with candidates: their texts, beam scores, folded tokens and per-clip counts,
    the greedy decode, and the V3 and oracle picks. Rows are matched by segment_id throughout."""
    recs = {}
    for shard in files_under(f"{BASE}/{HEADROOM}/{split}/shards/"):
        for r in fetch_lines(shard):
            recs[r["segment_id"]] = r
    voters = [{r["segment_id"]: r["text"] for r in fetch_lines(f"{TEACHER}/harness/{split}.jsonl")}]
    for s in VOTERS:
        lines = fetch_lines(f"{STUDENTS_PREFIX}/{s}/harness/{split}.jsonl", repo=STUDENTS_REPO)
        voters.append({r["segment_id"]: r["text"] for r in lines})
    rows = [r for r in splits[split] if r["segment_id"] in recs]
    ids = [r["segment_id"] for r in rows]
    missing = [i for i in ids if any(i not in v for v in voters)]
    assert not missing, f"{split}: {len(missing)} clips lack a voter's transcript"
    cands = [[c["text"] for c in recs[i]["candidates"]] for i in ids]
    greedy = [voters[0][i] for i in ids]
    pairs = [(r["text"], t) for r, cs in zip(rows, cands, strict=True) for t in cs]
    pairs += [(r["text"], g) for r, g in zip(rows, greedy, strict=True)]
    with ftkit.timed(f"{split}: aligning {len(pairs)} transcripts on {WORKERS} processes"):
        aligned = judgekit.parallel_map(align_one, pairs, workers=WORKERS)
    per, at = [], 0
    for cs in cands:
        per.append(aligned[at:at + len(cs)])
        at += len(cs)
    toks = [[fold_tokens(t) for t in cs] for cs in cands]
    vote = [[fold_tokens(v[i]) for v in voters] for i in ids]
    return {"rows": rows, "ids": ids, "at": {i: k for k, i in enumerate(ids)}, "cands": cands,
            "beam": [[c["score"] for c in recs[i]["candidates"]] for i in ids], "toks": toks,
            "per": per, "greedy": aligned[at:], "greedy_text": greedy,
            "v3": judgekit.mbr_index(toks, Levenshtein.distance, voters=vote),
            "oracle": judgekit.oracle_index(per, max(len(c) for c in cands))}


def wer_of(d, idx, ks=None):
    ks = range(len(d["ids"])) if ks is None else ks
    return score.summarize([d["per"][k][j] for k, j in zip(ks, idx, strict=True)])["wer"]


data = {"val": prepare("val")}
val = data["val"]
v3_val = wer_of(val, val["v3"])
print(f"val, {len(val['ids'])} clips: greedy {score.summarize(val['greedy'])['wer']:.2f} | "
      f"top@1 {wer_of(val, [0] * len(val['ids'])):.2f} | V3 {v3_val:.2f} | oracle {wer_of(val, val['oracle']):.2f}")
if not SMOKE:
    assert abs(v3_val - V3_VAL) <= 0.02, f"V3 on val is {v3_val:.2f}, published {V3_VAL}: the vote is not the same"

SAMPLE = judgekit.sample(val["ids"], SAMPLE_N, SAMPLE_SEED)
in_sample = set(SAMPLE)
print(f"sample: {len(SAMPLE)} clips from {len({val['rows'][val['at'][i]]['episode_id'] for i in SAMPLE})} episodes")

with ftkit.timed(f"n-gram: order {NGRAM_ORDER} on {len(splits['train'])} train labels"):
    lm = judgekit.KneserNey([fold_tokens(r["text"]) for r in splits["train"]], order=NGRAM_ORDER)


def add_ngram(d, lam):
    d["lm"] = [[lm.logprob(t) for t in ts] for ts in d["toks"]]
    d["words"] = [[len(t) for t in ts] for ts in d["toks"]]
    if lam is not None:
        d["ngram"] = judgekit.lm_index(d["beam"], d["lm"], d["words"], lam)


add_ngram(val, None)
tune = [k for k, i in enumerate(val["ids"]) if i not in in_sample]


def tune_wer(lam):
    idx = judgekit.lm_index([val["beam"][k] for k in tune], [val["lm"][k] for k in tune],
                            [val["words"][k] for k in tune], lam)
    return wer_of(val, idx, tune)


curve = {lam: tune_wer(lam) for lam in LAMBDAS}
LAMBDA = judgekit.choose_lambda(LAMBDAS, curve.__getitem__)
val["ngram"] = judgekit.lm_index(val["beam"], val["lm"], val["words"], LAMBDA)
print(f"n-gram weight on the {len(tune)} val clips outside the sample:",
      " | ".join(f"{lam:g}: {w:.2f}" for lam, w in curve.items()), f"-> lambda {LAMBDA:g}")
'''

JUDGE_NOTE = """
## The judges

One model on the GPU at a time, bf16, greedy decoding. The prompt is `judgekit.prompt`; the audio
judge gets the clip as a 16 kHz WAV beside it. Batches are sorted by length, and every answer is
matched back to its clip by segment_id.
"""

JUDGE = r'''
from transformers import AutoProcessor

WAV = FT / "wav"
WAV.mkdir(exist_ok=True)
loaded = {"id": None}


def load_judge(model_id):
    """`judge_model` and `processor` for `model_id`, freeing the model before it."""
    global judge_model, processor, stop_ids
    if loaded["id"] == model_id:
        return
    for name in ("judge_model", "processor"):
        globals().pop(name, None)
    gc.collect()
    torch.cuda.empty_cache()
    with ftkit.timed(f"loading {model_id}"):
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        for cls in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
            auto = getattr(transformers, cls, None)
            if auto is None:
                continue
            try:
                judge_model = auto.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"no auto class loads {model_id}")
    eos = judge_model.generation_config.eos_token_id
    stop_ids = {processor.tokenizer.pad_token_id, *(eos if isinstance(eos, list) else [eos])} - {None}
    loaded["id"] = model_id
    print(model_id, type(judge_model).__name__, f"| {torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU")


def wav_path(row):
    path = WAV / (hashlib.sha1(row["segment_id"].encode()).hexdigest()[:16] + ".wav")
    if not path.exists():
        sf.write(path, store.clip_f32(row).numpy(), ftkit.SR)
    return str(path)


def conversation(row, texts, judge):
    content = [{"type": "text", "text": judgekit.prompt(texts, audio=judge["audio"])}]
    if judge["audio"]:
        content.append({"type": "audio", "audio": wav_path(row)})
    return [{"role": "user", "content": content}]


def encode(convs, judge):
    return processor.apply_chat_template(
        convs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
        padding=True, enable_thinking=judge["think"], **judge.get("chat", {})).to(judge_model.device)


def generate(enc, max_new):
    with torch.no_grad():
        out = judge_model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return out[:, enc["input_ids"].shape[1]:]


def trim(ids):
    ids = list(ids)
    while ids and ids[-1] in stop_ids:
        ids.pop()
    return ids


def force(conv, judge, ids):
    """Budget forcing: the clip's prompt, the thinking so far and the model's end-of-thinking
    marker, then `ANSWER_TOKENS` more. Every per-token input (the mask, token types) is extended."""
    enc = encode([conv], judge)
    tail = processor.tokenizer(judge["force"], add_special_tokens=False, return_tensors="pt").input_ids
    tail = torch.cat([torch.tensor([ids]), tail], 1).to(judge_model.device)
    length = enc["input_ids"].shape[1]
    for key, value in list(enc.items()):
        if torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == length:
            pad = tail if key == "input_ids" else (torch.ones_like(tail) if "mask" in key else torch.zeros_like(tail))
            enc[key] = torch.cat([value, pad.to(value.dtype)], 1)
    return trim(generate(enc, ANSWER_TOKENS)[0].tolist())


def judge_batch(d, rows, judge):
    shown = [judgekit.present(d["cands"][d["at"][r["segment_id"]]], r["segment_id"], SHOW_SEED) for r in rows]
    convs = [conversation(r, [d["cands"][d["at"][r["segment_id"]]][j] for j in s], judge)
             for r, s in zip(rows, shown, strict=True)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen = generate(encode(convs, judge), THINK_BUDGET + ANSWER_TOKENS if judge["think"] else ANSWER_TOKENS)
    torch.cuda.synchronize()
    each = (time.perf_counter() - t0) / len(rows)
    out = []
    for r, s, conv, ids in zip(rows, shown, convs, gen.tolist(), strict=True):
        ids = trim(ids)
        text = processor.tokenizer.decode(ids, skip_special_tokens=False)
        answer, finished = judgekit.split_answer(text, judge.get("close"))
        extra = 0.0
        if not finished:
            t1 = time.perf_counter()
            text += judge["force"] + processor.tokenizer.decode(force(conv, judge, ids), skip_special_tokens=False)
            answer, _ = judgekit.split_answer(text, judge["close"])
            extra = time.perf_counter() - t1
        pos = judgekit.parse_pick(answer, len(s))
        out.append({"segment_id": r["segment_id"], "shown": s, "answer": answer[:200],
                    "pick": s[pos] if pos is not None else 0, "parsed": pos is not None,
                    "finished": finished, "skipped": False, "new_tokens": len(ids),
                    "judge_s": round(each + extra, 4), "batch": len(rows), "response": text})
    return out


def run_judge(name, split, rows):
    """`name`'s pick for every clip of `rows`, read back from its shards where they exist; the rest
    judged, uploaded every `SHARD_CLIPS` clips."""
    judge, d = JUDGES[name], data[split]
    prefix = f"{DEST}/{split}/{name}/"
    done = {}
    for shard in files_under(prefix):
        for rec in fetch_lines(shard):
            done[rec["segment_id"]] = rec
    todo = [r for r in rows if r["segment_id"] not in done]
    print(f"{split}/{name}: {len(rows) - len(todo)} clips already judged, {len(todo)} to judge")
    if todo:
        load_judge(judge["model"])
    todo.sort(key=lambda r: (ftkit.duration(r), sum(map(len, d["cands"][d["at"][r["segment_id"]]]))))
    number = len(files_under(prefix))
    for k in range(0, len(todo), SHARD_CLIPS):
        part, number = todo[k:k + SHARD_CLIPS], number + 1
        out = []
        many = []
        for r in part:
            if len(judgekit.present(d["cands"][d["at"][r["segment_id"]]], r["segment_id"], SHOW_SEED)) == 1:
                out.append({"segment_id": r["segment_id"], "shown": [0], "answer": "", "pick": 0, "parsed": True,
                            "finished": True, "skipped": True, "new_tokens": 0, "judge_s": 0.0, "batch": 0,
                            "response": ""})
            else:
                many.append(r)
        with ftkit.timed(f"{split}/{name} shard {number}: {len(many)} clips judged, {len(part) - len(many)} with one candidate"):
            i, size = 0, BATCH
            while i < len(many):
                chunk = many[i:i + size]
                try:
                    out += judge_batch(d, chunk, judge)
                except torch.cuda.OutOfMemoryError:
                    if size == 1:
                        raise
                    size //= 2
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"out of memory: batch {size} from here")
                    continue
                i += len(chunk)
        local = OUT / split / name / f"part-{number:05d}.jsonl"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), "utf-8")
        api.upload_file(path_or_fileobj=str(local), path_in_repo=prefix + local.name, repo_id=OUT_REPO,
                        commit_message=f"{prefix}: shard {number} ({len(out)} clips)")
        failed = sum(not o["parsed"] for o in out)
        forced = sum(not o["finished"] for o in out)
        print(f"  parse failures {failed}, forced answers {forced}, "
              f"{sum(o['judge_s'] for o in out) / max(len(many), 1):.2f} s a judged clip")
        done.update({o["segment_id"]: o for o in out})
    return {r["segment_id"]: done[r["segment_id"]] for r in rows}


sample_rows = [val["rows"][val["at"][i]] for i in SAMPLE]
picks = {"val": {name: run_judge(name, "val", sample_rows) for name in JUDGES}}
'''

SCORE_NOTE = """
## Every row on the sample, and the decision

Differences are in points of folded WER (and raw WER, CER), a negative number fewer errors, with
95% intervals from resampling episodes; the kill rule reads its own Bonferroni interval
(`LEVEL`). The report goes to `<split>/report.json`, the decision to `decision.json`.
"""

SCORE = r'''
TOP = "oracle@8"


def as_units(clips, errors, units):
    return [{"errors": c[errors], "words": c[units]} for c in clips]


def diff(base, pick, groups):
    """Folded WER, raw WER and CER of `pick` minus `base`, each with its 95% interval."""
    return {metric: list(sweep.paired_bootstrap(as_units(base, e, u), as_units(pick, e, u), groups))
            for metric, e, u in (("wer", "errors", "words"), ("raw_wer", "raw_errors", "raw_words"),
                                 ("cer", "char_errors", "chars"))}


def score_split(split, ids, names, decide):
    d = data[split]
    ks = [d["at"][i] for i in ids]
    rows = [d["rows"][k] for k in ks]
    refs, groups = [r["text"] for r in rows], [r["episode_id"] for r in rows]
    index = {"top1": [0] * len(ks), "v3": [d["v3"][k] for k in ks], "ngram": [d["ngram"][k] for k in ks]}
    index.update({n: [picks[split][n][i]["pick"] for i in ids] for n in names})
    index[TOP] = [d["oracle"][k] for k in ks]
    counts = {"greedy": [d["greedy"][k] for k in ks]}
    texts = {"greedy": [d["greedy_text"][k] for k in ks]}
    for row, idx in index.items():
        counts[row] = [d["per"][k][j] for k, j in zip(ks, idx, strict=True)]
        texts[row] = [d["cands"][k][j] for k, j in zip(ks, idx, strict=True)]
    report = {"split": split, "clips": len(ks), "groups": len(set(groups)), "rows": {},
              "row_order": ["greedy", *index]}
    for row in report["row_order"]:
        m = score.summarize(counts[row])
        m["plain_wer"] = evalkit.plain_wer(refs, texts[row])
        m["plain_cer"] = evalkit.plain_cer(refs, texts[row])
        if row != "greedy":
            m["vs_greedy"] = diff(counts["greedy"], counts[row], groups)
            m["vs_top1"] = diff(counts["top1"], counts[row], groups)
            m["ranks"] = {str(r): c for r, c in sorted(Counter(index[row]).items())}
        if row in names or row == "ngram":
            m["vs_v3"] = diff(counts["v3"], counts[row], groups)
            m["spread_vs_v3"] = judgekit.spread(counts["v3"], counts[row], groups)
            m["agree_v3"] = judgekit.agreement(index[row], index["v3"])
            m["agree_top1"] = judgekit.agreement(index[row], index["top1"])
        if row in names:
            recs = [picks[split][row][i] for i in ids]
            called = [(rec, d["rows"][d["at"][i]]) for rec, i in zip(recs, ids, strict=True) if not rec["skipped"]]
            m["judged"] = len(called)
            m["parse_failures"] = sum(not rec["parsed"] for rec, _ in called)
            m["forced"] = sum(not rec["finished"] for rec, _ in called)
            m["new_tokens_mean"] = sum(rec["new_tokens"] for rec, _ in called) / max(len(called), 1)
            m["timing"] = judgekit.timing([rec["judge_s"] for rec, _ in called], [ftkit.duration(r) for _, r in called])
            if decide:
                m["verdict"] = judgekit.verdict(counts[row], counts["v3"], groups, level=LEVEL)
        report["rows"][row] = m
    report["by_class"] = {row: distill.by_class(rows, counts[row], score.summarize, keys=evalkit.REPORT_KEYS)
                          for row in report["row_order"]}
    report["distinct"] = Counter(len(judgekit.present(d["cands"][k], d["ids"][k], SHOW_SEED)) for k in ks)
    report.update({"teacher": TEACHER, "dataset_export": export["exported_at"], "fold_version": score.fold_version,
                   "kits": evalkit.kit_digests(FT), "notebook": NOTEBOOK, "judges": JUDGES,
                   "think_budget": THINK_BUDGET, "sample_n": SAMPLE_N, "sample_seed": SAMPLE_SEED,
                   "show_seed": SHOW_SEED, "ngram": {"order": NGRAM_ORDER, "lambda": LAMBDA, "curve": curve},
                   "level": LEVEL, "transformers": transformers.__version__,
                   "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    local = OUT / split / "report.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), "utf-8")
    api.upload_file(path_or_fileobj=str(local), path_in_repo=f"{DEST}/{split}/report.json", repo_id=OUT_REPO,
                    commit_message=f"{DEST}/{split}: report")
    return report


reports = {"val": score_split("val", SAMPLE, list(JUDGES), decide=True)}
passed = [n for n in JUDGES if reports["val"]["rows"][n]["verdict"]["pass"]]
decision = {"decision": "continue" if passed else "stop", "passed": passed, "level": LEVEL,
            "verdicts": {n: reports["val"]["rows"][n]["verdict"] for n in JUDGES},
            "v3_wer": reports["val"]["rows"]["v3"]["wer"], "sample_n": len(SAMPLE)}
local = OUT / "decision.json"
local.write_text(json.dumps(decision, indent=1), "utf-8")
api.upload_file(path_or_fileobj=str(local), path_in_repo=f"{DEST}/decision.json", repo_id=OUT_REPO,
                commit_message=f"{DEST}: decision, {decision['decision']}")
print(f"DECISION (D118, val sample, {LEVEL:.2%} interval): {decision['decision'].upper()}",
      f"(passed: {', '.join(passed)})" if passed else "(no judge beats V3)")
'''

REPORT_NOTE = """
## The report

One table per question. Differences are in points with 95% intervals unless the column says
otherwise; a negative number is fewer errors.
"""

REPORT = r"""
def ci(d):
    return f"{d[0]:+6.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"


def show(rep):
    rr, order = rep["rows"], rep["row_order"]
    names = [n for n in order if "timing" in rr[n]]
    print(f"\n=== {rep['split']}: {rep['clips']} clips, {rep['groups']} episodes ===")
    print(f"{'row':<19} {'WER (S/D/I)':>22} {'raw WER':>8} {'CER':>6} {'plain':>13}")
    for r in order:
        m = rr[r]
        print(f"{r:<19} {m['wer']:>7.2f} ({m['sub']:.1f}/{m['del']:.1f}/{m['ins']:.1f})".ljust(42)
              + f"{m['raw_wer']:>8.2f} {m['cer']:>6.2f} {m['plain_wer']:>6.2f}/{m['plain_cer']:<6.2f}")
    for metric, label in (("wer", "folded WER"), ("raw_wer", "raw WER"), ("cer", "CER")):
        print(f"\n{label}: minus V3, minus top@1, minus greedy")
        for r in ["ngram", *names]:
            m = rr[r]
            print(f"  {r:<19} {ci(m['vs_v3'][metric]):>25} {ci(m['vs_top1'][metric]):>25} {ci(m['vs_greedy'][metric]):>25}")
    if any("verdict" in rr[n] for n in names):
        print(f"\nThe kill rule: judge minus V3, folded WER, {LEVEL:.2%} interval")
        for n in names:
            v = rr[n]["verdict"]
            print(f"  {n:<19} {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]  {'PASS' if v['pass'] else 'no'}")
    print("\nIs it a fluke? Change against V3 over episodes: better/same/worse, net errors removed, top-5 share")
    for r in ["ngram", *names]:
        s = rr[r]["spread_vs_v3"]
        share = "n/a" if s["top_share"] is None else f"{s['top_share']:.0%}"
        print(f"  {r:<19} {s['improved']}/{s['unchanged']}/{s['worse']}  net {s['net_errors_removed']:+d}  top-5 {share}")
    print("\nWhat the judges did")
    print(f"  {'row':<19} {'judged':>6} {'parse fail':>10} {'forced':>6} {'tokens':>7} {'= V3':>6} {'= top@1':>8} "
          f"{'s/clip':>7} {'p90 s':>6} {'s/audio s':>9}  ranks picked")
    for n in names:
        m, t = rr[n], rr[n]["timing"]
        print(f"  {n:<19} {m['judged']:>6} {m['parse_failures']:>10} {m['forced']:>6} {m['new_tokens_mean']:>7.0f} "
              f"{m['agree_v3']:>6.0%} {m['agree_top1']:>8.0%} {t['mean_s']:>7.2f} {t['p90_s']:>6.2f} "
              f"{t['per_audio_s']:>9.3f}  {m['ranks']}")
    print(f"  {'ngram':<19} lambda {rep['ngram']['lambda']:g}; = V3 {rr['ngram']['agree_v3']:.0%}; ranks {rr['ngram']['ranks']}")
    print(f"  V3 ranks {rr['v3']['ranks']}; oracle ranks {rr[TOP]['ranks']}; distinct candidates per clip {dict(sorted(rep['distinct'].items()))}")
    for key, values in rep["by_class"]["greedy"].items():
        print(f"\nby {key}\n  {'class':<14} {'clips':>6}" + "".join(f"{r[:12]:>13}" for r in order))
        for v, g in values.items():
            print(f"  {v:<14} {g['clips']:>6}" + "".join(f"{rep['by_class'][r][key][v]['wer']:>13.2f}" for r in order))


show(reports["val"])
"""

GOLD_NOTE = """
## Gold, for a judge that passed

Only a row that passed on val runs here, on every gold clip, with the n-gram weight chosen on val.
Gold is reported, never chosen on (D105); there is no rule here. A smoke run exercises the cell
with the first judge on a few clips.
"""

GOLD = r"""
names = passed or (list(JUDGES)[:1] if SMOKE else [])
if GOLD_FOR_PASSED and names:
    data["gold"] = prepare("gold")
    add_ngram(data["gold"], LAMBDA)
    gold_ids = data["gold"]["ids"][:8] if SMOKE else data["gold"]["ids"]
    gold_rows = [data["gold"]["rows"][data["gold"]["at"][i]] for i in gold_ids]
    picks["gold"] = {n: run_judge(n, "gold", gold_rows) for n in names}
    reports["gold"] = score_split("gold", gold_ids, names, decide=False)
    show(reports["gold"])
else:
    print("no judge passed on val: gold is not run (D118)")
"""

cells = [
    md(INTRO + nbkit.SMOKE_NOTE),
    md("## Config"),
    config("08b_Judge_Pilot", CONFIG),
    md("## Setup"),
    nbkit.setup("rapidfuzz duckdb soundfile librosa accelerate -U transformers"),
    *nbkit.kits("ftkit", "evalkit", "sweep", "distill", "judgekit"),
    load(("val", "gold"), flex=False),
    md(INPUTS_NOTE),
    code(INPUTS),
    md(JUDGE_NOTE),
    code(JUDGE),
    md(SCORE_NOTE),
    code(SCORE),
    md(REPORT_NOTE),
    code(REPORT),
    md(GOLD_NOTE),
    code(GOLD),
]

NOTEBOOKS = {"08b_Judge_Pilot.ipynb": cells}

if __name__ == "__main__":
    nbkit.write(NOTEBOOKS, OUT_DIR)
