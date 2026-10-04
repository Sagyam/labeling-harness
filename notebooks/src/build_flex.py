"""Build the Flex notebooks, 03a to 03e: `python notebooks/src/build_flex.py` (D105).

One notebook per GPU session, each starting from the hub and ending with an upload:
03a trains the vanilla recipe on two seeds, 03b scores runs on the public Nepali sets, 03c ablates
the augmentations, 03d blends the winner's weights with base, 03e freezes the teacher and exports
it for the CPU. Shared code lives in ftkit.py, evalkit.py, sweep.py, distill.py, xtalk.py,
augment.py and cpukit.py, written out by %%writefile cells."""

import sys
from pathlib import Path

import nbkit
from nbkit import GPU_NOTE, SMOKE_NOTE, code, md

HERE = Path(__file__).parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent

# --- what every Flex notebook shares -----------------------------------------------------------

COMMON_CONFIG = r"""
RUN_PREFIX = "flex-2026-09-30"   # OUT_REPO/<RUN_PREFIX>/<run>/: one folder per run, shared by 03a-03e
DATASET_EXPORT = "2026-09-30"    # the export every run trains and is scored on; another is refused
MODEL_ID = "bodhan-ai/indic-transcribe-flex"
LANG, MODE = "ne", "mixed"
OUT_REPO = "Sagyam/nepanglish-asr-flex-ft"  # private HF model repo
MODEL_NAME = "Indic-Transcribe-Flex FT"     # shown on the harness's Models page (D83)
SMOKE = False                    # True: a few hundred clips, one epoch, under <RUN_PREFIX>-smoke
PAD_TO_S = 1.0
EVAL_BUDGET_S, EVAL_ITEMS = 1200.0, 96
DATA_LOCAL = None                # a local export in the HF layout instead of the download
"""

TRAIN_CONFIG = r"""
EPOCHS, LR, WARMUP = 6, 1e-5, 0.1
EFFECTIVE_S = 720.0       # ~12 min of audio per optimizer step
PROBE_FRACTION = 0.9
GRAD_CKPT = False         # checkpoint the conformer layers if the probe finds a small batch
"""


def config(notebook: str, *bodies: str) -> dict:
    parts = [f'NOTEBOOK = "{notebook}"', COMMON_CONFIG, *bodies]
    return code("\n".join(p.strip("\n") for p in parts))


_LOAD = r'''
import gc
import math
import warnings

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

import evalkit
import ftkit
import sweep

# transformers' generate() warns on every call (max_new_tokens vs max_length); it is noise here and
# buries the training log. Errors still show.
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", module="transformers")
ftkit.fast_cuda()
TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)
api.create_repo(OUT_REPO, repo_type="model", private=True, exist_ok=True)

DATA = ftkit.download_dataset(DATA_LOCAL)  # labels and the scorer; the audio follows
export = json.loads((DATA / "training" / "manifest.json").read_text())
evalkit.check_export(export, DATASET_EXPORT)
splits = ftkit.load_splits(DATA)
if SMOKE:
    splits = {"train": splits["train"][::50], "val": splits["val"][::30], "gold": splits["gold"][::16]}
AUDIO_SPLITS = __AUDIO_SPLITS__  # the splits this notebook decodes or trains on
episodes = [r["episode_id"] for name in AUDIO_SPLITS for r in splits[name]]
with ftkit.timed(f"downloading {len(set(episodes))} recordings"):
    DATA = ftkit.download_dataset(DATA_LOCAL, episodes=episodes, analytics=__ANALYTICS__)
with ftkit.timed("reading them into RAM"):
    store = ftkit.AudioStore(DATA, episodes)
score = ftkit.harness_scorer(DATA, FT)
print({k: len(v) for k, v in splits.items()}, f"| audio in RAM: {store.gib:.1f} GiB | export",
      export["exported_at"][:10], "|", score.fold_version)

FLEX_DIR = snapshot_download(MODEL_ID)
sys.path.insert(0, FLEX_DIR)
from indic_transcribe import MODES, IndicTranscribe  # noqa: E402


def load_model(path):
    """Weights from `path` on the GPU in fp32, as `model` (which the decoder and the loss read)."""
    global asr, model, featurize
    asr = IndicTranscribe.from_pretrained(str(path), device="cuda", dtype=torch.float32)
    model, featurize = asr.model, asr.fe
    if globals().get("GRAD_CKPT"):
        for layer in model.model.encoder.layers:
            layer._forward = layer.forward
            layer.forward = lambda *a, _l=layer, **k: torch.utils.checkpoint.checkpoint(_l._forward, *a, use_reentrant=False, **k)
    return model


def free_model():
    for name in ("optimizer", "model", "asr"):
        globals().pop(name, None)
    gc.collect()
    torch.cuda.empty_cache()


load_model(FLEX_DIR)
tk = asr.tokenizer
itn, romanized = MODES[MODE]
PROMPT = tk.encode_prompt(LANG, itn=itn, romanized=romanized)
EOS, PAD, OFFSET = tk.eos_id, tk.pad_id, tk.spl_size
UNK_MULTI = tk.multi.unk_id()
'''


def load(audio_splits: tuple[str, ...], analytics: bool = False) -> dict:
    return code(
        _LOAD.replace("__AUDIO_SPLITS__", repr(audio_splits)).replace(
            "__ANALYTICS__", repr(analytics)
        )
    )


DECODE = r'''
import shutil

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize(text: str) -> str:
    """Map the characters the tokenizer lacks onto ones fold.py scores as identical."""
    return (text.replace("।", ".").translate(DEV_DIGITS).replace("—", "-")
            .replace("‍", "").replace("‌", ""))


def features(wav: torch.Tensor, lens: torch.Tensor):
    feats, flens = featurize(wav, lens)  # fp32 even inside autocast: it disables autocast itself
    mask = (torch.arange(feats.size(2), device=feats.device)[None, :] < flens[:, None]).long()
    return feats, flens, mask


def transcribe(rows, **gen):
    clips = [store.clip_f32(r) for r in rows]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(c) for c in clips), PAD_TO_S))
    for i, c in enumerate(clips):
        wav[i, :len(c)] = c
    feats, _, mask = features(wav.cuda(), torch.tensor([len(c) for c in clips], device="cuda"))
    prompt = torch.tensor([PROMPT] * len(rows), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(input_features=feats, attention_mask=mask, decoder_input_ids=prompt,
                             do_sample=False, num_beams=1, eos_token_id=EOS, pad_token_id=PAD,
                             **{"max_new_tokens": 300, **gen})
    return [tk.decode(tk.strip_prompt_and_trim(o.tolist(), PROMPT)) for o in out]


# Loop retry, fixed before scoring anything with it: greedy decoding occasionally sticks on a filler
# ("अँ. अँ. अँ.") to the length limit. A global repetition ban would also hit real repeats the
# references keep ("कोही छैन। कोही छैन"), so only a clip whose greedy output loops is decoded
# again, alone, with a repetition penalty, no repeated 6-token phrase, and a length cap from its
# duration (the densest training label is 12.6 tokens/s).
MAX_TOKENS_PER_S = 13.0
RETRY = {"repetition_penalty": 1.2, "no_repeat_ngram_size": 6}


def retry_one(row):
    cap = min(300, math.ceil(MAX_TOKENS_PER_S * ftkit.duration(row)))
    return transcribe([row], max_new_tokens=cap, **RETRY)[0]


def decode(rows):
    """The decoder every score in 03a-03e comes from: greedy, batched, plus the loop retry.
    Returns the texts, each clip's compute seconds and the retries."""
    model.eval()
    d = ftkit.RetryLoops(transcribe, retry_one)
    with torch.no_grad():
        texts, compute = ftkit.transcribe_rows(rows, d, budget_s=EVAL_BUDGET_S, max_items=EVAL_ITEMS,
                                               pad_to_s=PAD_TO_S)
    return texts, compute, d.log


def save_best(model):
    """`model` as `OUT/best/`, in bf16, with the code and tokenizer files IndicTranscribe loads."""
    dst = OUT / "best"
    model.save_pretrained(dst, state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    for f in Path(FLEX_DIR).iterdir():
        if f.suffix in {".py", ".model", ".md"} or f.name in {"tokenizer_config.json", "generation_config.json",
                                                             "feature_extractor.safetensors"}:
            shutil.copy(f, dst / f.name)


def card(description, **extra):
    """The model card's fields that do not depend on the scores: what the model is, and which
    export, fold and code the run used."""
    return {
        "name": MODEL_NAME,
        "description": description,
        "architecture": "Canary-style enc-dec: 32L conformer + 24L transformer decoder, 1.2B",
        "base_model": MODEL_ID,
        "decoder": "greedy+retry",
        "notebook": NOTEBOOK,
        "dataset_export": export["exported_at"],
        "train_export": {k: export.get(k) for k in ("exported_at", "git_commit", "row_count",
                                                    "normalization_version", "label_version")},
        "fold_version": score.fold_version,
        "kits": evalkit.kit_digests(FT),
        **extra,
    }


def fetch_json(remote):
    """A JSON file of OUT_REPO, or None when it is not there."""
    if not api.file_exists(OUT_REPO, remote):
        return None
    return json.loads(Path(hf_hub_download(OUT_REPO, remote, token=TOKEN)).read_text("utf-8"))


def uploaded(run):
    """A run's result row: OUT_REPO has it once the run is trained, scored and uploaded."""
    return fetch_json(f"{RUN_PREFIX}/{run}/result.json")


def run_counts(run):
    """A run's per-clip counts on gold and val, for another run to be paired against."""
    return fetch_json(f"{RUN_PREFIX}/{run}/per_clip.json")


def upload_run(out, run, message, weights):
    with ftkit.timed(f"uploading {run} ({'with' if weights else 'without'} weights)"):
        api.upload_folder(repo_id=OUT_REPO, folder_path=str(out), path_in_repo=f"{RUN_PREFIX}/{run}",
                          ignore_patterns=None if weights else ["best/*"], commit_message=f"{run}: {message}")


def download_weights(run):
    """`<run>/best/` of OUT_REPO as a local folder `load_model` reads."""
    with ftkit.timed(f"downloading the weights of {run}"):
        root = snapshot_download(OUT_REPO, allow_patterns=[f"{RUN_PREFIX}/{run}/best/*"], token=TOKEN)
    best = Path(root) / RUN_PREFIX / run / "best"
    if not (best / "model.safetensors").exists():
        raise FileNotFoundError(f"{OUT_REPO} holds no weights for {RUN_PREFIX}/{run}")
    return best


def line(m):
    return f"{m['wer']:6.2f} ({ftkit.sid(m)})"
'''

TARGETS = r"""
kept = []
for r in splits["train"]:
    ids = tk.multi.encode(normalize(r["text"]), out_type=int)
    if UNK_MULTI not in ids:
        r["ids"] = [OFFSET + i for i in ids] + [EOS]
        kept.append(r)
print(f"train: dropped {len(splits['train']) - len(kept)} clip(s) the tokenizer cannot write")
splits["train"] = kept
r = splits["train"][0]
assert tk.decode(r["ids"][:-1]) == normalize(r["text"]).strip(), "target encoding does not round-trip"
MAX_TARGET = max(len(r["ids"]) for r in splits["train"])
print("prompt", PROMPT, "| longest target", MAX_TARGET, "tokens")
"""

TRAIN = r'''
import traceback

# The probe runs on the base weights loaded in Setup; every run reloads fresh ones and uses the
# budget measured here, so all runs see the same batches.
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, fused=True)
ftkit.init_optimizer_state(model, optimizer)


def spec_augment(feats: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Canary's defaults: 2 frequency masks of <= 27 bins, 10 time masks of <= 5% of each clip."""
    b, f, t = feats.shape
    fr = torch.arange(f, device=feats.device)[None, :]
    tr = torch.arange(t, device=feats.device)[None, :]
    mask = torch.zeros(b, f, t, dtype=torch.bool, device=feats.device)
    for _ in range(2):
        w = torch.randint(0, 28, (b, 1), device=feats.device)
        s = (torch.rand(b, 1, device=feats.device) * (f - w)).long()
        mask |= ((fr >= s) & (fr < s + w))[:, :, None]
    for _ in range(10):
        w = (torch.rand(b, 1, device=feats.device) * 0.05 * lens[:, None]).long()
        s = (torch.rand(b, 1, device=feats.device) * (lens[:, None] - w).clamp(min=1)).long()
        mask |= ((tr >= s) & (tr < s + w))[:, None, :]
    return feats.masked_fill(mask, 0.0)


def forward_loss(feats, mask, inp, lab):
    logits = model(input_features=feats, attention_mask=mask, decoder_input_ids=inp, use_cache=False).logits
    return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), lab.flatten(), ignore_index=-100,
                                             reduction="sum")


longest = max(splits["train"], key=ftkit.duration)
max_len = ftkit.pad_len(round(ftkit.duration(longest) * ftkit.SR), PAD_TO_S)
max_t = len(PROMPT) + MAX_TARGET - 1


def probe_step(n):
    wav = torch.randn(n, max_len, device="cuda") * 0.1
    feats, _, mask = features(wav, torch.full((n,), max_len, device="cuda"))
    inp = torch.randint(OFFSET, OFFSET + 6000, (n, max_t), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = forward_loss(feats, mask, inp, inp)
    loss.backward()


max_items = ftkit.probe_max_items(probe_step, 1, 256, model)
BUDGET_S = max_items * max_len / ftkit.SR * PROBE_FRACTION
print(f"largest micro-batch at {max_len / ftkit.SR:.0f} s x {max_t} tokens: {max_items} clips -> "
      f"budget {BUDGET_S:.0f} s of padded audio per micro-batch")

N_PROMPT = len(PROMPT)
SEED = 0
AUG = None          # an augment.Augmenter, set per run by 03c; None trains on the clips as they are
STRETCH = 1.0  # 03c raises it when speed perturbation lengthens clips: batches are sized as if slowed


def make_augmenter(stages):
    """03c defines the real one; this notebook trains the clips as they are."""
    raise RuntimeError("no augmenter in this notebook: 03c_Flex_Augment trains the augmented recipes")


def train_example(row, rng):
    """(int16 audio, target ids) for one training clip: as it is, or through the run's augmenter.
    When the augmenter rewrites the label (two voices, everything said), the new text is encoded;
    if the tokenizer cannot write it, or it is longer than the probe measured, the clip stays clean."""
    clip = store.clip(row)
    if AUG is None:
        return clip, row["ids"]
    audio, info = AUG(row, clip, rng)
    if info.get("text") is None:
        return audio, row["ids"]
    ids = tk.multi.encode(normalize(info["text"]), out_type=int)
    if UNK_MULTI in ids or len(ids) + 1 > MAX_TARGET:
        return clip, row["ids"]
    return audio, [OFFSET + i for i in ids] + [EOS]


def collate(rows):
    # torch reseeds each DataLoader worker per epoch, so every epoch draws fresh augmentations, and
    # the run is still repeatable from its seed
    rng = np.random.default_rng(torch.randint(2**62, (1,)).item())
    examples = [train_example(r, rng) for r in rows]
    clips = [torch.from_numpy(np.array(a, dtype=np.int16)) for a, _ in examples]
    targets = [ids for _, ids in examples]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(c) for c in clips), PAD_TO_S), dtype=torch.int16)
    for i, c in enumerate(clips):
        wav[i, :len(c)] = c
    width = N_PROMPT + max(len(ids) for ids in targets) - 1
    inp = torch.full((len(rows), width), PAD, dtype=torch.long)
    lab = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, ids in enumerate(targets):
        full = PROMPT + ids
        inp[i, :len(full) - 1] = torch.tensor(full[:-1])
        lab[i, N_PROMPT - 1:len(full) - 1] = torch.tensor(ids)  # predict target + eos only
    return {"wav": wav, "lens": torch.tensor([len(c) for c in clips]), "inp": inp, "lab": lab,
            "tokens": sum(len(ids) for ids in targets),
            "seconds": sum(len(c) for c in clips) / ftkit.SR, "padded_seconds": wav.numel() / ftkit.SR}


def loss_fn(model, b):
    wav = b["wav"].cuda(non_blocking=True).float() / 32768.0
    feats, flens, mask = features(wav, b["lens"].cuda(non_blocking=True))
    feats = spec_augment(feats, flens)
    loss = forward_loss(feats, mask, b["inp"].cuda(non_blocking=True), b["lab"].cuda(non_blocking=True))
    return loss, b["tokens"]


def evaluate(model, rows=None):
    rows = rows or splits["val"]
    texts, _, _ = decode(rows)
    return score([r["text"] for r in rows], texts)


def make_batches(epoch):
    # seed 0 keeps the order every earlier run used; another seed also reorders the batches
    return ftkit.bucket_batches(splits["train"], budget_s=BUDGET_S, max_items=256, pad_to_s=PAD_TO_S,
                                shuffle=True, seed=epoch + 1000 * SEED, stretch=STRETCH)


def train_run(recipe, seed, *, stages=None, references=None, baseline=None):
    """One run from the base weights: train, score gold and val, upload. Returns its result row.

    Skipped when OUT_REPO has the run's result; when it has the weights but no result (the session
    ended while scoring), the weights are fetched and only the scoring is done again. `stages` is
    an augmentation recipe for `make_augmenter` (03c). With `baseline` (the vanilla rows) the run
    is judged as it finishes and its weights are uploaded only if it clears the rule; without, the
    weights always go up, as soon as training ends. A failure frees the GPU before it is raised, so
    the next run's cell can start."""
    global OUT, SEED, AUG, STRETCH, optimizer
    run = f"{recipe}-s{seed}"
    if (row := uploaded(run)) is not None:
        print(f"{run}: already in {OUT_REPO}, skipped")
        return row
    OUT = OUT_ROOT / run
    OUT.mkdir(parents=True, exist_ok=True)
    epochs = 1 if SMOKE else EPOCHS
    failure, stopped = None, False
    try:
        print(f"\n=== {run}" + (f": {json.dumps(stages)}" if stages else ""))
        if api.file_exists(OUT_REPO, f"{RUN_PREFIX}/{run}/best/model.safetensors"):
            print(f"{run}: weights already in {OUT_REPO}; scoring them")
            free_model()
            load_model(download_weights(run))
            history = fetch_json(f"{RUN_PREFIX}/{run}/history.json") or []
        else:
            SEED = seed
            AUG, STRETCH = make_augmenter(stages) if stages else (None, 1.0)
            free_model()  # the probe's weights and optimizer, or an interrupted run's
            load_model(FLEX_DIR).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, fused=True)
            ftkit.init_optimizer_state(model, optimizer)
            cfg = ftkit.TrainConfig(name=run, out=str(OUT), epochs=epochs, lr=LR, warmup_frac=WARMUP,
                                    effective_s=EFFECTIVE_S, patience=2, seed=seed)
            history = ftkit.train(model, cfg=cfg, rows=splits["train"], make_batches=make_batches,
                                  collate=collate, loss_fn=loss_fn, evaluate=evaluate, save_best=save_best,
                                  optimizer=optimizer, monitor=monitor)["history"]
            if baseline is None:
                upload_run(OUT, run, "best weights, before scoring", weights=True)
        evals = [h for h in history if "val_wer" in h]
        meta = {"recipe": recipe, "seed": seed, "stages": stages or {}, "epochs": epochs, "lr": LR,
                "best_epoch": min(evals, key=lambda h: h["val_wer"])["epoch"] if evals else None}
        description = f"{RUN_PREFIX}: {recipe}, seed {seed}" + (f", {json.dumps(stages)}" if stages else "")
        row = evalkit.evaluate_run(OUT, run, splits=splits, decode=decode, score=score,
                                   card=card(description), meta=meta, references=references)
        keep = True
        if baseline is not None:
            keep, why = sweep.beats_baseline(row, baseline)
            row.update(kept=keep, why=why)
            (OUT / "result.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
            print(f"{run}: {'KEPT' if keep else 'not kept'}: {why}")
        upload_run(OUT, run, "scores on gold and val", weights=keep and baseline is not None)
    except BaseException as e:
        failure, stopped = traceback.format_exc(), isinstance(e, KeyboardInterrupt)
    finally:
        AUG, STRETCH = None, 1.0
        free_model()
    if stopped:
        raise KeyboardInterrupt(f"{run} stopped by hand outside training; the GPU is freed")
    if failure:
        print(failure)
        raise RuntimeError(f"{run} failed (traceback above); the GPU is freed for the next run")
    return row


monitor = ftkit.GpuMonitor().start()
'''

SPEED_NOTE = """
## Speed check (before committing to the run)

This times forward+backward on 10 real training micro-batches, then one full batched val pass,
and projects the whole run. The val pass doubles as the **val WER before training**, the baseline
that fine-tuning has to beat. No optimizer step runs, so no weight moves. If the projection is
too long, or GPU utilisation is low, stop here and fix it.
"""

SPEED = r"""
OUT = OUT_ROOT
speed = ftkit.speed_check(model, rows=splits["train"], batches=make_batches(0), collate=collate,
                          loss_fn=loss_fn, evaluate=evaluate, val_rows=splits["val"], gold_rows=splits["gold"],
                          epochs=1 if SMOKE else EPOCHS, monitor=monitor)
(OUT_ROOT / "speed_check.json").write_text(json.dumps(speed, indent=1))
print(f"one run is about {speed['projected_h']:.1f} h of training and gold, plus a val pass to score")
"""

# The public sets, shared by 03b (any run) and 03d (each blend).
SETS = r'''
SETS = tuple(evalkit.BENCHMARKS)  # every public set; name fewer to skip some
LIMIT = 40 if SMOKE else None     # clips per set
# Each set's crosstalk and SNR, measured once by scripts/measure_benchmark_overlap.py
# (D111).
CONDITIONS = evalkit.fetch_conditions(OUT_REPO, TOKEN, FT / "conditions", SETS)


def done_sets(run):
    """The public sets OUT_REPO already holds a summary of for this run."""
    prefix = f"{RUN_PREFIX}/{run}/benchmarks/"
    return {Path(f).stem for f in api.list_repo_files(OUT_REPO)
            if f.startswith(prefix) and f.endswith(".json") and Path(f).stem in evalkit.BENCHMARKS}


def score_sets(run):
    """Decode the public sets with the weights in `model` and upload each as it finishes."""
    out = OUT_ROOT / run
    done = done_sets(run)
    for name in SETS:
        if name in done:
            print(f"{run} {name}: already in {OUT_REPO}, skipped")
            continue
        evalkit.run_benchmarks(out, decode=decode, score=score, work=FT / "benchmarks", token=TOKEN,
                               names=[name], limit=LIMIT, run=run, conditions_dir=CONDITIONS)
        api.upload_folder(repo_id=OUT_REPO, folder_path=str(out), path_in_repo=f"{RUN_PREFIX}/{run}",
                          allow_patterns=[f"benchmarks/{name}.json*", f"harness/errors/{name}.parquet"],
                          commit_message=f"{run}: {name}")


def set_files(run, name):
    """(summary, per-clip lines) of a run on a public set, from OUT_REPO; None when it has none."""
    remote = f"{RUN_PREFIX}/{run}/benchmarks/{name}.json"
    if not api.file_exists(OUT_REPO, remote):
        return None
    lines = Path(hf_hub_download(OUT_REPO, remote + "l", token=TOKEN)).read_text("utf-8").splitlines()
    return fetch_json(remote), [json.loads(x) for x in lines]


def public_table(runs, against="base"):
    """Every run on every public set, and each run minus `against` with the set's group resampled."""
    table = {}
    for name in SETS:
        files = {run: set_files(run, name) for run in runs}
        ref = files.get(against)
        print(f"\n{name}: {evalkit.BENCHMARKS[name].what}")
        for run, got in files.items():
            if got is None:
                print(f"  {run:<14} not scored")
                continue
            m, lines = got
            entry = {k: m[k] for k in ("wer", "raw_wer", "plain_wer", "sub", "del", "ins", "clips")}
            entry["plain_cer"] = m.get("plain_cer")  # absent from a set scored before 2026-10-04
            entry["by"] = {k: v["wer"] for k, v in (m.get("by") or {}).items()}
            cer = "" if entry["plain_cer"] is None else f" / CER {entry['plain_cer']:.2f}"
            text = f"  {run:<14} {line(m)}  plain {m['plain_wer']:.2f}{cer}"
            if ref is not None and run != against:
                entry["vs"] = evalkit.pair_benchmark(ref[1], lines)
                d, lo, hi = entry["vs"]["all"]
                text += f"  | minus {against} {d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            table.setdefault(name, {})[run] = entry
            print(text)
    means = {run: float(np.mean([table[n][run]["wer"] for n in SETS]))
             for run in runs if all(run in table.get(n, {}) for n in SETS)}
    print("\npublic mean, folded WER:", "  ".join(f"{run} {v:.2f}" for run, v in means.items()))
    return table, means
'''

# `model` as a blend of base and a fine-tune, shared by 03d (every alpha) and 03e (the chosen one).
BLEND = r'''
from safetensors.torch import load_file


def blend(alpha, tuned):
    """`model` = (1 - alpha) * base + alpha * fine-tuned (WiSE-FT), in place on fresh base weights.

    `tuned` is the fine-tune's state dict. Every float tensor it holds is interpolated with the
    same alpha; a buffer that is not float is taken from the fine-tune. A tensor two names share
    (the tied output head) is blended once. Returns how many tensors were blended."""
    load_model(FLEX_DIR)
    seen, n = set(), 0
    with torch.no_grad():
        for key, w in model.state_dict().items():
            if key not in tuned or w.data_ptr() in seen:
                continue
            seen.add(w.data_ptr())
            t = tuned[key].to(w.device)
            if w.is_floating_point():
                w.lerp_(t.to(w.dtype), alpha)
                n += 1
            else:
                w.copy_(t)
    return n
'''

# --- 03a: train the vanilla recipe -------------------------------------------------------------

TRAIN_INTRO = """
# 03a — Train Flex: the vanilla recipe, two seeds

Step 3 of the protocol (D105). A full fine-tune of `bodhan-ai/indic-transcribe-flex`: a 1.2 B
Canary-style encoder-decoder with a 32-layer conformer encoder and a 24-layer transformer decoder.
It trains on the train split with the `ne` + mixed-script prompt, uses val for model selection and
gold for the final score.

**Vanilla, twice.** The recipe is the one every earlier run used with no augmentation added, so
its numbers stay comparable to history. It runs on seeds 0 and 1, each from the same base weights,
export and batch budget. The gap between the two seeds on val is the noise every later comparison
has to clear: an augmentation in 03c is kept only if it beats the better seed by more than that,
and never by less than 0.15 points (`sweep.MIN_NOISE`, D109), since two seeds can agree by luck.

**What comes after.** 03b scores base Flex and these runs on the public Nepali sets (what
fine-tuning cost outside our domain). 03c ablates the augmentations. 03d blends the winner's
weights with base. 03e freezes one model as the teacher and exports it for the CPU.

**Why this model.** It led the zero-shot comparisons at 18.2% folded WER on gold (CI 14.0–21.2),
flat across code-mixing tiers. Its mixed mode already writes the corpus's convention: 33.6% Latin
tokens against the references' 36.0%.

**What had to be built, because the release is inference-only.**
- **Loss.** The model's `forward(labels=...)` raises `NotImplementedError`. The loss here is
  cross-entropy on teacher-forced logits. The input is the fixed 10-token prompt followed by the
  target, and loss is taken on the target tokens and the end-of-text only.
- **Target encoding.** The tokenizer has no text `encode()`. Targets are the multilingual
  SentencePiece ids offset by the 1,152 special tokens, which round-trips exactly under the
  model's own `decode`.
- **Regularisation.** The port has no dropout, so SpecAugment (2 frequency masks of ≤27 bins and
  10 time masks of ≤5%, Canary's defaults) is applied to the features on the GPU. It is part of
  vanilla.
- **Labels mapped to what the tokenizer can write.** Without this, 76% of labels would contain an
  unknown token. `।` becomes `.`, Devanagari digits become Latin, and ZWJ/ZWNJ are removed.
  `fold.py` scores each pair as identical, so WER is unaffected.

**Licence.** A fine-tuned model is a derivative under the Indic Open Model License v1.0. You can
use it privately. Giving it to anyone passes the same licence on, and hosting it as a service for
others needs Bodhan AI's written sign-off. Keep the weights private.

**Choices.** Peak LR 1e-5. Linear decay, 10% warmup, up to 6 epochs, stopping after 2 evaluations
without a gain on val.

**One evaluation** (`evalkit.evaluate_run`, the same code for every model in the protocol). Gold
and val, folded and raw, split into S, D and I (substitutions, deletions and insertions per 100
folded reference words, which add up to the folded WER), overall and per clip class, and paired
clip by clip against a reference run with episodes resampled. Read S, D and I together: a change
can move errors between kinds without moving WER.

**Decoding: greedy, plus a loop retry.** A clip whose greedy output repeats a 3-word sequence 5+
times is decoded again, alone, with a repetition penalty, no repeated 6-token phrase and a
length cap from its duration. The trigger reads only the model's own output (the idea of
Whisper's compression-ratio fallback), so it is usable on any audio. The settings were fixed
before scoring and never tuned; each split's metrics keep the greedy-only score from the same run.

**Outputs** go to `OUT_REPO/<RUN_PREFIX>/<run>/`: `best/` (bf16 weights), `history.json`, metrics,
`per_clip.json` (what a later run is paired against), `result.json` and the `harness/` folder for
the Models page (D83). The weights go up as soon as training ends, before scoring. A run whose
result is already there is skipped, so after a dropped session, run the notebook again.
"""

RUNS_NOTE = """
## Train, score and upload each seed

One cell per seed, so a crash takes down one run, not both. Per run: fresh base weights, training
with early stopping on val, the best weights uploaded, then gold and val decoded with them and
scored. Seed 1 is also paired against seed 0: on gold, that interval is what two runs of the same
recipe differ by.
"""

TRAIN_REPORT_NOTE = """
## Report

Both seeds on val and gold with S/D/I, gold per crosstalk bucket, and each run minus its
references. **The val gap between the seeds is the noise** 03c's rule uses, or 0.15 when the gap is
smaller (D109). The gold columns are
held-out scores: nothing here was chosen on gold.
"""

TRAIN_REPORT = r"""
rows = [r for r in (uploaded(f"vanilla-s{seed}") for seed in SEEDS) if r]
print(f"{'run':<14}{'epoch':>6}  {'val':<34}{'gold':<34}{'gold raw':>9}")
for r in rows:
    print(f"{r['run_name']:<14}{str(r.get('best_epoch') or '-'):>6}  {line(r['val']):<34}{line(r['gold']):<34}"
          f"{r['gold']['raw_wer']:9.2f}")
if len(rows) == 2:
    gap = abs(rows[0]["val_wer"] - rows[1]["val_wer"])
    bar = max(gap, sweep.MIN_NOISE)
    print(f"\nseed noise on val: {gap:.2f} points. An augmentation has to beat the better seed by more"
          f" than {bar:.2f} (the floor is {sweep.MIN_NOISE:.2f}, D109).")
print("\ngold by crosstalk bucket:")
for bucket in evalkit.BUCKETS:
    for r in rows:
        m = (r["gold_by_class"].get("overlap") or {}).get(bucket)
        if m:
            print(f"  {bucket:>6} {r['run_name']:<12} {m['clips']:4d} clips  {line(m)}")
for r in rows:
    for split in ("val", "gold"):
        for name, vs in r[split]["vs"].items():
            d, lo, hi = vs["all"]
            print(f"{r['run_name']} {split} minus {name}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] on {vs['clips']} clips")
"""

train_cells = [
    md(TRAIN_INTRO + SMOKE_NOTE + GPU_NOTE),
    md("## Config"),
    config(
        "03a_Flex_Train",
        TRAIN_CONFIG,
        r"""
SEEDS = (0, 1)            # both vanilla: the second measures run-to-run noise
""",
    ),
    md("## Setup"),
    nbkit.setup("rapidfuzz duckdb"),
    *nbkit.kits("ftkit", "evalkit", "sweep", "distill"),
    load(("train", "val", "gold")),
    code(DECODE),
    md("## Targets, the batch-size probe, loss and batches"),
    code(TARGETS),
    code(TRAIN),
    md(SPEED_NOTE),
    code(SPEED),
    md(RUNS_NOTE),
    md("### vanilla, seed 0"),
    code('train_run("vanilla", SEEDS[0])'),
    md("### vanilla, seed 1"),
    code(r"""
first = run_counts(f"vanilla-s{SEEDS[0]}")
train_run("vanilla", SEEDS[1], references={f"vanilla-s{SEEDS[0]}": first} if first else None)
"""),
    md(TRAIN_REPORT_NOTE),
    code(TRAIN_REPORT),
]

# --- 03b: the public sets ----------------------------------------------------------------------

BENCH_INTRO = """
# 03b — Flex on the public Nepali sets: what fine-tuning cost outside our domain

Step 3 of the protocol (D105). Decode only. Base Flex and each run in `RUNS` are scored on five
public Nepali test sets with the decoder and the scorer gold uses, and each run is paired with
base, clip by clip, resampling the set's own unit (speakers; sentences for FLEURS; videos for
nepali_cs).

**Why.** Fine-tuning on our 51 h narrows the vocabulary: on 2026-09-27 the fine-tune was about a
point worse than base on these sets, on words our train labels never contain (findings.md). That
loss is what "catastrophic forgetting" means here, and it is measured for every model the protocol
produces, before 03d's blend tries to recover it.

**The sets** (`evalkit.BENCHMARKS`): FLEURS `ne_np` test, OpenSLR 54 test, Common Voice 22
`ne-NP` test, IndicVoices `nepali` valid (it needs a token with access) and `nepali-cs-asr` test.
Read from the hub's parquet files; nothing of them is trained on, and none of them ever chooses a
model.

**Base Flex is scored here on gold and val too**, as the run `base`: no other notebook does it,
and 03d pairs every blend against it.

**Check the loader.** Base was scored on these sets on 2026-09-27. The report prints today's base
beside those numbers; a gap of more than a few tenths means a set is being read differently
(another reference column, another resampler), and the comparison with findings.md is off.

**Three numbers per set.** Folded WER with S/D/I (fold.py, as gold), raw WER, and plain WER and
CER (NFC, punctuation stripped, Latin lowercased: the normalisation published Nepali results use).
A word split differently costs plain WER two words and CER one space, so read the two together.

**Outputs**: `OUT_REPO/<RUN_PREFIX>/<run>/benchmarks/<set>.jsonl` (per clip) and `<set>.json`,
uploaded as each set finishes, and `OUT_REPO/<RUN_PREFIX>/benchmarks.json`. A set already there is
skipped. Run it again with the ablation's winner in `RUNS` once 03c has one.
"""

BENCH_RUN = r'''
def benchmark(run):
    """One run on the public sets; base Flex on gold and val as well."""
    global OUT
    OUT = OUT_ROOT / run
    OUT.mkdir(parents=True, exist_ok=True)
    gold_too = run == "base" and uploaded(run) is None
    if not gold_too and done_sets(run) >= set(SETS):
        print(f"{run}: every set already in {OUT_REPO}, skipped")
        return
    free_model()
    load_model(FLEX_DIR if run == "base" else download_weights(run)).eval()
    try:
        if gold_too:
            evalkit.evaluate_run(OUT, run, splits=splits, decode=decode, score=score,
                                 card=card("Indic-Transcribe-Flex as released, no fine-tuning",
                                           name="Indic-Transcribe-Flex (base)"),
                                 meta={"recipe": "base", "seed": None})
            upload_run(OUT, run, "base Flex on gold and val", weights=False)
        score_sets(run)
    finally:
        free_model()


for run in RUNS:
    benchmark(run)
'''

BENCH_REPORT_NOTE = """
## Report

Every run on every set, each run minus base with its 95% interval, the public mean, and base
beside its 2026-09-27 score.
"""

BENCH_REPORT = r"""
table, means = public_table(RUNS)
print("\nbase today against 2026-09-27 (the loader's check):")
for name in SETS:
    if "base" in table.get(name, {}):
        then = evalkit.BENCHMARKS[name].base_wer
        print(f"  {name:<14} {table[name]['base']['wer']:6.2f} vs {then:6.2f}  ({table[name]['base']['wer'] - then:+.2f})")
summary = {"runs": RUNS, "limit": LIMIT, "sets": table, "public_mean": means}
(OUT_ROOT / "benchmarks.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
api.upload_file(path_or_fileobj=str(OUT_ROOT / "benchmarks.json"), path_in_repo=f"{RUN_PREFIX}/benchmarks.json",
                repo_id=OUT_REPO, commit_message=f"{RUN_PREFIX}: public sets, {', '.join(RUNS)}")
print("\nuploaded", f"https://huggingface.co/{OUT_REPO}/tree/main/{RUN_PREFIX}")
"""

bench_cells = [
    md(BENCH_INTRO + SMOKE_NOTE),
    md("## Config"),
    config(
        "03b_Flex_Benchmarks",
        r"""
RUNS = ["base", "vanilla-s1"]   # "base" is Flex as released; the others are runs with best/ in OUT_REPO
""",
    ),
    md("## Setup"),
    nbkit.setup("rapidfuzz duckdb pyarrow"),
    *nbkit.kits("ftkit", "evalkit", "sweep", "distill"),
    load(("val", "gold")),
    code(DECODE),
    code(SETS),
    md("## Decode and score each run\n\nOne set at a time, uploaded as it finishes."),
    code(BENCH_RUN),
    md(BENCH_REPORT_NOTE),
    code(BENCH_REPORT),
]

# --- 03c: the augmentation ablation -------------------------------------------------------------

AUG_INTRO = """
# 03c — Flex: the augmentation ablation

Step 3 of the protocol (D105). Each stage of `augment.py` is switched on alone, on top of the
vanilla recipe of 03a, and trained from the same base weights on the same export and batch budget.
Then the stages that earned their place are trained together.

**The rule, fixed before any result of this notebook.** A stage is kept only if its val WER beats
vanilla's better seed by more than the gap between vanilla's two seeds, and never by less than
0.15 points (`sweep.beats_baseline`, `sweep.MIN_NOISE`). The floor was added after 03a and before
any stage ran (D109): 03a's seeds differed by 0.02, and two runs cannot be trusted to measure a
gap that small. Val alone decides; gold and the public sets are reported, never used to choose.
The winner of the whole ablation (`sweep.choose_recipe`) is vanilla unless a kept recipe beats it
by that margin.

**The stages** (`ABLATION` in Config; edit it before the first run, not after a result):
- **speed**: 0.9x or 1.1x, tempo and pitch together.
- **reverb**: a synthetic room.
- **channel**: a microphone's frequency response.
- **noise**: MUSAN's noise and music at 5 to 25 dB below the speech (OpenSLR 17, downloaded here).
- **gain**: a level change.
- **codec**: an MP3, Opus, AAC or 8 kHz mu-law round trip.
- **crosstalk**: a whole verified train clip laid over the clip, with **both voices' words in the
  label**, in time order, as gold is written since D100. D96's sweep kept the clip's own label,
  which teaches a model to leave the second voice out; that did not move real crosstalk.

SpecAugment is not a stage: it is part of vanilla. The settings in `ABLATION` are first guesses at
sensible strengths, not tuned values; a stage that fails at one strength has not been shown useless
at every strength.

**What the findings lead one to expect.** Reverberation should do nothing on gold (591 of 706 gold
clips were dry), and noise split WER only through crosstalk. They are measured anyway, with the
clean bucket as the no-harm check.

**Weights.** A run's weights are uploaded only if it is kept, so the losers cost no storage.

**Outputs**: `OUT_REPO/<RUN_PREFIX>/aug-<stage>-s<seed>/` per run and
`OUT_REPO/<RUN_PREFIX>/ablation.json`, which names the winner that 03d blends.
"""

AUG_KIT = r'''
import io
import tarfile
import urllib.request

import soundfile as sf
from IPython.display import Audio, display

import augment
import xtalk

n_turns = ftkit.attach_speaker_turns(DATA, splits["train"])
print(f"speaker turns for {n_turns} of {len(splits['train'])} train clips")
NOISE = None


def fetch(ep, s, e):
    return store.audio[ep][round(s * ftkit.SR):round(e * ftkit.SR)]


def noise_bank():
    """MUSAN's noise and music (OpenSLR 17): the first NOISE_SECONDS of each file, in RAM. The
    archive is read as it downloads (its speech third is skipped), and the bank is kept on the VM's
    disk. A smoke run uses generated noise instead of 11 GB."""
    global NOISE
    if NOISE is not None:
        return NOISE
    if SMOKE:
        rng = np.random.default_rng(0)
        NOISE = augment.NoiseBank([(f"generated-{k}", "noise", (rng.standard_normal(30 * ftkit.SR) * 2000).astype(np.int16))
                                   for k in range(4)])
        return NOISE
    cache = FT / "musan_bank.npz"
    if not cache.exists():
        names, audio = [], []
        with ftkit.timed("MUSAN: downloading and reading noise and music"):
            with urllib.request.urlopen(MUSAN_URL) as response, tarfile.open(fileobj=response, mode="r|gz") as tf:
                for member in tf:
                    parts = member.name.split("/")
                    if member.isfile() and member.name.endswith(".wav") and len(parts) > 2 and parts[1] in ("noise", "music"):
                        x, sr = sf.read(io.BytesIO(tf.extractfile(member).read()), dtype="int16",
                                        frames=NOISE_SECONDS * ftkit.SR)
                        assert sr == ftkit.SR and x.ndim == 1, (member.name, sr, x.shape)
                        names.append(member.name)
                        audio.append(x)
        np.savez(cache, names=np.array(names), lengths=np.array([len(a) for a in audio]), audio=np.concatenate(audio))
    with np.load(cache) as bank:
        names, lengths, audio = bank["names"], bank["lengths"], bank["audio"]
    ends = np.cumsum(lengths)
    NOISE = augment.NoiseBank([(str(name), str(name).split("/")[1], audio[end - length:end])
                               for name, length, end in zip(names, lengths, ends)])
    print(f"noise bank: {len(NOISE.sources)} sources, {ends[-1] / ftkit.SR / 3600:.1f} h")
    return NOISE


def make_augmenter(stages):
    """The run's `augment.Augmenter`, and how much longer its slowest clip can be (1 / 0.9 for
    speed), which `make_batches` sizes every clip by so a slowed clip still fits the probe's budget."""
    cfg = augment.AugmentConfig.from_dict(stages)
    donors = None
    if cfg.crosstalk.p:
        pool = xtalk.ClipDonorPool if cfg.crosstalk.donor == "clip" else xtalk.DonorPool
        donors = pool(splits["train"])
    aug = augment.Augmenter(cfg, noise=noise_bank() if cfg.noise.p else None, donors=donors, fetch=fetch)
    return aug, (1 / min(cfg.speed.factors) if cfg.speed.p else 1.0)
'''

AUG_BASELINE = r"""
vanilla = [r for r in (uploaded(f"vanilla-s{seed}") for seed in (0, 1)) if r]
if not vanilla:
    raise RuntimeError(f"no vanilla run under {OUT_REPO}/{RUN_PREFIX}: run 03a_Flex_Train first")
best_vanilla = min(vanilla, key=lambda r: r["val_wer"])
vanilla_counts = run_counts(best_vanilla["run_name"])
for r in vanilla:
    print(f"{r['run_name']}: val {line(r['val'])} | gold {line(r['gold'])}")
if len(vanilla) == 2:
    gap = abs(vanilla[0]["val_wer"] - vanilla[1]["val_wer"])
    print(f"seed noise on val: {gap:.2f} points; a stage is kept if it beats "
          f"{best_vanilla['val_wer']:.2f} by more than {max(gap, sweep.MIN_NOISE):.2f}")
else:
    print(f"one vanilla seed only: the noise is unmeasured, so the bar is the floor, {sweep.MIN_NOISE:.2f}")


def ablate(name, stages=None):
    return train_run(f"aug-{name}", SEED, stages=stages or ABLATION[name],
                     references={best_vanilla["run_name"]: vanilla_counts}, baseline=vanilla)
"""

AUG_CHECK_NOTE = """
## Listen before training

Each stage applied to two train clips with its chance set to 1, saved clean and augmented under
`OUT_ROOT/aug_examples/`, and the first one played. **Listen to them before starting a run.** A
stage that cannot be heard, or that ruins the speech, has the wrong strength. For crosstalk the new
label is printed: it should read as both voices in the order they speak.
"""

AUG_CHECK = r"""
rng = np.random.default_rng(0)
examples = OUT_ROOT / "aug_examples"
examples.mkdir(exist_ok=True)
picks = [r for r in splits["train"] if 6 <= ftkit.duration(r) <= 15 and r["overlap_spans"] == []][:40]
for name, stages in ABLATION.items():
    always, _ = make_augmenter({k: {**v, "p": 1.0} for k, v in stages.items()})
    shown = 0
    for r in picks:
        clean = store.clip(r)
        audio, info = always(r, clean, rng)
        if not info["stages"]:
            continue  # nothing was applied (no donor fits this clip)
        sf.write(examples / f"{name}_{shown}_clean.flac", clean, ftkit.SR)
        sf.write(examples / f"{name}_{shown}.flac", audio, ftkit.SR)
        if shown == 0:
            print(f"\n{name}: {r['segment_id']}: {r['text']}")
            for stage in info["stages"]:
                print("  ", {k: v for k, v in stage.items() if k != "windows"})
            if info.get("text"):
                print("   label:", info["text"])
            display(Audio(audio, rate=ftkit.SR))
        shown += 1
        if shown == 2:
            break
    if not shown:
        print(f"\n{name}: no example could be made; check the stage before training it")
"""

AUG_RUNS_NOTE = """
## One run per stage

Each cell trains one stage on top of vanilla, scores it, says whether it is kept, and uploads it.
A run already in `OUT_REPO` is skipped. The cells are independent: run the stages worth their
A100 time, in any order.
"""

AUG_COMBINED_NOTE = """
## The kept stages together

With two or more kept stages, one more run trains them all at once. With one or none there is
nothing to combine.
"""

AUG_COMBINED = r"""
done = {name: uploaded(f"aug-{name}-s{SEED}") for name in ABLATION}
kept = [name for name, row in done.items() if row and row.get("kept")]
print("kept:", kept or "none", "| not run:", [name for name, row in done.items() if row is None] or "none")
if len(kept) >= 2:
    merged = {}
    for name in kept:
        merged.update(ABLATION[name])
    ablate("combined", merged)
else:
    print("fewer than two kept stages: no combined run")
"""

AUG_REPORT_NOTE = """
## Report, and the winner

Every run on val and gold with S/D/I, gold per crosstalk bucket, each run minus vanilla on gold,
and the winner under the rule. `ablation.json` names it: 03d blends that run, and 03b scores it on
the public sets once its name is added to `RUNS` there.
"""

AUG_REPORT = r"""
rows = list(vanilla)
for name in [*ABLATION, "combined"]:
    if (row := uploaded(f"aug-{name}-s{SEED}")) is not None:
        rows.append(row)
winner, why = sweep.choose_recipe(rows)
print(f"{'run':<22}{'epoch':>6}  {'val':<34}{'gold':<34}")
for r in rows:
    tag = " <- winner" if r is winner else ("  kept" if r.get("kept") else "")
    print(f"{r['run_name']:<22}{str(r.get('best_epoch') or '-'):>6}  {line(r['val']):<34}{line(r['gold']):<34}{tag}")
print("\ngold by crosstalk bucket:")
for bucket in evalkit.BUCKETS:
    for r in rows:
        m = (r["gold_by_class"].get("overlap") or {}).get(bucket)
        if m:
            print(f"  {bucket:>6} {r['run_name']:<22} {m['clips']:4d} clips  {line(m)}")
print("\npaired on gold, WER points [95% CI, episodes resampled]:")
for r in rows:
    for name, vs in r["gold"]["vs"].items():
        cells = "  ".join(f"{k} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]" for k, v in vs.items()
                          if k == "all" or k.startswith("overlap="))
        print(f"  {r['run_name']} minus {name}: {cells}")
print("\nwinner:", winner["run_name"], "|", why)
summary = {"rule": "val WER; a recipe has to beat vanilla's better seed by more than vanilla's seed gap (D105)",
           "winner": winner["run_name"], "why": why,
           "kept": [r["run_name"] for r in rows if r.get("kept")], "ablation": ABLATION, "rows": rows}
(OUT_ROOT / "ablation.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
api.upload_file(path_or_fileobj=str(OUT_ROOT / "ablation.json"), path_in_repo=f"{RUN_PREFIX}/ablation.json",
                repo_id=OUT_REPO, commit_message=f"{RUN_PREFIX}: ablation, winner {winner['run_name']}")
"""

#: One stage per run. The strengths are first guesses, to be reviewed before the first run.
ABLATION = r"""
SEED = 0
# One stage per run, as augment.AugmentConfig.from_dict reads it. Fixed before the first run: do
# not edit it after reading a result. `p` is the chance a train clip gets the stage each epoch.
ABLATION = {
    "speed": {"speed": {"p": 0.5}},
    "reverb": {"reverb": {"p": 0.3}},
    "channel": {"channel": {"p": 0.3}},
    "noise": {"noise": {"p": 0.5}},
    "gain": {"gain": {"p": 0.5}},
    "codec": {"codec": {"p": 0.3}},
    # a whole verified train clip of 2-6 s over a clean clip, and both voices' words in the label
    "crosstalk": {"crosstalk": {"p": 0.3, "donor": "clip", "label": "everything", "seconds": [2.0, 6.0],
                                "overshoot": None}},
}
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"   # 11 GB; noise and music are kept
NOISE_SECONDS = 60        # of each MUSAN file
"""

aug_cells = [
    md(AUG_INTRO + SMOKE_NOTE),
    md("## Config"),
    config("03c_Flex_Augment", TRAIN_CONFIG, ABLATION),
    md("## Setup"),
    nbkit.setup("rapidfuzz duckdb"),
    *nbkit.kits("ftkit", "evalkit", "sweep", "distill", "xtalk", "augment"),
    load(("train", "val", "gold"), analytics=True),
    code(DECODE),
    md("## Targets, the batch-size probe, loss and batches"),
    code(TARGETS),
    code(TRAIN),
    md("## The augmenter, and vanilla to beat"),
    code(AUG_KIT),
    code(AUG_BASELINE),
    md(AUG_CHECK_NOTE),
    code(AUG_CHECK),
    md(AUG_RUNS_NOTE),
    *[
        cell
        for name in ("speed", "reverb", "channel", "noise", "gain", "codec", "crosstalk")
        for cell in (md(f"### {name}"), code(f'ablate("{name}")'))
    ],
    md(AUG_COMBINED_NOTE),
    code(AUG_COMBINED),
    md(AUG_REPORT_NOTE),
    code(AUG_REPORT),
]

# --- 03d: weight blending ------------------------------------------------------------------------

BLEND_INTRO = """
# 03d — Flex: blend the fine-tune's weights with base (WiSE-FT)

Step 3 of the protocol (D105). Decode only. Every weight is set to
`(1 - alpha) * base + alpha * fine-tuned`, the same alpha for every tensor, and the blend is scored
like any other model: gold and val, then the public sets.

**Why.** Fine-tuning erases vocabulary our train labels never contain, which costs about a point on
the public sets (03b). Averaging back towards base recovers it: on 2026-09-27, alpha = 0.5 kept our
domain (val +0.05, gold -0.32 against the fine-tune) and brought every public set back to base
(findings.md, *Weight blending recovers the loss*).

**The rule, fixed on 2026-09-27 before any blend was scored** (`sweep.choose_blend`): choose on
val, taking the blend closest to base whose val WER is within 0.3 points of the fine-tuned
model's. Gold and the public sets are reported, never used to choose. If no blend qualifies, the
fine-tuned model is kept.

**What is blended.** `SOURCE` names the fine-tuned run; left at `None` it is the winner in 03c's
`ablation.json`. To skip the ablation, set `SOURCE = "vanilla-s1"`.

**No weights are written here.** A blend is rebuilt in seconds from the two models; 03e rebuilds
the chosen one and writes it once.

**Needs** 03b's `base` run (base Flex on gold, val and the public sets) for the pairing, and the
source run's weights in `OUT_REPO`.

**Outputs**: `OUT_REPO/<RUN_PREFIX>/blend-<alpha>/` per blend (scores and transcripts) and
`OUT_REPO/<RUN_PREFIX>/blend.json`, which names the choice 03e freezes.
"""

BLEND_SOURCE = r"""
source = SOURCE or (fetch_json(f"{RUN_PREFIX}/ablation.json") or {}).get("winner")
if not source:
    raise RuntimeError(f"no ablation.json under {OUT_REPO}/{RUN_PREFIX}: run 03c, or set SOURCE in Config")
tuned_row = uploaded(source)
if tuned_row is None:
    raise RuntimeError(f"{source} has no result in {OUT_REPO}/{RUN_PREFIX}")
tuned = load_file(str(download_weights(source) / "model.safetensors"))  # bf16, on the CPU
base_row, base_counts, tuned_counts = uploaded("base"), run_counts("base"), run_counts(source)
if base_row is None:
    print("03b has not scored base on gold and val: the blends will not be paired against it")
print(f"blending base with {source}: val {line(tuned_row['val'])} | gold {line(tuned_row['gold'])}")

# The recipe's check: at alpha = 1 the blend is the fine-tune, tensor for tensor.
n = blend(1.0, tuned)
state = model.state_dict()
wrong = [k for k, t in tuned.items() if k in state and not torch.equal(state[k].cpu(), t.to(state[k].dtype))]
assert not wrong, f"alpha = 1 does not reproduce the fine-tune: {wrong[:5]}"
print(f"{n} float tensors blended; alpha = 1 reproduces {source} exactly")
free_model()


def blend_run(alpha):
    # Score one blend on gold, val and the public sets, and upload it (no weights).
    global OUT
    run = f"blend-{round(alpha * 100):03d}"
    row = uploaded(run)
    if row is not None and done_sets(run) >= set(SETS):
        print(f"{run}: already in {OUT_REPO}, skipped")
        return
    OUT = OUT_ROOT / run
    OUT.mkdir(parents=True, exist_ok=True)
    blend(alpha, tuned)
    try:
        if row is None:
            references = {k: v for k, v in (("base", base_counts), (source, tuned_counts)) if v}
            evalkit.evaluate_run(OUT, run, splits=splits, decode=decode, score=score,
                                 card=card(f"{1 - alpha:g} x base + {alpha:g} x {source} (WiSE-FT)",
                                           name=f"{MODEL_NAME} blend {alpha:g}"),
                                 meta={"recipe": "blend", "alpha": alpha, "source": source,
                                       "seed": tuned_row.get("seed")},
                                 references=references)
            upload_run(OUT, run, f"blend of base and {source}", weights=False)
        score_sets(run)
    finally:
        free_model()
"""

BLEND_TUNED_NOTE = """
## The fine-tune on the public sets

03b scores it only when it is named in its `RUNS`. If it was not, it is done here, so the report
has the alpha = 1 row.
"""

BLEND_TUNED = r"""
if done_sets(source) >= set(SETS):
    print(f"{source}: every public set already in {OUT_REPO}")
else:
    load_model(download_weights(source)).eval()
    try:
        score_sets(source)
    finally:
        free_model()
"""

BLEND_REPORT_NOTE = """
## Report, and the choice

Base, each blend and the fine-tune on val, gold and the public sets, then the blend the rule
keeps. `blend.json` names it for 03e.
"""

BLEND_REPORT = r"""
runs = ["base", *[f"blend-{round(a * 100):03d}" for a in ALPHAS], source]
table, means = public_table(runs)
rows = [{**tuned_row, "alpha": 1.0}]
for alpha in ALPHAS:
    if (row := uploaded(f"blend-{round(alpha * 100):03d}")) is not None:
        rows.append(row)
winner, why = sweep.choose_blend(rows, tolerance=TOLERANCE)
print(f"\n{'model':<16}{'alpha':>6}  {'val':<34}{'gold':<34}{'public mean':>12}")
for r in ([{**base_row, "alpha": 0.0}] if base_row else []) + sorted(rows, key=lambda r: r["alpha"]):
    mean = means.get(r["run_name"])
    tag = " <- chosen" if r is winner else ""
    print(f"{r['run_name']:<16}{r['alpha']:>6g}  {line(r['val']):<34}{line(r['gold']):<34}"
          f"{'-' if mean is None else format(mean, '.2f'):>12}{tag}")
for r in rows:
    for split in ("val", "gold"):
        for name, vs in r[split]["vs"].items():
            d, lo, hi = vs["all"]
            print(f"{r['run_name']} {split} minus {name}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
print("\nchosen:", winner["run_name"], "|", why)
summary = {"rule": f"val WER; the blend closest to base within {TOLERANCE:g} points of the fine-tune (2026-09-27)",
           "tolerance": TOLERANCE, "source": source, "winner": winner["run_name"], "alpha": winner["alpha"],
           "why": why, "rows": rows, "public": table, "public_mean": means}
(OUT_ROOT / "blend.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
api.upload_file(path_or_fileobj=str(OUT_ROOT / "blend.json"), path_in_repo=f"{RUN_PREFIX}/blend.json",
                repo_id=OUT_REPO, commit_message=f"{RUN_PREFIX}: blend, chosen {winner['run_name']}")
"""

blend_cells = [
    md(BLEND_INTRO + SMOKE_NOTE),
    md("## Config"),
    config(
        "03d_Flex_Blend",
        r"""
SOURCE = None               # the fine-tuned run to blend; None reads the winner from 03c's ablation.json
ALPHAS = (0.25, 0.5, 0.75)  # the grid, fixed in advance
TOLERANCE = 0.3             # points of val WER a blend may cost against the fine-tune
""",
    ),
    md("## Setup"),
    nbkit.setup("rapidfuzz duckdb pyarrow"),
    *nbkit.kits("ftkit", "evalkit", "sweep", "distill"),
    load(("val", "gold")),
    code(DECODE),
    code(SETS),
    md("## The two models, and the blend"),
    code(BLEND),
    code(BLEND_SOURCE),
    md(
        "## One blend per cell\n\nGold and val, then the public sets, each uploaded as it finishes."
    ),
    *[code(f"blend_run(ALPHAS[{i}])") for i in range(3)],
    md(BLEND_TUNED_NOTE),
    code(BLEND_TUNED),
    md(BLEND_REPORT_NOTE),
    code(BLEND_REPORT),
]

# --- 03e: freeze the teacher and ship ----------------------------------------------------------

SHIP_INTRO = """
# 03e — Flex: freeze the teacher, export for the CPU

Step 3 of the protocol ends here (D105): one named model leaves it. This notebook writes the
chosen model's weights, exports them for the CPU, and records the choice in `teacher.json` at the
root of `OUT_REPO`. 05 (the teacher's pseudo-labels), every student notebook and the report read
that file, so none of them names a model of its own.

**Which model.** The one 03d's `blend.json` chose on val: the blend closest to base within 0.3
points of the fine-tune, or the fine-tune itself. `TEACHER` in Config overrides it with a run's
name.

**Scored as shipped.** `best/` holds bf16 weights, while a run is scored from the fp32 weights in
memory and a blend is never written before this notebook. Gold and val are decoded here once more
with the weights as they were written, so the teacher's numbers and the per-clip counts every
student is paired against belong to the exact file that 05 and the playground load.

**Frozen means frozen.** Students are trained on this model's labels and scored against it.
Replacing it after that invalidates them, so an existing `teacher.json` that names another model is
not overwritten unless `REPLACE_TEACHER` is set.

**Outputs**, in `OUT_REPO/<RUN_PREFIX>/<run>/`: `best/` (bf16), `cpu/` (when int8 was accepted),
`int8/`, `cpu_bench.json` and the card with its `cpu` block; and `OUT_REPO/teacher.json`.
"""

SHIP_CHOICE = r"""
choice = fetch_json(f"{RUN_PREFIX}/blend.json")
if TEACHER:
    run, named = TEACHER, uploaded(TEACHER) or {}
    source, alpha, why = named.get("source"), named.get("alpha", 1.0), "named in Config"
elif choice:
    run, source, alpha, why = choice["winner"], choice["source"], choice["alpha"], choice["why"]
else:
    raise RuntimeError(f"no blend.json under {OUT_REPO}/{RUN_PREFIX}: run 03d, or set TEACHER in Config")
row = uploaded(run)
if row is None:
    raise RuntimeError(f"{run} has no result in {OUT_REPO}/{RUN_PREFIX}")
TEACHER_FILE = "teacher-smoke.json" if SMOKE else "teacher.json"
current = fetch_json(TEACHER_FILE)
if current and current["run"] != f"{RUN_PREFIX}/{run}" and not REPLACE_TEACHER:
    raise RuntimeError(f"{TEACHER_FILE} already names {current['run']}. Students trained on its labels would no "
                       "longer match the teacher; set REPLACE_TEACHER = True to freeze another model.")
print("teacher:", run, "|", why)
print(f"val {line(row['val'])} | gold {line(row['gold'])}")

OUT, HARNESS = OUT_ROOT / run, OUT_ROOT / run / "harness"
HARNESS.mkdir(parents=True, exist_ok=True)
for name in ("gold.jsonl", "val.jsonl", "model_card.json"):  # what 03a or 03d wrote for this run
    shutil.copy(hf_hub_download(OUT_REPO, f"{RUN_PREFIX}/{run}/harness/{name}", token=TOKEN), HARNESS / name)
free_model()
if alpha < 1:
    tuned = load_file(str(download_weights(source) / "model.safetensors"))
    n = blend(alpha, tuned)
    print(f"rebuilt the blend: {n} tensors at alpha = {alpha:g}")
    save_best(model)
    del tuned
    free_model()
else:
    shutil.copytree(download_weights(run), OUT / "best", dirs_exist_ok=True)
load_model(OUT / "best").eval()

# Score the weights as shipped. The run was scored from fp32 weights in memory (or, for a blend,
# never written at all); `best/` is bf16. The teacher's numbers, and the per-clip counts every
# student is paired against, are those of the exact weights 05 and the playground load.
before = row
card_on_hub = json.loads((HARNESS / "model_card.json").read_text())
references = {k: v for k, v in (("base", run_counts("base")),
                                (source, run_counts(source) if source and source != run else None)) if v}
row = evalkit.evaluate_run(OUT, run, splits=splits, decode=decode, score=score, card=card_on_hub,
                           meta={k: before[k] for k in ("recipe", "seed", "alpha", "source", "stages",
                                                        "best_epoch", "epochs", "lr") if k in before},
                           references=references)
print(f"as shipped (bf16): val {row['val_wer']:.2f} (was {before['val_wer']:.2f}), "
      f"gold {row['gold_wer']:.2f} (was {before['gold_wer']:.2f})")


def read_texts(name):
    hyps = evalkit.read_hyps(HARNESS / f"{name}.jsonl")
    return [hyps[r["segment_id"]] for r in splits[name]]


texts, val_texts = read_texts("gold"), read_texts("val")
"""

CPU_NOTE = """
## CPU export: quantize, benchmark, export

The target is a CPU at a desk, not this GPU: the harness's mic playground (D85).
**What was measured locally (Ryzen 7 7700X, 8 threads, 2026-09-15)**, base Flex on 10 gold clips:
- fp32 runs at RTF 0.41 and bf16 at RTF 0.18, with the same WER (21.8%). bf16 is realtime and
  free, and `best/` already stores bf16, so it is the CPU model if nothing else passes.
- PyTorch's **dynamic int8** (activations quantized too) broke the model: WER 97-113%, loops, and
  English written in Devanagari. It is not tried here.
- **Weight-only int8** (`cpukit.py`: int8 weights with one scale per output channel, bf16
  activations) decodes 1.5x faster than bf16 on a full-size Flex (16.5 vs 24.5 ms/token, random
  weights). Decoding is memory-bound, so halving the weight bytes is where CPU speed comes from.

**What this section measures** is the part that was not measured locally: what int8 costs in
accuracy on *this* model.
1. **Accuracy, on the GPU.** The model is quantized in place (this is its last use) and val and
   gold are decoded with the same decoder as above. The int8 arithmetic is dequantized on the GPU,
   so the numbers carry over to the CPU kernel.
2. **Rule, fixed before any run:** export int8 if its val WER is within **+0.3** of bf16 (the
   run-to-run noise) and it loops no more often. Otherwise the CPU model is `best/` in bf16.
3. **Speed, on this VM's CPU.** The same `CPU_TIMING_CLIPS` val clips are timed for each variant.
   Colab's CPU is not your machine (it may lack bf16 instructions entirely), so only the ratio
   between variants means anything. Measure the absolute speed where the model will run.
4. **Export.** An accepted int8 model goes to `OUT/cpu/` (~1.3 GB against 2.5 GB, with its code
   and `cpukit.py`; load it with `cpukit.load_int8`). `cpu_bench.json` holds every number, and the
   harness model card gets a `cpu` block.

ONNX is not needed: plain PyTorch bf16 is already realtime on the target CPU.
"""

CPU_ACCURACY = r"""
import cpukit

val_refs, gold_refs = [r["text"] for r in splits["val"]], [r["text"] for r in splits["gold"]]
bf16_scores = {"val": score(val_refs, val_texts), "gold": score(gold_refs, texts)}
cpukit.quantize_(model, dtype=torch.float32)  # in place: the GPU copy is not needed after this
int8_texts = {}
for name, rows in (("val", splits["val"]), ("gold", splits["gold"])):
    int8_texts[name], _, _ = decode(rows)
    evalkit.write_hyps(OUT / "int8" / f"{name}.jsonl", rows, int8_texts[name], [0.0] * len(rows))
int8_scores = {"val": score(val_refs, int8_texts["val"]), "gold": score(gold_refs, int8_texts["gold"])}
same = sum(a == b for a, b in zip(val_texts, int8_texts["val"]))
accept = (int8_scores["val"]["wer"] <= bf16_scores["val"]["wer"] + MAX_WER_COST
          and int8_scores["val"]["loops"] <= bf16_scores["val"]["loops"])
for name in ("val", "gold"):
    print(f"{name}: bf16 {bf16_scores[name]['wer']:.2f}% ({ftkit.sid(bf16_scores[name])}, "
          f"{bf16_scores[name]['loops']} loops) | int8 {int8_scores[name]['wer']:.2f}% "
          f"({ftkit.sid(int8_scores[name])}, {int8_scores[name]['loops']} loops)")
print(f"int8 text identical to bf16 on {same}/{len(val_texts)} val clips ->",
      "EXPORT int8" if accept else "int8 rejected; the CPU model is best/ in bf16")
"""

CPU_SPEED = r"""
import subprocess

step = max(1, len(splits["val"]) // CPU_TIMING_CLIPS)
timing_rows = sorted(splits["val"], key=ftkit.duration)[::step][:CPU_TIMING_CLIPS]
clips = [(r["segment_id"], torch.as_tensor(store.clip_f32(r)).float().cpu()) for r in timing_rows]
threads = os.cpu_count()
torch.set_num_threads(threads)
cpu_name = subprocess.run("lscpu | sed -n 's/^Model name: *//p'", shell=True, capture_output=True,
                          text=True).stdout.strip()
cpu_flags = open("/proc/cpuinfo").read()
host = {"cpu": cpu_name, "threads": threads, "avx512_bf16": "avx512_bf16" in cpu_flags,
        "amx_bf16": "amx_bf16" in cpu_flags}
print(host)

free_model()
timing = {}
cpu_asr = IndicTranscribe.from_pretrained(str(OUT / "best"), device="cpu", dtype=torch.bfloat16)
timing["bf16"] = cpukit.time_clips(cpu_asr, clips, lang=LANG, mode=MODE)
if accept:
    names = cpukit.quantize_(cpu_asr.model)
    cpukit.save_int8(cpu_asr.model, names, OUT / "best", OUT / "cpu")
    del cpu_asr
    cpu_asr = cpukit.load_int8(OUT / "cpu")  # time what was written, not what is in memory
    timing["int8"] = cpukit.time_clips(cpu_asr, clips, lang=LANG, mode=MODE)
del cpu_asr
for variant, t in timing.items():
    print(f"{variant}: RTF {t['rtf']:.3f}, {t['ms_per_token']:.1f} ms/token over {t['audio_s']} s of audio")

bench = {
    "variant": "int8-weight-only" if accept else "bf16",
    "export": "cpu/" if accept else "best/",
    "rule": f"int8 if val WER <= bf16 + {MAX_WER_COST} and no more loops",
    "scores": {"bf16": bf16_scores, "int8": int8_scores},
    "int8_identical_val_texts": same,
    "timing_host": host,
    "timing": {v: {k: x for k, x in t.items() if k != "texts"} for v, t in timing.items()},
}
(OUT / "cpu_bench.json").write_text(json.dumps(bench, indent=1))
card_now = json.loads((HARNESS / "model_card.json").read_text())
card_now["cpu"] = {
    "variant": bench["variant"],
    "export": bench["export"],
    "val_wer_bf16": bf16_scores["val"]["wer"],
    "val_wer_int8": int8_scores["val"]["wer"],
    "timing_host": host["cpu"],
    "rtf": {v: t["rtf"] for v, t in timing.items()},
}
card_now["teacher"] = True
(HARNESS / "model_card.json").write_text(json.dumps(card_now, indent=1, ensure_ascii=False))
print("wrote", OUT / "cpu_bench.json", "and the card's cpu block")
"""

FREEZE_NOTE = """
## Upload, and freeze

The run's whole folder goes to `OUT_REPO/<RUN_PREFIX>/<run>/`, then `teacher.json` to the root of
`OUT_REPO`. The VM can be deleted once this prints a commit.

**The token must be able to write.** The cell reads the `HF_TOKEN` secret again, because Setup
cached whatever token was there when it ran, so a secret swapped mid-session is picked up.

Then, in the harness checkout, the teacher goes on the Models page and into the mic playground:
```bash
hf download Sagyam/nepanglish-asr-flex-ft --include "<RUN_PREFIX>/<run>/harness/*" "<RUN_PREFIX>/<run>/cpu/*" --local-dir exports/<RUN_PREFIX>
mkdir -p data/models/asr/<run> && cp -r exports/<RUN_PREFIX>/<RUN_PREFIX>/<run>/harness/. data/models/asr/<run>/
cp -r exports/<RUN_PREFIX>/<RUN_PREFIX>/<run>/cpu data/models/asr/<run>/   # mic playground (D85)
```
Press **Rescan** on the Models page. The playground sidecar starts with `docker compose up -d`.
Any other run's `harness/` folder imports the same way. Its `errors/` (every aligned word pair of
gold, val and each public set, D110) comes with it: the Models page's Errors
section reads it after the rescan.

**Weights no longer needed.** Once the teacher is frozen, the `best/` folders of the runs that are
neither the teacher nor its source can be deleted from `OUT_REPO`; after a smoke run, so can the
whole `<RUN_PREFIX>-smoke/` folder.
"""

FREEZE = r"""
import datetime as dt

if IN_COLAB:
    os.environ["HF_TOKEN"] = TOKEN = userdata.get("HF_TOKEN")
    (Path.home() / ".cache" / "huggingface" / "token").write_text(TOKEN)
api = HfApi(token=TOKEN)
commit = api.upload_folder(repo_id=OUT_REPO, folder_path=str(OUT), path_in_repo=f"{RUN_PREFIX}/{run}",
                           commit_message=f"{run}: the teacher's weights and CPU export")
teacher = {
    "repo": OUT_REPO,
    "run": f"{RUN_PREFIX}/{run}",
    "alpha": alpha,
    "source": f"{RUN_PREFIX}/{source}" if source else None,
    "why": why,
    "val_wer": row["val_wer"],
    "gold_wer": row["gold_wer"],
    "dataset_export": export["exported_at"],
    "cpu": bench["variant"],
    "frozen_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
}
(OUT_ROOT / TEACHER_FILE).write_text(json.dumps(teacher, indent=1, ensure_ascii=False))
api.upload_file(path_or_fileobj=str(OUT_ROOT / TEACHER_FILE), path_in_repo=TEACHER_FILE, repo_id=OUT_REPO,
                commit_message=f"freeze the teacher: {RUN_PREFIX}/{run}")
size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
print(f"{size / 2**30:.2f} GiB -> https://huggingface.co/{OUT_REPO}/tree/main/{RUN_PREFIX}/{run} @ {commit.oid[:7]}")
print(json.dumps(teacher, indent=1, ensure_ascii=False))
"""

ship_cells = [
    md(SHIP_INTRO + SMOKE_NOTE),
    md("## Config"),
    config(
        "03e_Flex_Ship",
        r"""
TEACHER = None            # a run's name; None takes the choice in 03d's blend.json
REPLACE_TEACHER = False   # True to overwrite a teacher.json that names another model
CPU_TIMING_CLIPS = 8      # val clips timed on this VM's CPU; accuracy uses all of val and gold on the GPU
MAX_WER_COST = 0.3        # points of val WER int8 may cost against bf16 (the run-to-run noise)
""",
    ),
    md("## Setup"),
    nbkit.setup("rapidfuzz duckdb"),
    *nbkit.kits("ftkit", "evalkit", "sweep", "distill", "cpukit"),
    load(("val", "gold")),
    code(DECODE),
    md("## The chosen model, and its weights"),
    code(BLEND),
    code(SHIP_CHOICE),
    md(CPU_NOTE),
    code(CPU_ACCURACY),
    code(CPU_SPEED),
    md(FREEZE_NOTE),
    code(FREEZE),
]

NOTEBOOKS = {
    "03a_Flex_Train.ipynb": train_cells,
    "03b_Flex_Benchmarks.ipynb": bench_cells,
    "03c_Flex_Augment.ipynb": aug_cells,
    "03d_Flex_Blend.ipynb": blend_cells,
    "03e_Flex_Ship.ipynb": ship_cells,
}

if __name__ == "__main__":
    nbkit.write(NOTEBOOKS, OUT_DIR)
