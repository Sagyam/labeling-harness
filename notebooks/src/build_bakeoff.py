"""Build notebooks/09_SpeechLLM_Bakeoff.ipynb: `python notebooks/src/build_bakeoff.py` (D121).

Which open-weight LLMs that hear audio are worth training as students (roadmap B)? Each candidate
transcribes val and gold zero-shot and has its training step timed on the same GPU. The rules
(the gate, the failure counts, the projection, the verdict) are in bakeoffkit.py, tested in
backend/tests/test_bakeoff_kit.py; the scoring is evalkit's, like every other number (D105).
Omnilingual LLM-ASR runs as a script in the Python 3.12 environment fairseq2 needs, as 06e does."""

import sys
from pathlib import Path

import nbkit
from build_students import OMNI_ENV
from nbkit import code, md

HERE = Path(__file__).parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent

INTRO = """
# 09 — Speech-LLM bake-off: which audio LLMs are worth training as students?

Roadmap B trains students on the teacher's labels. All six so far are pure ASR models. This
notebook shortlists the LLMs that hear audio for the same recipe. It trains nothing, and it
searches breadth-first: every candidate is swept zero-shot on a val sample before any GPU-hours
go to one, and only the shortlist has its training step timed and its full val and gold decoded
(D121).

**Candidates** (`SPECS`, below the kits; `CANDIDATES` in Config picks and orders them):
- **Gemma 4 E2B and E4B** (~300M audio encoder, PLE) and **Gemma 4 12B** (no encoder: the waveform
  is projected straight into the LLM). No ASR language list is published for any of them.
- **Omnilingual LLM-ASR 1B and 3B** (Meta; wav2vec 2.0 encoder + LLM decoder; Nepali among its
  1,600+ languages, `nep_Deva`). With 06e's Omnilingual CTC 1B it makes the cleanest family
  comparison: same encoder family and pretraining languages, CTC head against LLM decoder. The 7B
  is swept as a zero-shot data point only (full fine-tuning does not fit in 96 GB).
- **Qwen3-ASR-1.7B** and **Voxtral Mini 3B**: cheap extra data points, neither trained on Nepali.

**Three rounds, breadth-first** (each its own cell, below):
1. **Sweep, every candidate.** Each of its prompts (or language codes) transcribes the same
   `SWEEP_CLIPS` val clips, greedy, bf16, scored with fold.py. The kept prompt is the one with the
   lowest folded WER among those that pass the gate, or the one nearest the gate when none passes
   (`bakeoffkit.pick_prompt`). Gold takes no part. It ends with a table to read before anything
   else is spent.
2. **Training step, the shortlist.** `SHORTLIST` in Config, or every candidate whose kept prompt
   passed the gate. Timed by `ftkit.speed_check` (06's `speed_check.json`): a probe finds the
   largest micro-batch, then real train clips run forward and backward. Full fine-tuning where it
   fits, LoRA for Gemma 4 12B. The verdict (`bakeoffkit.verdict`) reads the sweep's gate and the
   projected cost of both student stages against `BUDGET_H`.
3. **Full val and gold, the shortlist**, zero-shot with the kept prompt, through
   `evalkit.evaluate_run`: folded and raw WER split into S/D/I, CER, per clip class, error rows.
   For the record and the paper; the verdict does not wait for it.

Each answer is capped at 1.5× the densest training label's token rate in its own tokenizer, so a
runaway cannot fill the context, and is never retried: a loop is a finding here.

**The gate.** Among the clips whose reference is mostly Devanagari, the share whose answer is
mostly Devanagari too must be at least 90%. It is only a gate: zero-shot WER does not predict
trainability (Whisper-turbo read 123% zero-shot and 15.02 after training on teacher labels).
Hindi passes it, being Devanagari too; WER shows that. Also recorded: empty answers, runaways,
answers in another script, and the Latin share of the answers against the references'
(Omnilingual CTC wrote English in Devanagari).

**The projection.** Stage 1 trains on the human train hours for `HUMAN_EPOCHS`, stage 2 on a
`DISTILL_H` epoch (human plus pseudo-labels, as 06's mixture draws it) for `DISTILL_EPOCHS`, each
epoch with a val pass at the sweep's decode speed. Every epoch counted, so early stopping can only
make it cheaper. Whisper took about 5.6 A100-hours for both stages (findings, 2026-10-04).

**The bars for whoever continues** (D121, fixed now, read in the student notebooks, not here):
stage 1's first epoch at most 15.86 on val (Whisper's); the end of stage 1 at most 10.69 (Whisper)
or 13.65 (IndicConformer, the looser bar, for small models); worth the decode cost only if stage 2
beats Whisper's 15.02 on gold with an interval that excludes zero.

**Same GPU for everything.** Run it on the G4 (RTX PRO 6000, 96 GB): its speeds are not comparable
with the A100 numbers of 06.

**Small blast radius.** Every step uploads to `OUT_REPO/<RUN_PREFIX>/<candidate>/` as it finishes
(`prompts.json` after each prompt, `speed_check.json` and `bakeoff.json` in round 2, the zero-shot
run in round 3), and a rerun skips whatever is there. A candidate that fails is reported and the
next one runs.
"""

CONFIG = r"""
NOTEBOOK = "09_SpeechLLM_Bakeoff"
RUN_PREFIX = "speech-llm-bakeoff-2026-10-08"  # OUT_REPO/<RUN_PREFIX>/<candidate>/
DATASET_EXPORT = "2026-09-30"   # the export every model trains and is scored on; another is refused
DATASET_REPO = "Sagyam/nepanglish-asr"
OUT_REPO = "Sagyam/nepanglish-asr-students"  # private HF model repo
SMOKE = False                   # True: a few clips of each set, under <RUN_PREFIX>-smoke
# Round 1 sweeps all of these, in this order; leave one out to skip it. "omni-llm-7b" is zero-shot only.
CANDIDATES = ("gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b", "omni-llm-1b", "omni-llm-3b", "omni-llm-7b",
              "qwen3-asr-1.7b", "voxtral-mini-3b")
SWEEP_CLIPS = 300               # val clips each prompt is tried on in round 1, spread over val
SHORTLIST = None                # rounds 2 and 3: None for every candidate whose kept prompt passed the gate
SPEED_CLIPS = 600               # train clips the training step is timed on, spread over train
BUDGET_H = 24.0                 # GPU-hours both student stages may take, projected (D121)
HUMAN_EPOCHS, DISTILL_EPOCHS = 8, 6  # the epoch caps of 06's stages 1 and 2
DISTILL_H = 131.0               # one stage-2 epoch: 51.3 h human + 79.5 h pseudo-labels (06)
GEN_BUDGET_S, GEN_ITEMS = 600.0, 32  # audio seconds and clips per zero-shot batch
PAD_TO_S = 1.0
PROBE_FRACTION = 0.9
OMNI_VERSION = "0.2.0"          # omnilingual-asr; it needs Python <= 3.12 and fairseq2 0.6
RESCORE = False                 # True: score a candidate again even when its bakeoff.json is on the hub
DATA_LOCAL = None               # a local export in the HF layout instead of the download
"""

DATA = r"""
import gc
import hashlib
import shutil
import time
import traceback
import warnings

import numpy as np
import soundfile as sf
import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download

import bakeoffkit
import evalkit
import ftkit

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
TRAIN_TEXTS = [(r["text"], ftkit.duration(r)) for r in splits["train"] if r["text"].strip()]
HUMAN_H = sum(ftkit.duration(r) for r in splits["train"]) / 3600  # the projection reads the full set
VAL_H = sum(ftkit.duration(r) for r in splits["val"]) / 3600
if SMOKE:
    splits = {"train": splits["train"][::50], "val": splits["val"][::30], "gold": splits["gold"][::16]}
    SWEEP_CLIPS, SPEED_CLIPS = 24, 60


def spread(rows, n):
    # n rows evenly spaced through `rows`: every episode, not the first few
    step = max(1, len(rows) // n)
    return list(rows[::step][:n])


SWEEP_ROWS = spread(splits["val"], SWEEP_CLIPS)
SPEED_ROWS = [r for r in spread(splits["train"], SPEED_CLIPS) if r["text"].strip()]
episodes = [r["episode_id"] for rows in (splits["val"], splits["gold"], SPEED_ROWS) for r in rows]
with ftkit.timed(f"downloading {len(set(episodes))} recordings"):
    DATA = ftkit.download_dataset(DATA_LOCAL, episodes=episodes, analytics=True)
with ftkit.timed("reading them into RAM"):
    store = ftkit.AudioStore(DATA, episodes)
score = ftkit.harness_scorer(DATA, FT)
print({k: len(v) for k, v in splits.items()}, f"| prompt sample {len(SWEEP_ROWS)}, speed sample {len(SPEED_ROWS)}",
      f"| human train {HUMAN_H:.1f} h, val {VAL_H:.2f} h | audio in RAM: {store.gib:.1f} GiB |", score.fold_version)
"""

SPECS_NOTE = """
## The candidates

One entry per candidate: its family (which code loads and runs it), its checkpoint, the prompts
tried on the val sample, and how its training step is timed (`None`: zero-shot only). Gemma's
third prompt is the one its model card gives for ASR.
"""

SPECS = r"""
NEPALI = "Transcribe the following speech segment in Nepali into Nepali text."
MIXED = ("Transcribe the following speech segment exactly as it was said. The speakers talk in Nepali and often "
         "switch to English: write the Nepali words in Devanagari script and the English words in Latin script. "
         "Only output the transcription, with no newlines.")
CARD = ("Transcribe the following speech segment in its original language. Follow these specific instructions for "
        "formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, "
        "write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.")
GEMMA_PROMPTS = {"nepali": NEPALI, "mixed": MIXED, "card": CARD}
# Omnilingual takes a language code (None: none given); Qwen3-ASR a language name it is prefilled
# with ("language Nepali<asr_text>"; None: it names the language itself).
SPECS = {
    "gemma-4-e2b": {"family": "gemma", "model_id": "google/gemma-4-E2B-it", "prompts": GEMMA_PROMPTS,
                    "train": {"mode": "full", "optim": "adamw", "lr": 1e-5, "grad_ckpt": True},
                    "architecture": "Gemma 4 E2B: ~300M audio encoder + decoder-only LLM with per-layer embeddings"},
    "gemma-4-e4b": {"family": "gemma", "model_id": "google/gemma-4-E4B-it", "prompts": GEMMA_PROMPTS,
                    "train": {"mode": "full", "optim": "adamw8bit", "lr": 1e-5, "grad_ckpt": True},
                    "architecture": "Gemma 4 E4B: ~300M audio encoder + decoder-only LLM with per-layer embeddings"},
    "gemma-4-12b": {"family": "gemma", "model_id": "google/gemma-4-12B-it", "prompts": GEMMA_PROMPTS,
                    "train": {"mode": "lora", "rank": 16, "optim": "adamw", "lr": 1e-4, "grad_ckpt": True},
                    "architecture": "Gemma 4 12B Unified: no audio encoder, the waveform projected into the LLM"},
    "omni-llm-1b": {"family": "omni", "model_id": "omniASR_LLM_1B_v2", "prompts": {"nep_Deva": "nep_Deva", "none": None},
                    "train": {"mode": "full", "optim": "adamw", "lr": 1e-5},
                    "architecture": "Omnilingual LLM-ASR 1B: wav2vec 2.0 encoder + LLM decoder; character vocabulary"},
    "omni-llm-3b": {"family": "omni", "model_id": "omniASR_LLM_3B_v2", "prompts": {"nep_Deva": "nep_Deva", "none": None},
                    "train": {"mode": "full", "optim": "adamw", "lr": 1e-5},
                    "architecture": "Omnilingual LLM-ASR 3B: wav2vec 2.0 encoder + LLM decoder; character vocabulary"},
    "omni-llm-7b": {"family": "omni", "model_id": "omniASR_LLM_7B_v2", "prompts": {"nep_Deva": "nep_Deva", "none": None},
                    "train": None,
                    "architecture": "Omnilingual LLM-ASR 7B: wav2vec 2.0 encoder + LLM decoder; character vocabulary"},
    "qwen3-asr-1.7b": {"family": "qwen", "model_id": "Qwen/Qwen3-ASR-1.7B-hf",
                       "prompts": {"nepali": "Nepali", "hindi": "Hindi", "auto": None},
                       "train": {"mode": "full", "optim": "adamw", "lr": 1e-5, "grad_ckpt": False},
                       "architecture": "Qwen3-ASR 1.7B: AuT audio encoder + Qwen3 decoder"},
    "voxtral-mini-3b": {"family": "voxtral", "model_id": "mistralai/Voxtral-Mini-3B-2507",
                        "prompts": {"nepali": NEPALI, "mixed": MIXED},
                        "train": {"mode": "full", "optim": "adamw8bit", "lr": 1e-5, "grad_ckpt": True},
                        "architecture": "Voxtral Mini 3B: Whisper encoder + Ministral decoder"},
}
unknown = [c for c in CANDIDATES if c not in SPECS]
assert not unknown, f"not in SPECS: {unknown}"
print("candidates:", ", ".join(CANDIDATES))
"""

HF_NOTE = """
## Gemma, Qwen3-ASR and Voxtral: transformers, in this kernel

Each clip is a 16 kHz WAV handed to the processor's chat template, the audio after the text for
Gemma (its card asks for that order) and before it for Voxtral. Decoding is greedy, left-padded,
in batches of similar length. A training example is the generation prompt exactly as decoding
sees it, then the transcript's tokens and the token that closes the model's turn (read off its chat
template). It is not a rendered two-turn conversation: Gemma 4 12B's prompt opens an empty thought
channel its rendered history leaves out, and Voxtral's template refuses a conversation that ends
on the model's turn. The loss is on the transcript and the closing token only, and the logits are
computed from the first labelled position on (`logits_to_keep`), so a 262k-word head does not run
over the audio.
"""

HF = r'''
import re

WAV = FT / "wav"
WAV.mkdir(exist_ok=True)
DEVICE = "cuda"
SPECIAL = re.compile(r"<\|[^|>]*\|>")


def wav_path(row):
    key = f"{row['episode_id']}/{row['segment_id']}"
    path = WAV / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".wav")
    if not path.exists():
        tmp = path.parent / f"{path.stem}.{os.getpid()}.tmp"
        sf.write(tmp, store.clip_f32(row).numpy(), ftkit.SR, format="WAV")
        tmp.replace(path)
    return str(path)


def free_model():
    for name in ("model", "optimizer", "processor"):
        globals().pop(name, None)
    gc.collect()
    torch.cuda.empty_cache()


def load_hf(spec, train=None):
    """`model` and `processor` for `spec`: bf16 for decoding; for training, fp32 master weights
    (full) or a frozen bf16 base with LoRA adapters."""
    global model, processor, MODEL_DTYPE, TOKENS_PER_S, END_IDS, PAD_ID
    free_model()
    mode = train["mode"] if train else None
    with ftkit.timed(f"loading {spec['model_id']}" + (f" for {mode} training" if mode else "")):
        processor = AutoProcessor.from_pretrained(spec["model_id"])
        dtype = torch.float32 if mode == "full" else torch.bfloat16
        for cls in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForSpeechSeq2Seq",
                    "AutoModelForCausalLM"):
            auto = getattr(transformers, cls, None)
            if auto is None:
                continue
            try:
                model = auto.from_pretrained(spec["model_id"], dtype=dtype, device_map=DEVICE)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"no auto class loads {spec['model_id']}")
    for key in ("temperature", "top_p", "top_k"):  # decoding here is always greedy
        setattr(model.generation_config, key, None)
    MODEL_DTYPE = dtype
    if mode == "full":
        for m in model.modules():  # embeddings (Gemma's per-layer ones included) stay frozen
            if isinstance(m, torch.nn.Embedding):
                m.weight.requires_grad_(False)
    if train and train.get("grad_ckpt"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=train["rank"], lora_alpha=2 * train["rank"], lora_dropout=0.0,
                                                 target_modules="all-linear"))
    model.eval()
    tok = processor.tokenizer
    TOKENS_PER_S = max(len(tok.encode(t, add_special_tokens=False)) / d for t, d in TRAIN_TEXTS)
    END_IDS, PAD_ID = end_ids(spec), tok.pad_token_id if tok.pad_token_id is not None else 0
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{type(model).__name__}: {params / 1e9:.2f}B parameters" + (f", {trainable / 1e9:.3f}B trainable" if mode else "")
          + f" | densest train label {TOKENS_PER_S:.1f} tokens/s | {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    return model


def conversation(spec, prompt, row):
    """The generation prompt for one clip, as (messages, continue_final_message): Qwen3-ASR's
    answer is prefilled with its language, which an empty prefill leaves it to name itself."""
    family = spec["family"]
    if family == "qwen":
        prefill = f"language {prompt}<asr_text>" if prompt else ""
        return ([{"role": "user", "content": [{"type": "audio", "path": wav_path(row)}]},
                 {"role": "assistant", "content": [{"type": "text", "text": prefill}]}], True)
    if family == "gemma":
        user = [{"type": "text", "text": prompt}, {"type": "audio", "audio": wav_path(row)}]
    else:  # voxtral
        user = [{"type": "audio", "path": wav_path(row)}, {"type": "text", "text": prompt}]
    return [{"role": "user", "content": user}], False


def encode(spec, convs, side):
    """The chat template over a batch of `conversation`s, tokenized and padded on `side`."""
    msgs = [m for m, _ in convs]
    cont = convs[0][1]
    kw = {"tokenize": True, "return_dict": True, "return_tensors": "pt", "padding": True, "padding_side": side}
    kw["continue_final_message" if cont else "add_generation_prompt"] = True
    if spec["family"] == "gemma":
        kw["enable_thinking"] = False
    processor.tokenizer.padding_side = side
    try:
        return processor.apply_chat_template(msgs, **kw)
    except TypeError:  # a processor that takes no padding_side: the tokenizer's attribute holds it
        kw.pop("padding_side")
        return processor.apply_chat_template(msgs, **kw)


def end_ids(spec):
    """The tokens that close the model's turn, read off its chat template (Gemma `<turn|>`,
    Qwen `<|im_end|>`); end-of-sequence when the template will not render a finished turn
    (Voxtral's: `</s>`)."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ANSWER"}]}]
    kw = {"enable_thinking": False} if spec["family"] == "gemma" else {}
    try:
        text = processor.apply_chat_template(msgs, tokenize=False, **kw)
        ids = processor.tokenizer.encode(text.split("ANSWER", 1)[1].strip(), add_special_tokens=False)
        if ids:
            return ids
    except Exception:
        pass
    return [processor.tokenizer.eos_token_id]


def answer(spec, ids):
    if spec["family"] == "qwen":  # "language X<asr_text>" when it named the language itself
        text = processor.tokenizer.decode(ids, skip_special_tokens=False).split("<asr_text>")[-1]
        return SPECIAL.sub("", text)
    return processor.tokenizer.decode(ids, skip_special_tokens=True)


def on_device(inputs):
    # every tensor on the GPU, the floating ones (audio features) in the model's dtype
    return {k: (v.to(DEVICE, non_blocking=True).to(MODEL_DTYPE) if torch.is_floating_point(v)
                else v.to(DEVICE, non_blocking=True)) for k, v in inputs.items() if torch.is_tensor(v)}


def hf_transcribe(spec, prompt, rows):
    enc = on_device(encode(spec, [conversation(spec, prompt, r) for r in rows], "left"))
    cap = max(bakeoffkit.token_cap(TOKENS_PER_S, ftkit.duration(r)) for r in rows)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        out = model.generate(**enc, max_new_tokens=cap, do_sample=False, num_beams=1, use_cache=True)
    return [bakeoffkit.clean(answer(spec, ids)) for ids in out[:, enc["input_ids"].shape[1]:].tolist()]


# --- the training step ---------------------------------------------------------------------------


def per_token(key, value, length):
    # input_ids and the masks and type ids beside them; not the audio features or their masks
    return (torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == length
            and not any(w in key for w in ("feature", "audio", "pixel")))


def sft_collate(rows):
    """The generation prompt (right-padded), then the transcript and the closing token; labels on
    those only. Per-token inputs are extended alongside: the mask with ones, type ids with zeros
    (text)."""
    prefix = encode(SPEC, [conversation(SPEC, TRAIN_PROMPT, r) for r in rows], "right")
    starts = prefix["attention_mask"].sum(dim=1).tolist()
    answers = [processor.tokenizer.encode(r["text"], add_special_tokens=False) + END_IDS for r in rows]
    width = max(n + len(a) for n, a in zip(starts, answers, strict=True))
    length = prefix["input_ids"].shape[1]
    inputs = {}
    for key, value in prefix.items():
        if not per_token(key, value, length):
            inputs[key] = value
            continue
        new = torch.full((len(rows), width), PAD_ID if key == "input_ids" else 0, dtype=value.dtype)
        for i, (n, a) in enumerate(zip(starts, answers, strict=True)):
            new[i, :n] = value[i, :n]
            if key == "input_ids":
                new[i, n:n + len(a)] = torch.tensor(a, dtype=value.dtype)
            elif "mask" in key:
                new[i, n:n + len(a)] = 1
        inputs[key] = new
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (n, a) in enumerate(zip(starts, answers, strict=True)):
        labels[i, n:n + len(a)] = torch.tensor(a)
    lens = [len(store.clip(r)) for r in rows]
    return {"inputs": inputs, "labels": labels, "first": min(starts), "lens": torch.tensor(lens),
            "seconds": sum(lens) / ftkit.SR, "padded_seconds": len(rows) * max(lens) / ftkit.SR}


def sft_loss(model, b):
    """Summed token cross-entropy over the labelled positions, with logits only from the first
    labelled position of the batch on."""
    kw = on_device(b["inputs"])
    labels = b["labels"].to(DEVICE, non_blocking=True)
    keep = labels.shape[1] - b["first"] + 1
    try:
        logits = model(**kw, logits_to_keep=keep, use_cache=False).logits
    except TypeError:  # a model without logits_to_keep: every position, cut here
        logits = model(**kw, use_cache=False).logits[:, -keep:]
    target = labels[:, b["first"]:]
    sel = target != -100
    loss = torch.nn.functional.cross_entropy(logits[:, :-1][sel].float(), target[sel], reduction="sum")
    return loss, int(sel.sum())


def make_optimizer(model, train):
    params = [p for p in model.parameters() if p.requires_grad]
    if train["optim"] == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=train["lr"], weight_decay=0.0)
    return torch.optim.AdamW(params, lr=train["lr"], weight_decay=0.0, fused=True)


def allocate_state(model, optimizer):
    # ftkit.init_optimizer_state, for an optimizer whose step counter may be a plain int (bitsandbytes)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        if torch.is_tensor(state.get("step")):
            state["step"].zero_()
        elif "step" in state:
            state["step"] = 0


def hf_probe(rows, optimizer):
    """The largest micro-batch at the longest clip carrying the wordiest transcript, and at the
    wordiest of the shortest tenth (each clip brings its own prompt): (seconds, clips)."""
    def probe_at(clip):
        def step(n):
            b = sft_collate([clip] * n)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss, _ = sft_loss(model, b)
            loss.backward()

        return ftkit.probe_max_items(step, 1, 256, model)

    longest = max(rows, key=ftkit.duration)
    wordiest = max(rows, key=lambda r: len(r["text"]))
    model.train()
    allocate_state(model, optimizer)
    n = probe_at({**longest, "text": wordiest["text"]})
    if n == 0:
        raise RuntimeError("not even one clip fits a micro-batch")
    by_length = sorted(rows, key=ftkit.duration)
    short = max(by_length[: max(1, len(by_length) // 10)], key=lambda r: len(r["text"]))
    n_short = probe_at(short)
    budget, items = n * ftkit.duration(longest) * PROBE_FRACTION, max(1, int(n_short * PROBE_FRACTION))
    print(f"largest micro-batch: {n} clips at {ftkit.duration(longest):.1f} s, {n_short} at "
          f"{ftkit.duration(short):.1f} s -> {items} clips or {budget:.0f} s of audio")
    return budget, items


def hf_speed(spec, prompt):
    global SPEC, TRAIN_PROMPT, optimizer
    # A target is written as the model answers: Qwen3-ASR's opens with a language, Nepali when the
    # kept prompt let it name its own.
    SPEC, TRAIN_PROMPT = spec, prompt if prompt is not None else "Nepali"
    load_hf(spec, spec["train"])
    optimizer = make_optimizer(model, spec["train"])
    budget, items = hf_probe(SPEED_ROWS, optimizer)
    batches = ftkit.bucket_batches(SPEED_ROWS, budget_s=budget, max_items=items, pad_to_s=PAD_TO_S,
                                   shuffle=True, seed=0)
    sample = SWEEP_ROWS[:16]

    def evaluate(m):
        texts = hf_transcribe(spec, prompt, sample)
        return score.summarize(score.per_clip([r["text"] for r in sample], texts))

    speed = ftkit.speed_check(model, rows=SPEED_ROWS, batches=batches, collate=sft_collate, loss_fn=sft_loss,
                              evaluate=evaluate, val_rows=sample, gold_rows=sample, epochs=HUMAN_EPOCHS,
                              monitor=monitor)
    return {**speed, "micro_batch_s": budget, "micro_batch_items": items, "train": spec["train"]}
'''

OMNI_ENV_NOTE = """
## Omnilingual LLM-ASR: its own environment

omnilingual-asr needs Python 3.12 and fairseq2, which pin a torch, numpy and huggingface_hub
that do not install into this kernel (06e). Its candidates run `bakeoff_omni.py` there: this
kernel writes each set's clips to one int16 array, the script writes its transcripts and its
training-step timing back, and the scoring happens here like everyone else's. Skip this cell when
no Omnilingual candidate is in `CANDIDATES`.
"""

OMNI_SCRIPT = r'''%%writefile /content/ft/bakeoff_omni.py
"""Omnilingual LLM-ASR for the speech-LLM bake-off (D121), in the environment fairseq2 needs.

    bakeoff_omni.py <card> transcribe <clips> <lang|none> <out.jsonl> <items> <budget_s>
    bakeoff_omni.py <card> speed <clips> <lang> <out.json> <probe_fraction> <pad_to_s>

<clips> names `<clips>.npy` (every clip's int16 samples, end to end) and `<clips>.json` (each
clip's segment_id, offset, length and text), written by the notebook."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ftkit  # noqa: E402

CARD, ACTION = sys.argv[1], sys.argv[2]


def read_clips(stem):
    meta = json.loads(Path(f"{stem}.json").read_text("utf-8"))
    audio = np.load(f"{stem}.npy", mmap_mode="r")
    rows = [{**m, "start_time": 0.0, "end_time": m["length"] / ftkit.SR} for m in meta]
    return rows, [np.asarray(audio[m["offset"]:m["offset"] + m["length"]]) for m in meta]


def transcribe(stem, lang, out, items, budget_s):
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    pipe = ASRInferencePipeline(model_card=CARD, dtype=torch.bfloat16)  # greedy: nbest 1
    rows, clips = read_clips(stem)
    batches = ftkit.bucket_batches(rows, budget_s=budget_s, max_items=items, pad_to_s=1.0, shuffle=False)
    t_start, done, next_print = time.perf_counter(), 0, 0
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        for batch in batches:
            inp = [{"waveform": clips[i].astype(np.float32) / 32768.0, "sample_rate": ftkit.SR} for i in batch]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            texts = pipe.transcribe(inp, lang=[lang] * len(batch) if lang else None, batch_size=len(batch))
            torch.cuda.synchronize()
            took = time.perf_counter() - t0
            audio_s = sum(ftkit.duration(rows[i]) for i in batch)
            for i, text in zip(batch, texts, strict=True):
                fh.write(json.dumps({"segment_id": rows[i]["segment_id"], "text": text,
                                     "compute_s": took * ftkit.duration(rows[i]) / audio_s}, ensure_ascii=False) + "\n")
            done += len(batch)
            if done >= next_print or done == len(rows):
                print(f"  {done}/{len(rows)} clips, {time.perf_counter() - t_start:.0f} s", flush=True)
                next_print = done + 500
    Path(out + ".tmp").replace(out)


def speed(stem, lang, out, probe_fraction, pad_to_s):
    from fairseq2.datasets.batch import Seq2SeqBatch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    pipe = ASRInferencePipeline(model_card=CARD, dtype=torch.float32)
    model, tok = pipe.model, pipe.tokenizer
    encode, pad = tok.create_encoder(), tok.vocab_info.pad_idx
    for p in model.encoder_frontend.feature_extractor.parameters():  # frozen in the official recipe
        p.requires_grad_(False)
    rows, clips = read_clips(stem)
    for k, r in enumerate(rows):
        r["k"] = k
        r["ids"] = encode(r["text"].replace("‍", "").replace("‌", "")).tolist()

    def batch_of(chosen):
        wavs = [torch.nn.functional.layer_norm(t, t.shape) for t in
                (torch.from_numpy(clips[r["k"]].astype(np.float32) / 32768.0) for r in chosen)]
        wav = torch.zeros(len(chosen), ftkit.pad_len(max(len(w) for w in wavs), pad_to_s))
        for i, w in enumerate(wavs):
            wav[i, :len(w)] = w
        targets = torch.full((len(chosen), max(len(r["ids"]) for r in chosen)), pad, dtype=torch.long)
        for i, r in enumerate(chosen):
            targets[i, :len(r["ids"])] = torch.tensor(r["ids"])
        batch = Seq2SeqBatch(source_seqs=wav.cuda(), source_seq_lens=[len(w) for w in wavs], target_seqs=targets.cuda(),
                             target_seq_lens=[len(r["ids"]) for r in chosen], example={"lang": [lang] * len(chosen)})
        return batch, sum(len(w) for w in wavs) / ftkit.SR, wav.numel() / ftkit.SR

    def step(chosen):
        batch, _, _ = batch_of(chosen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)
        loss.backward()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0, fused=True)
    model.train()
    ftkit.init_optimizer_state(model, optimizer)
    longest = max(rows, key=ftkit.duration)
    wordiest = max(rows, key=lambda r: len(r["ids"]))
    n = ftkit.probe_max_items(lambda m: step([{**longest, "ids": wordiest["ids"]}] * m), 1, 256, model)
    if n == 0:
        raise RuntimeError("not even one clip fits a micro-batch")
    budget = n * ftkit.duration(longest) * probe_fraction
    print(f"largest micro-batch at {ftkit.duration(longest):.1f} s: {n} clips -> {budget:.0f} s of audio", flush=True)
    batches = ftkit.bucket_batches(rows, budget_s=budget, max_items=256, pad_to_s=pad_to_s, shuffle=True, seed=0)[:10]
    monitor = ftkit.GpuMonitor().start()

    def run():
        audio = padded = 0.0
        count = 0
        for idx in batches:
            chosen = [rows[i] for i in idx]
            batch, a, p = batch_of(chosen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch)
            loss.backward()
            model.zero_grad(set_to_none=False)
            audio, padded, count = audio + a, padded + p, count + len(chosen)
        torch.cuda.synchronize()
        return audio, padded, count

    run()  # warm-up: kernel selection, allocator growth
    monitor.window()
    t0 = time.perf_counter()
    audio, padded, count = run()
    took = time.perf_counter() - t0
    gpu = monitor.window()
    monitor.stop()
    rec = {"train_x_realtime": audio / took, "train_ms_per_clip": 1000 * took / count, "padding_waste": 1 - audio / padded,
           **gpu, "micro_batch_s": budget, "micro_batch_items": 256,
           "train": {"mode": "full", "optim": "adamw", "lr": 1e-5},
           "params_b": sum(p.numel() for p in model.parameters()) / 1e9, "trainable_b": sum(p.numel() for p in params) / 1e9}
    print(f"train: {rec['train_x_realtime']:.0f}x realtime, {rec['train_ms_per_clip']:.0f} ms per clip, pad waste "
          f"{rec['padding_waste']:.0%}, GPU {rec['gpu_util']:.0f}% util, {rec['gpu_mem_gib']:.1f} GiB", flush=True)
    Path(out).write_text(json.dumps(rec, indent=1))


if ACTION == "transcribe":
    transcribe(sys.argv[3], None if sys.argv[4] == "none" else sys.argv[4], sys.argv[5], int(sys.argv[6]),
               float(sys.argv[7]))
elif ACTION == "speed":
    speed(sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), float(sys.argv[7]))
else:
    raise SystemExit(f"unknown action {ACTION}")
'''

OMNI_KERNEL = r"""
CLIPS = OUT_ROOT / "clips"  # under the run folder: a smoke run's clips never stand in for a real run's
CLIPS.mkdir(exist_ok=True)
SCRIPT = FT / "bakeoff_omni.py"


def dump_clips(tag, rows):
    # The clips of `rows` for the script: one int16 array end to end, and where each one starts.
    stem = CLIPS / tag
    if not Path(f"{stem}.json").exists():
        clips = [store.clip(r) for r in rows]
        offsets = np.cumsum([0] + [len(c) for c in clips[:-1]]).tolist()
        np.save(f"{stem}.npy", np.concatenate(clips).astype(np.int16))
        Path(f"{stem}.json").write_text(json.dumps(
            [{"segment_id": r["segment_id"], "offset": o, "length": len(c), "text": r["text"]}
             for r, o, c in zip(rows, offsets, clips, strict=True)], ensure_ascii=False), "utf-8")
    return stem


def omni_transcribe(spec, lang, rows, tag):
    stem = dump_clips(tag, rows)
    card, lang_arg = spec["model_id"], lang or "none"
    out = OUT_ROOT / "omni" / f"{card}-{tag}-{lang_arg}.jsonl"
    out.parent.mkdir(exist_ok=True)
    if not out.exists():
        !{OMNI_PY} {SCRIPT} {card} transcribe {stem} {lang_arg} {out} {GEN_ITEMS} {GEN_BUDGET_S}
    if not out.exists():
        raise RuntimeError(f"{spec['model_id']} wrote no transcripts: read the script's output above")
    by_id = {}
    for line in out.read_text("utf-8").splitlines():
        rec = json.loads(line)
        by_id[rec["segment_id"]] = rec
    return ([bakeoffkit.clean(by_id[r["segment_id"]]["text"]) for r in rows],
            [by_id[r["segment_id"]]["compute_s"] for r in rows])


def omni_speed(spec, lang):
    stem = dump_clips("speed", SPEED_ROWS)
    card, lang_arg = spec["model_id"], lang or "nep_Deva"  # an LID model needs a language code to train
    out = OUT_ROOT / "omni" / f"{card}-speed.json"
    out.parent.mkdir(exist_ok=True)
    if not out.exists():
        !{OMNI_PY} {SCRIPT} {card} speed {stem} {lang_arg} {out} {PROBE_FRACTION} {PAD_TO_S}
    if not out.exists():
        raise RuntimeError(f"{spec['model_id']} measured no training step: read the script's output above")
    return json.loads(out.read_text())
"""

RUN_NOTE = """
## Shared by the rounds

The search is breadth-first, so no GPU-hours go to a candidate before its zero-shot numbers have
been read. Each round is its own cell and reads what the earlier rounds left on the hub, so it runs
in a fresh kernel after the cells above.

1. **Sweep**: every candidate in `CANDIDATES`, every prompt, on `SWEEP_CLIPS` val clips. Zero-shot
   only, about a minute of decoding per prompt. It ends with a table and the default shortlist.
2. **Training step**: only the shortlist (`SHORTLIST` in Config, or the candidates whose kept prompt
   passed the gate). A few minutes each; it gives the projected cost and the verdict.
3. **Full val and gold**: only the shortlist again, through `evalkit.evaluate_run`. The numbers
   for the record and the paper. The verdict does not wait for them.

One candidate at a time on the GPU. Each output is uploaded as it lands and a rerun skips it; a
failure is printed with its traceback, the GPU is freed, and the next candidate runs.
"""

RUN = r'''
monitor = ftkit.GpuMonitor().start()


def fetch_json(remote):
    if not api.file_exists(OUT_REPO, remote):
        return None
    return json.loads(Path(hf_hub_download(OUT_REPO, remote, token=TOKEN)).read_text("utf-8"))


def upload(folder, dest, message):
    with ftkit.timed(f"uploading {dest}: {message}"):
        api.upload_folder(repo_id=OUT_REPO, folder_path=str(folder), path_in_repo=dest, commit_message=f"{dest}: {message}")


def where(key):
    out = OUT_ROOT / key
    out.mkdir(parents=True, exist_ok=True)
    return f"{RUN_PREFIX}/{key}", out


def decode_rows(spec, prompt, rows, tag):
    """(texts, compute seconds) for `rows` in their order, with progress as it goes."""
    if spec["family"] == "omni":
        return omni_transcribe(spec, prompt, rows, tag)
    if globals().get("LOADED") != (spec["model_id"], "decode"):
        load_hf(spec)
        globals()["LOADED"] = (spec["model_id"], "decode")
    texts, compute = [""] * len(rows), [0.0] * len(rows)
    batches = ftkit.bucket_batches(rows, budget_s=GEN_BUDGET_S, max_items=GEN_ITEMS, pad_to_s=PAD_TO_S, shuffle=False)
    t_start, done, next_print = time.perf_counter(), 0, 0
    for batch in batches:
        chunk = [rows[i] for i in batch]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = hf_transcribe(spec, prompt, chunk)
        torch.cuda.synchronize()
        took = time.perf_counter() - t0
        audio_s = sum(ftkit.duration(r) for r in chunk)
        for i, text in zip(batch, out, strict=True):
            texts[i], compute[i] = text, took * ftkit.duration(rows[i]) / audio_s
        done += len(batch)
        if done >= next_print or done == len(rows):
            elapsed = time.perf_counter() - t_start
            print(f"  {tag}: {done}/{len(rows)} clips, {elapsed:.0f} s, about {elapsed * (len(rows) - done) / done:.0f} s left",
                  flush=True)
            next_print = done + 500
    return texts, compute


def sweep(key):
    """Round 1: every prompt of `key` on the val sample; prompts.json is uploaded after each one."""
    spec, (dest, out) = SPECS[key], where(key)
    found = fetch_json(f"{dest}/prompts.json") or {}
    refs = [r["text"] for r in SWEEP_ROWS]
    audio_s = sum(ftkit.duration(r) for r in SWEEP_ROWS)
    for name, prompt in spec["prompts"].items():
        if name in found and not RESCORE:
            continue
        with ftkit.timed(f"{key}: prompt {name!r} on {len(SWEEP_ROWS)} val clips"):
            texts, compute = decode_rows(spec, prompt, SWEEP_ROWS, "sweep")
        m = score.summarize(score.per_clip(refs, texts))
        found[name] = {**bakeoffkit.zero_shot_report(refs, texts), **{k: m[k] for k in evalkit.KEEP if k in m},
                       "prompt": prompt, "rtf": sum(compute) / audio_s,
                       "examples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, texts))[:8]]}
        (out / "prompts.json").write_text(json.dumps(found, indent=1, ensure_ascii=False))
        upload(out, dest, f"sweep, prompt {name}")
    return found


def sweeps():
    """Every candidate's round-1 reports from the hub ({} for one not yet swept)."""
    return {k: fetch_json(f"{RUN_PREFIX}/{k}/prompts.json") or {} for k in CANDIDATES}


def kept(key, found):
    name = bakeoffkit.pick_prompt(found)
    return name, SPECS[key]["prompts"][name]


def shortlisted():
    """Rounds 2 and 3 run these: `SHORTLIST` when Config names one, else the sweep's default."""
    keys = list(SHORTLIST) if SHORTLIST is not None else bakeoffkit.shortlist(sweeps())
    unknown = [k for k in keys if k not in SPECS]
    assert not unknown, f"not in SPECS: {unknown}"
    return keys


def guarded(fn, key):
    """`fn(key)`, with a failure printed and the GPU freed, so the next candidate still runs."""
    try:
        return fn(key)
    except Exception:
        traceback.print_exc()
        print(f"{key}: FAILED, the next candidate runs")
        return None
    finally:
        globals()["LOADED"] = None
        free_model()


def card(key, spec, prompt):
    return {
        "name": f"{key} (zero-shot)",
        "description": f"{RUN_PREFIX}: {spec['model_id']} zero-shot, prompt {prompt!r} (D121)",
        "architecture": spec["architecture"],
        "base_model": spec["model_id"],
        "decoder": "greedy, no retry",
        "notebook": NOTEBOOK,
        "dataset_export": export["exported_at"],
        "fold_version": score.fold_version,
        "kits": evalkit.kit_digests(FT),
    }
'''

SWEEP_NOTE = """
## Round 1: the sweep

Every candidate, every prompt, on the same val sample. In the table: folded WER with S/D/I, CER,
the share of Nepali clips answered in Devanagari (the gate is 90%), empty answers, runaways,
answers in another script, the Latin share of the answers against the references, and the decode's
real-time factor. `*` marks the prompt each candidate keeps. Two of its answers follow, beside
their references. **Read this before running round 2**, and set `SHORTLIST` in Config to overrule
the default.
"""

SWEEP = r"""
for key in CANDIDATES:
    print(f"\n=== {key}: {SPECS[key]['model_id']}")
    guarded(sweep, key)

found = sweeps()
print(f"\n{'candidate':<16} {'prompt':<10} {'WER':>7} {'S/D/I':>17} {'CER':>6} {'Nepali':>6} {'empty':>5} "
      f"{'runaway':>7} {'other':>5} {'Latin':>11} {'RTF':>6}")
for key, prompts in found.items():
    if not prompts:
        print(f"{key:<16} not swept")
        continue
    best = bakeoffkit.pick_prompt(prompts)
    for name, f in prompts.items():
        mark = "*" if name == best else " "
        print(f"{key:<16}{mark}{name:<10} {f['wer']:7.2f} {ftkit.sid(f):>17} {f['cer']:6.2f} {f['writes_nepali']:6.0%} "
              f"{f['empty']:5d} {f['runaway']:7d} {f['other_script_clips']:5d} {f['latin_hyp']:4.0%} / {f['latin_ref']:3.0%} "
              f"{f['rtf']:6.3f}")
for key, prompts in found.items():
    if prompts:
        name = bakeoffkit.pick_prompt(prompts)
        print(f"\n{key} ({name}):")
        for ex in prompts[name]["examples"][:2]:
            print(f"  ref: {ex['ref']}\n  hyp: {ex['hyp']}")
print("\ndefault shortlist (kept prompt passes the gate):", ", ".join(bakeoffkit.shortlist(found)) or "none")
print("rounds 2 and 3 will run:", ", ".join(shortlisted()) or "nothing")
"""

SPEED_NOTE = """
## Round 2: the training step and the verdict

The shortlist only. Each candidate is loaded for training (full fine-tuning, or LoRA for Gemma 4
12B), a probe finds its largest micro-batch, and real train clips run forward and backward. The
projection covers every epoch of both student stages at the measured speed, with val passes at the
sweep's decode speed. The verdict reads the gate on the sweep's val sample and the projection
against `BUDGET_H`.
"""

SPEED = r"""
def time_training(key):
    spec, (dest, out) = SPECS[key], where(key)
    found = fetch_json(f"{dest}/prompts.json")
    if not found:
        raise RuntimeError(f"{key} was not swept: run round 1 first")
    name, prompt = kept(key, found)
    speed = fetch_json(f"{dest}/speed_check.json")
    if bakeoffkit.needs_timing(speed, rescore=RESCORE) and spec["train"] is not None:
        print(f"\n=== {key}: timing the training step, prompt {name!r}")
        try:
            speed = omni_speed(spec, prompt) if spec["family"] == "omni" else hf_speed(spec, prompt)
        except Exception as exc:  # recorded, not raised: the verdict says it was not measured
            traceback.print_exc()
            speed = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        (out / "speed_check.json").write_text(json.dumps(speed, indent=1))
        upload(out, dest, "training step timed")
    cost = None
    if speed and "train_x_realtime" in speed:
        cost = bakeoffkit.projection(train_x_realtime=speed["train_x_realtime"], val_x_realtime=1 / found[name]["rtf"],
                                     human_h=HUMAN_H, human_epochs=HUMAN_EPOCHS, distill_h=DISTILL_H,
                                     distill_epochs=DISTILL_EPOCHS, val_h=VAL_H)
    verdict = bakeoffkit.verdict(found[name], cost, BUDGET_H)
    row = {**(fetch_json(f"{dest}/bakeoff.json") or {}), "candidate": key, "model_id": spec["model_id"],
           "family": spec["family"], "prompt": name, "sweep": {k: v for k, v in found[name].items() if k != "examples"},
           "speed": speed, "cost": cost, "budget_h": BUDGET_H, "verdict": verdict}
    (out / "bakeoff.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
    upload(out, dest, "verdict: " + ("continue" if verdict["continue"] else "stop"))
    return row


for key in shortlisted():
    guarded(time_training, key)

print(f"\n{'candidate':<16} {'train x':>7} {'pad':>5} {'GPU':>5} {'GiB':>5} {'stage 1 h':>9} {'stage 2 h':>9} {'total h':>8}  verdict")
for key in shortlisted():
    r = fetch_json(f"{RUN_PREFIX}/{key}/bakeoff.json")
    if r is None:
        print(f"{key:<16} no verdict")
        continue
    s, c = r["speed"] or {}, r["cost"] or {}
    nan = float("nan")
    print(f"{key:<16} {s.get('train_x_realtime', nan):7.0f} {s.get('padding_waste', nan):5.0%} {s.get('gpu_util', nan):4.0f}% "
          f"{s.get('gpu_mem_gib', nan):5.1f} {c.get('stage1_h', nan):9.1f} {c.get('stage2_h', nan):9.1f} "
          f"{c.get('total_h', nan):8.1f}  {'continue' if r['verdict']['continue'] else 'stop'}")
    for reason in r["verdict"]["reasons"]:
        print(f"{'':<18}{reason}")
    if "error" in s:
        print(f"{'':<18}training step failed: {s['error']}")
"""

FULL_NOTE = """
## Round 3: full val and gold, zero-shot

The shortlist only, with each candidate's kept prompt: every val and gold clip through
`evalkit.evaluate_run` (folded and raw WER with S/D/I, CER, per clip class, error rows, the
Models page's files), and the gate on the full sets. This is the costly round, for the record
and the paper. Skip it if round 2 left nothing worth writing up.
"""

FULL = r"""
def full_zero_shot(key):
    spec, (dest, out) = SPECS[key], where(key)
    found = fetch_json(f"{dest}/prompts.json")
    if not found:
        raise RuntimeError(f"{key} was not swept: run round 1 first")
    name, prompt = kept(key, found)
    result = fetch_json(f"{dest}/result.json")
    if result is None or RESCORE:
        print(f"\n=== {key}: val and gold, prompt {name!r}")
        result = evalkit.evaluate_run(
            out, key, splits={"val": splits["val"], "gold": splits["gold"]},
            decode=lambda rows: (*decode_rows(spec, prompt, rows, "val" if rows is splits["val"] else "gold"), []),
            score=score, card=card(key, spec, name), meta={"candidate": key, "stage": "zero-shot", "prompt": name})
        upload(out, dest, f"zero-shot val and gold, prompt {name}")
    else:
        (out / "harness").mkdir(exist_ok=True)
        for split in ("val", "gold"):
            shutil.copy(hf_hub_download(OUT_REPO, f"{dest}/harness/{split}.jsonl", token=TOKEN),
                        out / "harness" / f"{split}.jsonl")
    row = fetch_json(f"{dest}/bakeoff.json") or {"candidate": key, "model_id": spec["model_id"], "prompt": name}
    for split in ("val", "gold"):
        hyps = evalkit.read_hyps(out / "harness" / f"{split}.jsonl")
        gate = bakeoffkit.zero_shot_report([r["text"] for r in splits[split]], [hyps[r["segment_id"]] for r in splits[split]])
        row[split] = {**result[split], "gate": gate}
    (out / "bakeoff.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
    upload(out, dest, "full val and gold in the verdict file")
    return row


for key in shortlisted():
    guarded(full_zero_shot, key)
"""

REPORT_NOTE = """
## The shortlist

Every candidate the rounds reached, read back from the hub: the sweep's numbers for all of them,
the training step and the verdict for the shortlist, and val and gold for those round 3 ran. The
summary goes to `<RUN_PREFIX>/summary.json`.
"""

REPORT = r"""
nan = float("nan")
found = sweeps()
rows = {k: fetch_json(f"{RUN_PREFIX}/{k}/bakeoff.json") or {} for k in CANDIDATES}
print(f"{'candidate':<16} {'prompt':<10} {'sweep':>7} {'Nepali':>6} {'val':>7} {'gold':>7} {'gold CER':>8} {'train x':>7} "
      f"{'total h':>7}  verdict")
for key, prompts in found.items():
    if not prompts:
        print(f"{key:<16} not swept")
        continue
    name = bakeoffkit.pick_prompt(prompts)
    f, r = prompts[name], rows[key]
    s, c, v = r.get("speed") or {}, r.get("cost") or {}, r.get("verdict")
    verdict = "-" if v is None else ("continue" if v["continue"] else "stop")
    print(f"{key:<16} {name:<10} {f['wer']:7.2f} {f['writes_nepali']:6.0%} {r.get('val', {}).get('wer', nan):7.2f} "
          f"{r.get('gold', {}).get('wer', nan):7.2f} {r.get('gold', {}).get('cer', nan):8.2f} "
          f"{s.get('train_x_realtime', nan):7.0f} {c.get('total_h', nan):7.1f}  {verdict}")
summary = {"run_prefix": RUN_PREFIX, "budget_h": BUDGET_H, "gate_share": bakeoffkit.GATE_SHARE,
           "sweep_clips": len(SWEEP_ROWS), "default_shortlist": bakeoffkit.shortlist(found),
           "continue": [k for k, r in rows.items() if (r.get("verdict") or {}).get("continue")],
           "sweeps": {k: {n: {x: y for x, y in p.items() if x != "examples"} for n, p in ps.items()} for k, ps in found.items()},
           "candidates": rows}
(OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
api.upload_file(path_or_fileobj=str(OUT_ROOT / "summary.json"), path_in_repo=f"{RUN_PREFIX}/summary.json",
                repo_id=OUT_REPO, commit_message=f"{RUN_PREFIX}: summary")
print("\ngo on to a student notebook:", ", ".join(summary["continue"]) or "none")
"""

cells = [
    md(INTRO + nbkit.SMOKE_NOTE),
    md("## Config"),
    code(CONFIG),
    md("## Setup"),
    nbkit.setup(
        'rapidfuzz duckdb soundfile librosa accelerate peft bitsandbytes "mistral-common[audio]>=1.12" -U "transformers>=5.13.0"'
    ),
    *nbkit.kits("ftkit", "evalkit", "sweep", "distill", "bakeoffkit"),
    code(DATA),
    md(SPECS_NOTE),
    code(SPECS),
    md(HF_NOTE),
    code("from transformers import AutoProcessor\n" + HF),
    md(OMNI_ENV_NOTE),
    code(OMNI_ENV),
    code(OMNI_SCRIPT),
    code(OMNI_KERNEL),
    md(RUN_NOTE),
    code(RUN),
    md(SWEEP_NOTE),
    code(SWEEP),
    md(SPEED_NOTE),
    code(SPEED),
    md(FULL_NOTE),
    code(FULL),
    md(REPORT_NOTE),
    code(REPORT),
]

NOTEBOOKS = {"09_SpeechLLM_Bakeoff.ipynb": cells}

if __name__ == "__main__":
    nbkit.write(NOTEBOOKS, OUT_DIR)
