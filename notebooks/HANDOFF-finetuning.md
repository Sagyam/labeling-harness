# Handoff: fine-tuning ASR on the Nepanglish corpus

For the next agent picking this up. Written 2026-09-12. Read it top to bottom before touching a
notebook; each warning below cost time to find.

## Where things stand

| step | status |
|---|---|
| 20 h corpus exported to the private HF dataset `Sagyam/nepanglish-asr` | done; training set is D76-clean (HF commit `6e904fe`) |
| `03-asr-bakeoff.ipynb`: zero-shot scores on 706 gold clips | **done**; findings are in its last cell |
| `04a-finetune-whisper-turbo.ipynb` | written, **never run** |
| `04b-finetune-omnilingual-ctc.ipynb` | written, **never run** |
| `04c-finetune-indic-transcribe.ipynb` | written, **never run** |

Bake-off results: folded WER on gold, measured as distance from the fused reference, with 95%
episode-bootstrap intervals.

| model | WER | CER | note |
|---|---|---|---|
| Indic-Transcribe-Flex, `ne`, mixed mode | **18.2%** (14.0–21.2) | 14.0% | flat across code-mixing tiers; writes Latin English like the labels |
| Omnilingual CTC-1B v2 | 41.3% (33.5–47.3) | 43.7% | transliterates English into Devanagari (12% Latin tokens vs 36%) |
| Whisper-large-v3-turbo, `ne` / `hi` | 123% / 121% | ~100% | repetition loops on 271 / 283 of 706 clips |
| Scribe / Gemini / MAI | 12–13% | | fused into the reference, so flattered |

**Your job.** Run 04a, 04b and 04c on a Colab A100, fix what breaks, and then re-run the
bake-off's **Score** cell so the fine-tuned models appear in the same table as the zero-shot ones.
The owner asked for all three; do not drop one without asking.

## Before running anything

1. **The owner sets up the runtime.** You can't change the runtime type or add secrets. Ask the
   owner for:
   - an **A100** runtime (*Runtime → Change runtime type*);
   - the Colab secret **`HF_TOKEN`** with notebook access switched on;
   - acceptance of the gated terms on `bodhan-ai/indic-transcribe-flex` (already done on their
     account).
2. **Google Drive.** Each notebook has `USE_DRIVE = True`, and the mount opens a consent popup
   that only the owner can click. Ask them to click it, or set `USE_DRIVE = False`. With it off,
   outputs live in `/content` and are lost when the runtime is deleted.
3. **Edit the sources, not the notebooks.** The notebooks are generated from `notebooks/src/`:
   - `ftkit.py`: shared kit, embedded in all three 04 notebooks via `%%writefile`;
   - `build_finetune.py`: generates the 04 notebooks;
   - `build_bakeoff.py`: generates 03.

   Edit there, then run `python notebooks/src/build_finetune.py` (or `build_bakeoff.py`). Editing
   one notebook's copy of `ftkit` by hand makes the three drift apart. If you fix a cell live in
   Colab, port the fix back to the source and rebuild.

## Driving Colab from Claude Code

The official Colab MCP server is in `.mcp.json`, pinned to `googlecolab/colab-mcp@v1.0.2`.

- **Connect first.** Call `open_colab_browser_connection`; the editing tools then appear, and you
  load their schemas with ToolSearch. If a tool reports "Unknown tool", reconnect.
- **The owner's notebook is the target.** The MCP edits whatever notebook they have open.
  - To run a 04 notebook there, add its cells with `add_code_cell` / `add_text_cell` (read them
    from the `.ipynb`).
  - Or ask the owner to open the file from the repo.
- **Long cells.** `run_code_cell` waits up to the `timeout` in `.mcp.json`, now 3,600,000 ms. If
  a call still times out, the cell keeps running in Colab.
  - Poll with `get_cells(cellIndexStart=i, cellIndexEnd=i, includeOutputs=true)`.
  - Pace the polls with a background `sleep` in Bash (`run_in_background`), not tight loops.
- **Image outputs overflow the tool result.** For numbers, add a temporary cell that prints text,
  read it, then delete the cell.
- **The kernel is single-threaded.** A cell queued behind a long run won't start until the run
  finishes.

## What each notebook does, and what will probably break

All three share `ftkit.py`:
- the dataset in RAM as int16;
- duration-bucketed micro-batches padded to 1 s (about 6% padding waste);
- a probe that finds the largest micro-batch fitting forward+backward on the longest clip, with
  AdamW state already allocated; training uses 90% of it;
- gradient accumulation to about 12 min of audio per step;
- bf16 autocast, TF32 and fused AdamW;
- val WER after every epoch, with the best weights kept;
- gold scored at the end and written to `hyps/<run>.jsonl` in the bake-off's cache format.

Every 10 steps the log prints throughput (× realtime), padding waste, GPU utilisation and memory.
**Check these in the first minutes.** Below about 70% utilisation, look at `workers`, then at the
probe result.

Peak LR is 1e-5 in all three, on purpose, so the comparison is between models. Sweep LR only
after all three have a baseline, and select on val, never on gold.

### 04a — Whisper-large-v3-turbo

- Full fine-tune, 5 epochs, language token `ne`, Whisper's own SpecAugment.
- Log-mel is computed batched on the GPU. An assert compares it to `WhisperFeatureExtractor`
  before training; if that fails, fix it before anything else.
- **About 60% padding waste is unavoidable.** The encoder always takes 30 s and the median clip
  is 12 s. That is Whisper, not a bug.
- One label exceeds the 448-token decoder window and is dropped.
- **What to watch:** repetition loops. `evaluate` reports a `loops` count (clips with a trigram
  repeated 5 or more times); zero-shot it was 271 of 706. If loops survive fine-tuning, try
  `no_repeat_ngram_size` or a compression-ratio fallback at decode time. Change decoding only, and
  report it.

### 04b — Omnilingual CTC-1B v2

- **It cannot run in the Colab kernel.** `omnilingual-asr` requires Python <= 3.12, and
  `fairseq2` 0.6 pins `torch==2.8.0`, numpy 1.x and `huggingface_hub` 0.x. The notebook builds a
  `uv` Python 3.12 venv at `/content/omni-env`, writes `train_omni.py`, and runs it as a
  subprocess. The zero-shot bake-off ran fine this way.
- The recipe choices come from Meta's fairseq2 recipe (`workflows/recipes/wav2vec2/asr/`):
  - frozen conv feature extractor;
  - per-clip waveform layer norm;
  - the model's built-in masker;
  - summed CTC normalised per clip;
  - tri-stage LR;
  - AdamW with betas (0.9, 0.98).
- The model and tokenizer come from `ASRInferencePipeline(model_card=..., dtype=torch.float32)`,
  the same path that already worked.
- **Untested:**
  - the training forward `model(wav, BatchLayout, targets, BatchLayout)`, taken from the recipe's
    `criterion.py` and fairseq2's `Wav2Vec2AsrModel.forward`;
  - the probe;
  - `tokenizer.vocab_info.unk_idx`.
- If the forward signature errors, read `fairseq2/models/wav2vec2/asr/model.py` in the venv.
- The best weights are saved as a bf16 `state_dict` (`best/model.pt`), not a fairseq2 checkpoint.
  To reload, `load_model(card)` then `load_state_dict`.

### 04c — Indic-Transcribe-Flex

- **The release is inference-only.** `forward(labels=...)` raises `NotImplementedError`. The
  notebook computes cross-entropy itself on teacher-forced logits: the input is the 10-token
  `ne`/mixed prompt plus the target, and loss is taken on the target and EOS only.
- **The tokenizer has no `encode`.** Targets are `tk.multi.encode(text)` offset by 1,152, the
  number of special tokens, plus EOS. Verified: 5,863 of 5,864 train/val labels round-trip exactly
  through the model's own `decode`.
- **The labels are normalised first.** `।`, Devanagari digits, ZWJ/ZWNJ and `—` are unknown to
  its vocabulary; without this, 76% of labels would carry an unknown token. They map to `.`, Latin
  digits, nothing and `-`, which `fold.py` scores as identical.
- **The port has no dropout.** SpecAugment (Canary's defaults) is applied to the features on the
  GPU instead.
- **Untested:** batched `model.generate` for eval. Zero-shot ran one clip at a time.
  - If batched outputs look wrong, check a clip against `asr.transcribe(path, lang="ne",
    mode="mixed")`.
  - If they still disagree, fall back to per-clip decoding for val. It's slow: about 1.5 s a clip,
    so about 10 min per val pass.
- **Licence: Indic Open Model License v1.0.** The fine-tuned weights are a derivative. Keep them
  private. Giving them to anyone passes the licence on, and hosting them as a service for others
  needs Bodhan AI's sign-off.

## Rules that apply to this work

- **Gold never trains** (decision D76 in `docs/decisions.md`).
  - The exporter separates clips by id and by audio, and `ftkit.load_splits` re-checks the ids.
  - Gold is *not* speaker-held-out: every gold voice also trains. Say so whenever you report a
    gold number.
  - The owner declined episode- and voice-level gold for now; don't propose it again.
- **Measure, don't speculate.** Back every claim with a number from a run: what broke, why one
  model beats another, what the GPU was doing. Say plainly when an idea didn't work.
- **References are lightly edited LLM fusion output.** A WER is a distance from that fusion, not
  an accuracy.
- **The dataset is private** because the audio is YouTube content. Never make the HF repo or any
  audio public.
- **Git.** Commit only when the owner asks, and on a branch. The work so far is on
  `d76-gold-audio-separation`, unmerged.

## After the three runs

1. Re-run the bake-off **Score** cell, which picks up `/content/bakeoff/hyps/*-ft.jsonl`, and
   report the fine-tuned models against zero-shot, overall and by CMI tier.
2. Record the outcome in the bake-off's Findings cell, via `notebooks/src/build_bakeoff.py`.
3. Worth trying next, in this order, each measured on val first:
   - an LR sweep for the winner;
   - a KenLM decoder for the CTC model;
   - for Omnilingual, whether it learns to write English in Latin script.
