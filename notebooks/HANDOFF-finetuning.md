# Handoff: fine-tuning ASR on the Nepanglish corpus

For the next agent picking this up. Last updated 2026-09-12, at the end of the session that ran
04c. Read it top to bottom before touching a notebook; each warning below cost time to find.

## Where things stand

| step | status |
|---|---|
| 20 h corpus on the private HF dataset `Sagyam/nepanglish-asr` | done; training set is D76-clean (HF commit `6e904fe`) |
| `03-asr-bakeoff.ipynb`: zero-shot scores on 706 gold clips | done; findings in its last cell |
| `04c-finetune-indic-transcribe.ipynb` (Flex) | **run 2026-09-12**; results below |
| `04a-finetune-whisper-turbo.ipynb` | written, **not run**; needs the loop retry first (next step 1) |
| `04b-finetune-omnilingual-ctc.ipynb` | written, **not run** |
| bake-off table with the fine-tuned models | **not done** (next step 3) |

All of this is on `master`: branch `finetune-04c-flex` was fast-forwarded into it on 2026-09-12.
`master` is ahead of `origin/master`; nothing has been pushed.

### 04c result (Indic-Transcribe-Flex, `ne`, mixed mode, A100 40 GB)

| | zero-shot (bake-off) | fine-tuned, greedy | fine-tuned + loop retry |
|---|---|---|---|
| gold folded WER | 18.2% (14.0–21.2) | 13.20% | **11.44%** |
| gold raw WER | | 16.15% | 14.41% |
| gold CER | 14.0% | 9.40% | 8.09% |
| gold looping clips | 4 | 7 | 0 |
| val folded WER | 15.06% (before training) | 7.60% | 7.60% (no val clip loops) |

- **Training.** 6 epochs, best at epoch 5: val by epoch 8.88 / 8.11 / 7.97 / 8.24 / 7.60 / 8.22.
  About 18 min of training, 21 ms per clip, ~630x realtime, ~90% GPU utilisation, 36.8 GiB.
- **Where the results are.** Drive, under `MyDrive/nepanglish-asr/indic-transcribe-flex-ft/`:
  - `best/`: bf16 weights, loadable with `IndicTranscribe`; keep private (licence);
  - `history.json`, `speed_check.json`, `config.json`;
  - `gold_metrics.json`: both scores and every retried clip's before/after text;
  - `hyps/indic-transcribe-flex-ft.jsonl`: the retry version, in the bake-off format;
  - `gold_per_clip.jsonl`: per-clip errors and loop flags, from the greedy run.
- **The loop retry.** A clip whose greedy output repeats a 3-word sequence 5 or more times is
  decoded again, alone, with repetition penalty 1.2, no repeated 6-token phrase, and a cap of 13
  tokens per second of audio. The code is `ftkit.is_loop` / `ftkit.RetryLoops` plus `retry_one`
  in 04c.
  - The trigger reads only the model's output, so the rule is usable in production; it is the
    same idea as Whisper's compression-ratio fallback.
  - The settings were fixed before scoring and never tuned. The idea came from looking at gold's
    loops, so **tune any change to it on val, never on gold.**
- **Comparisons must use the same decoder.** 18.2% → 11.44% mixes fine-tuning with the retry.
  Zero-shot Flex also looped on 4 gold clips, so give it the same retry before calling it the
  fine-tuning gain (next step 3).

### What the numbers mean (measured this session)

- **Error depends on the type of episode.** Gold WER (greedy):
  - tech reviews, 1 speaker (19 episodes, 12% of gold words): **3.5%**;
  - 2-speaker podcasts (15 episodes, 78%): **13.5%**;
  - the one 3-speaker podcast (`on_air_with_sanjay_796_…`): **22.2%**.

  The median episode's WER is 4.6%; the worst 10% of clips hold 46% of the errors.
- **The loops.** All 7 were in podcasts; 5 were stuck on fillers (`अँ.`, `उम्.`), which the
  references transcribe.
- **The gold/train overlap is by design, not a leak.** Checked on the rows that trained: 0
  shared clip ids, 0.00 s of shared audio, 0 identical transcripts. All 36 gold episodes also
  have train or val clips (D76). Gold WER is therefore **not speaker-held-out**; say so whenever
  you report it.
- **Val is 4 whole episodes held out from train.** It is small and noisy: two identical decodes
  differed by 0.3 points through bf16 numerics. It is still the only unseen-episode number.

## The Colab notebook the owner has open (it is still running)

When you arrive, the owner's Colab tab holds the **04c cells**, and its kernel is alive on an
A100 40 GB with:
- the fine-tuned epoch-5 model in `model`;
- `splits`, `store`, `score`, `transcribe`, `retry_one`;
- `texts`, the gold transcripts with the retry.

Drive is mounted at `/content/drive`.

- **Do not re-run the Train cell.** It retrains, and `save_best` overwrites `best/` on Drive.
- **The kernel's `/content/ft/ftkit.py` (md5 `daac1653…`) predates a formatting-only commit**
  (line wrapping in `speed_check`'s print, for ruff). Its behaviour is identical. Write the
  committed version before running anything new, and its md5 will then match
  `md5sum notebooks/src/ftkit.py`.
- **The HF token is cached on this VM** in `~/.cache/huggingface/token`, where Setup put it.
  - A *kernel restart* needs no owner click: run `os.kill(os.getpid(), 9)` in a cell, wait about
    15 s, re-run Config and Setup through the MCP.
  - A *new runtime* (a new VM) needs the owner to run the Setup cell by hand once. Secrets cannot
    be read from cells the MCP runs: "Secrets can only be fetched when running from the Colab UI".
- **To run 04a in this same runtime** (the cheapest path):
  1. restart the kernel;
  2. delete the 04c cells and add 04a's, read from `notebooks/04a-finetune-whisper-turbo.ipynb`;
  3. run Config, then Setup, then the rest.

  The GPU is freed by the restart, and the token and Drive carry over. Ask the owner first
  whether they want to keep the 04c cells' logs (*File → Save a copy in Drive*). The results
  themselves are already on Drive.

## Next steps, in order

1. **04a (Whisper-turbo): add the loop retry, then run it.** Zero-shot it looped on 271 of 706
   gold clips.
   - Either port 04c's `RetryLoops` wrapper, or use Whisper's own fallback in HF `generate`:
     `temperature=(0.0, 0.2, 0.4, …)`, `compression_ratio_threshold` (~1.35 on tokens),
     `logprob_threshold`.
   - For a length cap, measure the densest train label in *Whisper* tokens per second first; 04c's
     13/s is in Flex tokens.
   - Record greedy-only and with-retry gold numbers from the same run, as 04c's Gold cell does.
   - The GPU log-mel assert now runs with autocast disabled; if it fails, fix it before training.
2. **04b (Omnilingual CTC).** Unchanged from the original plan; see its section below. CTC cannot
   loop the way autoregressive decoding does, so it needs no retry.
3. **Bake-off table.** The owner was asked whether the old bake-off runtime (zero-shot
   transcripts in `/content/bakeoff/hyps`) is still alive, and never answered. **Ask again.**
   - If it's gone, re-run 03's model cells. 03 still decodes Flex one clip per call (~1.4 s per
     clip, ~17 min for gold); batching it the way 04c does would be much faster.
   - Give zero-shot Flex the same loop retry for a like-for-like row.
   - The Score cell now mounts Drive and also reads `MyDrive/nepanglish-asr/*/hyps/*.jsonl`, so
     the fine-tuned transcripts are picked up automatically.
   - Report overall and by CMI tier; record it in the Findings cell via `build_bakeoff.py`.
4. **Owner's data questions.** The analysis above answers these in part; the owner asked
   directly, so measure before advising.
   - **Will more data help?** A learning curve answers it: retrain on 25% and 50% of the train
     *episodes*, then score val and gold. That is about 25 min of A100 time. Val is the cleaner
     signal: gold also loses familiar voices when episodes are removed.
   - **New voices.** The owner plans to add their own voice, friends' voices and smaller
     YouTubers. Advice given:
     - to make the score *realistic*, those voices must go into a **held-out test set**, never
       into training;
     - it must be **conversation, not reading**: solo scripted speech scores 3.5%, conversation
       13.5–22%;
     - use enough speakers (rule of thumb: 8 or more across several conversations);
     - label it with the same pipeline;
     - more training data should be multi-speaker conversation; more tech reviews won't move
       the number.
5. **Later, each measured on val first:** an LR sweep for the winner; beam search against greedy
   (loops are mainly a greedy problem); a KenLM decoder for the CTC model; whether Omnilingual
   learns to write English in Latin script.

## Driving Colab from Claude Code

The official Colab MCP server is in `.mcp.json`, pinned to `googlecolab/colab-mcp@v1.0.2`.

- **Connecting.**
  - Call `open_colab_browser_connection` first; the editing tools then appear, and you load their
    schemas with ToolSearch.
  - It edits and runs whatever notebook the owner has open, on that notebook's runtime.
  - If a tool reports "Unknown tool", reconnect.
- **Long cells.** `run_code_cell` moves to a background task after 120 s and notifies you when
  the cell finishes. For progress in between, use `get_cells` on **one index** with
  `includeOutputs=true`, paced by a background `sleep`. A full-range `get_cells` dumps the 22 KB
  ftkit cell.
- **Image outputs overflow the tool result.** For numbers, add a temporary cell that prints text,
  read it, then delete the cell. Leave the notebook matching the built one.
- **The kernel is single-threaded.** A cell queued behind a long run won't start until the run
  finishes.

## Before running anything

1. **Edit the sources, not the notebooks.** The notebooks are generated from `notebooks/src/`:
   - `ftkit.py`: the shared kit, embedded in all three 04 notebooks via `%%writefile`;
   - `build_finetune.py`: generates the 04 notebooks;
   - `build_bakeoff.py`: generates 03.

   Edit there, then run `python notebooks/src/build_finetune.py` (or `build_bakeoff.py`). If you
   fix a cell live in Colab, port the fix back to the source and rebuild.
2. **Check ftkit after writing it to Colab.** Compare `md5sum notebooks/src/ftkit.py` against
   the md5 of `/content/ft/ftkit.py`, then `importlib.reload(ftkit)` in a kernel that already
   imported it. The copies have matched every time so far.
3. **Run the speed check before every run.** Every notebook has one before its Train cell (in
   04b it is inside the script, with `speed_check_only`). It times 10 real training micro-batches,
   then one full val pass (which gives the val WER before training), and projects the run. If
   utilisation is low or the projection is long, stop and fix it first.

## Traps found so far (all verified)

- **The batch-size probe must keep the gradients allocated.** Freeing them between trials
  measured activations without the 4.8 GB gradient buffer and picked 19 clips, which ran out of
  memory once training accumulated. `probe_max_items(step, lo, hi, params)` now holds the buffer
  (15 clips); the notebook's `probe_step` must not touch the gradients.
- **A crash can leave GPU memory unfreeable.** After the out-of-memory error, `del` and `gc`
  left 37 GiB referenced. Restart the kernel instead of hunting for the reference.
- **The Colab kernel lacks `rapidfuzz`.** The Setup cell now pip-installs it.
- **Setup sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** before torch is imported; it
  only takes effect in a fresh kernel.
- **Speed.** The bake-off's 1.4 s per clip was Flex decoded one clip per call. Batched
  `generate` is 21x faster (169 ms per clip at batch 16), with identical folded text on 16 val
  clips.
- **One episode id starts with an invisible U+FE0F:**
  `️building_a_career_after_motherhood_the_challenges`. Copy it; don't retype it.
- **The `max_new_tokens … max_length` warning from transformers is harmless noise.**
- **fold.py from the HF cache fails.** `normalize.py` resolves its config through the cache's
  symlinks, so `harness_scorer` copies `harness/` to a real directory first.

## What each notebook does

All three share `ftkit.py`:
- audio in RAM as int16;
- duration-bucketed micro-batches padded to 1 s;
- the probe above, with training at 90% of the probe result;
- gradient accumulation to about 12 min of audio per step;
- bf16 autocast, TF32, fused AdamW, and no host sync per micro-batch;
- val WER after every epoch, with the best weights kept;
- gold scored at the end and written in the bake-off's cache format.

Peak LR is 1e-5 in all three, on purpose, so the comparison is between models. Sweep LR only
after all three have a baseline, and select on val, never on gold.

### 04a — Whisper-large-v3-turbo

- Full fine-tune, 5 epochs, language token `ne`, Whisper's own SpecAugment.
- Log-mel is computed batched on the GPU, in fp32 even inside autocast. An assert compares it
  with `WhisperFeatureExtractor` before training.
- **About 60% padding waste is unavoidable.** The encoder always takes 30 s and the median clip
  is 12 s. The longest clip in the data is 20.2 s, so nothing is cut.
- One label exceeds the 448-token decoder window and is dropped.
- Its eval `generate` passes `use_cache=True` explicitly, because training switches the cache
  off in the model config.

### 04b — Omnilingual CTC-1B v2

- **It cannot run in the Colab kernel.** `omnilingual-asr` requires Python <= 3.12, and
  `fairseq2` 0.6 pins `torch==2.8.0`, numpy 1.x and `huggingface_hub` 0.x. The notebook builds a
  `uv` Python 3.12 venv at `/content/omni-env`, writes `train_omni.py`, and runs it as a
  subprocess. The subprocess inherits `HF_TOKEN` from the kernel.
- The recipe follows Meta's fairseq2 recipe:
  - frozen conv feature extractor;
  - per-clip waveform layer norm;
  - the model's built-in masker;
  - summed CTC normalised per clip;
  - tri-stage LR;
  - AdamW (0.9, 0.98).
- **Untested:**
  - the training forward `model(wav, BatchLayout, targets, BatchLayout)`;
  - the inference forward's return value;
  - `tokenizer.vocab_info.unk_idx`.

  If a signature errors, read `fairseq2/models/wav2vec2/asr/model.py` in the venv. Set
  `speed_check_only: True` in `CONFIG` to test everything up to training cheaply.
- The best weights are saved as a bf16 `state_dict` (`best/model.pt`). To reload, `load_model`
  then `load_state_dict`.

### 04c — Indic-Transcribe-Flex (done; how it works, for reference)

- **The release is inference-only.** `forward(labels=...)` raises, so the notebook computes
  cross-entropy on teacher-forced logits: the input is the 10-token `ne`/mixed prompt plus the
  target, and loss is taken on the target and EOS.
- **Targets.** They are `tk.multi.encode(text)` offset by 1,152, plus EOS. `।`, Devanagari
  digits, ZWJ/ZWNJ and `—` are mapped to `.`, Latin digits, nothing and `-`, which `fold.py`
  scores as identical. The model therefore writes `.` where the references write `।`; map it back
  for display if it matters.
- **Regularisation.** The port has no dropout, so SpecAugment (Canary's defaults) is applied on
  the GPU. Flex's featurizer disables autocast itself, so features are fp32.
- **Batched `generate` works.** It reuses its cache, and matched one-clip-per-call decoding.
- **Licence: Indic Open Model License v1.0.** The fine-tuned weights are a derivative. Keep them
  private. Giving them to anyone passes the licence on, and hosting them as a service for others
  needs Bodhan AI's sign-off.

## Rules that apply to this work

- **Gold never trains** (decision D76 in `docs/decisions.md`). The owner declined episode- and
  voice-level gold for now; don't propose it again.
- **Measure, don't speculate.** Back every claim with a number from a run: what broke, why one
  model beats another, what the GPU was doing. Say plainly when an idea didn't work.
- **References are lightly edited LLM fusion output.** A WER is a distance from that fusion, not
  an accuracy.
- **The dataset is private** because the audio is YouTube content. Never make the HF repo, any
  audio, or the Flex-derived weights public.
- **Git.** Commit only when the owner asks, and on a new branch cut from `master`. Merge into
  `master` only when asked; the owner merged `finetune-04c-flex`. Push only when asked.
