# ASR Roadmap & Fine-Tuning Findings

Roadmap and consolidated technical findings for fine-tuning ASR on the Nepanglish corpus. Replaces and supersedes `notebooks/HANDOFF-finetuning.md`.

---

## 1. Execution Roadmap

We execute work in this strict order:

### Phase 1: Fix the Best Performing Model (Indic-Transcribe-Flex)
Focus exclusively on the winning architecture before touching secondary models or collecting more data.
1. **Decoder Search & Loop Elimination (No retraining needed initially):**
   - The fine-tuned weights (`best/`) are saved on Drive (`MyDrive/nepanglish-asr/indic-transcribe-flex-ft/best/`).
   - Replace the post-hoc greedy + single-clip loop-retry fallback with native robust decoding:
     - **Beam search** (`num_beams=2` and `num_beams=4`).
     - **Native repetition penalty** (`repetition_penalty=1.1` to `1.2`).
     - **N-gram blocking** (`no_repeat_ngram_size=6`).
     - **Dynamic audio length cap** ($\le 13$ tokens/s of audio) to prevent runaway generation during trailing silence.
   - Goal: Push Gold folded WER below 10.5% and CER below 8.0% without looping on fillers (`अँ`, `उम्`) or reduplications (`mixed-mixed`).
   - **Done 2026-09-13.** Gold was treated as a dev set. The experiment code was discarded after
     the run, and its per-clip outputs are on Drive (`flex-decoding/`).
     - **Outcome: post-processing is not worth it beyond the cheap parts. The standard decoder is
       greedy + length cap + loop retry,** for every model, run and report from here on.
     - **Best decoder found: `beam4+cap+rp1.1+retry`.** Gold WER 11.51 → 11.10% (Δ −0.41, 95%
       episode interval [−0.60, −0.19]), CER 8.11 → 7.85%. Val 7.57 → 7.27%.
       - **Rejected.** The gain is real but small, and it costs about 4x greedy's decode time.
       - **Optimistic.** The arm was chosen from ~20 on gold; val, which played no part in the
         choice, shows −0.30.
       - **At the noise level.** Identical training runs already differ by ~0.3 on val.
       - The experiment's decision rule picked it because the rule had no cost term; the owner
         overrode it on cost.
     - **The standard is free.** `greedy+cap+retry` equals 04c's decoder: 11.53 vs 11.51, Δ +0.02
       [+0.00, +0.06]. The cap only stops a looping clip at 13 tokens/s of audio instead of
       300 tokens.
     - **The cap is still not in the code; add it with the next notebook.** Phase 2 built it and
       verified it, then its code was discarded with the experiment. It is a logits processor
       passed to `generate`:
       - Each clip's cap is `min(300, ceil(13 × seconds))` tokens.
       - Once a row's generated length (`input_ids.shape[1]` minus the 10-token prompt) reaches its
         cap, set that row's scores to −inf except end-of-text.
       - Under beam search, row `r` belongs to clip `r // num_beams`.
       - The retry already caps its single-clip decode with `max_new_tokens`.
       - A plain class with `__call__(input_ids, scores)` in a `transformers.LogitsProcessorList`
         works with the Flex port's `generate`. On the A100, every clip of a batch stopped at
         exactly its own cap.
       - On the 2026-09-12 weights it reproduced gold 11.53% and val 7.57%.
     - **What earns its keep is the retry.** Greedy 13.31 → 11.51 by decoding 8 looping clips a
       second time.
     - **Global anti-repetition makes greedy worse:** repetition penalty 1.2 +1.11 points (202
       clips worse, 81 better), n-gram blocking +0.51. 25% of gold references repeat a 6-token
       span, which blocking can never reproduce.
     - **Beam alone does not replace the retry:** beam 4 still loops on 2 gold clips.
     - **Against the goal:** WER below 10.5% is out of reach by decoding; more is only
       available from training or data (Phases 1.2–2).
     - **Beam 4 is the footnote:** about 0.3 points at ~4x decode time, if a final report ever
       needs the best achievable number. Report it beside the standard, never instead of it.
2. **Training Hyperparameter Polish (If retrained):**
   - Sweep peak LR ($5 \times 10^{-6}$ vs $1 \times 10^{-5}$) and add label smoothing ($0.1$) to prevent over-confident predictions on repetitive filler tokens.

### Phase 2: Measure if More Data Helps (The Learning Curve)
Determine empirically whether data volume or distribution diversity is the bottleneck:
1. Retrain Flex on **25%** of the training episodes (~4.5 h).
2. Retrain Flex on **50%** of the training episodes (~9 h).
3. Compare against the 100% (18.3 h) baseline on the held-out **Val set** (4 unseen episodes).
   - If Val WER scales steeply ($12\% \to 9.5\% \to 7.6\%$), volume helps.
   - If Val WER is flat ($8.1\% \to 7.8\% \to 7.6\%$), the current distribution is saturated.
4. **Stopped 2026-09-13 with one draw's 25% and 50% points.** The owner judged the answer clear:
   **more hours of the same distribution do not help.** Go to Phase 3.
   - **Design.** The experiment code was discarded after the run at the owner's request.
     - Whole episodes are drawn per genre (podcast / tech review), each to the nearest fraction
       of its hours in a seeded order. Within a draw the subsets nest: 25 ⊂ 50 ⊂ 75 ⊂ 100.
     - Every point gets the full run's optimizer steps: its subset repeats within each epoch.
     - Each run keeps its best epoch by val.
     - Decision rule, fixed before the runs: a val gain from 50% to 100% of ≥ 1.0 means volume
       helps; < 0.5 means saturated.
     - Draw 1 and both 75% points were not run.

     | point | hours | val WER | gold WER |
     |---|---|---|---|
     | 25%, draw 0 | 4.6 | 7.61 | 12.07 |
     | 50%, draw 0 | 9.3 | 7.45 | 11.84 |
     | 100% (2026-09-12 weights) | 18.3 | 7.57 | 11.53 |

   - **Val is flat**, within the ~0.3 run-to-run noise. The rule reads −0.12 for 50%→100%:
     saturated.
   - **Gold falls ~0.25 per doubling.** Part of that gain is gold episodes entering training,
     which new episodes would not give.
   - **Paired gold test, 25%→50%.**
     - Clips whose episode stays unseen gain +0.24 [−0.29, +0.77] from doubling the hours.
     - Clips whose episode joins training gain +0.87 [+0.37, +1.42].
     - So an episode's own speakers are worth +0.62 [−0.16, +1.39] beyond the hours. That points
       at Phase 3 (new speakers), but the interval includes zero.
   - **Caveats.**
     - One draw only, so the spread between draws is unmeasured.
     - Val is 83% one podcast.
     - The curve is flat for this distribution (27 recurring voices, one tech-review host). That is
       why new gold must be new speakers.

### Phase 3: Clean Gold Pot + Reallocate Current Gold to Train
Fix episode/speaker leakage by constructing an uncompromised test benchmark:
1. **Record & Harvest New Gold (~2 hours):**
   - Record author and friends' voices + harvest fresh internet creators.
   - Strict criteria: brand new speakers, new acoustic environments, new microphones, and spontaneous conversational topics.
2. **Move Current Gold into Train:**
   - Once the new Gold set is verified, move all 706 clips of the current Gold set into the training pot using `set_segment_pot`.
   - This recycles ~2.3 h of human-verified audio into training data.
   - The new benchmark becomes **100% speaker-held-out, episode-held-out, and microphone-held-out**.

### Phase 4: Sanity Checks Against Popular Benchmarks
- Benchmark the optimized Flex model against standard open Nepali/Indic ASR baselines and datasets.
- Report folded and raw WER side-by-side with 95% bootstrap confidence intervals.

Whisper-turbo and Omnilingual CTC will not be fine-tuned further; their notebooks (`04a`, `04b`)
and the bake-off notebook (`03`) were removed 2026-09-14.

---

## 2. Where Things Stand (Benchmark Snapshot)

### Models Scored on Gold (706 clips, 2.3 h)

Folded WER is under `fold-v1` unless the column says otherwise. `fold-v2` (D84, 2026-09-15) also
drops fillers and folds numbers, contractions and colloquial Nepali; every system gains 1.7–2.3
points, and the fine-tuned Flex no longer leads Gemini. The fold-v2 column was measured against
`exports/gold/gold.jsonl`, with the recognisers' text read from the harness. It gives Scribe /
Gemini / MAI 13.11 / 11.96 / 11.99 under fold-v1, not the bake-off's figures below, which came
from another copy of their text. Raw WER does not depend on the fold.

| Model | Setup | Folded WER % | Folded WER % (fold-v2) | Raw WER % | CER % | Loops | Notes |
|---|---|---|---|---|---|---|---|
| **Scribe v2** | Cloud API (in ref) | 12.73% | 11.40% | 20.14% | 9.93% | 0 | Commercial; fused into reference |
| **Gemini 3.8 Flash** | Cloud API (in ref) | 12.03% | 9.66% | 20.64% | 10.69% | 0 | Commercial; fused into reference |
| **MAI Transcribe 2** | Cloud API (in ref) | 12.47% | 10.21% | 20.98% | 11.38% | 0 | Commercial; fused into reference |
| **Indic-Transcribe-Flex** | Zero-shot | 18.2% | — | — | 14.0% | 4 | Baseline |
| **Indic-Transcribe-Flex** | Fine-tuned, greedy | 13.20% | — | 16.15% | 9.40% | 7 | 6 epochs (best ep 5) |
| **Indic-Transcribe-Flex** | **Fine-tuned + retry** | **11.44%** | — | **14.41%** | **8.09%** | **0** | **Best model; 04c's decoder** |
| **Indic-Transcribe-Flex** | Fine-tuned, greedy + cap + retry (reloaded weights) | 11.53% | **9.65%** (val 5.59%, CER 7.62%) | 14.52% | 8.07% | 0 | **Standard decoder from Phase 1** (reloaded bf16 weights: 04c's decoder gives 11.51% on them) |
| **Indic-Transcribe-Flex** | Fine-tuned, beam 4 + cap + rp1.1 + retry | 11.10% | — | 14.01% | 7.85% | 0 | Not adopted: ~4x decode time; chosen on gold (dev score) |
| **Whisper-large-v3-turbo** | Zero-shot | 123% | — | — | — | 271 | Unusable zero-shot |
| **Whisper-large-v3-turbo** | Fine-tuned, greedy | 14.62% | — | 17.72% | 8.68% | 0 | 5 epochs (still improving) |
| **Omnilingual CTC-1B v2**| Fine-tuned (partial)| ~16.6% (val)| — | — | — | 0 | Stopped at epoch 6 |

Every model from here on should also go to the harness's **Models** page (D83): 04c writes
`OUT/harness/` (card plus gold/val transcripts), which is copied to `data/models/asr/<slug>/`. The
page reproduces this table's numbers and lists each model's worst clips with audio and the diff.

### Where Model Weights and Artefacts Live (Google Drive)
All trained weights and evaluation logs persist on Google Drive under `MyDrive/nepanglish-asr/`:
- **Flex:** `indic-transcribe-flex-ft/`
  - `best/`: bf16 weights loadable with `IndicTranscribe` (Private — Indic Open Model License v1.0).
  - `gold_metrics.json`: complete scores, greedy vs. retry metrics, and before/after text of all retried clips.
  - `hyps/indic-transcribe-flex-ft.jsonl`: cached hypotheses in bake-off format.
  - `eval/{val,gold}.jsonl`, `eval/metrics.json`: the same weights under the standard decoder
    (greedy + cap + retry), from the Phase 2 run. The files are per-clip text in bake-off format.
- **Diarization:** `diarization/diarization.json`. It holds pyannote community-1 for all 42
  episodes: overlapping and exclusive turns, and per-speaker embeddings.
- **Flex learning curve (Phase 2):** `flex-learning-curve/f{25,50}-d0/`
  - `best/` weights, `history.json`, `subset.json` (the episodes it trained on), `eval/`.
  - `f75-d0/` holds only a stopped run's `subset.json` and log; it has no weights and no `eval/`.
- **Flex decoder search (Phase 1):** `flex-decoding/`
  - `arms/<gold|val>/<arm>.jsonl`: per-clip text and error counts for every decoder (standard: `greedy+cap+retry`; best but not adopted: `beam4+cap+rp1.1+retry`).
  - `summary.json`: the results table, paired deltas and noise floor. Its `pick` field holds the rule's choice, which the owner overrode.
- **Whisper-turbo:** `whisper-turbo-ft/`
  - `best/`: bf16 weights, tokenizer, and feature extractor loadable with `WhisperForConditionalGeneration`.
  - `gold_metrics.json`, `history.json`, `hyps/whisper-turbo-ft.jsonl`.
- **Omnilingual CTC:** `omni-ctc-1b-v2-ft/`
  - `history.json`, `speed_check.json` (partial run logs).

---

## 3. Key Findings & Diagnostic Insights

### The Ground Truth Circularity Paradox
- The references are an LLM fusion of Scribe, Gemini, and MAI. Only 3 of 1,170 verified labels were ever edited; 82% of clips were screened.
- The commercial models' ~12% WER is their distance from their own consensus.
- **Flex at 11.44% WER and 8.09% CER beat the commercial engines on their own consensus benchmark.** In acoustic precision (CER), Flex has an ~18–29% relative error reduction over the commercial APIs.

### Error Skew & Long-Tail Distribution
- **By Genre/Episode Type:**
  - 1-speaker tech reviews (19 episodes): **3.5% WER** (near-human parity).
  - 2-speaker podcasts (15 episodes): **13.5% WER**.
  - 3-speaker podcast (`on_air_with_sanjay_796`): **22.2% WER**.
- **Long Tail:** The median episode WER is **4.6%**. The worst 10% of clips hold **46% of all errors**.

### Crosstalk explains the podcast errors; speaking rate and CMI do not (2026-09-13)
- **Method.**
  - Diarization with overlap detection on all 42 whole episodes: pyannote
    `speaker-diarization-community-1`, at the declared speaker count, 17 min on an A100.
  - Its output is on Drive at `diarization/diarization.json`.
  - Clip-level features for all 1,109 gold and val clips, joined to Flex's per-clip errors
    (standard decoder, 2026-09-12 weights). The analysis code was discarded.
- **Speaker attribution is reliable except where it matters most.**
  - pyannote and the EDA's ECAPA voice prints are independent systems. They agree on who is
    talking in 97.3% of 41,285 three-second windows in multi-speaker episodes.
  - They agree on 99.2% of the windows pyannote calls single-speaker (82% of windows).
  - The disagreements sit in the other 18%: turn changes and overlap.
- **Crosstalk is common inside clips.**
  - In multi-speaker episodes, 25% of clips contain some overlap and 27% have two speakers
    talking ≥ 0.5 s each.
  - Overlap is 2.1% of clip time overall and up to 10% in `on_air_with_sanjay_796`, the
    3-speaker episode.

  | overlap share of clip (podcasts) | clips | WER % [episode 95% CI] | share of errors |
  |---|---|---|---|
  | none | 747 | 9.01 [7.42, 11.53] | 50% |
  | 0–5% | 143 | 12.59 [10.23, 16.56] | 24% |
  | 5–15% | 77 | 17.81 [14.21, 20.18] | 19% |
  | > 15% | 26 | 24.29 [16.78, 27.91] | 8% |

- **Within an episode** (Poisson with episode fixed effects and episode-clustered SEs), the error
  rate ratios are:
  - **1.49 [1.32, 1.69] per 10 points of overlap share;**
  - 1.22 [1.00, 1.50] for a clip with two speakers;
  - 1.28 [1.15, 1.41] per 10 points of filler share;
  - speaking rate 1.01 [0.93, 1.11] and CMI 1.06 [0.98, 1.14], i.e. no effect. That matches the
    EDA.

  Together the factors explain 14–17% of clip-level deviance.
- **What overlap does:** deletions go from 1.6 to 8.3 per 100 words and substitutions from 6.0
  to 12.9. Near-miss spellings stay flat (~1.5), so these are real misrecognitions and dropped
  words, not scoring artefacts.
- **Counterfactual:** if overlap clips had the no-overlap error rate, gold would be ~8.8% instead
  of 11.53%, and val 6.9% instead of 7.57%. That is the largest lever measured so far.
- **Not everything.** Across the 16 podcast episodes, overlap and WER correlate at only Spearman
  0.39. `ep_447` is at 20.7% WER with 1.8% overlap, so episode- or speaker-level effects remain.
- **Caveats.**
  - The overlap detector has not been checked by ear.
  - In overlap the fused reference is at its least reliable: whose words, and are backchannels
    kept? Some "deletions" may be reference choices.
  - Single-host episodes were diarized with one speaker, so they have no overlap by
    construction.

### CPU inference is realtime in bf16; dynamic int8 breaks Flex (2026-09-15)

- **Setup.** Base Flex, which has the same architecture and speed as the fine-tune. It ran on the
  owner's Ryzen 7 7700X with 8 threads, plain PyTorch and the KV cache, over 10 gold clips (117 s,
  2.5–20 s each). The experiment code was discarded after the run.

  | variant | RTF | ms/token | WER (base model) |
  |---|---|---|---|
  | fp32 | 0.41 | 54 | 21.8% |
  | **bf16** | **0.18** | **23** | **21.8%** (8/10 texts identical to fp32) |
  | dynamic int8, per-tensor | 0.11 | 15 | 96.7%, 4 loops |
  | dynamic int8, per-channel | 0.11 | 15 | 112.5%, 2 loops |

- **Decoding is the cost.** The encoder takes ~0.015 s per second of audio. Nepanglish runs about
  8 tokens per second of speech, and each token re-reads the 24 decoder layers, which is
  memory-bound.
- **bf16 is fast on this CPU.** Zen 4 has bf16 instructions, so bf16 is 2.3x fp32. The model card's
  "bf16 is slower on CPU" holds only on CPUs without them.
- **Dynamic int8 quantizes activations, and Flex cannot take it.** Besides the loops, the output
  lost the mixed-script convention: English came out in Devanagari.
- **Weight-only int8** (`notebooks/src/cpukit.py`) keeps activations in bf16. On a full-size Flex
  with random weights it decodes at 16.5 vs 24.5 ms/token (1.5x), i.e. about RTF 0.12 on the 7700X.
  Its accuracy is unmeasured; 04c's CPU export section measures it on each new model and exports
  int8 only if val WER stays within +0.3 of bf16.
- **Live use.** Flex is not a streaming model. Cut the microphone at pauses (Silero VAD) and
  transcribe each utterance: a 5 s sentence returns in about 1 s in bf16.

### The Anatomy of Autoregressive Loops
- Greedy decoding gets stuck in positive-feedback absorbing states when cross-attention has low energy (speaker pauses or filler hesitations like `अँ`, `उम्`) or during natural reduplication (`mixed-mixed`, `खोज्दै खोज्दै`, `21, 21`).
- The audio in these clips is clean. When forced past the repeated token using a mild repetition penalty (`repetition_penalty=1.2`), the decoder snaps back to cross-attention and transcribes the rest of the clip with near-zero errors.

### Leakage Verification (D76 Compliance)
- Verified on disk:
  - **Shared clip IDs:** 0
  - **Audio time overlap:** 0.0000 s
  - **Exact transcript overlap (>5 chars):** 0
- Ten clips share a 0.0000 s VAD cut boundary (e.g. clip 23 is Train, clip 24 is Gold, clip 25 is Gold, clip 26 is Train), but audio samples are strictly disjoint.
- Gold is not speaker- or episode-held-out (36 of 42 episodes span both pots). The held-out **Val set** (4 unseen episodes) scored **7.60% WER**, proving true out-of-sample generalization.

---

## 4. Technical Traps & Operational Gotchas

- **The notebook scores with the dataset's copy of `fold.py`.** `harness/` in the HF dataset still
  holds fold-v1 until `backend/app/services/fold.py` is re-uploaded there, and the Models page
  imports a run under whatever the harness runs. Check that `fold_version()` agrees on both sides
  before comparing a notebook number with the page or this table.
- **Edit sources, not notebooks:** Notebooks are generated from `notebooks/src/`:
  - `ftkit.py`: shared training kit embedded via `%%writefile`.
  - `cpukit.py`: weight-only int8 for the CPU export, embedded via `%%writefile` and copied into
    `OUT/cpu/` (load with `cpukit.load_int8`).
  - `build_finetune.py`: builds notebook 04c (`python notebooks/src/build_finetune.py`).
- **Batch probe gradient buffer:** The batch probe must keep gradients allocated (`probe_max_items`). Probing without the gradient buffer measures activations without the ~4.8 GB optimizer/grad buffer and causes out-of-memory errors during accumulated training.
- **CUDA memory allocator:** Setup must set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch is imported to avoid fragmentation across variable sequence lengths.
- **Flex tokenization offset:** Targets are SentencePiece IDs offset by 1,152 special tokens. `।` maps to `.`, Devanagari digits to Latin digits, and ZWJ/ZWNJ are stripped.
- **Licence:** Fine-tuned Flex weights are a derivative under the Indic Open Model License v1.0. Keep weights private.
- **Colab session lifecycle:** Every session starts with an empty runtime. Only Google Drive persists. Run Setup by hand for secrets (`HF_TOKEN`) and Drive mounting.
