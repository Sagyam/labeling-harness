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
     - **The cap is not in the code yet; add it with Phase 2's notebook.** It is a
       `LogitsProcessor` passed to `generate`:
       - Each clip's cap is `min(300, ceil(13 × seconds))` tokens.
       - Once a row's generated length (`input_ids.shape[1]` minus the 10-token prompt) reaches its
         cap, set that row's scores to −inf except end-of-text.
       - Under beam search, row `r` belongs to clip `r // num_beams`.
       - The retry already caps its single-clip decode with `max_new_tokens`.
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

### Phase 5: Revisit Secondary Models (Whisper-turbo & Omnilingual)
Once the performance ceiling is established with Flex:
- Give **Whisper-large-v3-turbo** a longer training schedule (it was still improving at epoch 5) and beam search decoding.
- Evaluate whether **Omnilingual CTC-1B v2** can be salvaged with a KenLM language model decoder.

---

## 2. Where Things Stand (Benchmark Snapshot)

### Models Scored on Gold (706 clips, 2.3 h)

| Model | Setup | Folded WER % | Raw WER % | CER % | Loops | Notes |
|---|---|---|---|---|---|---|
| **Scribe v2** | Cloud API (in ref) | 12.73% | 20.14% | 9.93% | 0 | Commercial; fused into reference |
| **Gemini 3.8 Flash** | Cloud API (in ref) | 12.03% | 20.64% | 10.69% | 0 | Commercial; fused into reference |
| **MAI Transcribe 2** | Cloud API (in ref) | 12.47% | 20.98% | 11.38% | 0 | Commercial; fused into reference |
| **Indic-Transcribe-Flex** | Zero-shot | 18.2% | — | 14.0% | 4 | Baseline |
| **Indic-Transcribe-Flex** | Fine-tuned, greedy | 13.20% | 16.15% | 9.40% | 7 | 6 epochs (best ep 5) |
| **Indic-Transcribe-Flex** | **Fine-tuned + retry** | **11.44%** | **14.41%** | **8.09%** | **0** | **Best model; 04c's decoder** |
| **Indic-Transcribe-Flex** | Fine-tuned, greedy + cap + retry (reloaded weights) | 11.53% | 14.52% | 8.07% | 0 | **Standard decoder from Phase 1** (reloaded bf16 weights: 04c's decoder gives 11.51% on them) |
| **Indic-Transcribe-Flex** | Fine-tuned, beam 4 + cap + rp1.1 + retry | 11.10% | 14.01% | 7.85% | 0 | Not adopted: ~4x decode time; chosen on gold (dev score) |
| **Whisper-large-v3-turbo** | Zero-shot | 123% | — | — | 271 | Unusable zero-shot |
| **Whisper-large-v3-turbo** | Fine-tuned, greedy | 14.62% | 17.72% | 8.68% | 0 | 5 epochs (still improving) |
| **Omnilingual CTC-1B v2**| Fine-tuned (partial)| ~16.6% (val)| — | — | 0 | Stopped at epoch 6 |

### Where Model Weights and Artefacts Live (Google Drive)
All trained weights and evaluation logs persist on Google Drive under `MyDrive/nepanglish-asr/`:
- **Flex:** `indic-transcribe-flex-ft/`
  - `best/`: bf16 weights loadable with `IndicTranscribe` (Private — Indic Open Model License v1.0).
  - `gold_metrics.json`: complete scores, greedy vs. retry metrics, and before/after text of all retried clips.
  - `hyps/indic-transcribe-flex-ft.jsonl`: cached hypotheses in bake-off format.
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

- **Edit sources, not notebooks:** Notebooks are generated from `notebooks/src/`:
  - `ftkit.py`: shared training kit embedded via `%%writefile`.
  - `build_finetune.py`: builds notebooks 04a, 04b, 04c (`python notebooks/src/build_finetune.py`).
  - `build_bakeoff.py`: builds 03.
- **Batch probe gradient buffer:** The batch probe must keep gradients allocated (`probe_max_items`). Probing without the gradient buffer measures activations without the ~4.8 GB optimizer/grad buffer and causes out-of-memory errors during accumulated training.
- **CUDA memory allocator:** Setup must set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch is imported to avoid fragmentation across variable sequence lengths.
- **Flex tokenization offset:** Targets are SentencePiece IDs offset by 1,152 special tokens. `।` maps to `.`, Devanagari digits to Latin digits, and ZWJ/ZWNJ are stripped.
- **Licence:** Fine-tuned Flex weights are a derivative under the Indic Open Model License v1.0. Keep weights private.
- **Colab session lifecycle:** Every session starts with an empty runtime. Only Google Drive persists. Run Setup by hand for secrets (`HF_TOKEN`) and Drive mounting.
