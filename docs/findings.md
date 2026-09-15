# ASR findings

What the ASR work on the Nepanglish corpus has measured so far. What is still to do lives in
[roadmap.md](roadmap.md).

Experiment code is discarded after each run, so these notes are the record. The raw artefacts
behind them were on Google Drive under `MyDrive/nepanglish-asr/`:
- the weights of every fine-tune;
- per-clip outputs of the decoder search and the learning curve;
- the diarization run.

That folder was **deleted on 2026-09-15** at the owner's request, so the numbers below can no
longer be re-derived from those files.

What survives:
- **The current model.** Its int8 CPU export and harness files are in
  `data/models/asr/indic-transcribe-flex-ft-2026-09-15/`. Its bf16 weights are gone. Re-running
  04c builds an equivalent model in about 40 minutes on an A100.
- **The diarization run.** `exports/flex-eval/diarization.json` is the only copy of the file, and
  `exports/` is gitignored. Its turns are also imported into the database (D78).

Folded WER is computed by `app/services/fold.py`. Every number names its fold version: `fold-v1`
until 2026-09-15, `fold-v2` (D84) since.

---

## The current model (2026-09-15)

`indic-transcribe-flex-ft-2026-09-15` is a 04c full fine-tune of Indic-Transcribe-Flex with the
standard settings (6 epochs, peak LR 1e-5, best epoch 6). Its epochs were picked under fold-v2.
It is on the Models page, and it is the model the mic playground runs (D85).

| weights | val WER | gold WER | gold raw WER | gold CER | loops |
|---|---|---|---|---|---|
| bf16 (GPU) | 5.58% | 9.54% | 13.76% | 7.62% | 0 |
| **weight-only int8 (CPU export)** | **5.54%** | **9.58%** | — | — | 0 |

- **Training.** Val by epoch: 6.63, 5.89, 5.90, 5.75, 5.62, 5.58. The run took about 20 minutes,
  at ~620x realtime and ~88% GPU utilisation.
- **Same model as 2026-09-12.** The earlier fine-tune, rescored under fold-v2, gave val 5.59 and
  gold 9.65. The gap is within the ~0.3-point noise between identical runs.
- **The loop retry still earns its keep.** Greedy alone gave gold 10.46% with 7 looping clips; the
  retry fixed all 7.
- **int8 is free.** It gave the same text as bf16 on 337 of 403 val clips, and 04c's rule
  (int8 only within +0.3 of bf16 on val, with no more loops) accepted it.
- **CPU speed** is in [CPU inference](#cpu-inference-2026-09-15).

---

## Benchmark snapshot: gold (706 clips, 2.3 h)

The fold-v2 column was measured against `exports/gold/gold.jsonl`, with the recognisers' text read
from the harness. That source gives Scribe / Gemini / MAI 13.11 / 11.96 / 11.99 under fold-v1,
not the figures in the fold-v1 column, which came from the bake-off's copy of their text. Raw WER
does not depend on the fold.

fold-v2 also drops fillers and folds numbers, contractions and colloquial Nepali. Every system
gains 1.7–2.3 points under it, and the fine-tuned Flex no longer leads Gemini: they tie.

| Model | Setup | Folded WER (fold-v1) | Folded WER (fold-v2) | Raw WER | CER | Loops | Notes |
|---|---|---|---|---|---|---|---|
| **Scribe v2** | Cloud API (in ref) | 12.73% | 11.40% | 20.14% | 9.93% | 0 | Commercial; fused into reference |
| **Gemini 3.8 Flash** | Cloud API (in ref) | 12.03% | 9.66% | 20.64% | 10.69% | 0 | Commercial; fused into reference |
| **MAI Transcribe 2** | Cloud API (in ref) | 12.47% | 10.21% | 20.98% | 11.38% | 0 | Commercial; fused into reference |
| **Indic-Transcribe-Flex** | Zero-shot | 18.2% | — | — | 14.0% | 4 | Baseline |
| **Indic-Transcribe-Flex** | 2026-09-12 FT, greedy | 13.20% | — | 16.15% | 9.40% | 7 | 6 epochs (best ep 5) |
| **Indic-Transcribe-Flex** | 2026-09-12 FT + retry | 11.44% | — | 14.41% | 8.09% | 0 | 04c's decoder |
| **Indic-Transcribe-Flex** | 2026-09-12 FT, greedy + cap + retry | 11.53% | 9.65% (val 5.59%) | 14.52% | 8.07% | 0 | Standard decoder, on reloaded bf16 weights |
| **Indic-Transcribe-Flex** | 2026-09-12 FT, beam 4 + cap + rp1.1 + retry | 11.10% | — | 14.01% | 7.85% | 0 | Not adopted: ~4x decode time; chosen on gold |
| **Indic-Transcribe-Flex** | **2026-09-15 FT + retry** | — | **9.54%** (val 5.58%) | **13.76%** | **7.62%** | **0** | **Current model** |
| **Whisper-large-v3-turbo** | Zero-shot | 123% | — | — | — | 271 | Unusable zero-shot |
| **Whisper-large-v3-turbo** | Fine-tuned, greedy | 14.62% | — | 17.72% | 8.68% | 0 | 5 epochs (still improving) |
| **Omnilingual CTC-1B v2** | Fine-tuned (partial) | ~16.6% (val) | — | — | — | 0 | Stopped at epoch 6 |

Flex won the fine-tuning comparison. It made fewer errors than Whisper on 361 gold clips and more
on 163, with 182 ties. The gap is in the podcasts; the tech reviews are even. Whisper-turbo and
Omnilingual CTC were not fine-tuned further. Their notebooks (`04a`, `04b`) and the bake-off
notebook (`03`) were removed on 2026-09-14.

---

## Decoder search (2026-09-13)

Gold was treated as a dev set.

- **The standard decoder is greedy + length cap + loop retry,** for every model, run and report.
  Post-processing is not worth it beyond these cheap parts.
- **Best decoder found: `beam4+cap+rp1.1+retry`.** Gold WER went 11.51 → 11.10% (Δ −0.41, 95%
  episode interval [−0.60, −0.19]) and CER 8.11 → 7.85%. Val went 7.57 → 7.27%. It was rejected:
  - **Too costly.** The gain is real but small, and it costs about 4x greedy's decode time.
  - **Optimistic.** The arm was chosen from ~20 on gold; val, which played no part in the choice,
    shows only −0.30.
  - **At the noise level.** Identical training runs already differ by ~0.3 on val.
  - The experiment's decision rule picked it because the rule had no cost term; the owner
    overrode it on cost.
- **The standard is free.** `greedy+cap+retry` equals 04c's decoder: 11.53 vs 11.51, Δ +0.02
  [+0.00, +0.06]. The cap only makes a looping clip stop at 13 tokens per second of audio instead
  of running to 300 tokens.
- **Where the cap is in the code.** The mic playground caps every decode. 04c caps only its retry:
  the first greedy pass still runs to 300 tokens, which changes nothing but the time a looping
  clip takes. A batched cap is a logits processor passed to `generate`:
  - each clip's cap is `min(300, ceil(13 × seconds))` tokens;
  - once a row's generated length reaches its cap, every score in that row except end-of-text
    goes to −inf;
  - under beam search, row `r` belongs to clip `r // num_beams`.

  A plain class with `__call__(input_ids, scores)` in a `transformers.LogitsProcessorList` worked
  with the Flex port's `generate`, and every clip of a batch stopped at exactly its own cap.
- **The retry is what earns its keep.** Decoding 8 looping clips a second time took greedy from
  13.31 to 11.51.
- **Global anti-repetition makes greedy worse.** Repetition penalty 1.2 cost +1.11 points (202 clips
  worse, 81 better), and n-gram blocking +0.51. 25% of gold references repeat a 6-token span,
  which blocking can never reproduce.
- **Beam alone does not replace the retry:** beam 4 still loops on 2 gold clips.
- **A WER below 10.5% (fold-v1) is out of reach by decoding.** Any further gain has to come from
  training or data.

### The anatomy of autoregressive loops

- **When greedy decoding sticks.** It falls into a self-reinforcing loop when cross-attention has
  little to attend to: a pause, or a filler such as `अँ` or `उम्`. Natural reduplication does it
  too: `mixed-mixed`, `खोज्दै खोज्दै`, `21, 21`.
- **The audio in these clips is clean.** Pushed past the repeated token with a mild repetition
  penalty (1.2), the decoder goes back to the audio and transcribes the rest of the clip with
  near-zero errors.

---

## Learning curve: more of the same data does not help (2026-09-13)

One draw was run of Flex trained on 25% and 50% of the training episodes. The owner stopped the
experiment there, judging the answer clear.

- **Design.**
  - Whole episodes were drawn per genre (podcast / tech review), each to the nearest fraction of
    its hours, in a seeded order. The subsets nest within a draw: 25 ⊂ 50 ⊂ 75 ⊂ 100.
  - Every point got the full run's optimizer steps: its subset repeats within each epoch.
  - Each run kept its best epoch by val.
  - Decision rule, fixed before the runs: a val gain of ≥ 1.0 from 50% to 100% means volume
    helps; < 0.5 means saturated.
  - Draw 1 and both 75% points were not run.

  | point | hours | val WER | gold WER |
  |---|---|---|---|
  | 25%, draw 0 | 4.6 | 7.61 | 12.07 |
  | 50%, draw 0 | 9.3 | 7.45 | 11.84 |
  | 100% (2026-09-12 weights) | 18.3 | 7.57 | 11.53 |

- **Val is flat**, within the ~0.3 run-to-run noise. The rule reads −0.12 for 50%→100%, which
  means saturated.
- **Gold falls ~0.25 per doubling.** Part of that gain is gold episodes entering training, which
  new episodes would not give.
- **Paired gold test, 25%→50%.**
  - Clips whose episode stays unseen gain +0.24 [−0.29, +0.77] from doubling the hours.
  - Clips whose episode joins training gain +0.87 [+0.37, +1.42].
  - So an episode's own speakers are worth +0.62 [−0.16, +1.39] beyond the hours. That points at
    new speakers, but the interval includes zero.
- **Caveats.**
  - One draw only, so the spread between draws is unmeasured.
  - Val is 83% one podcast: 336 of its 403 clips come from `ep_612`.
  - The curve is flat for *this* distribution: 27 recurring voices and one tech-review host. That
    is why new gold must be new speakers.

---

## Crosstalk explains the podcast errors; speaking rate and CMI do not (2026-09-13)

- **Method.**
  - pyannote `speaker-diarization-community-1`, with overlap detection, ran on all 42 whole
    episodes at the declared speaker count: 17 minutes on an A100.
  - Clip-level features for all 1,109 gold and val clips were joined to Flex's per-clip errors
    (standard decoder, 2026-09-12 weights). The analysis code was discarded.
- **Speaker attribution is reliable except where it matters most.**
  - pyannote and the EDA's ECAPA voice prints are independent systems. They agree on who is
    talking in 97.3% of 41,285 three-second windows in multi-speaker episodes.
  - They agree on 99.2% of the windows pyannote calls single-speaker, which are 82% of windows.
  - The disagreements sit in the other 18%: turn changes and overlap.
- **Crosstalk is common inside clips.**
  - In multi-speaker episodes, 25% of clips contain some overlap, and 27% have two speakers
    talking ≥ 0.5 s each.
  - Overlap is 2.1% of clip time overall, and up to 10% in `on_air_with_sanjay_796`, the
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
  - 1.01 [0.93, 1.11] for speaking rate and 1.06 [0.98, 1.14] for CMI, i.e. no effect, which
    matches the EDA.

  Together these factors explain 14–17% of clip-level deviance.
- **What overlap does.** Deletions go from 1.6 to 8.3 per 100 words, and substitutions from 6.0 to
  12.9. Near-miss spellings stay flat (~1.5), so these are real misrecognitions and dropped words,
  not scoring artefacts.
- **Counterfactual.** If overlap clips had the no-overlap error rate, gold would be ~8.8% instead
  of 11.53% (fold-v1), and val 6.9% instead of 7.57%. That is the largest lever measured so far.
- **Not everything.** Across the 16 podcast episodes, overlap and WER correlate at only Spearman
  0.39. `ep_447` is at 20.7% WER with 1.8% overlap, so episode- or speaker-level effects remain.
- **Caveats.**
  - The overlap detector has not been checked by ear.
  - In overlap, the fused reference is at its least reliable: whose words are they, and are
    backchannels kept? Some "deletions" may be reference choices.
  - Single-host episodes were diarized with one speaker, so they have no overlap by construction.

The harness now measures overlap on every clip at ingest (D77), and the Models page breaks each
run down by the same buckets.

### Error skew and the long tail (2026-09-12 weights, fold-v1)

- **By genre:**
  - 1-speaker tech reviews (19 episodes): **3.5% WER**, near-human.
  - 2-speaker podcasts (15 episodes): **13.5% WER**.
  - 3-speaker podcast (`on_air_with_sanjay_796`): **22.2% WER**.
- **Long tail:** the median episode WER is **4.6%**, and the worst 10% of clips hold **46% of all
  errors**.

---

## Clip classes on the current model: what splits WER and what is ruled out (2026-09-15)

- **Method.** Every clip classified (D87); the gold run of `indic-transcribe-flex-ft-2026-09-15`
  (706 clips, 36 episodes, fold-v2) split by each axis. The ratio is the Mantel-Haenszel
  within-episode error-rate ratio against the axis's baseline, and the interval resamples
  episodes. Each axis is taken alone: the ratio adjusts for the episode and nothing else. The
  Models page shows these numbers live.

  | axis | bucket (clips) | WER | ratio [95% CI] | verdict |
  |---|---|---|---|---|
  | crosstalk | none (511) | 6.80 | baseline | **splits** |
  | | 0–5% (101) | 11.01 | 1.37 [1.19, 1.66] | |
  | | 5–15% (67) | 14.87 | 2.11 [1.60, 2.73] | |
  | | >15% (27) | 22.40 | 3.25 [2.03, 4.65] | |
  | speakers in clip | 1 (470) | 6.74 | baseline | **splits** |
  | | 2 (228) | 12.53 | 1.45 [1.18, 2.05] | |
  | turn changes | 0 (470) | 6.74 | baseline | **splits** |
  | | 1 (53) | 9.02 | 0.97 [0.70, 1.25] | |
  | | 2+ (183) | 13.93 | 1.63 [1.28, 2.38] | |
  | bandwidth | 6.5+ kHz (578) | 9.96 | baseline | splits, *downwards* |
  | | 4.5–6.5 kHz (116) | 6.70 | 0.75 [0.40, 0.98] | |
  | pause share | <2% (412) | 9.44 | baseline | ruled out |
  | | 2–10% (217) / >10% (77) | 9.54 / 10.77 | 0.96 [0.82, 1.16] / 0.87 [0.66, 1.38] | |
  | clip length | 5–15 s (255) | 9.97 | baseline | ruled out |
  | | <5 s (165) / 15+ s (286) | 9.19 / 9.39 | 0.89 [0.61, 1.03] / 1.04 [0.76, 1.28] | |
  | CMI (descriptive) | 0 (101) | 6.27 | baseline | ruled out |
  | | <15 / 15–30 / 30+ | 10.12 / 11.66 / 8.54 | 1.35 [0.95, 2.25] / 1.10 [0.66, 1.58] / 1.15 [0.91, 1.78] | |

- **Crosstalk still leads**, and more steeply than under fold-v1: over 15% overlap triples the
  error rate within an episode. Two speakers and 2+ turn changes split WER too. The three axes
  travel together, so each ratio carries part of the others.
- **A single hand-over does not hurt** (0.97). Only repeated back-and-forth does.
- **Ruled out: pauses, clip length, CMI.** A clip's pauses were the suspected trigger for decoder
  loops, but the current decoder's loop retry leaves no trace of them in WER.
- **Bandwidth runs the wrong way.** The band-limited clips (4.5–6.5 kHz; 99 of the 116 are
  podcast clips) do *better* within their episodes, and the interval only just clears 1. Nothing says narrow audio is easy, so the likely reading is a confound with the axes above;
  listen before building on it.
- **Noise splits WER only through crosstalk.** By Brouhaha's speech-to-noise ratio, against
  45+ dB (357 clips):
  - 35–45 dB gives 1.28 [1.02, 1.80], 25–35 dB 1.77 [1.04, 2.14] and 15–25 dB 1.98
    [1.61, 2.95] within episodes, while the pooled WER per bucket is flat, because the noisier
    buckets are full of easy tech reviews.
  - Crosstalk rises as SNR falls, though, from 1.0% to 3.8% of the clip: a second voice is noise
    to Brouhaha. On the 511 gold clips with no crosstalk, every SNR interval holds 1
    (35–45 dB: 1.08 [0.74, 1.26]).
  - So by this measure the corpus's background noise is not a source of errors. Item 3's noise
    augmentation is insurance for audio the corpus does not have yet, not a fix for measured
    errors.
- **Reverb is ruled out.** C50 is 55+ dB (a dry room) for 591 of 706 gold clips, and no bucket's
  interval clears 1. The one reverberant episode, the six-speaker `ep_344` roundtable (median C50
  43 dB), has too few gold clips to say more.
- **Voice exposure cannot be measured on this gold.** Gold shares its speakers with train (see
  Leakage verification). Only one gold voice is unseen in train: the guest of `ep_612`, a val
  episode, 50 clips. The guest's ratio against the seen host of the same episode is 1.20, from
  one episode, so there is no interval. That is item 5's argument, now as a number.
- **Declared gender and age cannot be separated from genre.** Female-declared clips are at 4.38
  WER, but almost all of them are the tech-review host's clean monologues. With no within-episode
  contrast, there is no ratio.
- **Word classes.**

  | class | words | WER | CER |
  |---|---|---|---|
  | Devanagari | 14,722 (63%) | 9.04 | 10.20 |
  | Latin | 8,252 (36%) | 6.82 | 8.07 |
  | at a code-switch | 8,742 (38%) | 7.31 | 9.51 |
  | first or last word of the clip | 1,411 (6%) | **11.13** | 10.45 |
  | numbers | 699 (3%) | 8.01 | 23.39 |

  - **A code-switch is not where errors land.** Switch words are no worse than either script's
    words. With the clip-level CMI result, code-mixing is ruled out as a cause of errors twice.
  - **Clip edges are the worst words**, which points at VAD cuts through a word.
  - **Number CER is mostly spelling.** A digit against a number word is a match in folded WER
    but a full character error, as in the run's own CER.
  - **Caveat.** The references are still the fused consensus (below), least reliable in crosstalk.

---

## The joint fit: overlapped time and rapid hand-overs are two enemies; the second speaker is exonerated (2026-09-15)

The crosstalk study's joint Poisson — episode fixed effects, episode-clustered SEs, offset log
reference words — run on the current model's gold run (706 clips, 36 episodes, fold-v2+norm-v3)
with all covariates together instead of one axis at a time. Covariates from the stored spans and
the newest diarization run; filler share is fillers per raw reference token. Analysis code
discarded.

| covariate | alone | joint |
|---|---|---|
| overlap share, per 10 points | 1.61 [1.42, 1.82] | **1.52 [1.35, 1.71]** |
| 2+ speakers in clip | 1.48 [1.10, 1.99] | **0.85 [0.66, 1.09]** |
| 2+ turn changes | 1.64 [1.23, 2.18] | **1.51 [1.25, 1.83]** |
| filler share, per 10 points | — | 1.07 [0.93, 1.24] |
| pause share, per 10 points | — | 0.97 [0.88, 1.06] |

- **Validation.** The bucket-alone fit reproduces the clip-classes table's ratios (1.32 / 2.05 /
  3.03 against the page's 1.37 / 2.11 / 3.25), and the recomputed classes match every clip's
  stored ones.
- **Overlapped time barely attenuates** (1.61 → 1.52): it is not the other axes in disguise.
- **The second speaker collapses** to 0.85, interval holding 1. Since 2+ turn changes requires
  two speakers, its coefficient is read off clips with a second voice but at most one
  hand-over — exactly the clips the clip-classes table already found harmless (0.97). A polite
  second voice costs nothing.
- **Rapid hand-overs keep their weight** at zero measured overlap: 1.51. Filler share's 1.28
  from the old study does not replicate on this model; pause stays ruled out. Together the five
  explain 17.6% of within-episode deviance.
- **Counterfactuals (joint fit).** 9.54% now; **8.03%** with overlap set to zero — the extraction
  prize, 16% of errors; **7.32%** if hand-over churn goes too (23%). The earlier 11.53 → 8.8
  counterfactual credited overlap with hand-over difficulty that is not overlap.
- **The reference is bounded, not acquitted.** Scoring substitutions and insertions only,
  overlap still carries 1.36 [1.24, 1.50] and hand-overs 1.41. The deletions charged in overlap
  clips are dominated by response tokens — हजुर, त, नि, you, हो, Thank — both the hardest to hear
  under crosstalk and the reference's least reliable keep-or-drop choices. The by-ear audit
  listens for exactly these.
- **For the roadmap.** Extraction (item 2) pays only on the overlapped seconds (~1.5 points);
  the ~0.7 points of hand-over churn need augmentation or a better architecture (items 3–4).
  Extraction does not need to erase the other voice, only the overlapped seconds.
- **Caveats.** Identification comes from the 16 multi-speaker episodes; dispersion is about 2,
  and episode-clustered SEs are the crosstalk study's convention for that.

---

## The reference is a consensus of the systems it scores

- The references are an LLM fusion of Scribe, Gemini and MAI (D74). Only 3 of 1,170 verified
  labels were ever edited, and 82% of clips were screened.
- The commercial models' ~12% WER (fold-v1) is their distance from their own consensus.
- Under fold-v1, Flex at 11.44% WER and 8.09% CER beat all three on their own consensus. On CER,
  that is an ~18–29% relative error reduction.
- Under fold-v2, Flex (9.54%) ties Gemini (9.66%) and beats Scribe and MAI.

## Leakage verification (D76)

- **Verified on disk:** 0 shared clip IDs, 0.0000 s of shared audio, and 0 exact transcript
  overlaps longer than 5 characters.
- **Adjacent clips.** Ten clips share a VAD cut boundary with a clip in the other pot (clip 23 is
  train, 24 and 25 are gold, 26 is train), but their audio samples are strictly disjoint.
- **Gold is not held out by speaker or episode:** 36 of 42 episodes have clips in both pots. The
  val set (4 unseen episodes) is the out-of-sample check: 7.60% under fold-v1 on the 2026-09-12
  model.

---

## CPU inference (2026-09-15)

**Base Flex on the owner's Ryzen 7 7700X.** Base Flex has the same architecture and speed as the
fine-tune. It ran with 8 threads, plain PyTorch and the KV cache, over 10 gold clips (117 s, 2.5–20
s each).

| variant | RTF | ms/token | WER (base model) |
|---|---|---|---|
| fp32 | 0.41 | 54 | 21.8% |
| **bf16** | **0.18** | **23** | **21.8%** (8/10 texts identical to fp32) |
| dynamic int8, per-tensor | 0.11 | 15 | 96.7%, 4 loops |
| dynamic int8, per-channel | 0.11 | 15 | 112.5%, 2 loops |

- **Decoding is the cost.** The encoder takes ~0.015 s per second of audio. Nepanglish runs about
  8 tokens per second of speech, and each token re-reads the 24 decoder layers, which is
  memory-bound.
- **bf16 is fast on this CPU.** Zen 4 has bf16 instructions, so bf16 runs 2.3x faster than fp32.
  The model card's "bf16 is slower on CPU" holds only on CPUs without them.
- **Dynamic int8 quantizes activations, and Flex cannot take it.** Besides the loops, the output
  lost the mixed-script convention: English came out in Devanagari.
- **Weight-only int8** (`notebooks/src/cpukit.py`) keeps activations in bf16 and stores int8
  weights with one scale per output channel. On a full-size Flex with random weights it decoded at
  16.5 vs 24.5 ms/token (1.5x).

**Measured on the 2026-09-15 fine-tune:**
- **Accuracy.** Weight-only int8 costs nothing measurable. The table is in
  [The current model](#the-current-model-2026-09-15).
- **Speed on the 7700X, through the mic playground's sidecar (8 threads).** A 20 s gold clip
  decoded in 2.3 s (RTF 0.11). Loading the model takes 5.2 s, and it holds 1.6 GB. The owner found
  push-to-talk from a live microphone "very fast".
- **Speed on Colab's Xeon,** which has no bf16 instructions: int8 ran at RTF 0.41 against 0.57
  for bf16. Only the ratio carries over to another CPU.
- **Output on the CPU.** On the 20 s clip, the playground's text matched the notebook's GPU text
  except for one word, which is not in the reference either.

**Flex is not a streaming model.** It decodes a whole utterance at once, so live use means
cutting the microphone at pauses and transcribing each utterance. The playground is push-to-talk
for that reason.

---

## Technical traps and operational gotchas

- **Two copies of `fold.py`.** The notebook scores with the copy in the HF dataset's `harness/`,
  and the Models page scores with the harness's own. Both have held fold-v2 since 2026-09-15 (HF
  commit `b1e25c6`). Check that `fold_version()` agrees on both sides before comparing a notebook
  number with the page.
- **Edit sources, not notebooks.** Notebooks are generated from `notebooks/src/`:
  - `ftkit.py` is the shared training kit, embedded via `%%writefile`;
  - `cpukit.py` is the weight-only int8 export, embedded via `%%writefile` and copied into
    `OUT/cpu/` (load it with `cpukit.load_int8`);
  - `build_finetune.py` builds notebook 04c (`python notebooks/src/build_finetune.py`).
- **The batch probe must keep gradients allocated** (`probe_max_items`). Without the ~4.8 GB
  gradient buffer, the probe measures activations alone and picks a batch that runs out of memory
  once training accumulates.
- **CUDA memory allocator.** Setup sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before
  torch is imported, to avoid fragmentation across variable sequence lengths.
- **Flex tokenization.** Targets are SentencePiece IDs offset by 1,152 special tokens. `।` maps to
  `.`, Devanagari digits to Latin digits, and ZWJ/ZWNJ are stripped.
- **Licence.** Fine-tuned Flex weights are a derivative under the Indic Open Model License v1.0.
  Keep them private: `data/` is gitignored.
- **Colab sessions start with an empty runtime.** Only Google Drive persists. Run Setup by hand
  for the `HF_TOKEN` secret and the Drive mount. The token needs write access for anything that
  uploads to the dataset.
- **Moving weights to this machine.** A model folder is too large for any connector, so 04c packs
  the harness files and CPU weights into one `<RUN_NAME>-playground.tar` on Drive, downloaded by
  hand.
