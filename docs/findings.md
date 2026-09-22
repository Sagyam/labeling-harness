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
until 2026-09-15, `fold-v2` (D84) until 2026-09-17, `fold-v3` (D89) since.

---

## Synthetic crosstalk does not move real crosstalk (2026-09-22)

The D96 sweep on the 2026-09-21 export (8,575 train, 1,198 val, 750 gold clips), fold-v3, one
A100 run per point from the same base weights. WER is folded, with S/D/I per 100 reference words.
The 2026-09-17 model was decoded on the same val and gold as the reference.

| run | best epoch | val | gold | gold none (489) | gold 0–5% (56) | gold 5–15% (87) | gold >15% (118) |
|---|---|---|---|---|---|---|---|
| p = 0 | 5 | 7.19 | 11.56 | 6.58 (4.87/1.02/0.69) | 9.28 (6.61/1.39/1.28) | 17.26 (10.76/4.62/1.88) | 29.25 (17.46/7.67/4.12) |
| p = 0.1 | 5 | 7.22 | 11.41 | 6.51 (4.85/1.00/0.67) | 9.39 (6.76/1.31/1.31) | 16.99 (10.44/4.65/1.90) | 28.68 (17.25/7.59/3.85) |
| p = 0.2 | 3 | 7.25 | 11.73 | 6.65 (4.94/1.03/0.69) | 9.53 (6.87/1.46/1.21) | 17.53 (10.44/5.19/1.90) | 29.68 (17.21/8.48/4.00) |
| p = 0.5 | 5 | 7.21 | 11.40 | 6.45 (4.76/0.99/0.70) | 9.57 (6.32/1.64/1.61) | 17.26 (10.68/4.92/1.66) | 28.57 (16.81/8.40/3.35) |
| 09-17 model | — | 7.28 | 11.87 | 7.07 (5.02/0.97/1.07) | 10.15 (6.43/1.13/2.59) | 17.26 (10.66/4.81/1.79) | 28.74 (17.36/7.77/3.61) |

Paired on gold against p = 0, 95% interval from resampling episodes:

| gold | p = 0.1 | p = 0.2 | p = 0.5 | 09-17 model |
|---|---|---|---|---|
| all | −0.15 [−0.28, 0.00] | +0.17 [0.00, +0.36] | −0.16 [−0.38, +0.08] | +0.31 [−0.19, +1.11] |
| none | −0.06 [−0.22, +0.08] | +0.08 [−0.09, +0.24] | −0.12 [−0.26, +0.03] | +0.49 [−0.15, +1.68] |
| 5–15% | −0.27 [−0.70, +0.38] | +0.27 [−0.26, +1.19] | 0.00 [−0.54, +1.20] | 0.00 [−0.63, +0.82] |
| >15% | −0.57 [−1.12, +0.06] | +0.43 [−0.33, +1.85] | −0.69 [−1.41, +0.80] | −0.51 [−1.15, +0.63] |

**The sweep was cut short, so the D96 rule was not applied.** p = 0.3 was lost mid-run to a power
cut and the owner then stopped the sweep: p = 0.3 and the second p = 0 seed never ran. No winner
was chosen and no weights were kept (`best/` is uploaded only for a winner). The 2026-09-17
model stays the deployed one. Every point's metrics, transcripts and per-clip counts are in
`Sagyam/nepanglish-asr-flex-ft/flex-xtalk-sweep-2026-09-22/`.

- **No crosstalk bucket improved, and there is no dose response.** Every interval holds zero. At
  >15% the change runs −0.57, +0.43, −0.69 as p goes 0.1, 0.2, 0.5: putting synthetic crosstalk
  into up to about 36% of each epoch did nothing distinguishable from run-to-run noise. Val agrees
  (p = 0.5: +0.01 [−0.13, +0.13]).
- **No spillover.** Clean clips hold (`none` −0.12 at p = 0.5), and so do their deletions (1.02 →
  0.99), the D95 check that the model had not learnt to drop its own speaker's short words.
- **What the augmentation did change: fewer insertions, more deletions.** At >15%, p = 0 → 0.5 moves
  insertions 4.12 → 3.35 and substitutions 17.46 → 16.81, but deletions 7.67 → 8.40. The model
  writes less of the other voice, as the labels it trained on ask. Gold labels in overlap keep
  what was audible (D95), sometimes the other voice's words too, and leaving those out counts as
  deletions. So in these buckets gold WER cannot show the gain this augmentation was designed for,
  whether or not the model learned it.
- **Val cannot select for crosstalk.** It has 4 clips over 15% overlap. A crosstalk experiment
  needs a selection split that holds crosstalk, or its winner is chosen on clean speech.
- **The added data.** p = 0 against the 09-17 model (7,052 → 9,774 clips): gold −0.31 [−1.11,
  +0.19], clean clips −0.49 [−1.68, +0.15], both within noise. The crosstalk buckets are
  identical (5–15%: 17.26 against 17.26).
- **Verdict.** More of this augmentation does not help this model on real crosstalk. What is not
  ruled out: (1) the mismatch in shape, since the mixer's bursts have a median of 0.30 s while the
  >15% clips are sustained talk-over; (2) the mismatch in target, since training rewards leaving
  the other voice out while gold sometimes rewards writing it. A model that writes one stream of
  text for one speaker cannot be right both ways. Overlapped speech proper needs a
  speaker-attributed or serialized output, scored with cpWER, tcpWER or ORC-WER (roadmap, sections
  B and C).

---

## Can gold measure a crosstalk fix? Only a large one (2026-09-21)

The owner added 26 recordings (155 clips, all verified by ear, 12 edited) from `chill_pill_clips`
to gold on 2026-09-19 to measure the synthetic-crosstalk augmentation of roadmap item 3. The
09-17 fine-tune's int8 CPU export (playground sidecar, not stored as a run) transcribed all 660
labeled gold clips, scored under fold-v3 against the current labels. It agrees with the stored
bf16 run on the old clips (6.61 against 6.51 on the same 69 clips), and made 0 loops.

- **Gold now has crosstalk.** Before the batch it had 49 overlapped clips, 2 of them >15%. It now
  has 172 (45 episodes): 55 at 0–5%, 50 at 5–15%, 67 at >15%. Overlap is 3.4% of gold audio (the
  corpus is 3.1%); the new clips alone are 13.7%.

  | crosstalk | clips | episodes | WER [95% CI, episodes] |
  |---|---|---|---|
  | none | 488 | 89 | 6.97 [5.79, 8.20] |
  | 0–5% | 55 | 26 | 9.30 [6.18, 12.63] |
  | 5–15% | 50 | 16 | 16.57 [11.74, 22.73] |
  | >15% | 67 | 23 | 28.85 [24.52, 32.86] |

- **Most of the gap is the show, not the overlap.** The new show's clean clips score 17.33 (old
  gold's 6.48). Within the same episode (Mantel-Haenszel) the new clips' ratio against clean is
  0.86 at 0–5%, 1.00 at 5–15% and **1.54 at >15%**; across all gold, 1.23 / 1.24 / 1.61. Only the
  >15% bucket carries a clear crosstalk penalty, and eliminating it entirely would take that
  bucket from ~29 to ~18, about 11 points.
- **The bucket is narrow.** 41 of the 67 >15% clips, and 62% of that bucket's errors, come from
  three long episodes (183–185) of one show, which share a host (v002). Only 8 of the new episodes
  have both clean and crosstalk clips.
- **Smallest detectable paired change** (two-sided 5%, 80% power, clustered by episode). The
  per-clip noise is the disagreement between the 09-16 and 09-17 fine-tunes on old gold. It is
  heavy-tailed, since a handful of clips where one model collapses carry most of the variance, so
  the standardized differences were resampled rather than assumed Gaussian. Simulating clips
  independently understates the noise 3x: it predicts a 0.64-point CI on old gold where the real
  09-16 to 09-17 comparison has 1.93. The limits below are scaled by that 3x. Voice clustering
  gives the same numbers.

  | bucket | clips | WER | detectable change |
  |---|---|---|---|
  | none (regression guard) | 488 | 6.97 | ~1.5 pts (21% rel) |
  | any crosstalk | 172 | 18.95 | ~3.7 pts (20% rel) |
  | ≥5% | 117 | 23.74 | ~5.1 pts (21% rel) |
  | >15% | 67 | 28.85 | ~7.1 pts (25% rel) |

- **Verdict.** Gold can confirm an augmentation that removes about two thirds of the >15%
  penalty (7 of ~11 points). A 10–15% relative gain is within noise. Any result would be mostly
  a statement about one show's roundtables. Halving the detectable change takes about four times
  as many independent rooms with heavy crosstalk (many shows, many hosts), not more clips from
  183–185. Read any augmentation result as a within-episode contrast (crosstalk clips against
  clean clips from the same episodes) so the show's own difficulty cancels.
- **Caveats.** The 3x calibration comes from the whole of old gold and may be worse on a bucket
  dominated by three episodes. The overlap detector is still unchecked by ear.

**Update the same day: 86 more crosstalk clips from one episode.** The owner added one Prime
Television talk show (72.5 min, two voices v192/v193), kept 86 clips (17.9 min, 85 with
crosstalk, 18% of their audio overlapped) as gold and deleted the rest. Labels keep the stronger
voice and never invent words, but drop some of the weaker one. The same int8 model and noise
model were used, and the recalibrated clustering factor is 3.07.

| bucket | clips (before → now) | episodes | Kish effective episodes | WER now | detectable change |
|---|---|---|---|---|---|
| any crosstalk | 176 → 261 | 45 → 46 | 12.2 → 8.3 | 20.67 | 3.56 → 3.13 pts (15% rel) |
| ≥5% | 120 → 205 | 27 → 28 | 7.5 → 5.3 | 24.04 | 4.80 → 3.84 pts (16% rel) |
| >15% | 70 → 118 | 24 → 25 | 6.7 → 5.3 | 28.68 | 6.82 → 5.46 pts (19% rel) |

- The new episode scores 19.29 at 5–15% and 27.99 at >15%, close to chill_pill's 29.46. It has
  one clean clip, so it gives no within-episode baseline.
- **More clips, fewer effective rooms.** The detectable change shrinks by about a fifth if noise is
  per clip. But the new episode now holds 41% of the >15% clips, so the bucket's effective number
  of independent episodes fell from 6.7 to 5.3. Whether a fix generalises across rooms is no better
  known than before.
- **The references penalise hearing the second voice.** A model that transcribes the weaker
  speaker is charged with insertions for the words the label dropped (93 of the model's 783
  errors on these clips are insertions). Score augmentation by deletions and substitutions as well
  as WER, or a real gain can read as a loss.
- Verdict unchanged in kind: gold can confirm a gain of about 5.5 points (a fifth) at >15%, not
  a 10% one, and it speaks for about five rooms.

---

## The retrain on fold-v3 and the redrawn val (2026-09-17)

`indic-transcribe-flex-ft-2026-09-17` is 04c with the standard settings, trained on the D90 split
(6,417 train clips; val 635 clips / 2.18 h in 6 episodes) and scored under fold-v3.

| | val WER | val CER | gold WER | gold CER | gold raw WER | loops |
|---|---|---|---|---|---|---|
| base Flex | 13.15 | | | | | 0 |
| fine-tune 2026-09-17 (bf16) | **5.75** | 4.69 | **6.94** [5.8, 8.3] | 6.93 | 12.95 | 0 |
| same, int8 (accepted) | 5.78 | | 6.92 | | | 0 |
| fine-tune 2026-09-16, rescored under fold-v3 | | | 6.25 | 6.25 | | |

- **No clear change from 09-16.** Paired on the same 505 gold clips, it is +0.68 [−0.04, +1.82]
  points of WER when bootstrapping episodes. 68 clips are better, 70 worse and 367 tied.
- **Val kept falling.** By epoch: 6.84 / 6.18 / 6.06 / 5.89 / 5.84 / 5.75. The best epoch was the
  last and the early stop never fired.
- **CPU:** int8 runs at RTF 0.39 against 0.56 for bf16, on Colab's Xeon.
- **Where the errors are (raw rates per class, not within-episode ratios):**
  - Crosstalk: val goes from 4.5 with no overlap to 11.5 at 5–15%, and deletions grow fastest
    (0.9 → 3.8 per 100 words).
  - Noise: gold below 15 dB SNR scores 9.8 against ~6 above.
  - Reverb: gold with C50 below 40 dB scores 9.8 against 6.1 for dry rooms.
  - Clips under 5 s are the worst on both splits (gold 13.9, val 8.6).
  - A second speaker adds ~4 points on both splits. Code-mixing shows no trend.
- **Outputs** are in the private HF model repo `Sagyam/nepanglish-asr-flex-ft` under the run name
  (commit a13eb37): `best/`, `cpu/`, `harness/`, `hyps/`, `int8/`. From this run on, 04c uploads
  there instead of Drive.

---

## fold-v3: colloquial Nepali folded (2026-09-17)

D89 folds ten kinds of colloquial form (contracted verbs, `-या`, progressive, benefactive, first
person plural, pronouns, `लाउनु`, emphatic `-ै`, loose pairs, unseen forms). Same references, same
text:

| system | gold fold-v2 | gold fold-v3 | val fold-v2 | val fold-v3 |
|---|---|---|---|---|
| Flex fine-tune 2026-09-16 | 6.58 | **6.25** | 12.59 | **12.05** |
| Scribe | 7.08 | 6.68 | | |
| Gemini | 6.38 | 6.16 | | |
| MAI | 6.00 | 5.81 | | |

- **Small.** Item 6's ceiling was 2.02 points of same-word substitutions. Only ~115 of those 1,268
  errors were colloquial-against-standard pairs; the rest are grammar (`थियो`/`थिएँ`,
  `हामी`/`हामीले`, `भने`/`भनेर`).
- **Not fitted to the fine-tune.** Scribe gains most (0.40), Flex 0.33.
- **In the references**, spoken forms are common, not rare: `हैन` 2,230 against `होइन` 992,
  `थिएन` 170 against `थिइनँ` 10, `लाएर` 36 against `लगाएर` 11, `नि` 2,234 against `पनि` 4,671.

---

## Speaker-held-out gold: the retrain (2026-09-16)

The first measurement on a gold pot no training voice speaks in. The HF dataset
(`Sagyam/nepanglish-asr`, commit 5345974, harness `a45d467`) was re-exported:
- **gold**: 505 clips (2.17 h) from 82 shorts/reels, every clip listened to, 23 edited (4.6%).
  None of those recordings feeds train or val. Voice linking finds 93 gold voices, none among
  train's 32; the links are unchecked by ear.
- **train**: 6,131 clips (20.55 h), the old 706 gold clips included. **val**: 921 clips (3.94 h),
  of which 849 come from two podcasts.

`indic-transcribe-flex-ft-2026-09-16` is 04c with the standard settings, fold-v2:

| | gold WER (95% CI by recording) | gold raw WER | gold CER | val WER |
|---|---|---|---|---|
| base Flex, zero-shot | 12.26% | 22.36% | 11.60% | 22.38% |
| **fine-tune** | **6.58% (5.61–7.59)** | 12.36% | 6.25% | 12.59% |

- **Fine-tuning transfers to new voices.** It roughly halves gold WER, with 0 loops before or
  after, so the retry did nothing on this gold.
- **Per recording** WER has a median of 5.3%, a p90 of 11.4% and a maximum of 18.5% (`cars_in_nepal`).
  The CI resamples the 82 recordings. It is 2 points wide, against 4.4 points for the old gold
  resampled by its 27 voices.
- **Training.** Val by epoch: 14.20, 12.94, **12.59**, 12.65, 12.93, then early stopping (patience 2).
  Val is no longer comparable with the 5.58 of 2026-09-15: it now contains the bodybuilders
  roundtable, and 46% of val clips have crosstalk.
- **Gold has almost no crosstalk.** 10% of gold clips have any overlap, and overlap is 0.4% of
  gold audio (train 1.6%, val 7.3%). Gold measures clean single-voice speech, not the podcast
  failure mode that item 2 of the roadmap targets.
- **Recognisers on the same gold** (fold-v2): MAI 6.00%, Gemini 6.38%, Scribe 7.08%. The references
  are their fusion (the fusion itself scores 0.04%), so these are flattered. The fine-tune's 6.58%
  is independent of the references.
- **Selection.** Clips where Gemini returned SAFETY were dropped before gold was labeled (~16%), so
  gold holds only clips all three recognisers transcribed.

**Why val is twice gold: crosstalk (measured the same day).** Folded WER by the overlapped share of
each clip:

| overlap share | gold clips | gold WER | val clips | val WER | train clips (seen) | train WER |
|---|---|---|---|---|---|---|
| none | 456 | 6.40 | 497 | **5.71** | 200 | 7.01 |
| < 5% | 35 | 7.22 | 122 | 9.11 | 116 | 11.65 |
| 5–15% | 12 | 9.86 | 133 | 14.08 | 84 | 13.82 |
| > 15% | 2 | 16.47 | 169 | **25.45** | 151 | 20.30 |

- **With crosstalk removed, val and gold agree** (5.71 against 6.40), so the pipeline is not at fault.
  On val, 81% of errors are in overlapped clips and 49% in the >15% bucket. Deletions rise from 0.9%
  to 5.0% of reference words.
- **One episode dominates val.** The bodybuilders roundtable has 16.52% WER and 83% of val's errors;
  its clean clips alone score 8.54%. ep_612 scores 6.97%, the tech reviews 1.9–3.3%.
- **The model does not fit crosstalk even on clips it trained on.** Train clips with >15% overlap
  still score 20.30% after 3 epochs (deletions 7.0%). That is why val sat at ~12.6 from epoch 3:
  it is not a generalisation gap that more epochs would close. Only 2.5% of train clips are >15%
  overlapped. The references in overlap are screened fusion output, and nobody has listened to
  whether they are a consistent target.

**What the 6.58% is made of (error mining, 2026-09-17).** The 1,483 folded errors, each assigned
to the first rule that fits. The votes compare the word with the three recognisers' own
transcripts. Those recognisers are what the reference was fused from, so a vote is a proxy, not
ground truth.

| bucket | errors | WER pts | what it looks like |
|---|---|---|---|
| clip edge (first/last word) | 169 | 0.75 | reference holds a cut fragment (`सक्नुहु`), the model completes it; edge-word WER 12.7% |
| Devanagari spelling variant | 56 | 0.25 | vowel length, nasal marks, halant, स/श/ष, ब/व, ण/न |
| number / unit | 153 | 0.68 | `साढे नौ फिट` → `9.9 ft` (a real error: 9.5), `किलो` → `kg`; also catches छ-forms read as six |
| ≥2 recognisers side with the model | 132 | 0.59 | reference suspects: `गको` / `गएको`, dropped fillers |
| all 3 recognisers agree with the reference | 266 | 1.18 | clear model errors: brands (`ElevenLabs` → `11 lamps`, `Api` → `Appy`), English (`firm` → `form`) |
| split vote | 707 | 3.14 | suffixes (`हामी`/`हामीले`, `कुरा`/`कुराहरू`), colloquial against standard forms (`देछु`/`दिएछु`, `चै`/`चाहिँ`) |

- **Noise shows at the low end.** 39 gold clips below 10 dB SNR score 12.58%, against ~6% everywhere
  above. `api_base_camp`, recorded beside a waterfall, has a median SNR of 10.6 dB and 11.6% WER.
- **Concentration.** 113 clips are error-free; the worst 10% of clips hold 34% of errors. Clips
  the owner edited score 8.79%, against 6.48% for clips accepted unchanged.

Outputs are on Drive under `MyDrive/nepanglish-asr/indic-transcribe-flex-ft-2026-09-16/`: `best/`,
`harness/` for the Models page, and gold hyps. No CPU export or playground bundle was built.

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

## How big gold has to be: voices, not hours (2026-09-16)

Measured on run 5 (the current model on the 706-clip gold pot) by bootstrapping its per-clip
errors 4,000 times. The question was what size gold has to reach before its WER is worth
publishing; the answer is that the number of clips stopped being the binding constraint a while
ago.

The same 9.54% carries three different intervals depending on what a resample treats as
independent:

| Resampled unit | 95% CI | Width |
|---|---|---|
| Clips | 8.79 – 10.30 | 1.52 pts |
| Episodes (36) | 7.26 – 11.33 | 4.07 pts |
| **Voices (27)** | **7.34 – 11.70** | **4.36 pts** |

Clips of one speaker are not independent evidence about the next speaker, so the voice-clustered
interval is the one a corpus can claim. It is nearly three times the clip interval, because
per-voice WER runs from 2.14% (v017) to 22.05% (v011) — between-voice variance dominates
everything else, which is the same thing the voice-exposure axis says (see the clip-classes
section).

That makes the clip curve misleading and the voice curve the real one:

| Clips (resampling clips) | CI width | Voices (resampling voices) | CI width |
|---|---|---|---|
| 706 (2.3 h) | 1.58 pts | 27 | 4.27 pts |
| 1,500 (4.9 h) | 1.06 pts | 60 | 2.92 pts |
| 3,000 (9.8 h) | 0.72 pts | 100 | 2.29 pts |
| 5,000 (16.3 h) | 0.58 pts | 150 | 1.85 pts |

A two-stage bootstrap (draw voices, then clips within each) prices the trade-off directly. At a
fixed budget of 700 clips:

| Shape | CI width |
|---|---|
| 7 voices x 100 clips | 8.33 pts |
| 27 voices x 26 clips (what gold was) | 4.46 pts |
| 70 voices x 10 clips | 2.86 pts |
| 140 voices x 5 clips | 2.19 pts |

Depth per voice saturates at about 20 clips: going from 20 to 80 clips a voice buys 0.2–0.5
points at any voice count, while going from 27 to 60 voices buys 1.6. **Roughly 60 voices x 20
clips (~1,200 clips, ~3.9 h) puts the published interval near +/-1.5 points; 100–120 voices is
what it takes to get near +/-1.**

Two consequences for how gold is filled:

- A clip is worth adding in proportion to how new its speaker is. Twenty clips from a voice
  already in gold are worth less than five from a voice that is not.
- Diversity on the other axes is only free when it does not cost voices. Hunting the extremes
  (crosstalk, low SNR, narrow band) concentrates on a few noisy episodes — sorting the queue by
  crosstalk returns one roundtable's clips for pages — so the hunt has to be run per episode, or
  gold buys rare conditions by spending voices.

Separately, this is measured on gold as the seed built it, and gold's clips were chosen by hand
from episodes that also fed train. It bounds precision, not leakage; see the leakage section.

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

## Overlap windows: how long they are and who is louder (2026-09-15)

A corpus-wide characterisation of crosstalk, to ground the synthetic mixing of roadmap item 3 in
what real overlap looks like rather than in LibriMix-style guesses. Analysis code discarded; every
number here is re-derivable from the stored spans and the newest diarization run.

- **Method.** Overlap windows are the measured D77 detector spans on **all 7,075 clips** (not just
  gold and val, which is what the 2026-09-13 study's 25% of clips and 2.1% of clip time cover).
  Speaker identity comes from the current pyannote community-1 run (D78/D79). Loudness is
  broadband frame RMS (25 ms window, 10 ms hop) of each speaker's **solo** speech — frames where
  only that speaker is active — referenced locally within ±15 s of each window, with an
  episode-global reference as fallback. No source separation ran, so "who is louder" is an energy
  model over solo reference levels, not a measurement of the two voices inside the window.
- **Overlap is common in clips but rare in time.** 1,887 of 7,075 clips (27%) carry 4,250 measured
  windows, 2,714 s in total — **3.1% of all clip audio**. The diarizer's own turn overlaps agree
  with the detector within 1% (2,694 s vs 2,714 s inside clips).
- **Windows are short.** 83% are under 1 s; only 84 windows (2%) exceed 3 s. The shape is a burst
  distribution — backchannels, interjections, laughter over speech — not extended duets.

  | percentile (s) | d10 | d20 | d30 | d40 | d50 | d60 | d70 | d80 | d90 | p95 | p99 | max |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | window duration | 0.07 | 0.17 | 0.27 | 0.35 | **0.42** | 0.52 | 0.66 | 0.89 | 1.33 | 1.93 | 4.03 | 8.94 |

- **Per overlapped clip.** Median total overlap 0.81 s (5.4% of the clip); p75 1.74 s (10.8%), p90
  3.48 s (20.2%), p99 9.3 s (51%). Median 2 windows per clip.
- **The two voices are nearly always at comparable levels.** Over the 3,450 windows (≥ 0.2 s) where
  both speakers have a solo reference — 33 (episode, pair) units — the **per-pair median level gap
  is 1.59 dB, IQR 1.17–2.06 dB, p90 2.51 dB, and no pair exceeds ~3.0 dB**. One voice is louder in
  ≥ 90% of a pair's windows in only 5 of the 33 pairs (gaps 0.9–3.05 dB); elsewhere the direction
  flips between windows, which at sub-dB gaps is estimate noise.
- **The mix itself confirms near-equality.** The measured level inside an overlap window sits
  **+0.78 dB above the louder speaker's solo reference** (p5 −4.2, p95 +6.2), and ~1.5 dB below the
  incoherent power-sum of the two solo references. That is the signature of two comparable sources
  actually summing: the "quieter" voice is fully audible, only ~7 dB down, not masked.
- **The negative finding: there is no consistently louder voice to train on.** Real crosstalk in
  this corpus is near-equal-level mixing, so a "focus on the dominant voice" model would discard
  roughly half the overlap energy on data that does not resemble the corpus. Target-speaker
  conditioning, if used, must key on voice identity, not level.
- **What item 3's mixer should sample** (all from these measurements): 2 speakers per window;
  duration from the decile table above (median ~0.4 s); level gap over 0–3 dB with mass at 1–2 dB;
  incidence such that ~27% of clips carry ≥ 1 window but only ~3% of frames are overlapped.
- **Caveats.** Loudness is solo-frame RMS, so sub-1-dB direction calls are noise and per-speaker
  levels inside a window were never separated; the diarizer labels are model outputs; the overlap
  detector remains unchecked by ear; and the fused reference is at its least reliable exactly in
  these windows (as the studies above warn).

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
