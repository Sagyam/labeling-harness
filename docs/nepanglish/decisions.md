# Decisions: the Nepanglish research

Each entry: the decision, why, and what it would cost to reverse.

These are the research project's decisions: how a model is trained and evaluated in the
notebooks, and which experiments ran. The harness's own are in
[../harness/decisions.md](../harness/decisions.md). D-numbers are shared by both files: a new
entry takes the next number after the highest in either.

## Retired entries

Deleted entries keep one line here so code comments citing them still resolve.

- **D118** The LLM-judge pilot (`08b_Judge_Pilot`): Qwen3.8-27B reading and Gemma 4 12B hearing
  pick one of the teacher's 8 beams on 300 random val clips, against the panel vote, with a
  Bonferroni interval as the kill rule; the thinking rows were dropped mid-run for their cost.
  It stopped G1; the notebook is gone (D120).
- **D119** After D118 stopped G1, both judges ran on every set for the record. Removed with the
  notebook (D120); its numbers are in findings.md.

## D95 — Synthetic crosstalk is mixed on the fly, labelled as the clip's own speaker, and cut from the same room

The fine-tune notebook (`notebooks/src/xtalk.py`, on when `XTALK_P > 0`) gives a fresh 30% of the
train clips measured clean short bursts of another voice every epoch, in the DataLoader workers.
Each mixed clip keeps its label unchanged.

- **The label is the clip's own speaker.** The references in real overlap keep what was audible:
  usually the main voice, sometimes the other one, sometimes both. The owner accepts any of these
  as long as nothing is written that was not said. A fixed target always takes one acceptable
  option, the main voice. Writing the other voice's words would need them transcribed, and the only
  text for a donor excerpt is an unverified recogniser's, which would teach invented words. Flex's
  decoder has no speaker-attributed output. Measured on 8,406 synthetic bursts against Scribe's
  word spans (2026-09-22): 58% hold only word fragments, 37% at least one whole word, and 6% no
  word, so most of what the model learns to leave out could not be written as a word anyway.
- **Measured shape, not LibriMix defaults** (findings.md, *Overlap windows*). Window durations
  come from the measured deciles, with a median of 0.42 s. The level gap is within ±3 dB (median
  |gap| 1.6), and either voice can be louder. The model cannot learn "drop the quieter voice".
  It has to learn "follow the voice that holds the clip".
- **The other voice comes from the same episode** when it has another voice with a solo stretch
  long enough. Same room, microphone and level, so the channel gives nothing away. Otherwise it
  comes from another train episode. On the 2026-09-21 export, 55% of windows come from the same
  episode: 27 of the 52 train episodes have one voice.
- **Weighted toward the error buckets.** An augmented clip draws its overlapped share with weights
  40/30/30 over 1–5%, 5–15% and 15–40%. Realised on the export, the split is 30/34/36. Real overlap
  is 3% of audio, but the errors sit above 5%.
- **What is never touched.** Clips with real crosstalk keep it, and unmeasured or undiarized clips
  are not assumed clean. Donors come only from train clips' solo speech: `DonorPool` raises on a
  val or gold row (D76). Val and gold are never mixed.

**Why on the fly.** It needs no new export and no new HF upload. Every epoch sees different mixes,
it costs 2.7 ms of worker CPU per clip, and `XTALK_P = 0` gives the old run back exactly.

**How it is judged:** by the sweep in D96. Gold WER in the crosstalk buckets also charges
acceptable choices (a label that kept the other voice counts its omission as deletions), so the
S/D/I split is read with it, and the deletions on clips without overlap are the check that the
model has not learnt to drop its own speaker's short words.

**Reversal:** set `XTALK_P = 0`. To remove it entirely, delete `xtalk.py`, its cell and the
mixer lines in `collate`.

## D96 — XTALK_P is swept on one export, and the kept weights are chosen on val by a rule fixed in advance

The fine-tune notebook trains one model per `(XTALK_P, seed)` point in `SWEEP`:
`(0, 0), (0.1, 0), (0.2, 0), (0.3, 0), (0.5, 0), (0, 1)`. Every point starts from the same base
weights, on the same export, with the same batch budget, and is scored on val and gold overall
and per crosstalk bucket, with S/D/I. The 2026-09-17 model is decoded on the same val and gold
under the same decoder. The results are meant for a paper, so what may be claimed is fixed here,
before any point has run.

- **The winner is chosen on val alone** (`sweep.choose_winner`). p = 0 is its better seed. The
  noise is the gap between the two p = 0 seeds. The best augmented run is kept only if it beats
  p = 0 on val by more than the noise; otherwise p = 0 is kept, as the simpler recipe. Gold never
  takes part, so gold stays a held-out score for every point. Choosing on gold would bias every
  gold number reported.
- **Two seeds at p = 0.** One run per point cannot tell an effect of p from a different random
  run. The only earlier estimate (09-16 against 09-17, +0.68 [−0.04, +1.82]) also changed the fold
  and the val draw. The second seed also reorders the batches: `make_batches` seeds its shuffle
  with `epoch + 1000 * seed`, so seed 0 keeps the order of every earlier run.
- **Up to 0.5.** 73% of train clips are eligible to be mixed. At p = 0.5, about 36% of each epoch
  is synthetic crosstalk, on top of the 27% of clips with real crosstalk, against about 3% of the
  audio in reality.
- **Gold comparisons are paired and resampled by episode** (`sweep.paired_bootstrap`). Clips of
  one episode share a room and voices, so resampling clips would overstate how much gold knows.
- **What the added data did** is p = 0 seed 0 against the 09-17 model, paired on gold. The export
  grew from 7,052 to 9,774 clips. Val grew with the new episodes too, which changes which epoch
  early stopping picks, so the gold pair is the reading.
- **Only the winner keeps its weights.** Every point uploads its metrics, transcripts, per-clip
  counts and harness folder as it finishes, so all of them can go on the Models page and a
  dropped session resumes. `best/` stays on the VM until the winner is known, and is uploaded and
  exported for the CPU only for the winner.
- **Crosstalk is a secondary analysis, not the paper's claim.** The model card says the model is
  trained for single-speaker recordings. Overlapped speech proper (cpWER, tcpWER, ORC-WER,
  diarization-conditioned models) is left to separate work.

**Run 2026-09-22: cut short, rule not applied.** Only p = 0, 0.1, 0.2 and 0.5 ran (seed 0), so
there was no seed noise to apply the rule with. No winner was chosen and no weights were kept.
Results are in findings.md.

**Reversal:** set `SWEEP` to one point, which runs a plain single fine-tune. Delete `sweep.py`, its
tests and the sweep cells to remove it entirely.

## D101 — Unlabelled audio for distillation is a second corpus of files, cut like ingest, screened against gold, never in the database

Roadmap §B step 1 needs unlabelled Nepali audio for Flex to pseudo-label (step 0, findings
2026-09-25: every student's gap to Flex is widest on pure Nepali). D86 lets an episode into the
harness only through the web ingest, and that ingest transcribes every clip three times on paid
routes and diarizes on Modal. Neither is wanted for audio no one will label, so this audio does
not become episodes at all.

- **Files, not rows.** The owner collects whole playlists as MP3 (or any audio ffmpeg reads),
  each with yt-dlp's info JSON beside it under the same name. `scripts/prepare_distill_audio.py`
  turns them into `data/distill/` (gitignored): one folder per source with `source.json` (video
  id, channel, playlist, title, duration: provenance) and its clips, and one `clips.jsonl` for the
  corpus. It is uploaded as a private HF dataset, `Sagyam/nepanglish-distill`, for Colab. Nothing
  is written to Postgres, no queue task exists, and no paid route or Modal call is made: the only
  model is Flex, as the teacher, on Colab.
- **Cut exactly like ingest.** Stage 1's loudness normalisation to 16 kHz mono FLAC, the same
  Silero VAD thresholds and the same 2–20 s slicing (`normalize_audio`, `SileroVAD`,
  `segment_audio_to_slices`, `extract_clips`). A student then trains on clips shaped like the
  labelled ones, and a clip is one lossy generation from the download, not two.
- **Kept away from gold and val (D76), in three layers.**
  - *A known recording.* A source whose YouTube id is already an episode in the harness is
    refused: it is labelled audio, and may be gold or val.
  - *A known channel.* Sources from channels listed in `distill.blocked_channels` (the shows gold
    was drawn from) are refused.
  - *A known voice.* Gold holds 82 shorts with no recorded source, so channels cannot cover it.
    Every source's clips are embedded in 2 s windows with the voiceprint model (D99), whose space
    is the diarizer's, and compared with the centroid of every voice diarized in an episode that
    holds a gold clip. A source with at least `distill.screen_min_seconds` of windows at or above
    `distill.screen_threshold` to one gold voice is quarantined: listed with the voice, the
    seconds and its best windows for the owner to hear, and left out of the corpus until cleared.
    The threshold is measured before it is used, on the corpus's own clips (a gold voice's own
    windows against its centroid, and other voices' windows against it), and the measurement goes
    in findings.md. D87's 0.6 was set on whole-speaker centroids and is not assumed to hold for
    2 s windows.
- **Val is never pseudo-labelled either.** Val is episodes in the harness, so the first layer
  covers it.

- **Amended 2026-09-25: cut in Colab from the owner's zip.** The owner uploads `distill.zip`
  (`<channel_name>_<NN>.mp3`, no info JSON) to the root of `Sagyam/nepanglish-asr`, and
  `notebooks/04_PreDistill.ipynb` cuts it there with ingest's own `normalize_audio` and Silero VAD,
  fetched from GitHub at a pinned commit and checked by sha256. It stores each recording whole
  (`distill/episodes/<id>.flac`) with its clip times (`distill/sources/<id>.json`,
  `distill/clips.jsonl`), the layout `ftkit.AudioStore` reads, in that repo; the exports' downloads
  never fetch `distill/`. `05_Teacher.ipynb` reads this layout. The owner vouches that these channels
  are new, so this path has no voiceprint screen and no known-recording check (there are no video
  ids); gold's shows are still refused by file name.
- **Amended 2026-09-25: the local path is deleted** at the owner's request, with its voiceprint
  screen and the `distill:` settings (`scripts/prepare_distill_audio.py`,
  `scripts/screen_distill_audio.py`, `scripts/upload_distill_corpus.py`,
  `app/services/distill_corpus.py`, `app/services/distill_prep.py`). The Colab path is the only
  one. The bullets above describe what the local path did; the screen's calibration stays in
  findings.md. To screen a channel against gold again, restore them from git (commit 024c8e7).

**Reversal:** cheap. Delete `distill/` in the dataset repo; nothing else refers to it.

## D105 — The notebooks are one numbered protocol: how a model is trained and evaluated

The 2026-09-30 export (57 h of labels) came with every fine-tuned weight deleted, so everything is
trained again. Until then `notebooks/` held seven notebooks named by history, two copies of the
scoring cell, a teacher notebook pointing at weights that no longer existed, and no repository
code at all for the public benchmarks or weight blending (both had been scratch Colab cells). From
now on a model is trained and evaluated by running the notebooks in the order of their numbers.

| Step | Notebook | What it does |
|---|---|---|
| 1 | `01_EDA` | the corpus, and the recognisers on it (written by hand) |
| 2 | `02_Sociolinguistics` | what the corpus records about its speakers (written by hand) |
| 3 | `03a_Flex_Train` | Flex, the vanilla recipe, seeds 0 and 1 |
| | `03b_Flex_Benchmarks` | base Flex and any run on the five public sets |
| | `03c_Flex_Augment` | one `augment.py` stage per run, then the kept ones together |
| | `03d_Flex_Blend` | `(1 − a)·base + a·fine-tuned` for a = 0.25, 0.5, 0.75 |
| | `03e_Flex_Ship` | freezes the teacher in `teacher.json`, exports it for the CPU |
| 4 | `04_PreDistill` | cuts the unlabelled recordings (D101) and measures each clip's overlap |
| 5 | `05_Teacher` | the frozen teacher pseudo-labels them; the labels are filtered |
| 6 | `06a`–`06f_Student_*` | one notebook per ASR student, three stages each (D106) |
| 7 | `07a`–`07b_Student_LLM_*` | one notebook per speech-LLM student, the same three stages (D122) |
| 8 | `08_Judge_Headroom` | how much a judge picking among the teacher's beam candidates could gain (D117) |
| 9 | `09_Report` | every model's scores read back into one table; no GPU; always last (D122) |

- **One notebook per GPU session.** Each starts from the hub and ends with an upload, a run whose
  result is already there is skipped, and a run whose weights are there without its scores is
  scored without training again. The order inside step 3 is train, evaluate, ablate, blend,
  freeze: blending is the remedy for what 03b measures, and the ablation can change which run is
  blended.
- **One evaluation** (`notebooks/src/evalkit.py`), for Flex, its blends and every student stage:
  gold and val, folded and raw, S/D/I, per clip class, paired against named runs on the clips both
  scored with episodes resampled; then the five public sets of 2026-09-27, folded, raw and plain,
  paired against base Flex with each set's own unit resampled. On the public sets "forgetting" is
  only meaningful for a model that knew Nepali; for the others they measure generalisation.
- **Every choice is made on val, by a rule fixed before the result** (`sweep.py`). An augmentation
  is kept only if it beats vanilla's better seed by more than the gap between vanilla's two seeds,
  and never by less than 0.15 points (D109).
  The blend is the one closest to base whose val WER is within 0.3 of the fine-tune's. Gold and
  the public sets are reported and never choose.
- **One teacher, frozen.** `teacher.json` at the root of the Flex model repo names the model that
  leaves step 3. Steps 5, 6 and 7 read it; none names a model of its own. An existing file is not
  overwritten with another model unless `REPLACE_TEACHER` is set, because students trained on the
  old teacher's labels would no longer match it.
- **One export, pinned by its date.** Every notebook refuses a dataset whose training manifest is
  not `DATASET_EXPORT`. A date and not a commit: the dataset repo's history is squashed after each
  upload, which would orphan a pinned commit.
- **Crosstalk in the unlabelled audio is measured in step 4 and dropped in step 5.** The harness's
  own overlap detector (D77) gives each clip its overlapped share, and step 4's summary prices each
  threshold in hours per channel. `MAX_OVERLAP_SHARE` in step 5's filter applies the one the owner
  picks; changing it never needs another pass. It is `None` until picked. Round-table shows are
  the most overlapped and bring the most voices, which is why nothing is dropped at cut time.
- **The augmentation ablation runs on Flex only**, where a run is cheap. A student gets the recipe
  that won there, switched on, against the same stage without it; not an ablation of its own.
  SpecAugment is part of vanilla, so the ablation covers `augment.py`'s seven waveform stages. The
  strengths in `ABLATION` are first guesses: a stage that fails at one strength has not been shown
  useless at every strength.
- **Generated, and checked.** `python notebooks/src/build_all.py` writes every notebook from
  `build_*.py`. `test_notebook_builds.py` fails when a committed notebook is not what its builder
  writes (a kit edited without a rebuild left the fine-tuning notebook running old code in
  September 2026), when a cell does not parse, or when a cell uses a name no earlier cell defines.
- **A smoke run first.** `SMOKE = True` runs a notebook end to end on a few hundred clips for one
  epoch under `<RUN_PREFIX>-smoke`. No notebook is known to work on a GPU until its smoke run has
  passed; the test suite cannot reach that. A smoke run reads its predecessor's smoke output
  when there is one and the real output otherwise (`distill.smoke_source`), so 05, 06 and 07 can
  be smoke-tested after a predecessor that ran for real; it never writes outside its smoke names.
- **Names in older entries.** "04c" is the fine-tuning notebook that is now 03a–03e, "04a" and
  "04b" the Whisper and Omnilingual ones that are now 06a and 06e.

**Reversal:** cheap. The notebooks are generated, so another order is a rename in the builders;
the kits and their tests do not depend on the numbering.

## D106 — A student's second stage continues its human-label fine-tune

The owner's choice (2026-09-30), reversing the roadmap's rule of 2026-09-26 that every student
start from its pretrained checkpoint on the mixture of pseudo-labels and human labels, with the
human-only fine-tune kept apart as the 0 h point.

- **Three stages per student.** *human*: from the pretrained weights, on the verified train labels.
  *distill*: those weights continued on the teacher's filtered pseudo-labels mixed with the human
  labels. *distill-aug*: the distill stage again, from the human stage's weights, with the recipe
  that won Flex's ablation; skipped when vanilla won.
- **The human labels stay in the mixture.** They take `HUMAN_SHARE` (0.5) of every epoch's draws;
  the rest is split between the unlabelled channels by the square root of their hours, as settled
  on 2026-09-26 (`distill.mixture_weights`). So the model does not end on the teacher's errors
  alone, which was one of the two objections to continuing.
- **What the comparison can no longer say.** Distill minus human is the pseudo-labels *and* the
  extra training together; the two cannot be separated, which the fresh-weights rule existed to
  avoid. What it still says: whether the student a human-label fine-tune produces gets closer to
  the teacher when it is given the teacher's labels. Distill-aug minus distill is the
  augmentation alone, since both start from the same weights.
- **Resumable.** A stage saves a resume point to a scratch model repo after every epoch
  (`ftkit.HubResume`): the weights and the counters, not the optimizer's moments. Each save
  replaces the last, and the superseded blobs are deleted from that repo's storage.

**Reversal:** cheap in code: `train_stage` in `notebooks/src/build_students.py` loads the human
stage's weights for the later stages, and passing none restores fresh weights. Costly in GPU once
the students are trained, since every distill stage would be run again.

## D109 — An augmentation has to beat vanilla by at least 0.15 points on val

D105 kept a 03c stage if it beat vanilla's better seed on val by more than the gap between
vanilla's two seeds. 03a, the first run of the protocol, gave seeds of 6.95 and 6.93: a gap of
0.02. Under the rule as written, any single-seed stage below 6.91 would have been kept, and a kept
stage goes into the combined run, into the teacher and into every student's recipe.

- **Two runs do not measure a spread.** The same pair differed by 0.08 on gold, with a paired
  interval of −0.15 to +0.33. 0.02 says the seeds happened to agree, not that val resolves 0.02.
- **The bar is `max(seed gap, 0.15)`** (`sweep.MIN_NOISE`). 0.15 is a judgement, not derived: it
  sits just under the half-widths of the paired val intervals measured so far (0.17 to 0.26 in
  findings.md), so it is a floor against luck, not a significance test. A margin equal to the bar is a
  tie and keeps vanilla, the simpler recipe. With one vanilla seed the bar is the floor rather
  than zero.
- **Changed after 03a and before any 03c stage ran.** The owner chose it on 2026-10-01 over
  keeping the rule as written and over a second seed for every stage that clears the bar (about an
  hour of A100 each). Nothing read from an augmented run informed it.
- **Named recipes only.** D96's crosstalk sweep (`sweep.choose_winner`) keeps its own rule, which
  was fixed before its run, so its history reads the same. The blend rule (`choose_blend`) is
  unchanged.

**Reversal:** cheap before 03c's results exist: set `MIN_NOISE` to 0. After them, a reversal
re-decides which stages were kept and would have to retrain the combined run.

## D115 — The Conformer from scratch runs its own stage 3 augmentation

The owner's choice (2026-10-05), narrowing D106's stage 3 for one student. Stage 3 runs the
augmentation that won Flex's ablation, and since 03c kept none, every student so far skipped it.
The Conformer from scratch (06f) is the exception: Flex was pretrained on 1.7 M hours, which is the
reason 03c gave for augmentation having little to fix, and a 33 M model from random weights on
~130 h is the case where the literature says augmentation pays.

- **`STAGE3_RECIPE` in a student's Config.** `None` keeps D106's rule (Flex's winner, or no stage
  3). A dict, as `augment.AugmentConfig.from_dict` reads it, is that student's own recipe. Only
  06f sets one; a test holds every other student at `None`.
- **06f's recipe is 03c's six acoustic stages together, at the strengths 03c tested them at**:
  speed 0.9x/1.1x (p 0.5), reverb (p 0.3), channel (p 0.3), MUSAN noise (p 0.5), gain (p 0.5),
  codec (p 0.3). Not tuned for this model, and none of them won on Flex.
- **Crosstalk is left out** (owner): its mixing and labels were flawed (findings.md, *Augmentation
  on Flex*), and heavy crosstalk is not chased word by word (D100, D114).
- **What the comparison says.** Stage 3 minus stage 2 is the augmentation alone, as D106 already
  provides, since both start from stage 1's weights. It can say nothing about another student,
  and 06f's stages 1 and 2 stay comparable with 06c and 06d, which ran without augmentation.
- **Augmentation is the remedy for overfitting, not undertraining.** If 06f's train loss and val
  WER are both still falling when its epochs run out, the answer is more epochs, and stage 3 will
  read worse than it should.
- **Stopped after one epoch (owner, 2026-10-06).** Starting from stage 1, it could only measure a
  few points of augmentation on a model 16 points from the teacher (findings.md, *The Conformer
  from scratch*). The recipe stays in Config; nothing of stage 3 was published.

**Reversal:** cheap: set `STAGE3_RECIPE` back to `None` in `build_students.py` and rebuild. A
trained stage 3 stays in the students repo as a result, under `conformer-distill-aug`.

## D116 — A micro-batch never holds more audio than an optimizer step, and 06f takes 2-minute steps

2026-10-05, after 06f's second full run. The Conformer from random weights wrote nothing at all
(val WER 100.00, all deletions) for 14 epochs while its train loss crawled from 6.2 to 3.7, and
the owner stopped it.

- **The recipe's step size was not what it said.** `ftkit.group_steps` builds a step from whole
  micro-batches, and the batch probe sizes a micro-batch to the GPU. On the 96 GB G4 that was
  ~1,490 s of padded audio, so every step was one micro-batch of ~1,400 s, about 130 steps an
  epoch, not the ~12 min `EFFECTIVE_S` claims. `micro_budget` now caps a seconds budget at
  `EFFECTIVE_S`; a clip-count budget (the HF students') is left alone. The probe budgets of
  06a–06e on the A100 were not recorded, so whether any of them ran steps above 12 minutes is
  unknown; a rerun of theirs now gets the documented size.
- **06f takes 2-minute steps** (`EFFECTIVE_S = 120`, `Student.effective_s`), about 6x the
  optimizer steps for the same audio. A transducer from random weights needs many updates before
  it emits a token; the pretrained students only had to adjust. The other students keep 720 s.
- **What it costs the comparison.** 06f's stages 1 and 2 no longer share 06c's and 06d's step
  size, on top of their weights. Its question (does pretraining matter at ~150 h?) is answered by
  its own curve against theirs, not by an identical recipe, which random weights never allowed.
- **Smaller steps were not enough.** The third run, at 120 s steps, was still all blanks at epoch 5
  (loss 4.07). On its epoch-5 weights the CTC head wrote the same filler for every clip: the
  model ignored the audio.
- **06f's stage 1 trains at peak lr 3e-4 without SpecAugment** (`SPECAUG_IN_HUMAN = False`), the
  owner's choice after these diagnostics, on 2,000 short train clips (2.5 h), 2,000 steps of 32
  clips, CER from the CTC head:

  | peak lr | CTC weight | SpecAugment | train CER | val CER |
  |---|---|---|---|---|
  | 1e-3 | 1.0 | off | 81.7 | 81.7 (loss unstable) |
  | 3e-4 | 1.0 | off | **30.8** | **72.3** |
  | 1e-4 | 1.0 | off | 50.1 | 72.2 |
  | 3e-4 | 0.3 | off | 71.4 | 79.4 |
  | 3e-4 | 1.0 | on | 79.3 | 81.1 |
  | 3e-4 | 0.3 | on | 79.7 | 81.3 |

  1e-3 never let the encoder learn; SpecAugment from step 1 kept it from starting, whatever the
  CTC weight. Stage 2 starts from stage 1's weights, which already listen, and keeps SpecAugment.
  The CTC weight stays 0.3, as in 06c and 06d. A fixed batch of 32 clips was memorised by every
  setting, which shows the plumbing works and nothing more: 32 clips can be told apart by length.

- **Early stopping does not count warmup** (2026-10-06). Stage 2 starts from stage 1's weights
  and re-warms its rate over three epochs; val went 31.68, 31.84, 32.28 as it rose, so patience
  stood at 2 and the first evaluation at peak rate had to beat 31.48 or end the stage. An
  evaluation that ends before the warmup's last step now counts for nothing, like one at 100;
  the best weights are still kept from every epoch. A resumed run rebuilds its count from its
  stored history, so a resume point saved under the old rule is counted by the new one.

**Reversal:** cheap: `effective_s` back to 720 for 06f; the cap is a bug fix and stays. The
learning rate and `specaug_in_human` are `Student` fields in `build_students.py`. Counting warmup
again is `warming=False` in `ftkit.train`.

## D117 — The judge pilot starts with its ceiling: oracle@8 on val must beat the top candidate by a point

Roadmap G1's pilot (2026-10-04) has an LLM pick one of the teacher's beam candidates. Before any
judge is chosen, `08_Judge_Headroom` measures the most any picker could gain. The teacher in
`teacher.json` decodes val with beam 8 under the standard length cap. The oracle takes, per clip,
the candidate with the fewest folded errors, a tie going to the higher-ranked one.

- **The rule, fixed before any candidate was decoded:** G1 goes on only if oracle@8 is at least
  1.0 point of folded val WER below the beam's own first choice (`judgekit.decide`,
  `GAIN_TO_CONTINUE`). Under that, the pilot stops and the null is reported. One point is the
  roadmap's "about a point", and about three times the run-to-run noise on val.
- **The rule reads val alone.** Gold and the public sets never take part in a choice (D105).
- **Reported beside the decision, on val, gold and the five public sets** (the owner asked for the
  whole picture before any judge is built, 2026-10-06): the standard greedy decode, the top
  candidate, a random pick, an MBR pick (the candidate closest to the others, which reads no
  reference and needs no LLM) and oracle@2/4/8, each with folded WER and S/D/I, raw WER, CER and
  plain WER and CER, each paired against greedy with episodes or speakers resampled; per clip
  class on val and gold and per the public sets' own column; how the gain spreads over the
  episodes (how many improve, the share the five most improved carry); how many candidates
  still differ once folded; the rank the oracle picked from; and the beam's real-time factor. The
  decode cost enters the later choice of a judge, not this rule.
- **Step 8 of the protocol.** G1 runs after the students and the report, so the numbered order
  gains a step. Each set's candidates are uploaded in shards to the Flex model repo
  (`<RUN_PREFIX>/judge-headroom/<set>/`); they are the judge's input if G1 goes on.
- **No judge model is named yet.** That choice is made on val against a KenLM rescorer, and only if
  this notebook says there is something to choose.

**Reversal:** cheap. The threshold is one Config value; changing it after the result is read
would need a new entry saying why.

## D120 — The LLM judge is a dead end: G1 is closed, and the vote is the panel vote

The owner's verdict (2026-10-06), after D118's pilot and D119's run on every set: not worth
writing up and not worth the compute. On all of val and gold both judges were worse than the
panel vote on folded WER, raw WER and CER; on nepali_cs Qwen tied it and Gemma lost, both lost
on Common Voice, and only Qwen won, on FLEURS, whose sentences come from Wikipedia
(findings.md, *An LLM judge is a dead end*).

- **G1 is closed.** The roadmap ran fusion (its steps 1–4) only if the pilot showed promise, and
  the n-gram picker on our train labels gained nothing either (its weight went to 0.05, which is
  the top candidate).
- **`08b_Judge_Pilot` and its kit code are removed** (the sample, the prompt and answer parsing,
  the Kneser-Ney picker, the kill rule), as experiment code is once its findings are written.
  Kept: `judgekit.mbr_index`'s voters (the panel vote), `sweep.paired_bootstrap`'s `level`, and
  `build_flex.load(..., flex=False)` for a notebook that runs another model. The judges' picks
  and reports stay on the hub under `flex-2026-09-30/judge-pilot/`.
- **V3 is the panel vote.** The pick among the teacher's 8 beams by least folded edit distance to
  the teacher's greedy decode and the four students' stage-2 transcripts was named V3 for being
  the third variant tried. Its numbers do not change: val 6.50, gold 10.16. Files already on the
  hub keep the key `v3`.

**Reversal:** moderate: the notebook and kit are in git history before this entry, and the
picks are on the hub; a reasoning or fine-tuned judge would need its own entry and a faster
decoder first.

## D121 — Speech-LLM students are shortlisted by a zero-shot gate and a training budget, not by zero-shot WER

Roadmap B's students so far are pure ASR models. The question for the paper is whether an
open-weight LLM that hears audio can be a student under the same recipe, and whether that family
does better. Before any was trained, `09_SpeechLLM_Bakeoff` shortlisted them (2026-10-07), with the
rules fixed here before any candidate transcribed a clip. It ran, its results are in findings.md
(*Speech-LLMs as students*), and the notebook and its kit were removed (D122); they are in git
history before that entry.

- **Candidates:** Gemma 4 E2B, E4B and 12B, Omnilingual LLM-ASR 1B and 3B (`nep_Deva`; the 7B as a
  zero-shot data point only, since full fine-tuning does not fit in 96 GB), Qwen3-ASR-1.7B and
  Voxtral Mini 3B. Omnilingual LLM-ASR 1B against 06e's Omnilingual CTC 1B is the cleanest family
  comparison (same encoder family and pretraining languages, CTC head against LLM decoder).
- **Breadth-first** (the owner, 2026-10-07: no GPU-hours before the performance is known). Round 1
  sweeps every candidate and every prompt zero-shot on a 300-clip val sample and stops at a table;
  round 2 times the training step of the shortlist alone (the owner's pick, by default those whose
  kept prompt passed the gate); round 3 decodes the shortlist's full val and gold for the record.
- **The gate, on the sweep's val sample** (`bakeoffkit.zero_shot_report`): among clips whose
  reference is at least half Devanagari letters, at least 90% must be answered at least half in
  Devanagari. Zero-shot WER ranks nothing: Whisper-turbo read 123% zero-shot and 15.02 on gold
  after training. Each candidate's prompt is chosen on the same sample. Gold is scored in round 3
  and reported, never used to choose (D105).
- **The budget:** training speed measured by `ftkit.speed_check` on the G4, projected over every
  epoch of both stages (`HUMAN_EPOCHS` 8 on the human hours, `DISTILL_EPOCHS` 6 on a 131 h
  mixture epoch, each with a val pass), must fit `BUDGET_H`, 24 GPU-hours, about four times what
  Whisper's two stages took. A candidate whose training step cannot be timed does not continue.
- **Recorded beside it:** empty answers, runaways (more than twice the reference's words plus
  ten), answers in another script, and the Latin share of answers against references. Answers
  are greedy, capped at 1.5× the densest train label's token rate, never retried, and cleaned only
  of whitespace and wrapping quotes.
- **The bars for the student notebooks that follow:** stage 1's first epoch at most 15.86 on val
  (Whisper's); the end of stage 1 at most 10.69 (Whisper), or 13.65 (IndicConformer) for a small
  model; worth its decode cost only if stage 2 beats Whisper's 15.02 on gold with an interval
  that excludes zero.
- **Framing for the paper:** "the recipe transfers to speech-LLM students" (stage 2 minus stage 1
  per family), not "LLM students are better": size, pretraining data and decoder differ at once
  across families.

**Reversal:** cheap. The gate share, the budget and the epoch caps were constants in
`bakeoffkit.py` and Config (git history); changing one after the results are read needs a new
entry saying why.

## D122 — Two speech-LLM students, the smaller sibling of each kind, and the report goes last

The bake-off (D121) ran on 2026-10-07 (findings.md, *Speech-LLMs as students*). The owner's
choice the same day:

- **Students: Gemma 4 E2B (`07a`) and Omnilingual LLM-ASR 1B (`07b`).** The rest are out. Gemma
  12B and Voxtral failed on language (Hindi, loops); Qwen3-ASR-1.7B started at 67 on gold.
- **No bigger sibling when the smaller does the job.** E4B read 3.1 points better than E2B on gold
  zero-shot and trains 1.6× slower; Omnilingual 7B read 1.4 better than the 1B. The rule is about
  kind, not size: two models of the same kind keep the smaller one, and models of different kinds
  are both kept. So Qwen3-ASR stays as 06b's 0.6B, which is the same model as the 1.7B at a
  smaller size; and Omnilingual LLM-ASR 1B joins 06e's Omnilingual CTC 1B, because a CTC head and
  an LLM decoder are different kinds (and the pair is the cleanest family comparison, D121).
- **Numbering.** The speech-LLM students are step 7 (`07a`, `07b`, `<NN><x>_Student_LLM_<name>`),
  the judge's headroom stays step 8, and the report moves from 07 to **09**: it reads every run
  of every earlier step, so it always sits last.
- **The bake-off notebook and `bakeoffkit.py` are removed**, as experiment code is once its
  findings are written. What the students still use moved to `llmkit.py` (tested in
  `test_llm_kit.py`): the answer's cleaning and length cap, `fit_items` (a micro-batch sized from
  two measured peaks; a run of out-of-memory probes leaked memory in the bake-off), where each
  answer sits after its own prompt, and Omnilingual LLM-ASR's loss turned back into a sum. Its
  outputs stay on the hub under `speech-llm-bakeoff-2026-10-08/`.
- **Each student keeps the bake-off's prompt** (Gemma's "Nepali" instruction, Omnilingual's
  `nep_Deva`), chosen on the val sample, and its training-step code, which ran on the G4. The
  stages around them follow D105/D106 unchanged, with D121's projected epoch caps (8 and 6).

**Reversal:** cheap. A student notebook is one `Student` entry in `build_students.py`; adding
the 1.7B Qwen or E4B back is one entry and a smoke run.

## D123 — Every set's WER is also given within each word-rarity bucket

The owner asked on 2026-10-08 for a WER breakdown by how often each word occurs in the training
corpus, in the standard evaluation, on val, gold and the public sets, backfilled for the best
weights of each model.

- **Each bucket's own WER, against the set's labels.** For one model on one set, a bucket's WER is
  its errors per 100 of the set's reference words in that bucket, split into S/D/I. The labels are
  the truth; no model is a reference, and no bucket is read as a difference from another model.
- **Buckets** never, 1-9, 10-49, 50-99, 100+ (the owner's 1-10, 10-50, 50-100 with each edge given
  to the upper bucket). A substitution and a deletion are the reference word's, an insertion the
  inserted word's, which is the only word it has; a bucket with insertions and no reference words
  has no WER.
- **One corpus for every model**: the export's human train labels (stage 1) plus the frozen
  teacher's filtered pseudo-labels (stage 2), counted by `fold.spelling_key` after the scorer's
  tokenizer, so a bucket holds the same words in every model's table. The file names the export,
  the labels and the fold; a count file from another fold is refused, because spelling keys move
  with it.
- **The counts live in the dataset's `harness/word_counts.json`**, beside the scorer, so
  `ftkit.download_dataset` brings them to every notebook with no new fetch. 05_Teacher writes them
  when its labels are final; before any teacher has run there are none, and runs are scored
  without the block and say so.
- **Computed from the error rows, not stored in them.** The counts depend on the teacher and the
  export; the rows only on the transcripts. So the miner version did not change, and a backfill is
  a read of each run's error files (`scripts/backfill_rarity.py`).

**Reversal:** cheap. Remove the `rarity` block from `evalkit` and the report; the error files are
untouched by it.

## D124 — Overlapped speech and the diarizer are retired; the last experiments are dosing, the council, the commercial recognisers and switch points

On 2026-10-20 the owner closed the project's crosstalk and diarization work: no model is
fine-tuned for overlapped speech, and the diarizer is not fiddled with. D100 had put both on hold,
with the overlap models waiting on a diarizer that could be trusted, and nothing since has made
one more likely. Crosstalk stays what the gold measures it as: single-stream, everything said, the
largest remaining error (17% at 5–15% overlap, 29% above).

- **Retired from the roadmap:** D (DiCoW, SOT-DiCoW, target-voice extraction, the multitalker
  Parakeet method, diarizing commercial baselines), F (the episode-205 diarizer test), and C1 and
  C2 (the synthetic-mix bench and conversation timing that existed to develop D). Each keeps a line
  in the roadmap's retired list, so citations resolve. `docs/custom-arch.md`, whose revisit trigger
  was F, is deleted (its text is in git history); the one idea in it that needed no diarization,
  a Flex with a cut-down decoder, became roadmap K.
- **Kept as they are:** ingest's diarization (D79) and the speaker turns as automatic metadata
  (D100); the D98/D99 code, which nothing uses now; `augment.py`'s crosstalk mixer.
- **What is left, in the owner's order:** data dosing on Flex, hours against voices (roadmap H);
  the council once every student is in, with a word-level vote as a variant and no commercial
  voter on val or gold (B step 6); the three commercial recognisers on the public sets, the only
  references they did not write (I); WER at code-switch points (J); a smaller Flex distilled
  with the teacher's per-token probabilities (K). Still open from before: 07b,
  06b and the low-risk fold rules (G2).

**Reversal:** cheap in code, nothing was deleted. Reopening D or F needs a new decision entry and
the sections restored from git history.
