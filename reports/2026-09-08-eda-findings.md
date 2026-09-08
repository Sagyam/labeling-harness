# EDA findings — 2026-09-08

Analysis of the full labeled export (`exports/analytics/analytics.jsonl`, exported
2026-09-08T12:28:11Z at commit `30e916e`) and a re-run of `notebooks/01-EDA.ipynb`, plus
three checks the notebook does not currently perform: an episode-clustered bootstrap, a
language-profile breakdown per episode, and a screening-selection check.

Export: analytics 4,797 rows / training 3,858 / gold 927 / error_mining 12. All 66 files
pass sha256.

## Where the corpus moved since the previous committed run

| | previous run | now |
|---|---|---|
| labeled | 3,490 seg / 11.86 h / 21 eps | **4,797 seg / 15.49 h / 28 eps** |
| gold | 1,171 seg / 3.98 h / 9 eps | **927 seg / 2.80 h / 5 eps** (93% of the 3 h target) |
| verified edit rate | 26.9% | 23.9% |
| mean CMI | 23.1 | 20.3 |
| no-English segments | 7.4% | **16.1%** |

The gold shrink is the four `demote_from_gold` runs on 2026-09-08 (show-level host overlap),
all reasoned from coverage metadata before any WER existed. No show now sits in both pots —
that leak is closed.

## Findings

### 1. One 34-segment episode decides the system ranking

`pixel_11_pro_vs_galaxy_s26_ultra` is **100% English** — zero Devanagari tokens in the
reference. mai-transcribe-2 scores 85.7% WER on it. Removing that one episode from gold:

| system | gold WER as exported | gold minus pixel_11 |
|---|---|---|
| elevenlabs-scribe-v2 | 24.29% | 26.28% |
| gemini-3.8-flash | 29.41% | 31.03% |
| mai-transcribe-2 | **35.82%** | **29.80%** |

mai moves from clear last to second. Corpus-wide the same holds: 30.12% → 25.88% once
English-dominant references are dropped.

### 2. mai's error is orthographic, not acoustic

On the 390 segments whose reference is >=90% Latin, mai emits **78.3% Devanagari** — it
transliterates English into Devanagari (`So throughout 2026` -> `सो थ्रूआउट 2026`). WER there
is 80.4%, against elevenlabs 5.5% and gemini 14.0%. This is the D51 failure mode, live on a
route with no `restore_script` applied. The current number measures mai's spelling
convention, not its recognition.

Share of each system's measured error attributable to script rather than recognition
(8.2% of the corpus is English-dominant):

| system | WER all | WER non-English | delta |
|---|---|---|---|
| elevenlabs-scribe-v2 | 22.94% | 23.57% | -0.63 pts |
| gemini-3.8-flash | 28.60% | 29.64% | -1.04 pts |
| mai-transcribe-2 | 30.12% | 25.88% | **+4.23 pts** |

### 3. The CMI metric conflates two opposite populations

Of the 1,231 segments at CMI <= 5: **801 are near-pure Nepali, 349 are near-pure English.**
Section 8b's "WER by code-switching density" therefore reads mai at 54.97% in the `none`
tier — that is the English-only material, not monolingual Nepali. The apparent
"WER falls as code-mixing rises" pattern for mai is that artifact.

### 4. The disagreement gate improved slightly and still cannot clear the bar

Best signal AUC 0.665 (up from 0.652). At 80% auto-approval, 21.2% edited clips are
admitted; a perfect ranking still admits 4.9%. The base edit rate binds, not the ordering.

## Concerning trends, ranked by threat to the paper

### 1. The confidence intervals are computed the wrong way, and fixing them erases the headline claim

The notebook bootstraps over *segments*, but segments within an episode are not independent
— same speaker, same mic, same acoustics. With an episode-level cluster bootstrap over the
5 gold recordings (4,000 resamples):

| system | WER | segment CI (reported) | **episode-cluster CI** |
|---|---|---|---|
| elevenlabs-scribe-v2 | 24.29 | [22.8, 25.7] (±2.9) | **[14.7, 31.2] (±16.5)** |
| gemini-3.8-flash | 29.41 | [27.9, 30.8] (±2.9) | **[22.1, 36.7] (±14.6)** |
| mai-transcribe-2 | 35.82 | [33.0, 38.8] (±5.8) | **[24.5, 61.7] (±37.2)** |

The notebook's claim *"its interval does not overlap the runner-up — the ordering is
resolved at this corpus size"* does not survive. The intervals overlap heavily. Per-episode
WERs confirm it — elevenlabs ranges 1.8% to 36.9% across the five recordings:

| episode | segs | mean CMI | elevenlabs | gemini | mai |
|---|---|---|---|---|---|
| best_burger_in_ktm | 50 | 31.4 | 17.8 | 39.4 | 39.8 |
| mailo_bajeko_kalo_maya_herne_katha | 299 | 4.7 | 36.9 | 39.8 | 42.6 |
| minister_mahabir_pun_unfiltered | 281 | 2.6 | 20.9 | 25.5 | 26.6 |
| pixel_11_pro_vs_galaxy_s26_ultra | 34 | 0.0 | 1.8 | 4.7 | 85.7 |
| हिमालयन (nepal_vox) | 263 | 25.2 | 24.7 | 27.1 | 22.8 |

A reviewer who asks for speaker-clustered intervals will find this. It is the most likely
reason the ASR comparison gets rejected. **A 5-recording test set cannot resolve system
ordering.**

### 2. Gold contains ~1.2 h of actually code-mixed audio

Gold's 2.79 h by Latin-token share of the reference:

| episode | hours | Latin share |
|---|---|---|
| minister_mahabir_pun | 0.730 | 1.4% |
| mailo_bajeko | 0.724 | 3.3% |
| pixel_11_pro | 0.184 | 100.0% |
| हिमालयन (nepal_vox) | 0.969 | 25.0% |
| best_burger_in_ktm | 0.186 | 44.9% |

That is **1.45 h near-monolingual Nepali, 0.18 h pure English, and 1.16 h genuinely
code-switched** — 42% of the benchmark by duration, for a paper about code-switching.

### 3. One show is 63% of the training pot

`sushant_pradhan`: 4 episodes, 8.02 h labeled, out of 12.69 h in train (and 44% of all
segmented audio). Anything trained or fine-tuned on this is a model of one podcast host and
one acoustic setup. Cross-pot leakage is closed; within-pot concentration is not.

### 4. The edit-rate numbers have a selection problem

Screening is applied to 31.3% of the train pot and 0% of gold, and it targets
low-disagreement clips — so the "verified only" remainder is enriched for hard clips.
Three estimates:

- train, all rows (screened counted as accepted): **17.7%** — too low; unlistened accepts
  are not evidence the hypothesis was right
- train, verified only: **25.7%** — too high; the easy clips were skimmed off first
- **gold, every clip listened to, nothing screened: 18.9%** — the only unbiased edit rate
  in the corpus

The notebook calls 23.9% "the honest figure". It is honest about screening but still biased
upward as an estimate of the corpus edit rate. Quote 18.9% with the gold caveat, or the
paper over-claims how often the ASR seed was wrong.

### 5. Screening is wildly non-uniform per episode

0% on eleven episodes; 65.6% on `nepal_has_money`, 65.2% on `nepals_most_terrifying_flood`,
63.3% on `ep_623`, 60.2% on `ashima_poudyal`. Whatever drove that (annotator fatigue,
session ordering) is an uncontrolled variable inside the corpus.

### 6. The corpus is drifting monolingual

No-English segments went 7.4% -> 16.1% and mean CMI 23.1 -> 20.3 in one ingest round.
Annotation budget is being spent on material the paper is not about.

### Smaller items

- `nepal_lead_tv` has 336 segments / 0.86 h and is still `pot=unassigned`.
- `notes` and `script_conflict_rate` are 100% null across the export.

## What to acquire next

Binding constraints: gold can only grow from **new** episodes of **shows not already in
train** (D63 rule 3), and an episode holding any screened label is excluded from gold
selection (D65). So the list is about *new shows*, not more hours.

**Highest priority — gold breadth.** Roughly 10–15 gold episodes from 10–15 distinct shows,
to get episode-clustered intervals narrow enough to separate systems. Episode count matters
more than hours: 15 x 12-minute videos beats 3 x 1-hour ones. Annotate fully verified,
never screened.

**Target profile:**

- **Genre:** anything but podcast. Podcast is 11.25 h of 15.49 h. Interview, commentary,
  review, documentary, talkshow and cooking are all under the 1 h floor; vlogs, news
  bulletins and street interviews are absent entirely.
- **Latin share 20–50%** — the `हिमालयन` / `best_burger` / `claude_for_beginners` band.
  Skip pure-Nepali documentary and political monologue (1.45 h already in gold), and skip
  English-only tech reviews — they teach nothing about code-switching and actively corrupt
  the ASR comparison (finding 1).
- **Speakers:** female is 1.75 h of 15.49 h (11%) across 8 episodes. Under-20, 60–79 and
  80+ are one episode each. Female-hosted lifestyle, beauty, food and student content covers
  gender, age and the thin `lifestyle_relationships` / `health_wellness` /
  `entertainment_media` topics at once.
- **Hard rule:** one episode per show until there are ~25 shows. No more
  `sushant_pradhan`, `a_pause_in_time`, or `gadgetbyte_nepali`.

**Highest-value single acquisition:** a female-hosted, non-podcast, 20–50% Latin episode
from a show never ingested, 10–20 minutes, fully verified into gold. That moves gold episode
count, gender, genre, topic and code-mixing density at once.

### 5. mai's orthography can be scored around, read-only (section 8d)

`script_restore.py` is an LLM rewrite — a call per segment, `llm_requests` rows, real money — so
it cannot run from a notebook. A deterministic alternative does the same job at analysis time
without touching the database or the export.

Align each hypothesis to the reference with the same edit script the WER already uses, then take
every **substitution** where one side is Devanagari and the other Latin. Romanize the Devanagari,
reduce both sides to a coarse consonant skeleton (Nepali writes English `v`/`w` as `भ`, `f` as
`फ`, `z` as `ज`, so those classes merge) and, if they are close enough, count the substitution as
the same word in the other script rather than a recognition error.

Corpus-wide, seed-excluded:

| system | raw WER | script-folded | oracle floor | fold moves it |
|---|---|---|---|---|
| elevenlabs-scribe-v2 | 22.94% | 21.67% | 20.74% | 1.27 |
| gemini-3.8-flash | 28.60% | 25.99% | 24.33% | 2.61 |
| mai-transcribe-2 | **30.12%** | **24.69%** | 20.53% | **5.43** |

mai moves from third to second. On English-dominant references it goes 79.98% → 42.88%, with an
oracle floor of 7.32%.

Three guards keep this from being a thumb on the scale:

- **It is applied identically to all three systems.** The two that do not transliterate move 1.27
  and 2.61 points; mai moves 5.43. A general WER discount would move all three equally.
- **The threshold is swept, not chosen.** The notebook prints folded WER at 0.0 / 0.25 / 0.34 /
  0.5 / 0.6 and plots the curve against each system's oracle floor, so no single setting is
  load-bearing.
- **The matcher is conservative.** It still rejects `the`/`द`, `of`/`ऑफ`, `they`/`दे`, `now`/`नाउ`
  — all real matches — so the folded column *overstates* the error a perfect restoration would
  leave. The truth is between the folded number and the oracle floor. The notebook prints the
  accepted and rejected pairs so a reader can audit them.

The one arguable acceptance is `and`/`अनि`: mai wrote the Nepali word, which is a translation
rather than a transliteration. It is 22 of 3,346 accepted pairs.

**Raw WER stays the headline number.** Anything consuming these transcripts receives the
Devanagari, so the raw figure is the one that describes the artifact; the folded figure describes
the recogniser. Which one belongs in the paper depends on which claim is being made — "this
system is a good Nepali-English recogniser" is the folded number, "this system produces usable
code-switched transcripts" is the raw one.

## Repository changes

Applied to `notebooks/01-EDA.ipynb` (re-run clean, 37 cells, no errors):

1. **Episode-clustered bootstrap** beside the segment-level one in section 8a. Both intervals
   are reported and plotted; the verdict on whether the ordering is resolved now reads the
   clustered one, and the notebook warns when gold holds fewer than 10 recordings. Section 8a
   also gained a per-episode gold WER table showing the between-recording spread the cluster
   interval is reading. The point estimate is now the actual micro-averaged WER rather than
   the mean of the bootstrap draws.
2. **The monolingual bucket is split** in section 8b: `monolingual Nepali` and
   `monolingual English` replace the single `none (0-5)` tier, with a `latin %` column, and
   section 6 reports the no-Nepali share alongside the no-English one.
3. **New section 8c, "Is it recognition error, or is it the wrong script?"** — scores every
   system with English-dominant references dropped, reports each system's Devanagari share of
   its own output on that material, and tests whether the gold ordering survives. It reports
   `mai-transcribe-2 ... 79% Devanagari ... dropping that material CHANGES the gold ordering`.
4. **New section 8d, "Scoring the words instead of the spelling"** — the deterministic
   script fold described above, with the control, the threshold sweep, an audit sample of
   accepted and rejected pairs, and `figures/eda_05b_script_folding.png`.
5. The summary in section 12 now quotes episode-clustered intervals and prints a
   `SCRIPT, NOT RECOGNITION` block, with the folded range, whenever a system moves by more
   than a point.

Still outstanding, outside the notebook:

- Decide whether `restore_script` should be applied to the `mai-transcribe-2` route at ingest.
  Section 8d scores around the problem read-only; it does not change what the export contains,
  and a downstream consumer of these transcripts still receives Devanagari English.
- Assign a pot to `nepal_lead_tv` (336 segments, 0.86 h, still `unassigned`).
- Decide whether `pixel_11_pro_vs_galaxy_s26_ultra` — 100% English — belongs in a
  Nepali-English code-switching benchmark at all.

Working scripts for the original ad-hoc analyses are in the session scratchpad
(`deep.py`, `deep2.py`, `deep3.py`); their substance is now in the notebook.
