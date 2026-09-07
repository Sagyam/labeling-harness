# Next experiments: lowering the base edit rate

## Why these two

The screening goal is that ~80% of clips be good enough to accept without listening. As of
2026-09-07 that is not reachable by ranking, and the reason is arithmetic rather than algorithm
quality. Measured over the 1,363 verified labels:

| auto-approve | clips | edits leaked | error rate in the auto pool |
|---|---|---|---|
| 50% | 681 | 72 | 10.6% |
| 70% | 954 | 157 | 16.5% |
| **80%** | **1090** | **225** | **20.6%** |
| 90% | 1226 | 301 | 24.6% |

The base edit rate is 29.7% (405 of 1363). You cannot auto-approve 80% of a population where 30%
needs work, however well you sort it — D67 moved the 80% figure from 22.8% to 20.6% and that is
most of what sorting has left to give.

So the quantity to attack is **how often the seed needs editing at all**, not the order clips are
shown in. Both experiments below try to lower it. Both are measured the same way, against the same
number:

> **Success metric.** Over the 1,363 verified labels, the share where `final_text` differs from
> what the annotator was shown. Baseline 29.7%. Anything that does not move this number does not
> help, whatever else it improves.

Run the measurement step of each experiment **before** writing any code. Both measurements are
free and use data already in the database; experiment 2's implementation is not free.

---

## Experiment 1 — apply the normalizer on the way in, not only on the way out

### The idea

D64 applies `config/normalization.yaml` (norm-v3, 142 rules over 75 stems) to `final_text` at
export. The annotator never sees it. So they are hand-fixing `चैँ` → `चाहिँ` and loanword spellings
that the exporter would have fixed anyway, and every such fix is a full open-listen-edit-save
cycle spent on a change that was already automated.

A previous check found only 4.8% of past edits were *entirely* redundant with the table — that is,
seed and final became identical after normalization. That is the wrong question. The right one is
how many edits were **partly** redundant: how much of the typing was avoidable, and how many clips
would have dropped below the threshold where the annotator bothers to intervene.

### Step 1 — measure (free, no API calls)

For each of the 1,363 verified labels, compare three strings: the seed as stored, the seed with
`normalize_text` applied, and `final_text`.

```python
from app.services.normalize import load_ruleset, normalize_text

rs = load_ruleset()
# per label:
#   raw   = seed.text_raw
#   shown = normalize_text(raw, rs)          # what the annotator would see under this experiment
#   final = normalize_text(label.final_text, rs)   # normalize both sides; export does anyway
```

Report four numbers:

1. **Clips where `shown == final`** but `raw != final`. These are edits the experiment deletes
   outright. Expect this to be near the 4.8% already measured — it is the floor, not the result.
2. **Token-level edit distance**, `raw→final` versus `shown→final`, summed. This is the typing
   saved. A large drop here with no movement in (1) still matters: it is the difference between
   retyping a sentence and fixing one word.
3. **Clips where the remaining diff is a single token.** These are candidates for a "one-token
   diff" fast path in the UI, which is a separate and probably larger win than the normalization
   itself.
4. **Clips where `shown` is *further* from `final` than `raw` was.** This is the failure mode that
   kills the experiment. It means a rule fires somewhere the annotator deliberately kept the other
   spelling. Any non-zero count here needs the specific rules named before proceeding.

Number (4) is the one that decides whether to continue. The table was reviewed token by token
across 6,205 distinct Devanagari tokens, so it should be small — but `config/normalization_candidates.yaml`
exists precisely because some tokens (`बस` = "bus" or the verb "to sit", `भोट` = "vote" or Tibet,
`डे` = the जे → day trap) cannot be decided without seeing the occurrence. If any of those leaked
into the live table, this measurement finds them.

### Step 2 — implement, if the measurement holds

**Do not write the normalized text back.** Invariant 2 says hypotheses are immutable, and D64's
whole argument is that the table stays revisable because nothing depends on having been applied.
Normalizing `asr_hypotheses.text_raw` in place would forfeit both.

Apply it at serialization instead — the seed text as it leaves the API for the editor:

- `app/api/serializers.py` builds the task and queue payloads; that is where the seed text is
  chosen for display.
- `app/services/labeling.py` handles the decision path — check whether an `accepted_unchanged`
  decision needs to record what was shown rather than what was stored.

The consequential design question: **what does `accepted_unchanged` mean once the shown text is
not the stored text?** Today it means "the seed was right". Under this change it would mean "the
normalized seed was right", which is a different claim, and the export already normalizes so the
two agree in the output. Decide explicitly and record it as a decision entry; do not let it be
implied by the code.

### Cost and risk

Free to measure, free to run. The risk is not cost but silent semantic drift in what a label
asserts — see the question above. Reversible by removing one call.

---

## Experiment 2 — repair the Devanagari-English in gemini seeds

### The idea

`asr_gemini_flash` writes English words in Devanagari far more than the other two. Measured on the
pending gold pot:

| seed system | n | mean `roman_gap` | mean `seed_orphan_rate` |
|---|---|---|---|
| elevenlabs-scribe-v2 | 187 | 0.082 | 0.183 |
| mai-transcribe-2 | 187 | 0.110 | 0.204 |
| **gemini-3.8-flash** | **175** | **0.423** | **0.227** |

Five times the rate. Since the corpus policy is English-in-Latin (D64), every one of those is an
edit the annotator will make. Gemini seeds are a third of the gold pot, so this is a large,
concentrated, *mechanically identifiable* slice of the base rate.

`app/llm/script_restore.py` was built for exactly this and kept when D51 removed the composite
system it served. `restore_script(session, tokens, route=..., config=...)` returns one output token
per input token, in order, and rejects any rewrite that does not — which is what lets each restored
word keep its own acoustic span with no re-alignment.

### Step 1 — measure the ceiling (free)

Before spending anything, find out how much is actually there. The other two systems already tell
you what the Latin form should be for most of these tokens, for free:

- For every gemini-seeded pending clip, compute `roman_gap` (`app/services/lexical.py`).
- For each Latin token the other two agree on and gemini lacks, check whether gemini has a
  Devanagari token that is plausibly the same word. A cheap proxy: the token is Devanagari, the
  others agree on a Latin token, and neither appears in the other's text.
- Count how many clips would drop below the review threshold if every one of those were repaired.

That count is the **ceiling** on this experiment. If it is small, stop here — the concentration in
the table above could be gemini being wrong in ways romanization does not fix.

### Step 2 — prefer a table over an LLM call

The obvious implementation is one `restore_script` call per gemini seed. Resist it first.

norm-v3 was built by reviewing 6,205 distinct Devanagari tokens by hand and writing 142 rules. The
same method applied to gemini's Devanagari-English tokens gives a deterministic, free, re-runnable,
reviewable table — with no per-clip cost, no failure mode, and no new dependency. Mine the
candidates automatically, but **approve them by hand**: the header of `config/normalization.yaml`
records that the automated miner proposed `जे → day` from a single mishearing, and `जे` is a common
Nepali word.

Only reach for `restore_script` for what the table cannot cover — tokens whose correct form depends
on the sentence rather than the token.

### Step 3 — if you do run the model

Read the gotchas in `AGENTS.md` before the first call; several are specific to this path.

- **It spends real money per clip.** Use `dry_run: true` in `config/llm_routes.yaml` while wiring
  it up, and a handful of clips for the first real run.
- It needs a text route in `config/llm_routes.yaml`, and every call writes an `llm_requests` row
  (invariant 6). D66 moved this route to Vertex; check it still resolves before a batch run.
- It raises `LlmRequestFailed` when it cannot return an aligned token list after
  `MAX_ALIGNMENT_ATTEMPTS`. Decide up front whether that fails the clip or leaves it unrepaired —
  D46's "half an episode is not a cheaper episode" argument applies to seeds too.
- Same immutability rule as experiment 1: repair at serialization, or write a *new* hypothesis
  row for the repaired text. Never overwrite `text_raw`.

### Cost and risk

Steps 1 and 2 are free. Step 3 is one paid call per gemini-seeded clip — 175 in the current gold
pot. The real risk is that a repaired seed is *wrong* in a new way and the annotator accepts it
without listening, which puts an error into gold. Gate it: repaired seeds must not be screenable
(invariant 5 already refuses a screened decision on gold with a 409) until a sample has been
verified by ear.

---

## Order

Run experiment 1's measurement first. It is free, it takes an afternoon, and it is the one that
could turn out to already be most of the answer — the normalizer exists, is reviewed, and is
currently doing nothing for the person doing the work.

Then experiment 2 step 1, also free, which tells you whether step 2 is worth building.

Do not start experiment 2 step 3 until both measurements are in. It is the only one that spends
money and the only one that can put a new class of error into the corpus.

## Open item from the same session

Not an experiment, but unresolved and cheap: the D67 score pushes gemini-seeded clips up the gold
queue (54 of the top 100, from 17% of the queue). Gold rotates seeds deterministically so the
benchmark is not anchored to one system; working top-down and stopping partway would verify mostly
gemini seeds. Options were: leave it, stratify the gold queue across seed systems, or drop
`roman_gap` for gold only. Both experiments above reduce gemini's `roman_gap`, so a decision here
may become moot — which is a reason to decide it after, not before.
