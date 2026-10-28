# Sociolinguistic value of the corpus

The corpus was collected to fine-tune an ASR model. This note records what it can and cannot
support as material for a sociolinguistic study of Nepali–English code-mixing, what to collect
next to make such a study robust, and what metadata is worth adding at ingest. The measurements
behind it are in `notebooks/02_Sociolinguistics.ipynb`, run on the export of 2026-09-30, and every
number below is printed by that notebook, so a rerun on a newer export revises it. Who a voice is
(gender, age bracket, role) is resolved once, by the harness, and written on every turn of the
export (D107): the Voices page and the notebook read the same answer, and a difference between
them means the export is stale. The export also carries the show each recording was filed under
and every label's words with their timings (D108).

## What the corpus supports today

The unit of a sociolinguistic claim is a speaker, not a clip. On the 2026-09-30 export (342
recordings, 358 voices):

| | |
|---|---|
| voices with 300+ words attributable to one speaker | 171 |
| of those, gender known | 171 (35 women, 136 men) |
| of those, age bracket known | 171 |
| of those, host/guest role resolved | 105 |
| female voices aged 40–59 / 60–79, any talk time | 21 / 2 |
| genres with three or more such voices | 14 of 17 |
| multi-voice episodes with both genders present | 24 |
| recurring hosts with 5+ host–guest episodes | 3 |
| shows those 171 voices mostly speak on, a short-form upload counted as its own | 97 |

What is recorded about a person is deliberately narrow: gender and a twenty-year age bracket,
typed by the annotator from watching the episode (D58) or set on the voice after listening to it
(D104), and a role. The owner listened to the voices and set the gender of 175 of the 358 by
ear, which is what took gender from 125 of those 171 voices to all of them; where a declared row
had already reached a voice, the two agreed on every gender and on all but four ages. Name, dialect and
origin were refused (D56). Everything else the notebook uses is derived from the reference
transcript and the diarizer's turns: English share, switch points, English run lengths
(single-word insertion versus multi-word alternation), English stems carrying Nepali morphology
(*phonesहरूको*), English function-word share, second-person address forms (*hajur / tapāī̃ /
timī / tã*), discourse markers in both languages, host–guest accommodation, and the gender
composition of the room.

**Supported.** A descriptive paper on the *form* of Nepali–English mixing in produced media
speech: how much of the English is lexical insertion and how much is alternation into English
syntax (37% of the English inside a mixed sentence is a single word, 18% sits in runs of six or
more), which Nepali suffixes attach to English stems, which discourse markers are borrowed (*so*
is the most frequent English token in the corpus, ahead of *the*), how address forms differ
between host and guest, and how all of these differ between the fourteen genres that hold enough
voices. To these the timed words add what happens at the switch itself (next section).

What the corpus says about *people* is thinner than the group medians suggest. The notebook fits
one model per measure, with age, gender and genre together and a random intercept per show, and
corrects for every coefficient at once. Of 24 coefficients five are under 0.05 on their own and
none survives the correction. Two things are still worth carrying forward:

- **Accommodation.** Over 34 host–guest episodes a host's English share tracks the guest's
  (Spearman 0.81), and still does with the topic's usual English share taken out of both (0.79).
  But half of those episodes belong to three hosts, and within a host it holds for two of the
  three (1.00 over six episodes, 0.90 over five); the third hosts an interview series whose
  guests barely mix at all (0.14 over six). Suggestive, on very few episodes per host.
- **Younger speakers may switch more often at the same English share.** English share itself
  does not move with age once genre is fixed (−0.6 points per twenty-year bracket, interval −3.8
  to +2.5): the older voices sit in the interviews, speeches and news, which are nearly all
  Nepali. The switch rate, with English share held fixed, falls by 1.2 switches per hundred
  words per bracket over all voices (p = 0.03, which the correction does not leave standing).
  Where there is English to switch into it is much clearer: −4.3 inside the podcasts (48 voices
  on 17 shows, p = 0.0006), −3.9 among guests alone, −2.6 on the clips that were played and read
  rather than screened, and between −1.7 and −0.7 whichever show is left out. Those checks were
  chosen after looking, so this is a hypothesis for a planned test, not a result.

**Not supported.** A Labovian social-stratification study. There is no region, class, education
or first language by design. Gender is confounded with genre: women hold between none and 74% of
the speech, depending on the genre, and only two of the 35 usable female voices are over 60.
With every voice's gender known and genre and show held fixed, women and men differ by −0.4
points of English share (interval −5.6 to +4.9), and by nothing on any other measure that
survives the correction. At the observed between-speaker spread a plain two-group comparison
detects a difference of about 9 points of English share (it was 12 before the by-ear values,
and 21 on the 2026-09-18 export), and a five-point difference would need about 165 voices per
group.

## The moment of the switch

The label's words carry their timings (D108), from the harness's forced aligner. A single word
boundary is known to about one 20 ms frame, and against the one recogniser that reports its own
timings a word's start differs by 37 ms on average and its end by 66 ms. That is too coarse for
one pause and ample for the average of tens of thousands: a pause a listener hears is 200 ms or
more.

- **Which words.** 52,900 switches into English inside a sentence use 7,059 different words and
  are spread thin: the ten commonest are 6% of them, the hundred commonest 24%, and 38% of the
  words occur once. 68% of the switches are one word long. The commonest are *phone, I, video,
  like, time, price, first, type*; *phone* comes from 16 voices where *video* comes from 82. An English word that opens a sentence is a different list: *so* alone is 19%
  of them, then *I, okay, thank, and, but*. Before the English stands a demonstrative, a
  genitive or a topic marker (*चाहिँ, को, यो, त्यो*); after it a postposition or a suffix (*मा, को, हरू*), and
  one time in ten a form of *गर्नु*, which is how an English word is made a Nepali verb.
- **There is a hesitation before a switch into English.** A gap of 200 ms or more comes before
  16% of switches into English against 7% of Nepali-to-Nepali boundaries, in the median voice
  of the 100 with enough of both, and the gap is longer in every one of those voices. On the
  second recogniser's own words and timings it is 15% against 7%. It is not the word either:
  the *same* English word waits 42 ms longer (35 to 48) after a Nepali word than after an
  English one, in 88% of the 260 words common enough to compare. And it grows with rarity, as a
  search for a word should: 14% long gaps before the 50 commonest English words, 21% before
  the rarest. A filler (*uh, um, अँ*) precedes 10.5 in a thousand English words and 6.1 in a
  thousand Nepali ones.
- **Going back to Nepali is a little easier, not much.** The raw gap after English is shorter,
  but most of that is spelling: *मा* and *को* are written onto a Nepali noun and after an English
  one. With the word held fixed the same Nepali word comes 7 ms sooner (3 to 10) after English.
- **The pace does not change.** The same Nepali word lasts 1.6 ms longer just before English
  and 1.4 ms less just after it, both inside 4 ms of zero on a word that typically lasts 221 ms:
  under 2%, and the data would have shown a change of that size. An English word that opens a
  run is about 5 ms longer than the same word inside one, and one that closes a run about 10 ms
  shorter.

## What to collect next

Every item below adds *speakers who break a confound*. Hours of an existing speaker add
nothing to a paper's n.

1. **Female and older guests on the podcasts already in the corpus.** Of the 171 usable voices
   35 are women and two are women over 60; most of the voices over 60 are interview guests.
   Senior professionals, women entrepreneurs, teachers, doctors and retired officials as podcast
   guests fill those cells inside a genre where age and gender can be compared at all.
2. **Mixed-gender rooms.** 24 episodes have both genders in the room against 64 that are men
   only and 4 that are women only. Female guests on the male-hosted series, and a female-hosted
   podcast with male guests, separate gender from show.
3. **More voices in the genres one person dominates.** Review has seven usable voices but one
   of them holds 72% of its words; commentary, audiobook, sketch and streaming are each over
   half one voice, and audiobook, street interview and FM radio have fewer than three usable
   voices. Two or three more voices in each turn an anecdote into a group.
4. **More episodes per recurring host, and the same guest on more than one show.** Ten or more
   episodes per host, a third recurring host, and guests who appear with different hosts give
   the accommodation design where the guest is held constant and the host varies.
5. **Speech that is not produced media.** Vox pops, call-in shows, live streams, panel
   discussions and classroom recordings add register range; news interviews and press
   conferences anchor the Nepali-heavy end, since topic drives English share and the current
   topics skew to tech and business.
6. **Peer and intimate talk.** The low address form *tã* appears in 38 of the 171 voices, a
   third of them in podcasts, and *timī* in 60. Friends chatting, comedy and gaming streams
   sample the *timī* and *tã* registers that interview formats never reach.

## Metadata worth adding at ingest

In order of value. Each is about the recording or is a closed, coarse fact; none reopens D56.

1. **Link declared speaker rows to diarized voices.** Done for gender and age. The forcing
   rules (one row meets one voice; a single host row meets a single recurring voice; the
   remaining rows agree) live in the harness (`app/services/inventory/resolve.py`, D91), the
   owner gave every voice the rules could not reach a gender and an age bracket by ear on the
   Voices page (D104), and the export writes the result on every turn (D107): 357 of 358 voices.
   What is still open is **role**: it comes from the declared rows alone, and reaches 105 of the
   171 usable voices.
2. **Upload date and view count.** The show is done: every episode was filed under one at
   ingest and the export carries it (D108), 80 in all, though the 131 short-form uploads share
   one bucket and are not a show. The upload date is not stored. `episodes.published_at` holds
   the day of ingest, and the YouTube probe still reads the upload date and the uploader and
   drops them; storing them, and probing the source URLs again for the episodes already in,
   gives change over time and a real channel for the short-form uploads. A view-count bucket
   is an audience proxy. All describe the recording, not the person.
3. **Interaction format and relationship, closed vocabularies.** Format: monologue, interview,
   panel, call-in. Relationship: first meeting, acquainted, unknown. Both are visible in the
   episode and both predict address form and accommodation.
4. **Facts the speaker states about themselves on air, with a provenance flag.** Medium of
   schooling and time lived abroad are the strongest known predictors of South Asian
   code-mixing, and neither can be seen. Guests often say them. A field that records only
   what was said in the episode, defaulting to unknown, is honest data and stays clear of the
   guessing D58 refused.
5. **Coarse professional domain of a guest** as the host introduces them: tech, business,
   medicine, arts, politics, education, other. A class proxy without being class. Keep it
   closed and coarse: with the voice and the topic it could become a re-identification key in
   a corpus this small.

Not to add: dialect, region or origin (D56), name (D56), anything inferred from audio (D58).

## A threat to validity to settle before writing

The mixing variable is the script of each reference word, and script is exactly what the three
recognisers disagree on: whether a word is written *phone* or *फोन* is partly a transcription
convention inherited from the fused seed (D74), not a fact about the speaker. Most of train was
screened on the disagreement signal without being played (D63). For a paper, define the mixing
measures on the verified tier, or hand-check script choice on a sample, and state the
convention explicitly. The gold pot is fully verified and is the natural sample for that check.
The notebook makes that check for the one age result it carries forward, which holds on the
verified clips alone; it has not been made for the rest.

The timings have their own version of the same worry: the pause before a switch is measured by
an aligner, and an aligner could treat the two scripts differently. That is why the pause is
measured a second time on a recogniser that reports its own timings for its own words, and why
it is tested on the same word in two contexts. A third instrument, or a hand-marked sample of a
few hundred boundaries, would settle it.
