/**
 * Putting the annotator's own words on the karaoke line.
 *
 * The line used to render the seed hypothesis, which is what the transcriber said and not what
 * the box says once anything has been edited. Only the transcriber's words carry timings,
 * though, so the current text has to inherit them:
 *
 *  - a word left alone keeps its own span, exactly;
 *  - a word swapped for another one takes the span of the word it replaced -- which is what makes
 *    replacing a disputed word land on the clock, since the replacement occupies that moment;
 *  - a rewritten stretch splits the span it covers between its words, by length;
 *  - a word inserted where the transcriber heard nothing gets no span at all, and renders dim
 *    like any other untimed word. A guessed timing would light a word at a moment nobody said it.
 *
 * When the text is another hypothesis verbatim -- loading a model's output over the seed -- that
 * model's own timings are used instead of the seed's, so a wholesale swap stays exact rather than
 * inheriting spans from a transcript it has nothing to do with.
 */

import type { Hypothesis, HypothesisWord } from '@/types'

/** Word position given to text the annotator wrote; it belongs to no seed word. */
export const WRITTEN_IN = -1

// `\p{M}` is not optional. Devanagari vowel signs are combining marks, so a class of only
// letters and numbers treats the `ो` of `भयो` as trailing punctuation and strips it.
// `analysis.py` carries the same warning about `\w`.
const EDGE = /^[^\p{L}\p{N}\p{M}]*|[^\p{L}\p{N}\p{M}]*$/gu

/** A token without the punctuation hanging off either end: `"भयो,"` and `"भयो"` are one word. */
export function stripEdgePunctuation(token: string): string {
  return token.replace(EDGE, '')
}

/** Whitespace-separated tokens, which is what both the transcript and the word list are made of. */
export function tokenize(text: string): string[] {
  return text.match(/\S+/g) ?? []
}

/** Compared without their edge punctuation, so retyping a comma is not a changed word. */
const same = (a: string, b: string) =>
  stripEdgePunctuation(a).normalize('NFC') === stripEdgePunctuation(b).normalize('NFC')

/** Longest common subsequence of two token lists, as the pairs that matched. */
function matchedPairs(seed: string[], current: string[]): Array<[number, number]> {
  const rows = seed.length
  const cols = current.length
  const lengths: number[][] = Array.from({ length: rows + 1 }, () => new Array(cols + 1).fill(0))
  for (let i = rows - 1; i >= 0; i -= 1) {
    for (let j = cols - 1; j >= 0; j -= 1) {
      lengths[i][j] = same(seed[i], current[j])
        ? lengths[i + 1][j + 1] + 1
        : Math.max(lengths[i + 1][j], lengths[i][j + 1])
    }
  }

  const pairs: Array<[number, number]> = []
  let i = 0
  let j = 0
  while (i < rows && j < cols) {
    if (same(seed[i], current[j])) {
      pairs.push([i, j])
      i += 1
      j += 1
    } else if (lengths[i + 1][j] >= lengths[i][j + 1]) {
      i += 1
    } else {
      j += 1
    }
  }
  return pairs
}

const isTimed = (word: HypothesisWord) => word.start_time !== null && word.end_time !== null

/** Share the span the replaced words covered between the words that replaced them, by length. */
function spread(replaced: HypothesisWord[], tokens: string[]): HypothesisWord[] {
  const timed = replaced.filter(isTimed)
  if (timed.length === 0) {
    return tokens.map((token) => untimed(token))
  }
  const start = timed[0].start_time as number
  const end = timed[timed.length - 1].end_time as number
  const weight = (token: string) => Math.max(stripEdgePunctuation(token).length, 1)
  const total = tokens.reduce((sum, token) => sum + weight(token), 0)

  let cursor = start
  return tokens.map((token) => {
    const span = ((end - start) * weight(token)) / total
    const word: HypothesisWord = {
      position: WRITTEN_IN,
      word: token,
      start_time: cursor,
      end_time: cursor + span,
      confidence: null,
    }
    cursor += span
    return word
  })
}

function untimed(token: string): HypothesisWord {
  return { position: WRITTEN_IN, word: token, start_time: null, end_time: null, confidence: null }
}

/**
 * The transcript as it stands, on the transcriber's clock.
 *
 * A returned word keeps its seed `position` only while it is still the seed's word, so a caller
 * marking disputed positions underlines the words that are genuinely still in dispute and lets
 * go of the ones the annotator has already decided.
 *
 * @param text What the annotator has in the box.
 * @param seedWords The seed hypothesis's words, in position order.
 * @param hypotheses Every hypothesis for the segment, consulted for an exact-match swap.
 */
export function karaokeWords(
  text: string,
  seedWords: HypothesisWord[],
  hypotheses: Hypothesis[] = [],
): HypothesisWord[] {
  const tokens = tokenize(text)
  if (tokens.length === 0) return []
  if (seedWords.length === 0) return tokens.map((token) => untimed(token))

  // A verbatim swap to another model's output: use that model's own timings, one for one.
  const verbatim = hypotheses.find(
    (hyp) => hyp.text.trim() === text.trim() && (hyp.words?.length ?? 0) === tokens.length,
  )

  const seedTokens = seedWords.map((word) => word.word)
  const pairs = matchedPairs(seedTokens, tokens)

  const aligned: HypothesisWord[] = []
  let seedCursor = 0
  let tokenCursor = 0
  const flushTo = (seedEnd: number, tokenEnd: number) => {
    if (tokenEnd > tokenCursor) {
      aligned.push(
        ...spread(seedWords.slice(seedCursor, seedEnd), tokens.slice(tokenCursor, tokenEnd)),
      )
    }
    seedCursor = seedEnd
    tokenCursor = tokenEnd
  }

  for (const [seedIndex, tokenIndex] of pairs) {
    flushTo(seedIndex, tokenIndex)
    const seedWord = seedWords[seedIndex]
    aligned.push({ ...seedWord, word: tokens[tokenIndex] })
    seedCursor = seedIndex + 1
    tokenCursor = tokenIndex + 1
  }
  flushTo(seedWords.length, tokens.length)

  if (!verbatim?.words) return aligned
  // Positions still come from the seed above -- they say which words are untouched -- but every
  // span is the swapped-in model's own.
  return aligned.map((word, index) => ({
    ...word,
    start_time: verbatim.words![index].start_time,
    end_time: verbatim.words![index].end_time,
  }))
}
