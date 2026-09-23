/**
 * The multitrack editor's blocks (D98).
 *
 * The server hands over the clip's words already on their diarized lanes; everything here is
 * bookkeeping over that list -- ids for React, a provisional place for a word nobody measured a
 * span for, the 10 ms grid, and what the lanes add up to. Validation that matters (a lane per
 * word, a span inside the clip, speakers the episode has) is repeated by the server, which is the
 * authority; this is only what lets the editor say so before a round trip.
 */

import type { LaneCandidate, LaneWord, SpeakerLanes } from '@/types'

/** The editor's grid, in seconds. */
export const GRID = 0.01
/** The shortest block the server accepts. */
export const MIN_BLOCK = 0.02
/** How long a typed word starts out, before its edges are dragged to fit. */
export const TYPED_LENGTH = 0.3

/** A word on the lanes, with an identity that survives moves. */
export interface Block extends LaneWord {
  id: string
  /** Its span was put there by the editor, not measured; drawn dashed until it is moved. */
  estimated: boolean
}

export interface CandidateBlock extends LaneCandidate {
  id: string
}

/** Snap to the 10 ms grid, as a number with no float dust: 0.30000000000000004 is 0.3. */
export function snap(seconds: number): number {
  return Math.round(seconds / GRID) / Math.round(1 / GRID)
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value))
}

let counter = 0
export function newId(prefix = 'w'): string {
  counter += 1
  return `${prefix}${counter}`
}

/**
 * The served words as blocks.
 *
 * A word without a span -- one the annotator wrote into the verified text that no system has a
 * timing for -- is given the gap between its timed neighbours, shared evenly with any untimed
 * neighbours of its own, and flagged `estimated`. It stays on no lane, so it cannot be saved
 * until somebody has placed it.
 */
export function initialBlocks(lanes: SpeakerLanes, duration: number): Block[] {
  const words = lanes.words
  const blocks: Block[] = words.map((w) => ({ ...w, id: newId(), estimated: false }))
  let i = 0
  while (i < blocks.length) {
    if (blocks[i].start !== null && blocks[i].end !== null) {
      i += 1
      continue
    }
    let j = i
    while (j < blocks.length && (blocks[j].start === null || blocks[j].end === null)) j += 1
    const before = i > 0 ? (blocks[i - 1].end as number) : 0
    const after = j < blocks.length ? (blocks[j].start as number) : duration
    const count = j - i
    const gap = after - before
    const each = gap >= count * MIN_BLOCK * 2 ? gap / count : TYPED_LENGTH
    for (let k = 0; k < count; k += 1) {
      const start = clamp(snap(before + k * each), 0, Math.max(0, duration - MIN_BLOCK))
      const end = clamp(snap(start + each), start + MIN_BLOCK, duration)
      blocks[i + k] = { ...blocks[i + k], start, end, speaker: null, estimated: true }
    }
    i = j
  }
  return blocks
}

export function initialCandidates(lanes: SpeakerLanes): CandidateBlock[] {
  return lanes.candidates.map((c) => ({ ...c, id: newId('c') }))
}

const startOf = (b: LaneWord) => b.start ?? Number.POSITIVE_INFINITY

/** Blocks by start, then end, then lane: the order the server stores and flattens them in. */
export function inTimeOrder<T extends LaneWord>(blocks: T[]): T[] {
  return [...blocks].sort(
    (a, b) =>
      startOf(a) - startOf(b) ||
      (a.end ?? 0) - (b.end ?? 0) ||
      (a.speaker ?? 0) - (b.speaker ?? 0),
  )
}

/** What the server is sent: the blocks without the editor's own fields. */
export function toSubmission(blocks: Block[]): LaneWord[] {
  return inTimeOrder(blocks).map(
    ({ word, start, end, speaker, proposed_speaker, source, suggested_speaker }) => ({
      word,
      start,
      end,
      speaker,
      proposed_speaker,
      source,
      suggested_speaker: suggested_speaker ?? null,
    }),
  )
}

export interface EditSummary {
  /** Verified-text words now on a different lane than the diarization put them on. */
  moved: number
  /** Words the verified text did not have: typed, copied or taken from a recogniser. */
  added: number
  /** Words on no lane yet; saving is refused while there are any. */
  unplaced: number
  /** Words a voiceprint hears as another speaker than the lane they are on (D99). */
  suggested: number
}

export function summarize(blocks: Block[]): EditSummary {
  let moved = 0
  let added = 0
  let unplaced = 0
  let suggested = 0
  for (const b of blocks) {
    if (b.speaker === null) unplaced += 1
    if (hasSuggestion(b)) suggested += 1
    if (b.source !== 'label') added += 1
    else if (b.speaker !== null && b.speaker !== b.proposed_speaker) moved += 1
  }
  return { moved, added, unplaced, suggested }
}

/** Whether a voiceprint suggests a lane other than the one the block is on (D99). */
export function hasSuggestion(block: LaneWord): boolean {
  return block.suggested_speaker != null && block.suggested_speaker !== block.speaker
}

/** One speaker's words in time order: the stream cpWER will score for that speaker. */
export function laneText(blocks: Block[], speaker: number): string {
  return inTimeOrder(blocks.filter((b) => b.speaker === speaker))
    .map((b) => b.word)
    .join(' ')
}

/** Categorical colour slot for a display number, 1-4; a fifth voice reuses slot 1. */
export function speakerSlot(speaker: number): number {
  return ((speaker - 1) % 4) + 1
}
