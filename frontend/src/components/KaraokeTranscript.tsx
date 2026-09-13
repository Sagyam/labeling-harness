import { useCallback, useEffect, useMemo, useRef } from 'react'

import type { WordVoice } from '@/lib/karaoke'
import { cn } from '@/lib/utils'
import type { HypothesisWord } from '@/types'

interface KaraokeTranscriptProps {
  /** Words of the transcript being followed, in position order. */
  words: HypothesisWord[]
  /**
   * Identity of the clip being transcribed. The highlight is cleared when this changes rather
   * than when `words` does: the word list is rebuilt on every keystroke while the annotator
   * edits, and restarting the animation under them each time would make the line stutter.
   */
  resetKey: string | number
  /** The element actually playing; read directly so the highlight runs at frame rate. */
  audioRef: React.RefObject<HTMLAudioElement | null>
  /** Seek here when a word is clicked. */
  onSeekWord: (time: number) => void
  /** Words other systems disputed, by word position. Underlined whether lit or not. */
  contestedPositions?: ReadonlySet<number>
  /** Called instead of seeking when a disputed word is clicked. */
  onPickContested?: (position: number, anchor: HTMLElement) => void
  /** Who speaks each word and whether it is overlapped, aligned with `words` (D77, D78). */
  voices?: WordVoice[]
  className?: string
}

/** Speaker tints and swatches, categorical slots 1-4 in fixed order; a fifth voice reuses slot 1. */
const SPEAKER_TINT = ['bg-speaker-1/15', 'bg-speaker-2/15', 'bg-speaker-3/20', 'bg-speaker-4/20']
const SPEAKER_SWATCH = ['bg-speaker-1', 'bg-speaker-2', 'bg-speaker-3', 'bg-speaker-4']
const slot = (speaker: number) => (speaker - 1) % SPEAKER_TINT.length

/** A speaker number never travels on colour alone: its swatch sits beside the text "S2". */
function SpeakerTag({ speaker }: { speaker: number }) {
  return (
    <span
      title={`Speaker ${speaker}`}
      className="inline-flex select-none items-center gap-1 self-center font-mono text-[10px] text-muted-foreground"
    >
      <span className={cn('size-1.5 rounded-full', SPEAKER_SWATCH[slot(speaker)])} />S{speaker}
    </span>
  )
}

/** How much the word being spoken grows at its peak. */
const ACTIVE_SCALE = 0.19
/** Fraction of a word's span spent growing to full size; the rest holds. */
const GROW_FRACTION = 0.4
/** How long a word takes to settle back once its successor has started, in ms. */
const RELEASE_MS = 190

const easeOutCubic = (t: number) => 1 - (1 - t) ** 3

/**
 * The seed transcript, with the word being spoken growing and lit as the clip plays.
 *
 * Both halves of the animation are driven from one requestAnimationFrame loop, and neither is
 * a CSS transition. The rise is locked to the audio -- a word's size is a function of how far
 * into its own span the playhead is -- because anything time-based there shows up as the
 * highlight lagging the sound. The fall is time-based instead, decaying over RELEASE_MS once
 * the next word starts, because a trailing animation cannot be late by definition. Putting a
 * CSS transition on `transform` as well would interpolate towards a target that moves every
 * frame, which is what made the bloom arrive ~150 ms late. Colour is the exception: it flips
 * discretely, so CSS may transition it.
 */
export function KaraokeTranscript({
  words,
  resetKey,
  audioRef,
  onSeekWord,
  contestedPositions,
  onPickContested,
  voices,
  className,
}: KaraokeTranscriptProps) {
  const spanRefs = useRef<(HTMLSpanElement | null)[]>([])
  const frameRef = useRef<number | null>(null)
  const activeRef = useRef<number>(-1)
  /** Index to current size, 0-1, for every word still settling. Usually one or two entries. */
  const heatRef = useRef<Map<number, number>>(new Map())
  const lastTickRef = useRef<number>(0)

  // Words the transcriber placed on the clock. One without both boundaries cannot be lit, but
  // it must still be rendered: dropping it would show the annotator a transcript missing words
  // the hypothesis actually contains. Memoized so `paint` keeps its identity across renders.
  const timed = useMemo(
    () => words.map((w) => w.start_time !== null && w.end_time !== null),
    [words],
  )

  // A tag goes before the first word of each run by one speaker. Overlapped and unknown words do
  // not break a run: a backchannel inside someone's turn is not a change of speaker.
  const { tagged, present, anyOverlap } = useMemo(() => {
    const tags = new Set<number>()
    const speakers = new Set<number>()
    let last: number | null = null
    ;(voices ?? []).forEach((voice, index) => {
      if (voice.speaker === null) return
      speakers.add(voice.speaker)
      if (voice.speaker !== last) tags.add(index)
      last = voice.speaker
    })
    return {
      tagged: tags,
      present: [...speakers].sort((a, b) => a - b),
      anyOverlap: (voices ?? []).some((v) => v.overlap),
    }
  }, [voices])

  const paint = useCallback(
    (time: number, elapsedMs: number) => {
      // The last word to have *started*, not the word whose span contains `time`. Almost every
      // pair of words has a gap between them -- around 60 ms, on all three transcribers -- and
      // going dark in each one makes the line flicker and read as lagging. Holding the word
      // until its successor begins moves the highlight exactly on word onsets.
      let next = -1
      for (let i = 0; i < words.length; i += 1) {
        if (!timed[i]) continue
        if ((words[i].start_time as number) <= time) next = i
      }

      const heat = heatRef.current

      if (next >= 0) {
        const start = words[next].start_time as number
        const end = words[next].end_time as number
        const progress = Math.min(1, Math.max(0, (time - start) / Math.max(end - start, 1e-3)))
        heat.set(next, easeOutCubic(Math.min(1, progress / GROW_FRACTION)))
      }

      if (activeRef.current !== next) {
        const previous = spanRefs.current[activeRef.current]
        if (previous) previous.dataset.state = 'sung'
        activeRef.current = next
      }

      for (const [index, value] of heat) {
        const span = spanRefs.current[index]
        if (index !== next) {
          const faded = value - elapsedMs / RELEASE_MS
          if (faded <= 0.001) {
            heat.delete(index)
            if (span) span.style.transform = ''
            continue
          }
          heat.set(index, faded)
          if (span) span.style.transform = `scale(${(1 + ACTIVE_SCALE * faded).toFixed(4)})`
          continue
        }
        if (span) {
          span.dataset.state = 'active'
          span.style.transform = `scale(${(1 + ACTIVE_SCALE * value).toFixed(4)})`
        }
      }
    },
    [words, timed],
  )

  useEffect(() => {
    // The loop runs whether or not the clip is playing. Paused is not idle -- the annotator
    // scrubs, clicks a word, nudges by two seconds, and a word released just before the pause
    // still has to finish settling. With at most a couple of warm spans the frame costs
    // nothing, and browsers stop rAF entirely when the tab is hidden.
    const tick = (now: number) => {
      const element = audioRef.current
      const elapsed = lastTickRef.current ? Math.min(now - lastTickRef.current, 100) : 16
      lastTickRef.current = now
      if (element) paint(element.currentTime, elapsed)
      frameRef.current = requestAnimationFrame(tick)
    }
    frameRef.current = requestAnimationFrame(tick)

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
      frameRef.current = null
      lastTickRef.current = 0
    }
  }, [paint, audioRef])

  // A new clip brings a new transcript; clear the highlight so it cannot survive the swap.
  useEffect(() => {
    activeRef.current = -1
    heatRef.current.clear()
    spanRefs.current.forEach((span) => {
      if (!span) return
      span.style.transform = ''
      span.dataset.state = ''
    })
  }, [resetKey])

  if (words.length === 0) return null

  return (
    <div className={className}>
      {(present.length > 0 || anyOverlap) && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 pt-2 text-[11px] text-muted-foreground">
          {present.map((speaker) => (
            <span key={speaker} className="inline-flex items-center gap-1">
              <span className={cn('size-2 rounded-full', SPEAKER_SWATCH[slot(speaker)])} />
              Speaker {speaker}
            </span>
          ))}
          {anyOverlap && (
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2.5 w-4 rounded-sm outline-1 outline-dashed outline-warning" />
              overlapped speech
            </span>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-2 p-3 font-devanagari text-lg leading-loose">
        {words.map((word, index) => {
          const voice = voices?.[index]
          const isContested = contestedPositions?.has(word.position) ?? false
          const activate = (anchor: HTMLElement) => {
            if (isContested && onPickContested) onPickContested(word.position, anchor)
            else if (timed[index]) onSeekWord(word.start_time as number)
          }
          return (
            <span key={`${word.position}-${index}`} className="contents">
              {tagged.has(index) && voice?.speaker != null && <SpeakerTag speaker={voice.speaker} />}
              <span
                ref={(el) => {
                  spanRefs.current[index] = el
                }}
                role={timed[index] || isContested ? 'button' : undefined}
                tabIndex={timed[index] || isContested ? 0 : undefined}
                onClick={(e) => activate(e.currentTarget)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    activate(e.currentTarget)
                  }
                }}
                title={
                  timed[index]
                    ? `${(word.start_time as number).toFixed(2)}s - ${(word.end_time as number).toFixed(2)}s${
                        voice?.speaker != null ? ` - Speaker ${voice.speaker}` : ''
                      }${voice?.overlap ? ' - overlapped speech' : ''}${
                        isContested ? ' - the other systems heard this differently' : ''
                      }`
                    : 'no timing reported'
                }
                data-contested={isContested ? '' : undefined}
                data-overlap={voice?.overlap ? '' : undefined}
                className={cn(
                  'inline-block origin-bottom text-muted-foreground',
                  // Colour only; `transform` is written every frame from the loop above.
                  'transition-[color,opacity] duration-100 ease-out',
                  (timed[index] || isContested) && 'cursor-pointer hover:text-foreground',
                  // Words already sung stay readable; the active one is the only lit word.
                  'data-[state=sung]:text-foreground/70',
                  'data-[state=active]:font-semibold data-[state=active]:text-info',
                  // A disputed word keeps its underline whether or not it is lit.
                  'data-[contested]:underline data-[contested]:decoration-warning',
                  'data-[contested]:decoration-wavy data-[contested]:underline-offset-4',
                  // Who is speaking: a tint behind the word, never its text colour, so the sung
                  // and active states read the same for every speaker.
                  voice?.speaker != null && ['rounded-sm px-0.5', SPEAKER_TINT[slot(voice.speaker)]],
                  // Crosstalk: a dashed outline, distinct from the contested word's wavy underline.
                  'data-[overlap]:rounded-sm data-[overlap]:outline-1 data-[overlap]:outline-dashed',
                  'data-[overlap]:outline-offset-1 data-[overlap]:outline-warning',
                  !timed[index] && 'opacity-40',
                )}
              >
                {word.word}
              </span>
            </span>
          )
        })}
      </div>
    </div>
  )
}
