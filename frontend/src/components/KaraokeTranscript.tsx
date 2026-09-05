import { useCallback, useEffect, useMemo, useRef } from 'react'

import { cn } from '@/lib/utils'
import type { HypothesisWord } from '@/types'

interface KaraokeTranscriptProps {
  /** Words of the hypothesis being followed, in position order. */
  words: HypothesisWord[]
  /** The element actually playing; read directly so the highlight runs at frame rate. */
  audioRef: React.RefObject<HTMLAudioElement | null>
  /** Whether playback is running, so the loop can idle when it is not. */
  isPlaying: boolean
  /** Seek here when a word is clicked. */
  onSeekWord: (time: number) => void
  /** Words other systems disputed, by word position. Underlined whether lit or not. */
  contestedPositions?: ReadonlySet<number>
  className?: string
}

/** How much the word being spoken grows at its peak. */
const ACTIVE_SCALE = 0.19
/** Fraction of a word's span spent growing to full size; the rest holds. */
const GROW_FRACTION = 0.4

const easeOutCubic = (t: number) => 1 - (1 - t) ** 3

/**
 * The seed transcript, with the word being spoken growing and lit as the clip plays.
 *
 * Styling is written straight onto the spans from a requestAnimationFrame loop rather than
 * through React state: at 60 fps a state update per frame would re-render the whole editor,
 * and the only thing that actually changes is a transform and a colour on at most two spans.
 * `timeupdate` alone is far too coarse to drive this -- it fires about four times a second,
 * which reads as a stutter rather than a glide.
 */
export function KaraokeTranscript({
  words,
  audioRef,
  isPlaying,
  onSeekWord,
  contestedPositions,
  className,
}: KaraokeTranscriptProps) {
  const spanRefs = useRef<(HTMLSpanElement | null)[]>([])
  const frameRef = useRef<number | null>(null)
  const activeRef = useRef<number>(-1)

  // Words the transcriber placed on the clock. One without both boundaries cannot be lit, but
  // it must still be rendered: dropping it would show the annotator a transcript missing words
  // the hypothesis actually contains. Memoized because `paint` and the frame loop below hang
  // off its identity -- rebuilt every render, the loop would be town down four times a second.
  const timed = useMemo(
    () => words.map((w) => w.start_time !== null && w.end_time !== null),
    [words],
  )

  const paint = useCallback(
    (time: number) => {
      // The last word that has *started*, not the word whose span contains `time`. Almost every
      // pair of words has a gap between them -- around 60 ms, on all three transcribers -- and
      // going dark in each one makes the line flicker and read as lagging. Holding the word
      // until its successor begins moves the highlight exactly on word onsets.
      let next = -1
      for (let i = 0; i < words.length; i += 1) {
        if (!timed[i]) continue
        if ((words[i].start_time as number) <= time) next = i
      }

      if (activeRef.current !== next) {
        const previous = spanRefs.current[activeRef.current]
        if (previous) {
          previous.style.transform = ''
          previous.dataset.state = 'sung'
        }
        activeRef.current = next
      }
      if (next < 0) return

      const word = words[next]
      const span = spanRefs.current[next]
      if (!span) return

      const start = word.start_time as number
      const end = word.end_time as number
      const progress = Math.min(1, Math.max(0, (time - start) / Math.max(end - start, 1e-3)))
      const grow = easeOutCubic(Math.min(1, progress / GROW_FRACTION))

      span.dataset.state = 'active'
      // Written every frame, so nothing may interpolate it in CSS as well -- see the class list.
      span.style.transform = `scale(${(1 + ACTIVE_SCALE * grow).toFixed(4)})`
    },
    [words, timed],
  )

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    // Paused is not idle: the annotator scrubs the waveform, clicks a word, nudges by two
    // seconds. Those arrive as events rather than frames, so the highlight follows them here
    // instead of burning a rAF loop on a clip nobody is playing.
    if (!isPlaying) {
      paint(audio.currentTime)
      const onSeek = () => paint(audio.currentTime)
      audio.addEventListener('seeked', onSeek)
      audio.addEventListener('timeupdate', onSeek)
      return () => {
        audio.removeEventListener('seeked', onSeek)
        audio.removeEventListener('timeupdate', onSeek)
      }
    }

    const tick = () => {
      const element = audioRef.current
      if (element) paint(element.currentTime)
      frameRef.current = requestAnimationFrame(tick)
    }
    frameRef.current = requestAnimationFrame(tick)

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
      frameRef.current = null
    }
  }, [isPlaying, paint, audioRef])

  // A new task brings a new word list; clear the highlight so it cannot survive the swap.
  useEffect(() => {
    activeRef.current = -1
    spanRefs.current.forEach((span) => {
      if (!span) return
      span.style.transform = ''
      span.dataset.state = ''
    })
  }, [words])

  if (words.length === 0) return null

  return (
    <div
      className={cn(
        'flex flex-wrap items-baseline gap-x-1.5 gap-y-2 p-3 font-devanagari text-lg leading-loose',
        className,
      )}
    >
      {words.map((word, index) => (
        <span
          key={`${word.position}-${index}`}
          ref={(el) => {
            spanRefs.current[index] = el
          }}
          role={timed[index] ? 'button' : undefined}
          tabIndex={timed[index] ? 0 : undefined}
          onClick={() => timed[index] && onSeekWord(word.start_time as number)}
          onKeyDown={(e) => {
            if (timed[index] && (e.key === 'Enter' || e.key === ' ')) {
              e.preventDefault()
              onSeekWord(word.start_time as number)
            }
          }}
          title={
            timed[index]
              ? `${(word.start_time as number).toFixed(2)}s - ${(word.end_time as number).toFixed(2)}s`
              : 'no timing reported'
          }
          data-contested={contestedPositions?.has(word.position) ? '' : undefined}
          className={cn(
            'inline-block origin-bottom text-muted-foreground',
            // Colour only. `transform` is driven per frame from the rAF loop, and a CSS
            // transition over it would interpolate towards a target that moves every 16 ms --
            // the bloom then lands ~150 ms late and lingers as long after the word ends, which
            // at three words a second reads as the highlight running half a word behind.
            'transition-[color,opacity] duration-100 ease-out',
            timed[index] && 'cursor-pointer hover:text-foreground',
            // Words already sung stay readable; the active one is the only lit word.
            'data-[state=sung]:text-foreground/70',
            'data-[state=active]:font-semibold data-[state=active]:text-info',
            // A disputed word keeps its underline whether or not it is lit.
            'data-[contested]:underline data-[contested]:decoration-warning',
            'data-[contested]:decoration-wavy data-[contested]:underline-offset-4',
            !timed[index] && 'opacity-40',
          )}
        >
          {word.word}
        </span>
      ))}
    </div>
  )
}
