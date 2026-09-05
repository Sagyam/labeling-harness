import { useCallback, useEffect, useRef } from 'react'

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
  // the hypothesis actually contains.
  const timed = words.map((w) => w.start_time !== null && w.end_time !== null)

  const paint = useCallback(
    (time: number) => {
      let next = -1
      for (let i = 0; i < words.length; i += 1) {
        const word = words[i]
        if (!timed[i]) continue
        if (time >= (word.start_time as number) && time < (word.end_time as number)) {
          next = i
          break
        }
      }

      // Retire the word that just finished. Only ever touches the spans that changed.
      if (activeRef.current !== next && activeRef.current >= 0) {
        const previous = spanRefs.current[activeRef.current]
        if (previous) {
          previous.style.transform = ''
          const end = words[activeRef.current].end_time as number
          previous.dataset.state = time >= end ? 'sung' : ''
        }
      }
      activeRef.current = next
      if (next < 0) return

      const word = words[next]
      const span = spanRefs.current[next]
      if (!span) return

      const start = word.start_time as number
      const end = word.end_time as number
      const progress = Math.min(1, Math.max(0, (time - start) / Math.max(end - start, 1e-3)))
      const grow = easeOutCubic(Math.min(1, progress / GROW_FRACTION))

      span.dataset.state = 'active'
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
            'transition-[transform,color,opacity] duration-150 ease-out',
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
