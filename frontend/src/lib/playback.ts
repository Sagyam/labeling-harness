/**
 * Playback speed, shared by both editors and remembered between clips.
 *
 * Kept in the browser: it is one viewer's preference, not a fact about the corpus. Storage can be
 * missing or refuse (a private window, blocked site data), so every read and write is guarded and
 * the editors fall back to normal speed.
 */

import { useCallback, useState } from 'react'

export const SPEEDS = ['0.25', '0.5', '0.75', '1', '1.25'] as const
export type Speed = (typeof SPEEDS)[number]

const KEY = 'harness.playbackRate'

function read(): Speed {
  try {
    const stored = window.localStorage.getItem(KEY)
    if (stored && (SPEEDS as readonly string[]).includes(stored)) return stored as Speed
  } catch {
    // storage unavailable: normal speed
  }
  return '1'
}

/** The remembered speed and a setter that remembers the next one. */
export function usePlaybackRate(): [Speed, (value: string) => void] {
  const [rate, setRate] = useState<Speed>(read)
  const update = useCallback((value: string) => {
    if (!(SPEEDS as readonly string[]).includes(value)) return
    setRate(value as Speed)
    try {
      window.localStorage.setItem(KEY, value)
    } catch {
      // not remembered, still applied
    }
  }, [])
  return [rate, update]
}
