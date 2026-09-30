/**
 * Tag voices by ear, one after another (D104): the next voice without a gender or an age plays
 * its sample, one key sets each field, and the queue moves on once both are known.
 *
 * It writes what the voices page's own toggles write -- the by-ear pair through
 * `PUT /voices/{voice}/attributes` -- and nothing else. A voice that cannot be judged from what
 * is played is skipped, not guessed.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { RiArrowLeftLine, RiArrowRightLine, RiCloseLine, RiPauseFill, RiPlayFill, RiSkipForwardLine } from '@remixicon/react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { Spinner } from '@/components/ui/spinner'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { playStretch } from '@/components/VoiceDialog'
import { bucketLabel, minutes } from '@/components/analytics/labels'
import { POT_TEXT } from '@/components/analytics/primitives'
import { usePlaybackRate } from '@/lib/playback'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { VoiceClip, VoiceProfile } from '@/types'

const GENDERS = ['female', 'male'] as const
const AGES = ['under_20', '20_39', '40_59', '60_79', '80_plus'] as const
const GENDER_KEYS: Record<string, string> = { f: 'female', m: 'male' }
/** How many of a voice's stretches alone can be cycled through before judging. */
const SAMPLES = 8
/** Long enough to see the pick land before the next voice starts playing. */
const ADVANCE_MS = 350

type PotFilter = 'all' | 'train' | 'gold'
type Field = 'gender' | 'age_bracket'

const trainMinutes = (v: VoiceProfile) => (v.hours.train + v.hours.val) * 60
const goldMinutes = (v: VoiceProfile) => v.hours.gold * 60
const untagged = (v: VoiceProfile) => !v.gender || !v.age_bracket

/** The by-ear pair the API replaces as a whole: a field left to the declared rows stays null. */
function byEar(voice: VoiceProfile): Record<Field, string | null> {
  return {
    gender: voice.manual.includes('gender') ? voice.gender : null,
    age_bracket: voice.manual.includes('age_bracket') ? voice.age_bracket : null,
  }
}

/** The profile as the page should show it once `field` is set by ear to `value`. */
function withByEar(voice: VoiceProfile, field: Field, value: string | null): VoiceProfile {
  const next = { ...byEar(voice), [field]: value }
  return {
    ...voice,
    [field]: value,
    manual: (['gender', 'age_bracket'] as const).filter((k) => next[k] !== null),
    disagrees: voice.disagrees.filter((k) => k !== field),
    identity: value || voice.gender || voice.age_bracket ? 'resolved' : voice.identity,
  }
}

function Pick({
  label,
  options,
  keys,
  value,
  onSet,
}: {
  label: string
  options: readonly string[]
  keys: string[]
  value: string | null
  onSet: (value: string) => void
}) {
  return (
    <div className="space-y-1.5">
      <div className="text-xs font-semibold">{label}</div>
      <div className="flex flex-wrap gap-2">
        {options.map((o, i) => (
          <Button
            key={o}
            variant={value === o ? 'default' : 'outline'}
            className="h-11 gap-2 px-4 text-sm"
            onClick={() => onSet(o)}
            aria-pressed={value === o}
          >
            <Kbd>{keys[i]}</Kbd>
            {bucketLabel(o)}
          </Button>
        ))}
      </div>
    </div>
  )
}

export function VoiceTagger({
  voices,
  initialPot,
  onChange,
  onClose,
}: {
  voices: VoiceProfile[]
  initialPot: PotFilter
  onChange: (next: VoiceProfile) => void
  onClose: () => void
}) {
  const [pot, setPot] = useState<PotFilter>(initialPot)
  // The queue is fixed when the pot is picked, so a voice does not vanish under the cursor once
  // it is tagged and can be stepped back to.
  const queue = useMemo(
    () =>
      voices
        .filter((v) => untagged(v) && (pot === 'all' || (pot === 'gold' ? goldMinutes(v) : trainMinutes(v)) > 0))
        .sort((a, b) => b.talk_minutes - a.talk_minutes)
        .map((v) => v.voice),
    [pot],
  )
  const [index, setIndex] = useState(0)
  const [samples, setSamples] = useState<VoiceClip[] | null>(null)
  const [sample, setSample] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [playbackRate] = usePlaybackRate()
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const cancelRef = useRef<(() => void) | null>(null)
  const saveRef = useRef<Promise<unknown>>(Promise.resolve())
  const advanceRef = useRef<number | null>(null)

  const byId = useMemo(() => new Map(voices.map((v) => [v.voice, v])), [voices])
  const voice = byId.get(queue[index]) ?? null
  const left = queue.filter((id) => {
    const v = byId.get(id)
    return v ? untagged(v) : false
  }).length

  const stop = () => {
    cancelRef.current?.()
    cancelRef.current = null
    audioRef.current?.pause()
    setPlaying(false)
  }

  const play = (clip: VoiceClip) => {
    const audio = audioRef.current
    if (!audio) return
    cancelRef.current?.()
    cancelRef.current = playStretch(audio, clip, Number(playbackRate), () => setPlaying(false))
    setPlaying(true)
  }

  // Load the voice's stretches alone, best first; a voice never heard alone falls back to the
  // clips it shares, which are played whole.
  const id = voice?.voice ?? null
  useEffect(() => {
    stop()
    setSamples(null)
    setSample(0)
    if (!id) return
    let cancelled = false
    ;(async () => {
      try {
        let page = await api.getVoice(id, { limit: SAMPLES })
        if (!page.clips.length) page = await api.getVoice(id, { limit: SAMPLES, shared: true })
        if (cancelled) return
        const first = page.reference
        const clips = first ? [first, ...page.clips.filter((c) => c.segment_id !== first.segment_id)] : page.clips
        setSamples(clips)
        if (clips.length) play(clips[0])
      } catch (err: any) {
        if (!cancelled) {
          toast.error(err.detail || `Could not load ${id}`)
          setSamples([])
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [id])

  useEffect(
    () => () => {
      cancelRef.current?.()
      audioRef.current?.pause()
      if (advanceRef.current !== null) window.clearTimeout(advanceRef.current)
    },
    [],
  )

  const go = (to: number) => {
    if (advanceRef.current !== null) window.clearTimeout(advanceRef.current)
    advanceRef.current = null
    setIndex(Math.max(0, Math.min(queue.length, to)))
  }

  const set = (field: Field, value: string) => {
    if (!voice) return
    const before = voice
    const next = withByEar(voice, field, value)
    onChange(next)
    // One request at a time, each carrying the whole pair, so two quick keys cannot cross.
    saveRef.current = saveRef.current.then(() =>
      api.setVoiceAttributes(next.voice, byEar(next)).catch((err: any) => {
        toast.error(err.detail || `Could not save ${next.voice}`)
        onChange(before)
      }),
    )
    if (next.gender && next.age_bracket) {
      if (advanceRef.current !== null) window.clearTimeout(advanceRef.current)
      advanceRef.current = window.setTimeout(() => go(index + 1), ADVANCE_MS)
    }
  }

  const toggle = () => {
    const clip = samples?.[sample]
    if (!clip) return
    if (playing) stop()
    else play(clip)
  }

  const another = () => {
    if (!samples?.length) return
    const to = (sample + 1) % samples.length
    setSample(to)
    play(samples[to])
  }

  const handleKey = (e: KeyboardEvent) => {
    const key = e.key.toLowerCase()
    if (key === 'escape') onClose()
    else if (key === 'enter' || key === 'arrowright') go(index + 1)
    else if (key === 'arrowleft') go(index - 1)
    else if (!voice) return
    else if (key === ' ') toggle()
    else if (key === 's') another()
    else if (GENDER_KEYS[key]) set('gender', GENDER_KEYS[key])
    else if (/^[1-5]$/.test(key)) set('age_bracket', AGES[Number(key) - 1])
    else return
    e.preventDefault()
    e.stopPropagation()
  }
  const keyRef = useRef(handleKey)
  keyRef.current = handleKey
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      keyRef.current(e)
    }
    // Capture, so the digits set an age here before the app's own 1-8 switch the view.
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [])

  const clip = samples?.[sample] ?? null
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto rounded-lg border bg-card p-4">
      <audio ref={audioRef} onEnded={() => setPlaying(false)} onPause={() => setPlaying(false)} />
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <h2 className="font-heading text-sm font-semibold">Tag by ear</h2>
        <ToggleGroup type="single" variant="outline" size="sm" spacing={0} value={pot}
          onValueChange={(v) => {
            if (!v) return
            setPot(v as PotFilter)
            go(0)
          }}
        >
          <ToggleGroupItem value="all">All</ToggleGroupItem>
          <ToggleGroupItem value="train">Train + val</ToggleGroupItem>
          <ToggleGroupItem value="gold">Gold</ToggleGroupItem>
        </ToggleGroup>
        <span className="text-muted-foreground">
          {queue.length ? `${Math.min(index + 1, queue.length)} of ${queue.length} · ${left} still untagged` : 'nothing to tag'}
        </span>
        <Button variant="ghost" size="sm" className="ml-auto h-8 gap-1.5" onClick={onClose}>
          <RiCloseLine className="size-3.5" />
          Back to the list
        </Button>
      </div>

      {!voice ? (
        <p className="p-10 text-center text-sm text-muted-foreground">
          {queue.length
            ? `End of the queue: ${left} voice${left === 1 ? '' : 's'} skipped. Step back with ←, or return to the list.`
            : 'Every voice in this pot has a gender and an age.'}
        </p>
      ) : (
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-5">
          <div className="flex flex-wrap items-center gap-3">
            <Button size="icon-lg" variant={playing ? 'default' : 'outline'} onClick={toggle} disabled={!clip} aria-label={playing ? 'Pause' : 'Play the sample'}>
              {playing ? <RiPauseFill /> : <RiPlayFill />}
            </Button>
            <h3 className="font-mono text-3xl font-bold">{voice.voice}</h3>
            <span className="text-xs text-muted-foreground">
              {minutes(voice.talk_minutes)} of talk · {voice.episode_count} episode{voice.episode_count === 1 ? '' : 's'}
              {trainMinutes(voice) > 0 ? <span className={POT_TEXT.train}> · train</span> : null}
              {goldMinutes(voice) > 0 ? <span className={POT_TEXT.gold}> · gold</span> : null}
              {voice.genres.length ? ` · ${voice.genres.map(bucketLabel).join(', ')}` : ''}
            </span>
          </div>

          <div className="min-h-20 rounded-md border bg-background p-3">
            {samples === null ? (
              <Spinner />
            ) : !clip ? (
              <p className="text-sm text-muted-foreground">No clip holds this voice. Skip it.</p>
            ) : (
              <>
                <p className="line-clamp-3 font-devanagari text-sm leading-6">
                  {clip.text ?? <span className="text-muted-foreground">no verified text</span>}
                </p>
                <p className="mt-1 flex flex-wrap items-center gap-2 font-mono text-[10px] text-muted-foreground">
                  <span>
                    sample {sample + 1} of {samples.length}
                  </span>
                  <span className={cn(!clip.alone && 'text-amber-600 dark:text-amber-400')}>
                    {clip.alone ? `${(clip.end - clip.start).toFixed(1)} s alone` : `${clip.duration_seconds.toFixed(1)} s, other voices in it too`}
                  </span>
                  <span className="truncate">{clip.episode_external_id}</span>
                  {samples.length > 1 ? (
                    <button type="button" className="underline decoration-dotted hover:text-foreground" onClick={another}>
                      another sample
                    </button>
                  ) : null}
                </p>
              </>
            )}
          </div>

          <Pick label="Gender" options={GENDERS} keys={['F', 'M']} value={voice.gender} onSet={(v) => set('gender', v)} />
          <Pick label="Age" options={AGES} keys={['1', '2', '3', '4', '5']} value={voice.age_bracket} onSet={(v) => set('age_bracket', v)} />

          <div className="flex flex-wrap items-center gap-2">
            <Button variant="outline" size="sm" className="gap-1.5" onClick={() => go(index - 1)} disabled={index === 0}>
              <RiArrowLeftLine className="size-3.5" />
              Previous
            </Button>
            <Button variant="outline" size="sm" className="gap-1.5" onClick={() => go(index + 1)}>
              {untagged(voice) ? <RiSkipForwardLine className="size-3.5" /> : <RiArrowRightLine className="size-3.5" />}
              {untagged(voice) ? 'Skip' : 'Next'}
            </Button>
            <span className="ml-auto text-[11px] text-muted-foreground">
              <Kbd>Space</Kbd> play · <Kbd>S</Kbd> another sample · <Kbd>F</Kbd>/<Kbd>M</Kbd> gender · <Kbd>1</Kbd>–<Kbd>5</Kbd> age ·{' '}
              <Kbd>Enter</Kbd> skip · <Kbd>←</Kbd> back · <Kbd>Esc</Kbd> list
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
