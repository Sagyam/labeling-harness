import { useCallback, useEffect, useRef, useState } from 'react'
import {
  RiCheckLine,
  RiCloseLine,
  RiEraserLine,
  RiPauseFill,
  RiPlayFill,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Chip } from '@/components/Chip'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Kbd } from '@/components/ui/kbd'
import { Spinner } from '@/components/ui/spinner'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { usePlaybackRate } from '@/lib/playback'
import { cn } from '@/lib/utils'
import { api, resolveUrl } from '@/services/api'
import type { VoiceClip, VoicePage, VoiceVerdict } from '@/types'

interface VoiceDialogProps {
  /** The anonymous voice id (D87); null closes the dialog. */
  voice: string | null
  /** Show this episode's clips first; the annotator can widen to every episode. */
  episode?: string | null
  onClose: () => void
}

/**
 * Play ``[start, end]`` of a clip on ``audio``, stopping on a timer: ``timeupdate`` is too coarse
 * to end a stretch where the next voice comes in. Returns a cancel function.
 */
function playStretch(
  audio: HTMLAudioElement,
  clip: VoiceClip,
  rate: number,
  onStop: () => void,
): () => void {
  let timer: number | null = null
  const begin = () => {
    audio.currentTime = clip.start
    audio.play().catch(() => {})
    timer = window.setTimeout(
      () => {
        audio.pause()
        onStop()
      },
      ((clip.end - clip.start) / rate) * 1000,
    )
  }
  audio.src = resolveUrl(clip.audio_url)
  audio.defaultPlaybackRate = rate
  audio.playbackRate = rate
  if (audio.readyState >= 1) begin()
  else audio.addEventListener('loadedmetadata', begin, { once: true })
  return () => {
    if (timer !== null) window.clearTimeout(timer)
    audio.removeEventListener('loadedmetadata', begin)
  }
}

const minutes = (seconds: number) =>
  seconds >= 60 ? `${(seconds / 60).toFixed(1)} min` : `${seconds.toFixed(0)} s`

/**
 * One voice across the corpus (D99): where it speaks, and its longest stretch alone in each clip,
 * to listen to and confirm.
 *
 * "Only this voice" means the owner heard the stretch and nobody else speaks in it. Confirmed
 * stretches become the voice's print, which the multitrack editor's suggestions are scored
 * against; until there is one, the diarizer's own centroid stands in. A voice never gets a name (D56):
 * knowing what it sounds like is the point, not who it is.
 */
export function VoiceDialog({ voice, episode, onClose }: VoiceDialogProps) {
  const [page, setPage] = useState<VoicePage | null>(null)
  const [scope, setScope] = useState<'episode' | 'all'>(episode ? 'episode' : 'all')
  const [episodeFilter, setEpisodeFilter] = useState<string | null>(episode ?? null)
  const [loading, setLoading] = useState(false)
  const [focused, setFocused] = useState(0)
  const [playing, setPlaying] = useState<number | null>(null)
  const [playbackRate] = usePlaybackRate()
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const cancelRef = useRef<(() => void) | null>(null)
  const rowRefs = useRef<Map<number, HTMLDivElement>>(new Map())

  useEffect(() => {
    setScope(episode ? 'episode' : 'all')
    setEpisodeFilter(episode ?? null)
    setFocused(0)
  }, [voice, episode])

  const load = useCallback(async () => {
    if (!voice) return
    setLoading(true)
    try {
      setPage(
        await api.getVoice(voice, {
          episode: scope === 'episode' && episodeFilter ? episodeFilter : undefined,
          limit: 120,
        }),
      )
    } catch (err: any) {
      toast.error(err.detail || `Could not load ${voice}`)
      setPage(null)
    } finally {
      setLoading(false)
    }
  }, [voice, scope, episodeFilter])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    if (!voice) {
      cancelRef.current?.()
      audioRef.current?.pause()
      setPlaying(null)
    }
  }, [voice])

  const play = (clip: VoiceClip) => {
    const audio = audioRef.current
    if (!audio) return
    cancelRef.current?.()
    if (playing === clip.segment_id && !audio.paused) {
      audio.pause()
      return
    }
    cancelRef.current = playStretch(audio, clip, Number(playbackRate), () => setPlaying(null))
    setPlaying(clip.segment_id)
  }

  const judge = async (clip: VoiceClip, verdict: VoiceVerdict) => {
    if (!voice) return
    try {
      const out = await api.setVoiceVerdict(voice, clip.segment_id, verdict)
      setPage((p) =>
        p
          ? {
              ...p,
              confirmed: out.confirmed,
              rejected: out.rejected,
              print_source: out.confirmed > 0 ? 'confirmed' : 'diarizer',
              clips: p.clips.map((c) =>
                c.segment_id === clip.segment_id
                  ? { ...c, verdict: verdict === 'cleared' ? null : verdict }
                  : c,
              ),
            }
          : p,
      )
      if (verdict === 'confirmed' && !out.embedded) {
        toast.warning('Confirmed, but the voiceprint model is not loaded: it is not in the print yet')
      }
    } catch (err: any) {
      toast.error(err.detail || 'Could not save the verdict')
    }
  }

  const clips = page?.clips ?? []

  const onKeyDown = (e: React.KeyboardEvent) => {
    const clip = clips[focused]
    const move = (to: number) => {
      const next = Math.max(0, Math.min(clips.length - 1, to))
      setFocused(next)
      rowRefs.current.get(next)?.scrollIntoView({ block: 'nearest' })
    }
    if (e.key === 'j' || e.key === 'ArrowDown') move(focused + 1)
    else if (e.key === 'k' || e.key === 'ArrowUp') move(focused - 1)
    else if (e.key === ' ' && clip) play(clip)
    else if ((e.key === 'y' || e.key === 'Y') && clip) {
      judge(clip, 'confirmed')
      move(focused + 1)
    } else if ((e.key === 'n' || e.key === 'N') && clip) {
      judge(clip, 'rejected')
      move(focused + 1)
    } else if ((e.key === 'u' || e.key === 'U') && clip) judge(clip, 'cleared')
    else return
    e.preventDefault()
    e.stopPropagation()
  }

  return (
    <Dialog open={voice !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        className="flex max-h-[85vh] flex-col gap-4 sm:max-w-4xl"
        onKeyDown={onKeyDown}
      >
        <audio ref={audioRef} onEnded={() => setPlaying(null)} onPause={() => setPlaying(null)} />
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 font-mono">
            {voice}
            {page && (
              <Chip
                className={cn(
                  page.print_source === 'confirmed'
                    ? 'bg-success/15 text-success'
                    : 'bg-muted text-muted-foreground',
                )}
              >
                {page.print_source === 'confirmed'
                  ? `print from ${page.confirmed} confirmed clip${page.confirmed === 1 ? '' : 's'}`
                  : 'print from the diarizer'}
              </Chip>
            )}
          </DialogTitle>
          <DialogDescription>
            {page
              ? `${minutes(page.talk_seconds)} of talk in ${page.episodes.length} episode${
                  page.episodes.length === 1 ? '' : 's'
                }. Each row plays this voice's longest stretch alone in a clip. Confirm the ones where only this voice speaks: they become its print.`
              : 'Loading…'}
          </DialogDescription>
        </DialogHeader>

        {page && (
          <div className="flex flex-wrap items-center gap-1.5">
            {page.episodes.map((e) => (
              <button
                key={e.episode_id}
                type="button"
                onClick={() => {
                  setEpisodeFilter(e.external_id)
                  setScope('episode')
                  setFocused(0)
                }}
                title={`${e.title ?? e.external_id}\nSpeaker ${e.speaker_number} there · ${minutes(e.talk_seconds)} · ${e.solo_clips} solo clips`}
                className={cn(
                  'max-w-60 truncate rounded-sm border px-2 py-0.5 text-left text-[11px]',
                  scope === 'episode' && episodeFilter === e.external_id
                    ? 'border-info bg-info/10'
                    : 'border-border hover:bg-muted',
                )}
              >
                <span className="font-mono">S{e.speaker_number}</span> ·{' '}
                {e.title ?? e.external_id} · {minutes(e.talk_seconds)}
              </button>
            ))}
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              spacing={0}
              className="ml-auto"
              value={scope}
              onValueChange={(v) => {
                if (!v) return
                setScope(v as 'episode' | 'all')
                setFocused(0)
              }}
            >
              <ToggleGroupItem value="episode" disabled={!episodeFilter}>
                This episode
              </ToggleGroupItem>
              <ToggleGroupItem value="all">All episodes</ToggleGroupItem>
            </ToggleGroup>
          </div>
        )}

        <div className="scrollbar-thin -mx-2 min-h-40 flex-1 overflow-y-auto px-2" tabIndex={-1}>
          {loading && !page ? (
            <div className="flex justify-center py-10">
              <Spinner />
            </div>
          ) : clips.length === 0 ? (
            <p className="py-10 text-center text-muted-foreground">
              No clip has this voice alone for 1.5 s
              {scope === 'episode' ? ' in this episode — try all episodes.' : '.'}
            </p>
          ) : (
            <div className="flex flex-col">
              {clips.map((clip, index) => (
                <div
                  key={clip.segment_id}
                  ref={(el) => {
                    if (el) rowRefs.current.set(index, el)
                    else rowRefs.current.delete(index)
                  }}
                  onClick={() => setFocused(index)}
                  className={cn(
                    'flex items-center gap-3 border-b px-2 py-2 last:border-b-0',
                    index === focused && 'bg-muted/60',
                    clip.verdict === 'rejected' && 'opacity-50',
                  )}
                >
                  <Button
                    size="icon-sm"
                    variant={playing === clip.segment_id ? 'default' : 'outline'}
                    onClick={() => play(clip)}
                    aria-label={playing === clip.segment_id ? 'Pause' : 'Play'}
                  >
                    {playing === clip.segment_id ? <RiPauseFill /> : <RiPlayFill />}
                  </Button>
                  <div className="min-w-0 flex-1">
                    <div className="line-clamp-2 font-devanagari text-sm leading-6">
                      {clip.text ?? <span className="text-muted-foreground">no verified text</span>}
                    </div>
                    <div className="flex gap-2 font-mono text-[10px] text-muted-foreground">
                      <span title={clip.whole ? 'the whole clip is this voice' : 'the longest stretch of this voice alone in the clip'}>
                        {clip.whole
                          ? `${clip.duration_seconds.toFixed(1)} s, whole clip`
                          : `${(clip.end - clip.start).toFixed(1)} s alone at ${clip.start.toFixed(1)}–${clip.end.toFixed(1)} s of ${clip.duration_seconds.toFixed(1)} s`}
                      </span>
                      <span className="truncate">{clip.episode_external_id}</span>
                      {clip.pot === 'gold' && <span className="text-amber-600">gold</span>}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Button
                      size="sm"
                      variant={clip.verdict === 'confirmed' ? 'default' : 'outline'}
                      className={cn(clip.verdict === 'confirmed' && 'bg-success hover:bg-success/90')}
                      onClick={() => judge(clip, 'confirmed')}
                      title="Only this voice speaks in what was played (Y)"
                    >
                      <RiCheckLine data-icon="inline-start" />
                      Only this voice
                    </Button>
                    <Button
                      size="sm"
                      variant={clip.verdict === 'rejected' ? 'destructive' : 'outline'}
                      onClick={() => judge(clip, 'rejected')}
                      title="Someone else is in what was played, or it is not this voice (N)"
                    >
                      <RiCloseLine data-icon="inline-start" />
                      Not only
                    </Button>
                    {clip.verdict && (
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        onClick={() => judge(clip, 'cleared')}
                        aria-label="Take the verdict back"
                        title="Take the verdict back (U)"
                      >
                        <RiEraserLine />
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {page && (
          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
            <span>
              {clips.length} of {page.total_clips} clips shown · {page.confirmed} confirmed ·{' '}
              {page.rejected} rejected
            </span>
            <span>
              <Kbd>J</Kbd>/<Kbd>K</Kbd> move · <Kbd>Space</Kbd> play · <Kbd>Y</Kbd> only this voice
              · <Kbd>N</Kbd> not only · <Kbd>U</Kbd> undo
            </span>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

/**
 * Play what a voice sounds like, without leaving the page: its best clip, confirmed first.
 * Returns a player to call with the voice and, optionally, the episode to prefer.
 */
export function useVoiceSample() {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const cancelRef = useRef<(() => void) | null>(null)
  const [playingVoice, setPlayingVoice] = useState<string | null>(null)
  const [playbackRate] = usePlaybackRate()

  useEffect(() => {
    const audio = new Audio()
    audio.onended = () => setPlayingVoice(null)
    audio.onpause = () => setPlayingVoice(null)
    audioRef.current = audio
    return () => {
      audio.pause()
      audioRef.current = null
    }
  }, [])

  const playVoice = useCallback(
    async (voice: string, episode?: string | null) => {
      const audio = audioRef.current
      if (!audio) return
      cancelRef.current?.()
      if (playingVoice === voice && !audio.paused) {
        audio.pause()
        return
      }
      try {
        let page = await api.getVoice(voice, { episode: episode ?? undefined, limit: 1 })
        if (!page.reference && episode) page = await api.getVoice(voice, { limit: 1 })
        if (!page.reference) {
          toast.info(`No clip has ${voice} alone`)
          return
        }
        cancelRef.current = playStretch(audio, page.reference, Number(playbackRate), () =>
          setPlayingVoice(null),
        )
        setPlayingVoice(voice)
      } catch (err: any) {
        toast.error(err?.detail || `Could not play ${voice}`)
      }
    },
    [playingVoice, playbackRate],
  )

  const stop = useCallback(() => {
    cancelRef.current?.()
    audioRef.current?.pause()
  }, [])
  return { playVoice, playingVoice, stop }
}
