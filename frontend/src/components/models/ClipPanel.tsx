/**
 * One clip of a run: listen, then read what the model got wrong (D83).
 *
 * Audio and peaks come from the segment endpoints the editor uses; the crosstalk spans (D77) are
 * drawn over the waveform because they are the largest measured cause of error (roadmap §3). The
 * diff is the backend's folded alignment, so its marks add up to the error count shown above it.
 * `Space` plays or pauses and `r` replays from the start.
 */

import { useEffect, useRef, useState } from 'react'
import { RiErrorWarningLine, RiPauseFill, RiPlayFill, RiReplay5Line } from '@remixicon/react'

import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { Waveform } from '@/components/Waveform'
import { humanize } from '@/components/analytics/primitives'
import { AlignedDiff, DiffLegend } from '@/components/models/AlignedDiff'
import { api, resolveUrl } from '@/services/api'
import type { ModelClipDetail, PeaksPayload, Segment } from '@/types'

function Count({ label, value, tone = '' }: { label: string; value: number | string; tone?: string }) {
  return (
    <span className="flex items-baseline gap-1">
      <span className={`font-mono text-sm font-bold tabular-nums ${tone}`}>{value}</span>
      <span className="text-[10px] text-muted-foreground">{label}</span>
    </span>
  )
}

export function ClipPanel({ runId, segmentId }: { runId: number; segmentId: number }) {
  const [detail, setDetail] = useState<ModelClipDetail | null>(null)
  const [segment, setSegment] = useState<Segment | null>(null)
  const [peaks, setPeaks] = useState<PeaksPayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    let cancelled = false
    setDetail(null)
    setSegment(null)
    setPeaks(null)
    setError(null)
    setCurrentTime(0)
    Promise.all([api.getRunClip(runId, segmentId), api.getSegment(segmentId)])
      .then(([nextDetail, nextSegment]) => {
        if (cancelled) return
        setDetail(nextDetail)
        setSegment(nextSegment)
        setDuration(nextSegment.duration_seconds)
        if (nextSegment.peaks_url) {
          api
            .getPeaks(nextSegment.peaks_url)
            .then((p) => !cancelled && setPeaks(p))
            .catch(() => {})
        }
      })
      .catch((err) => !cancelled && setError(err.detail || err.message || 'Failed to load the clip'))
    return () => {
      cancelled = true
    }
  }, [runId, segmentId])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onTime = () => setCurrentTime(audio.currentTime)
    const onMeta = () => audio.duration && !isNaN(audio.duration) && setDuration(audio.duration)
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)
    audio.addEventListener('timeupdate', onTime)
    audio.addEventListener('loadedmetadata', onMeta)
    audio.addEventListener('play', onPlay)
    audio.addEventListener('pause', onPause)
    audio.addEventListener('ended', onPause)
    return () => {
      audio.removeEventListener('timeupdate', onTime)
      audio.removeEventListener('loadedmetadata', onMeta)
      audio.removeEventListener('play', onPlay)
      audio.removeEventListener('pause', onPause)
      audio.removeEventListener('ended', onPause)
    }
  }, [segment])

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) audio.play().catch(() => {})
    else audio.pause()
  }

  const seek = (time: number) => {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = Math.max(0, Math.min(duration, time))
    setCurrentTime(audio.currentTime)
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      // Buttons are not skipped: the clip rows are buttons, and after a click one has focus.
      // preventDefault below stops Space from also pressing it.
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return
      if (e.key === ' ') {
        e.preventDefault()
        togglePlay()
      } else if (e.key === 'r') {
        e.preventDefault()
        seek(0)
        audioRef.current?.play().catch(() => {})
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-lg border bg-card p-4 text-sm text-destructive">
        <RiErrorWarningLine className="size-4" /> {error}
      </div>
    )
  }
  if (!detail || !segment) {
    return (
      <div className="flex justify-center rounded-lg border bg-card p-8">
        <Spinner className="size-5 text-primary" />
      </div>
    )
  }

  const speakers = new Set(segment.speaker_turns.map((t) => t.speaker)).size
  const spans = segment.overlap_spans ?? []

  return (
    <div className="space-y-3 rounded-lg border bg-card p-4">
      <audio ref={audioRef} src={resolveUrl(segment.audio_url)} preload="auto" />

      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate font-mono text-xs font-semibold">{detail.external_id}</div>
          <div className="truncate text-[11px] text-muted-foreground">
            {humanize(detail.genre)} · {detail.duration_seconds.toFixed(1)} s ·{' '}
            {segment.overlap_spans === null
              ? 'crosstalk never measured'
              : `crosstalk ${((detail.overlap_share ?? 0) * 100).toFixed(0)}%`}
            {speakers > 0 ? ` · ${speakers} speaker${speakers > 1 ? 's' : ''}` : ''}
          </div>
        </div>
        <div className="flex flex-wrap items-baseline gap-3">
          <Count label="errors" value={detail.errors} tone="text-rose-600 dark:text-rose-400" />
          <Count label="WER" value={`${detail.wer.toFixed(1)}%`} />
          <Count label="sub" value={detail.substitutions} />
          <Count label="del" value={detail.deletions} />
          <Count label="ins" value={detail.insertions} />
          <Count label="raw" value={detail.raw_errors} />
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Button size="icon" className="size-8 shrink-0" onClick={togglePlay} aria-label={isPlaying ? 'Pause' : 'Play'}>
          {isPlaying ? <RiPauseFill className="size-4" /> : <RiPlayFill className="size-4" />}
        </Button>
        <Button
          size="icon"
          variant="outline"
          className="size-8 shrink-0"
          onClick={() => {
            seek(0)
            audioRef.current?.play().catch(() => {})
          }}
          aria-label="Replay from the start"
        >
          <RiReplay5Line className="size-4" />
        </Button>
        <div className="relative min-w-0 flex-1 overflow-hidden rounded border">
          {peaks ? (
            <Waveform peaks={peaks} currentTime={currentTime} duration={duration} onSeek={seek} />
          ) : (
            <div className="flex h-16 items-center bg-muted/30 px-3 text-[11px] text-muted-foreground">
              {currentTime.toFixed(1)} / {duration.toFixed(1)} s
            </div>
          )}
          {duration > 0 &&
            spans.map(([start, end], index) => (
              <div
                key={index}
                className="pointer-events-none absolute inset-y-0 bg-amber-500/25"
                style={{ left: `${(start / duration) * 100}%`, width: `${((end - start) / duration) * 100}%` }}
              />
            ))}
        </div>
      </div>

      {detail.fold_version_changed && (
        <div className="flex items-center gap-2 rounded border border-amber-500/50 px-2 py-1 text-[11px] text-amber-700 dark:text-amber-400">
          <RiErrorWarningLine className="size-3.5" />
          The fold rules changed since this run was imported ({detail.fold_version}); the marks
          below may not add up to the counts.
        </div>
      )}

      <div className="space-y-1.5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="font-heading text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Alignment
          </h3>
          <DiffLegend />
        </div>
        <AlignedDiff ops={detail.ops} />
      </div>

      <div className="grid gap-2 md:grid-cols-2">
        <div className="rounded border bg-muted/30 p-2">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Reference (label)
          </div>
          <p className="text-sm leading-relaxed">{detail.ref_text}</p>
        </div>
        <div className="rounded border bg-muted/30 p-2">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Model
          </div>
          <p className="text-sm leading-relaxed">{detail.hyp_text || '∅ (empty output)'}</p>
        </div>
      </div>
    </div>
  )
}
