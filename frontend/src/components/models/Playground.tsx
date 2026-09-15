/**
 * Try a model on your own voice (D85). Hold the button and talk, or click once to start and again
 * to stop, or pick an audio file. The recording goes to the backend, which prepares it the way
 * ingest prepares an episode and has the playground sidecar transcribe it on this machine's CPU.
 *
 * Nothing is kept but the `llm_requests` row: no clip, no label. The takes below live in this
 * tab only, so each one can be replayed next to what the model wrote.
 */

import { useEffect, useRef, useState } from 'react'
import { RiCloseLine, RiErrorWarningLine, RiMicLine, RiStopFill, RiUpload2Line } from '@remixicon/react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { AsrModel, PlaygroundResult } from '@/types'

/** The backend refuses more than 30 s; stop a little short of it. */
const MAX_RECORDING_S = 29
/** A press held at least this long is push-to-talk: releasing it stops the recording. */
const HOLD_MS = 350

interface Take {
  id: number
  url: string
  source: string
  result?: PlaygroundResult
  error?: string
}

function pickMimeType(): string | undefined {
  const candidates = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']
  return candidates.find((type) => MediaRecorder.isTypeSupported(type))
}

function extensionFor(mime: string): string {
  if (mime.includes('ogg')) return 'ogg'
  if (mime.includes('mp4')) return 'm4a'
  return 'webm'
}

function Timings({ result }: { result: PlaygroundResult }) {
  const raw = result.raw
  const parts = [`${result.audio_s.toFixed(1)} s of audio`]
  if (raw.compute_s !== undefined) {
    parts.push(`decoded in ${raw.compute_s.toFixed(1)} s${raw.rtf !== undefined ? ` (RTF ${raw.rtf.toFixed(2)})` : ''}`)
  }
  if (raw.load_s) parts.push(`model loaded in ${raw.load_s.toFixed(0)} s`)
  if (result.latency_ms !== null) parts.push(`${(result.latency_ms / 1000).toFixed(1)} s round trip`)
  parts.push(raw.variant ?? result.weights)
  return <div className="text-[10px] text-muted-foreground">{parts.join(' · ')}</div>
}

function TakeRow({ take, onRemove }: { take: Take; onRemove: () => void }) {
  const retried = take.result?.raw.retried
  return (
    <li className="space-y-1.5 rounded-md border bg-background p-2.5">
      <div className="flex items-center gap-2">
        <audio src={take.url} controls preload="metadata" className="h-8 min-w-0 flex-1" />
        <span className="shrink-0 text-[10px] text-muted-foreground">{take.source}</span>
        <Button variant="ghost" size="icon" className="size-7 shrink-0" onClick={onRemove} title="Remove this take">
          <RiCloseLine className="size-4" />
        </Button>
      </div>
      {!take.result && !take.error && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Spinner className="size-3.5" /> Transcribing on the CPU…
        </div>
      )}
      {take.error && (
        <div className="flex items-start gap-1.5 text-xs text-destructive">
          <RiErrorWarningLine className="mt-0.5 size-3.5 shrink-0" /> {take.error}
        </div>
      )}
      {take.result && (
        <>
          <p className="font-devanagari text-base leading-relaxed break-words">
            {take.result.text || <span className="text-muted-foreground italic">(no words)</span>}
          </p>
          {take.result.dry_run && (
            <div className="text-[10px] font-semibold text-amber-600">
              Dry run: canned text, the model was not called (config/llm_routes.yaml)
            </div>
          )}
          {retried && (
            <details className="text-[10px] text-amber-700 dark:text-amber-400">
              <summary className="cursor-pointer">Greedy decoding looped; this is the retry</summary>
              <p className="mt-1 font-devanagari text-xs text-muted-foreground">{take.result.raw.first_text}</p>
            </details>
          )}
          <Timings result={take.result} />
        </>
      )}
    </li>
  )
}

export function Playground({ model }: { model: AsrModel }) {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [takes, setTakes] = useState<Take[]>([])
  const recorderRef = useRef<MediaRecorder | null>(null)
  const pressedAtRef = useRef(0)
  const nextId = useRef(1)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const takesRef = useRef<Take[]>([])
  takesRef.current = takes

  // The page keys this panel by model, so leaving a model unmounts it: drop an unfinished
  // recording without sending it, release the microphone, and free the takes' object URLs.
  useEffect(() => {
    return () => {
      const recorder = recorderRef.current
      if (recorder) {
        recorder.onstop = null
        if (recorder.state !== 'inactive') recorder.stop()
        recorder.stream.getTracks().forEach((track) => track.stop())
      }
      takesRef.current.forEach((t) => URL.revokeObjectURL(t.url))
    }
  }, [])

  useEffect(() => {
    if (!recording) return
    const started = performance.now()
    const timer = window.setInterval(() => {
      const seconds = (performance.now() - started) / 1000
      setElapsed(seconds)
      if (seconds >= MAX_RECORDING_S) stop()
    }, 100)
    return () => window.clearInterval(timer)
  }, [recording])

  const send = async (blob: Blob, filename: string, source: string) => {
    const id = nextId.current++
    const url = URL.createObjectURL(blob)
    setTakes((current) => [{ id, url, source }, ...current])
    try {
      const result = await api.transcribeWithModel(model.slug, blob, filename)
      setTakes((current) => current.map((t) => (t.id === id ? { ...t, result } : t)))
    } catch (err: any) {
      const error = err.detail || err.message || 'Transcription failed'
      setTakes((current) => current.map((t) => (t.id === id ? { ...t, error } : t)))
    }
  }

  const start = async () => {
    if (recorderRef.current) return
    let stream: MediaStream
    try {
      // The browser's own processing is off: the backend normalises loudness the way ingest does,
      // and the point is to hear what the model makes of the microphone as it is.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      })
    } catch (err: any) {
      toast.error(`Microphone unavailable: ${err.message || err.name}`)
      return
    }
    const mimeType = pickMimeType()
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    const chunks: Blob[] = []
    recorder.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data)
    recorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop())
      const type = recorder.mimeType || mimeType || 'audio/webm'
      const blob = new Blob(chunks, { type })
      if (blob.size > 0) send(blob, `take.${extensionFor(type)}`, 'mic')
    }
    recorderRef.current = recorder
    recorder.start()
    setElapsed(0)
    setRecording(true)
  }

  function stop() {
    const recorder = recorderRef.current
    recorderRef.current = null
    setRecording(false)
    if (recorder && recorder.state !== 'inactive') recorder.stop()
  }

  // Press: start. Release after a hold: stop (push-to-talk). A quick click leaves it recording
  // until the next click.
  const onPointerDown = () => {
    if (recording) {
      stop()
      return
    }
    pressedAtRef.current = performance.now()
    start()
  }
  const onPointerUp = () => {
    if (recording && performance.now() - pressedAtRef.current >= HOLD_MS) stop()
  }

  if (!model.playground) {
    return (
      <section className="rounded-lg border border-dashed p-3 text-xs text-muted-foreground">
        <span className="font-semibold text-foreground">Try it on your voice:</span> no CPU weights for this
        model yet. Copy the notebook's <code className="font-mono">cpu/</code> folder (or{' '}
        <code className="font-mono">best/</code> if int8 was rejected) into{' '}
        <code className="font-mono">data/models/asr/{model.slug}/</code>, start the sidecar with{' '}
        <code className="font-mono">docker-compose --profile playground up -d playground</code>, and reload.
      </section>
    )
  }

  return (
    <section className="space-y-2 rounded-lg border bg-muted/30 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="font-heading text-sm font-semibold">Try it on your voice</h2>
        <span className="text-[10px] text-muted-foreground">
          {model.playground}/ on the CPU · up to {MAX_RECORDING_S} s · hold to talk, or click to start and stop
        </span>
      </div>
      <div className="flex items-center gap-2">
        <Button
          type="button"
          onPointerDown={onPointerDown}
          onPointerUp={onPointerUp}
          onContextMenu={(e) => e.preventDefault()}
          className={cn('h-10 min-w-40 gap-2 select-none', recording && 'bg-rose-600 hover:bg-rose-600/90')}
        >
          {recording ? <RiStopFill className="size-4" /> : <RiMicLine className="size-4" />}
          {recording ? `Recording ${elapsed.toFixed(1)} s` : 'Hold to talk'}
        </Button>
        {recording && (
          <div className="h-1.5 w-40 overflow-hidden rounded-full bg-muted">
            <div className="h-full bg-rose-500" style={{ width: `${Math.min(100, (elapsed / MAX_RECORDING_S) * 100)}%` }} />
          </div>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="gap-1.5"
          disabled={recording}
          onClick={() => fileRef.current?.click()}
        >
          <RiUpload2Line className="size-3.5" /> Audio file
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="audio/*,video/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) send(file, file.name, file.name)
            e.target.value = ''
          }}
        />
      </div>
      {takes.length > 0 && (
        <ul className="space-y-2">
          {takes.map((take) => (
            <TakeRow
              key={take.id}
              take={take}
              onRemove={() => {
                URL.revokeObjectURL(take.url)
                setTakes((current) => current.filter((t) => t.id !== take.id))
              }}
            />
          ))}
        </ul>
      )}
    </section>
  )
}
