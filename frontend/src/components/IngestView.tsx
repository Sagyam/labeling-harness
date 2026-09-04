import React, { useEffect, useRef, useState } from 'react'
import {
  RiArrowDownSLine,
  RiArrowRightSLine,
  RiCheckLine,
  RiDeleteBin6Line,
  RiErrorWarningLine,
  RiFileMusicLine,
  RiUploadCloud2Line,
  RiUserVoiceLine,
  RiYoutubeLine,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Field, FieldLabel } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Spinner } from '@/components/ui/spinner'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type {
  DiscardedSegment,
  IngestEvent,
  IngestJobStatus,
  IngestLogEntry,
  YouTubeProbe,
} from '@/types'

interface IngestViewProps {
  onComplete: (episodeId: string) => void
}

const STAGES = [
  { key: 'normalizing', label: 'Normalize audio', desc: 'FFmpeg loudnorm, 16 kHz mono FLAC' },
  { key: 'segmenting', label: 'Silero VAD', desc: 'CPU speech turn detection (2s–20s)' },
  { key: 'transcribing', label: 'Cloud ASR', desc: 'Every configured system, per clip' },
  { key: 'analyzing', label: 'Token analysis', desc: 'Devanagari/Latin tagging & CMI' },
  { key: 'importing', label: 'Direct import', desc: 'Database records & queue building' },
]

/** A URL job fetches its own audio first; an upload arrives with the request. */
const DOWNLOAD_STAGE = {
  key: 'downloading',
  label: 'Fetch audio',
  desc: 'yt-dlp download from YouTube',
}

const ALLOWED_EXTENSIONS = ['mp3', 'm4a', 'wav', 'flac', 'aac', 'ogg']

type SourceTab = 'file' | 'youtube'

/** How long to sit on a keystroke before asking the backend what the URL points at. */
const PROBE_DEBOUNCE_MS = 500

/**
 * Where the running job's id is parked.
 *
 * Ingestion is a page now rather than a modal, and a page can be navigated away from. The job
 * itself lives on the server and never noticed, but the SSE subscription dies with the unmount --
 * so the id is kept here and the page reattaches to the stream when it comes back, including
 * across a browser reload.
 */
const ACTIVE_JOB_KEY = 'harness.ingest.activeJobId'

const LOG_LEVEL_CLASS: Record<string, string> = {
  error: 'text-destructive',
  warn: 'text-warning',
  success: 'text-success',
}

function formatDuration(seconds: number | null) {
  if (seconds === null || !Number.isFinite(seconds)) return 'unknown length'
  const total = Math.round(seconds)
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  const pad = (n: number) => String(n).padStart(2, '0')
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`
}

function formatClock(seconds: number) {
  const total = Math.max(0, Math.round(seconds))
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

function slugify(text: string) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/[-\s]+/g, '_')
    .slice(0, 40)
}

/**
 * What was thrown away, and who threw it.
 *
 * A discarded segment is money already spent on a clip that will never reach the corpus, so it is
 * reported at the same weight as the things that worked -- with the responsible system named. A
 * run where one vendor accounts for every discard is a vendor problem, and that only shows up if
 * the blame is counted rather than buried in the log.
 */
function DiscardPanel({ discarded }: { discarded: DiscardedSegment[] }) {
  const [expanded, setExpanded] = useState(false)
  if (discarded.length === 0) return null

  const bySystem = discarded.reduce<Record<string, number>>((acc, seg) => {
    const systems = seg.failures.map((f) => f.system_id)
    for (const system of systems.length ? systems : [seg.stage]) {
      acc[system] = (acc[system] || 0) + 1
    }
    return acc
  }, {})
  const ranked = Object.entries(bySystem).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))

  return (
    <div className="border border-warning/40">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center justify-between gap-2 border-b border-warning/30 bg-warning/10 px-3 py-2 text-left"
      >
        <span className="flex items-center gap-1.5 font-heading text-[11px] font-semibold tracking-widest uppercase">
          <RiDeleteBin6Line className="size-3.5 text-warning" />
          {discarded.length} segment{discarded.length === 1 ? '' : 's'} discarded
        </span>
        {expanded ? (
          <RiArrowDownSLine className="size-4" />
        ) : (
          <RiArrowRightSLine className="size-4" />
        )}
      </button>

      <div className="flex flex-wrap gap-1.5 p-3">
        {ranked.map(([system, count]) => (
          <span
            key={system}
            className="flex items-center gap-1.5 border bg-muted/40 px-2 py-0.5 font-mono text-[11px]"
          >
            <span className="text-foreground">{system}</span>
            <span className="text-muted-foreground tabular-nums">{count}</span>
          </span>
        ))}
      </div>

      {expanded && (
        <div className="scrollbar-thin max-h-64 overflow-y-auto border-t">
          {discarded.map((seg) => (
            <div key={seg.segment_id} className="border-b px-3 py-2 last:border-b-0">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono text-[11px] text-foreground">{seg.segment_id}</span>
                <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
                  {formatClock(seg.start_time)}–{formatClock(seg.end_time)}
                </span>
              </div>
              {seg.failures.map((failure, i) => (
                <div key={i} className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
                  <span className="text-warning">{failure.system_id}</span> · {failure.error}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function IngestView({ onComplete }: IngestViewProps) {
  // Form state
  const [sourceTab, setSourceTab] = useState<SourceTab>('file')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [showId, setShowId] = useState<string>('nepanglish')
  const [episodeTitle, setEpisodeTitle] = useState<string>('')
  const [episodeId, setEpisodeId] = useState<string>('')
  const [isManualEpisodeId, setIsManualEpisodeId] = useState<boolean>(false)
  const [isDragging, setIsDragging] = useState<boolean>(false)

  // YouTube source state
  const [youtubeUrl, setYoutubeUrl] = useState<string>('')
  const [probe, setProbe] = useState<YouTubeProbe | null>(null)
  const [isProbing, setIsProbing] = useState<boolean>(false)
  const [probeError, setProbeError] = useState<string | null>(null)

  // Sociolinguistic metadata state
  const [showSociolinguistics, setShowSociolinguistics] = useState<boolean>(false)
  const [genre, setGenre] = useState<string>('podcast')
  const [topic, setTopic] = useState<string>('')
  const [spk0Name, setSpk0Name] = useState<string>('')
  const [spk0Gender, setSpk0Gender] = useState<string>('')
  const [spk0Origin, setSpk0Origin] = useState<string>('')
  const [spk1Name, setSpk1Name] = useState<string>('')
  const [spk1Gender, setSpk1Gender] = useState<string>('')
  const [spk1Origin, setSpk1Origin] = useState<string>('')

  const buildSpeakersJson = () => {
    const speakers: Record<string, any> = {}
    if (spk0Name.trim() || spk0Gender || spk0Origin.trim()) {
      speakers['spk0'] = {
        name: spk0Name.trim() || 'Host',
        role: 'host',
        gender: spk0Gender || undefined,
        origin: spk0Origin.trim() || undefined,
      }
    }
    if (spk1Name.trim() || spk1Gender || spk1Origin.trim()) {
      speakers['spk1'] = {
        name: spk1Name.trim() || 'Guest',
        role: 'guest',
        gender: spk1Gender || undefined,
        origin: spk1Origin.trim() || undefined,
      }
    }
    return Object.keys(speakers).length > 0 ? JSON.stringify(speakers) : ''
  }

  // Execution state. `jobId` is seeded from storage so a reload rejoins a running pipeline.
  const [jobId, setJobId] = useState<string | null>(() => {
    try {
      return window.localStorage.getItem(ACTIVE_JOB_KEY)
    } catch {
      return null
    }
  })
  const [jobSource, setJobSource] = useState<SourceTab>('file')
  const [jobStatus, setJobStatus] = useState<IngestJobStatus | null>(null)
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [logs, setLogs] = useState<IngestLogEntry[]>([])
  const [discarded, setDiscarded] = useState<DiscardedSegment[]>([])
  const [completedSummary, setCompletedSummary] = useState<Record<string, any> | null>(null)

  const logContainerRef = useRef<HTMLDivElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  //: The title the last probe filled in. A title still equal to it was not typed by hand, so a
  //: new URL may replace it; anything else is the annotator's and is left alone.
  const probedTitleRef = useRef<string>('')

  const rememberJob = (id: string | null) => {
    setJobId(id)
    try {
      if (id) window.localStorage.setItem(ACTIVE_JOB_KEY, id)
      else window.localStorage.removeItem(ACTIVE_JOB_KEY)
    } catch {
      // A browser with storage disabled loses only the reattach, not the job.
    }
  }

  const handleTitleChange = (val: string) => {
    setEpisodeTitle(val)
    if (!isManualEpisodeId) setEpisodeId(slugify(val))
  }

  const handleFileSelected = (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase() || ''
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      toast.error(`Unsupported format .${ext}`, {
        description: 'Please use MP3, M4A, WAV, FLAC, AAC or OGG.',
      })
      return
    }
    setSelectedFile(file)
    if (!episodeTitle) {
      const baseName = file.name.replace(/\.[^/.]+$/, '')
      setEpisodeTitle(baseName)
      if (!isManualEpisodeId) setEpisodeId(slugify(baseName))
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files?.length) handleFileSelected(e.dataTransfer.files[0])
  }

  const handleStartIngestion = async () => {
    if (!selectedFile) {
      toast.warning('Please select an audio file')
      return
    }
    if (!episodeTitle.trim()) {
      toast.warning('Please enter an episode title')
      return
    }

    setIsSubmitting(true)
    setLogs([])
    setDiscarded([])
    setCompletedSummary(null)

    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('episode_title', episodeTitle.trim())
    formData.append('show_id', showId.trim() || 'podcast')
    formData.append('episode_id', episodeId.trim())
    formData.append('genre', genre.trim())
    formData.append('topic', topic.trim())
    formData.append('speakers_json', buildSpeakersJson())

    try {
      const res = await api.startIngest(formData)
      setJobSource('file')
      rememberJob(res.job_id)
      toast.info(`Ingestion started for '${res.title}'`)
    } catch (err: any) {
      setIsSubmitting(false)
      toast.error(err.message || 'Failed to start ingestion')
    }
  }

  const handleStartYoutubeIngestion = async () => {
    if (!youtubeUrl.trim()) {
      toast.warning('Please paste a YouTube URL')
      return
    }

    setIsSubmitting(true)
    setLogs([])
    setDiscarded([])
    setCompletedSummary(null)

    try {
      const res = await api.startYouTubeIngest({
        url: youtubeUrl.trim(),
        episode_title: episodeTitle.trim(),
        show_id: showId.trim() || 'podcast',
        episode_id: episodeId.trim(),
        genre: genre.trim(),
        topic: topic.trim(),
        speakers_json: buildSpeakersJson(),
      })
      setJobSource('youtube')
      rememberJob(res.job_id)
      toast.info(`Ingestion started for '${res.title}'`)
    } catch (err: any) {
      setIsSubmitting(false)
      toast.error(err.message || 'Failed to start ingestion')
    }
  }

  // Look the URL up as it is typed, so the form fills itself in. The lookup downloads nothing,
  // and it front-loads every rejection the ingest call would make -- a private, live or
  // over-long video is refused here rather than after the annotator commits to it.
  useEffect(() => {
    const url = youtubeUrl.trim()
    if (sourceTab !== 'youtube' || !url) {
      setProbe(null)
      setProbeError(null)
      setIsProbing(false)
      return
    }

    let cancelled = false
    setIsProbing(true)
    setProbeError(null)

    const timer = window.setTimeout(() => {
      api
        .probeYouTube(url)
        .then((info) => {
          if (cancelled) return
          setProbe(info)
          setProbeError(null)
          // Only claim the title if it is still the one a previous probe wrote.
          setEpisodeTitle((current) => {
            if (current && current !== probedTitleRef.current) return current
            probedTitleRef.current = info.title
            if (!isManualEpisodeId) setEpisodeId(slugify(info.title))
            return info.title
          })
        })
        .catch((err: any) => {
          if (cancelled) return
          setProbe(null)
          setProbeError(err?.detail || err?.message || 'Could not read that URL')
        })
        .finally(() => {
          if (!cancelled) setIsProbing(false)
        })
    }, PROBE_DEBOUNCE_MS)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [youtubeUrl, sourceTab, isManualEpisodeId])

  // Attach to a job: its outcome from the status endpoint, its live progress from the stream.
  useEffect(() => {
    if (!jobId) return

    let cancelled = false
    let unsubscribe: (() => void) | null = null

    const handleEvent = (evt: IngestEvent) => {
      if (evt.type === 'log') {
        setLogs((prev) => [
          ...prev,
          { timestamp: evt.timestamp, level: evt.level, message: evt.message },
        ])
      } else if (evt.type === 'progress') {
        setJobStatus((prev) =>
          prev
            ? {
                ...prev,
                stage: evt.stage as IngestJobStatus['stage'],
                progress: evt.progress,
                active_segments: evt.active_segments,
                total_segments: evt.total_segments,
              }
            : prev,
        )
      } else if (evt.type === 'discard') {
        // The stream replays discards on connect, so a rejoin must not double-count them.
        setDiscarded((prev) =>
          prev.some((d) => d.segment_id === evt.segment.segment_id) ? prev : [...prev, evt.segment],
        )
      } else if (evt.type === 'complete') {
        setCompletedSummary(evt.summary)
        setJobStatus((prev) =>
          prev ? { ...prev, status: 'completed', stage: 'complete', progress: 100 } : prev,
        )
        setIsSubmitting(false)
        toast.success('Ingestion complete', { description: 'The episode is ready for annotation.' })
        // The server closes the stream on the last event and EventSource treats a closed stream
        // as an error to retry, so nothing closes this from the other end.
        unsubscribe?.()
        unsubscribe = null
      } else if (evt.type === 'error') {
        setJobStatus((prev) =>
          prev ? { ...prev, status: 'failed', stage: 'failed', error: evt.error } : prev,
        )
        setIsSubmitting(false)
        toast.error(`Ingestion error: ${evt.error}`)
        unsubscribe?.()
        unsubscribe = null
      }
    }

    // The snapshot decides whether there is anything to watch. Opening a stream on a job that has
    // already ended is not merely wasteful: the server closes it immediately, EventSource reads
    // that as a failure and reconnects every few seconds, and every reconnect replays the whole
    // history into the log. Measured before this: seven copies of a finished run's log, growing.
    api
      .getIngestStatus(jobId)
      .then((status) => {
        if (cancelled) return
        setJobStatus(status)
        setDiscarded(status.discarded_segments || [])

        const finished = status.status === 'completed' || status.status === 'failed'
        setIsSubmitting(!finished)
        if (finished) {
          setLogs(status.logs || [])
          setCompletedSummary(status.summary ?? null)
          return
        }

        // Live job: the stream replays the history, so it owns the log outright. Taking the
        // snapshot's copy as well would show everything so far twice.
        setLogs([])
        unsubscribe = api.subscribeIngestEvents(jobId, handleEvent)
      })
      .catch(() => {
        // A job id parked in storage may name one the server has since evicted.
        if (cancelled) return
        rememberJob(null)
        setJobStatus(null)
      })

    return () => {
      cancelled = true
      unsubscribe?.()
    }
  }, [jobId])

  // Auto-scroll the terminal console
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  const resetForNextRun = () => {
    if (isSubmitting && !window.confirm('Ingestion is still running. Start a new one anyway?')) {
      return
    }
    rememberJob(null)
    setJobStatus(null)
    setIsSubmitting(false)
    setSelectedFile(null)
    setYoutubeUrl('')
    setProbe(null)
    setProbeError(null)
    setIsProbing(false)
    probedTitleRef.current = ''
    setEpisodeTitle('')
    setEpisodeId('')
    setGenre('podcast')
    setTopic('')
    setSpk0Name('')
    setSpk0Gender('')
    setSpk0Origin('')
    setSpk1Name('')
    setSpk1Gender('')
    setSpk1Origin('')
    setShowSociolinguistics(false)
    setLogs([])
    setDiscarded([])
    setCompletedSummary(null)
  }

  const handleStartAnnotating = () => {
    const target = jobStatus?.episode_id || episodeId || 'new'
    resetForNextRun()
    onComplete(target)
  }

  const handleCopyLogs = () => {
    const text = logs
      .map((l) => `[${l.timestamp}] [${l.level.toUpperCase()}] ${l.message}`)
      .join('\n')
    navigator.clipboard.writeText(text)
    toast.info('Logs copied to clipboard')
  }

  const currentStage = jobStatus?.stage || 'upload'
  const isComplete = jobStatus?.status === 'completed' || currentStage === 'complete'
  const isFailed = jobStatus?.status === 'failed' || currentStage === 'failed'

  // A URL job fetches its own audio before stage 1, so its stepper carries one extra step.
  const stages = jobSource === 'youtube' ? [DOWNLOAD_STAGE, ...STAGES] : STAGES

  const getStageState = (stageKey: string) => {
    if (isComplete) return 'done'
    const stageOrder = stages.map((stage) => stage.key)
    const currentIndex = stageOrder.indexOf(currentStage)
    const thisIndex = stageOrder.indexOf(stageKey)

    if (currentIndex === -1) return 'pending'
    if (thisIndex < currentIndex) return 'done'
    if (thisIndex === currentIndex) return isFailed ? 'failed' : 'active'
    return 'pending'
  }

  const kept = completedSummary?.segments ?? null
  const detected = completedSummary?.segments_detected ?? jobStatus?.total_segments ?? null

  return (
    <div className="scrollbar-thin flex-1 overflow-y-auto">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-4 sm:p-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="flex flex-col gap-1">
            <h1 className="font-heading text-lg font-bold tracking-wide uppercase">
              Ingest episode
            </h1>
            <p className="text-xs text-muted-foreground">
              Upload a file or paste a YouTube URL. Loudnorm, Silero VAD segmentation, then every
              configured ASR system on every clip.
            </p>
          </div>

          {jobId && (
            <div className="flex items-center gap-2">
              {isSubmitting && (
                <span className="flex items-center gap-1.5 font-mono text-xs text-muted-foreground">
                  <Spinner className="size-3.5" />
                  Job #{jobId.slice(0, 8)}
                </span>
              )}
              <Button variant="outline" size="sm" onClick={resetForNextRun}>
                New ingestion
              </Button>
              {isComplete && <Button size="sm" onClick={handleStartAnnotating}>Start annotating</Button>}
            </div>
          )}
        </div>

        {!jobId ? (
          <div className="flex flex-col gap-6">
            <Tabs value={sourceTab} onValueChange={(value) => setSourceTab(value as SourceTab)}>
              <TabsList variant="line">
                <TabsTrigger value="file">
                  <RiUploadCloud2Line className="size-4" />
                  Upload file
                </TabsTrigger>
                <TabsTrigger value="youtube">
                  <RiYoutubeLine className="size-4" />
                  YouTube URL
                </TabsTrigger>
              </TabsList>

              <TabsContent value="file">
                <button
                  type="button"
                  className={cn(
                    'flex w-full flex-col items-center gap-3 border border-dashed p-8 text-center transition-colors',
                    isDragging ? 'border-foreground bg-accent' : 'hover:bg-muted/50',
                    selectedFile && 'border-solid bg-muted/30',
                  )}
                  onDragOver={(e) => {
                    e.preventDefault()
                    setIsDragging(true)
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    accept=".mp3,.m4a,.wav,.flac,.aac,.ogg"
                    onChange={(e) => {
                      if (e.target.files?.[0]) handleFileSelected(e.target.files[0])
                    }}
                  />

                  {selectedFile ? (
                    <div className="flex w-full items-center gap-4 text-left">
                      <RiFileMusicLine className="size-8 shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium">{selectedFile.name}</div>
                        <div className="font-mono text-xs text-muted-foreground">
                          {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB ·{' '}
                          {selectedFile.name.split('.').pop()?.toUpperCase()}
                        </div>
                      </div>
                      <span className="text-xs font-semibold tracking-widest uppercase">Change</span>
                    </div>
                  ) : (
                    <>
                      <RiUploadCloud2Line className="size-9 text-muted-foreground" />
                      <div className="text-sm">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                        podcast audio
                      </div>
                      <div className="flex gap-1.5 font-mono text-[10px] text-muted-foreground">
                        {ALLOWED_EXTENSIONS.map((ext) => (
                          <span key={ext} className="bg-muted px-1.5 py-0.5 uppercase">
                            {ext}
                          </span>
                        ))}
                      </div>
                    </>
                  )}
                </button>
              </TabsContent>

              <TabsContent value="youtube">
                <div className="flex flex-col gap-3">
                  <Field>
                    <FieldLabel htmlFor="youtube-url">Video URL</FieldLabel>
                    <Input
                      id="youtube-url"
                      className="font-mono text-xs"
                      placeholder="https://www.youtube.com/watch?v=..."
                      value={youtubeUrl}
                      autoComplete="off"
                      spellCheck={false}
                      onChange={(e) => setYoutubeUrl(e.target.value)}
                    />
                  </Field>

                  {isProbing && (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Spinner className="size-3.5" />
                      <span>Reading video details…</span>
                    </div>
                  )}

                  {!isProbing && probeError && (
                    <Alert variant="destructive">
                      <RiErrorWarningLine />
                      <AlertTitle>Cannot ingest this URL</AlertTitle>
                      <AlertDescription>{probeError}</AlertDescription>
                    </Alert>
                  )}

                  {!isProbing && probe && (
                    <div className="flex items-start gap-4 border bg-muted/30 p-3">
                      {probe.thumbnail && (
                        <img
                          src={probe.thumbnail}
                          alt=""
                          className="hidden w-32 shrink-0 object-cover sm:block"
                        />
                      )}
                      <div className="flex min-w-0 flex-1 flex-col gap-1">
                        <div className="truncate text-sm font-medium">{probe.title}</div>
                        <div className="font-mono text-xs text-muted-foreground">
                          {probe.uploader ? `${probe.uploader} · ` : ''}
                          {formatDuration(probe.duration_seconds)} · {probe.video_id}
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                          Audio is downloaded on the server, then normalized and segmented like any
                          upload.
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>

            {/* Metadata */}
            <div className="flex flex-col gap-4">
              <Field>
                <FieldLabel htmlFor="episode-title">Episode title</FieldLabel>
                <Input
                  id="episode-title"
                  placeholder="e.g. Episode 42: AI in Kathmandu"
                  value={episodeTitle}
                  onChange={(e) => handleTitleChange(e.target.value)}
                />
              </Field>

              <div className="grid gap-4 sm:grid-cols-2">
                <Field>
                  <FieldLabel htmlFor="show-id">Show ID</FieldLabel>
                  <Input
                    id="show-id"
                    placeholder="podcast"
                    value={showId}
                    onChange={(e) => setShowId(e.target.value)}
                  />
                </Field>

                <Field>
                  <div className="flex items-center justify-between gap-2">
                    <FieldLabel htmlFor="episode-id">Episode ID (slug)</FieldLabel>
                    <Button
                      variant="link"
                      size="xs"
                      className="h-auto px-0"
                      onClick={() => {
                        setIsManualEpisodeId(!isManualEpisodeId)
                        if (isManualEpisodeId) setEpisodeId(slugify(episodeTitle))
                      }}
                    >
                      {isManualEpisodeId ? 'Auto-generate' : 'Custom'}
                    </Button>
                  </div>
                  <Input
                    id="episode-id"
                    className="font-mono"
                    placeholder="ep_42_ai_in_kathmandu"
                    value={episodeId}
                    readOnly={!isManualEpisodeId}
                    onChange={(e) => setEpisodeId(e.target.value)}
                  />
                </Field>
              </div>

              {/* Sociolinguistics & Speaker Metadata Toggle */}
              <div className="rounded-lg border bg-card p-3">
                <button
                  type="button"
                  className="flex w-full items-center justify-between text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground"
                  onClick={() => setShowSociolinguistics((prev) => !prev)}
                >
                  <span className="flex items-center gap-1.5">
                    <RiUserVoiceLine className="size-4 text-primary" />
                    Speaker Demographics &amp; Topic (Sociolinguistics · Optional)
                  </span>
                  {showSociolinguistics ? (
                    <RiArrowDownSLine className="size-4" />
                  ) : (
                    <RiArrowRightSLine className="size-4" />
                  )}
                </button>

                {showSociolinguistics && (
                  <div className="mt-3 flex flex-col gap-3.5 border-t pt-3">
                    <div className="grid gap-3 sm:grid-cols-2">
                      <Field>
                        <FieldLabel htmlFor="genre">Genre / Category</FieldLabel>
                        <Input
                          id="genre"
                          placeholder="e.g. podcast_interview, tech_review"
                          value={genre}
                          onChange={(e) => setGenre(e.target.value)}
                        />
                      </Field>
                      <Field>
                        <FieldLabel htmlFor="topic">Topic / Domain</FieldLabel>
                        <Input
                          id="topic"
                          placeholder="e.g. tech_gadgets, business, lifestyle"
                          value={topic}
                          onChange={(e) => setTopic(e.target.value)}
                        />
                      </Field>
                    </div>

                    {/* Speaker 0 (Host) */}
                    <div className="rounded border bg-muted/20 p-2.5">
                      <div className="mb-2 text-xs font-medium text-foreground">
                        Speaker 0 (Host / Primary)
                      </div>
                      <div className="grid gap-2 sm:grid-cols-3">
                        <Input
                          placeholder="Name (e.g. Sushant)"
                          value={spk0Name}
                          onChange={(e) => setSpk0Name(e.target.value)}
                        />
                        <select
                          className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                          value={spk0Gender}
                          onChange={(e) => setSpk0Gender(e.target.value)}
                        >
                          <option value="">Gender (optional)</option>
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                          <option value="non_binary">Non-binary</option>
                          <option value="other">Other</option>
                        </select>
                        <Input
                          placeholder="Dialect / Origin (e.g. Kathmandu)"
                          value={spk0Origin}
                          onChange={(e) => setSpk0Origin(e.target.value)}
                        />
                      </div>
                    </div>

                    {/* Speaker 1 (Guest) */}
                    <div className="rounded border bg-muted/20 p-2.5">
                      <div className="mb-2 text-xs font-medium text-foreground">
                        Speaker 1 (Guest / Secondary)
                      </div>
                      <div className="grid gap-2 sm:grid-cols-3">
                        <Input
                          placeholder="Name (e.g. Kusang)"
                          value={spk1Name}
                          onChange={(e) => setSpk1Name(e.target.value)}
                        />
                        <select
                          className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                          value={spk1Gender}
                          onChange={(e) => setSpk1Gender(e.target.value)}
                        >
                          <option value="">Gender (optional)</option>
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                          <option value="non_binary">Non-binary</option>
                          <option value="other">Other</option>
                        </select>
                        <Input
                          placeholder="Dialect / Origin (e.g. Kathmandu)"
                          value={spk1Origin}
                          onChange={(e) => setSpk1Origin(e.target.value)}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Pipeline overview */}
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 bg-muted/40 p-3 font-mono text-[11px] text-muted-foreground">
              {(sourceTab === 'youtube' ? [DOWNLOAD_STAGE, ...STAGES] : STAGES).map(
                (stage, index) => (
                  <React.Fragment key={stage.key}>
                    {index > 0 && <span aria-hidden>→</span>}
                    <span>{stage.label}</span>
                  </React.Fragment>
                ),
              )}
            </div>

            <div className="flex items-center justify-end gap-2 border-t pt-4">
              <Button
                disabled={
                  isSubmitting ||
                  !episodeTitle.trim() ||
                  (sourceTab === 'file' ? !selectedFile : !probe || isProbing)
                }
                onClick={sourceTab === 'file' ? handleStartIngestion : handleStartYoutubeIngestion}
              >
                {isSubmitting && <Spinner data-icon="inline-start" />}
                {isSubmitting ? 'Starting…' : 'Start ingestion'}
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {/* Stepper */}
            <ol
              className={cn('grid gap-2', stages.length === 6 ? 'sm:grid-cols-6' : 'sm:grid-cols-5')}
            >
              {stages.map((stage) => {
                const state = getStageState(stage.key)
                return (
                  <li
                    key={stage.key}
                    className={cn(
                      'flex flex-col gap-1 border-t-2 pt-2',
                      state === 'done' && 'border-success',
                      state === 'active' && 'border-info',
                      state === 'failed' && 'border-destructive',
                      state === 'pending' && 'border-border',
                    )}
                  >
                    <span className="flex items-center gap-1.5 font-heading text-[11px] font-semibold tracking-wider uppercase">
                      {state === 'done' && <RiCheckLine className="size-3.5 text-success" />}
                      {state === 'active' && <Spinner className="size-3.5 text-info" />}
                      {state === 'failed' && (
                        <RiErrorWarningLine className="size-3.5 text-destructive" />
                      )}
                      {stage.label}
                    </span>
                    <span className="text-[11px] leading-snug text-muted-foreground">
                      {stage.desc}
                    </span>
                  </li>
                )
              })}
            </ol>

            {/* Progress */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-2">
                <span className="font-heading text-xs font-semibold tracking-widest uppercase">
                  {isComplete
                    ? 'Ingestion complete'
                    : isFailed
                      ? 'Pipeline failed'
                      : `Processing: ${currentStage}`}
                </span>
                <span className="font-mono text-xs text-muted-foreground tabular-nums">
                  {jobStatus?.active_segments
                    ? `${jobStatus.active_segments}${
                        jobStatus.total_segments ? `/${jobStatus.total_segments}` : ''
                      } segments · `
                    : ''}
                  {discarded.length > 0 ? `${discarded.length} discarded · ` : ''}
                  {Math.round(jobStatus?.progress || 0)}%
                </span>
              </div>
              <Progress value={Math.min(100, jobStatus?.progress || 0)} />
            </div>

            <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
              {/* Terminal console */}
              <div className="border">
                <div className="flex items-center justify-between gap-2 border-b bg-muted/40 px-3 py-1.5">
                  <span className="font-heading text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
                    Live pipeline log (SSE)
                  </span>
                  <Button variant="ghost" size="xs" onClick={handleCopyLogs}>
                    Copy
                  </Button>
                </div>
                <div
                  ref={logContainerRef}
                  className="scrollbar-thin h-[26rem] overflow-y-auto bg-muted/20 p-2 font-mono text-[11px] leading-5"
                >
                  {logs.length === 0 ? (
                    <div className="text-muted-foreground">Connecting to event stream…</div>
                  ) : (
                    logs.map((log, i) => (
                      <div key={i} className="flex gap-2">
                        <span className="shrink-0 text-muted-foreground">{log.timestamp}</span>
                        <span
                          className={cn(
                            'shrink-0 uppercase',
                            LOG_LEVEL_CLASS[log.level] ?? 'text-muted-foreground',
                          )}
                        >
                          [{log.level}]
                        </span>
                        <span className="break-all">{log.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Right rail: outcome */}
              <div className="flex flex-col gap-4">
                {isComplete && completedSummary && (
                  <Alert className="border-success/40 text-success">
                    <RiCheckLine />
                    <AlertTitle>Episode ready for review</AlertTitle>
                    <AlertDescription>
                      {kept} of {detected} segments kept, {completedSummary.tasks_created} tasks
                      queued.
                    </AlertDescription>
                  </Alert>
                )}

                {isFailed && (
                  <Alert variant="destructive">
                    <RiErrorWarningLine />
                    <AlertTitle>Ingestion error</AlertTitle>
                    <AlertDescription>
                      {jobStatus?.error || 'An unknown error occurred.'}
                    </AlertDescription>
                  </Alert>
                )}

                <DiscardPanel discarded={discarded} />

                {isComplete && discarded.length === 0 && (
                  <div className="border bg-muted/20 p-3 text-[11px] text-muted-foreground">
                    Every detected segment was transcribed by every configured system. Nothing was
                    discarded.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
