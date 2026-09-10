import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  RiAddLine,
  RiArrowDownSLine,
  RiArrowRightSLine,
  RiCheckLine,
  RiCloseLine,
  RiCpuLine,
  RiDeleteBin6Line,
  RiAlarmWarningLine,
  RiErrorWarningLine,
  RiExternalLinkLine,
  RiFileMusicLine,
  RiHistoryLine,
  RiListCheck2,
  RiRefreshLine,
  RiRobotLine,
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
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type {
  DiscardedSegment,
  IngestEvent,
  IngestJobStatus,
  IngestLogEntry,
  IngestQueueResponse,
  YouTubeBatchIngestOut,
  YouTubeProbe,
} from '@/types'

interface IngestViewProps {
  onComplete: (episodeId: string) => void
}

const STAGES = [
  { key: 'normalizing', label: 'Normalize audio', desc: 'FFmpeg loudnorm, 16 kHz mono FLAC' },
  { key: 'segmenting', label: 'Silero VAD', desc: 'CPU speech turn detection (2s–20s)' },
  { key: 'transcribing', label: 'Cloud ASR', desc: 'Every configured system, per clip' },
  { key: 'fusing', label: 'Fusion', desc: 'Reasoning model reconciles the recognisers' },
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

type SourceTab = 'file' | 'youtube' | 'batch_youtube'

/** How long to sit on a keystroke before asking the backend what the URL points at. */
const PROBE_DEBOUNCE_MS = 500

/**
 * How long the AZ-5 cover stays open after the first press.
 */
const SCRAM_ARM_TIMEOUT_MS = 8000

/** The show id a form starts on, before a probe offers the channel name instead. */
const DEFAULT_SHOW_ID = 'nepanglish'

/** How many speakers one episode may declare. Beyond four, a form is the wrong instrument. */
const MAX_SPEAKERS = 4

type SpeakerDraft = { gender: string; ageBracket: string }

const emptySpeaker = (): SpeakerDraft => ({ gender: '', ageBracket: '' })

const GENDER_OPTIONS = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
]

const AGE_BRACKET_OPTIONS = [
  { value: 'under_20', label: 'Under 20' },
  { value: '20_39', label: '20-39' },
  { value: '40_59', label: '40-59' },
  { value: '60_79', label: '60-79' },
  { value: '80_plus', label: '80+' },
]

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

function formatTimestamp(ts: number | string | null | undefined) {
  if (ts === undefined || ts === null) return ''
  try {
    const ms = typeof ts === 'number' ? (ts < 1e11 ? ts * 1000 : ts) : Date.parse(ts)
    const d = new Date(ms)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return String(ts)
  }
}

function slugify(text: string) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\p{M}\s_-]/gu, '')
    .trim()
    .replace(/[-\s]+/g, '_')
    .slice(0, 40)
    .replace(/^_+|_+$/g, '')
}

/**
 * What was thrown away, and who threw it.
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
    <div className="border border-warning/40 bg-card">
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
  // Top navigation view
  const [mainView, setMainView] = useState<'queue' | 'new'>('queue')

  // Ingestion queue & background management
  const [queueData, setQueueData] = useState<IngestQueueResponse | null>(null)
  const [isRefreshingQueue, setIsRefreshingQueue] = useState<boolean>(false)
  const [historyFilter, setHistoryFilter] = useState<'all' | 'completed' | 'failed'>('all')

  // Form state
  const [sourceTab, setSourceTab] = useState<SourceTab>('youtube')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [showId, setShowId] = useState<string>(DEFAULT_SHOW_ID)
  const [episodeTitle, setEpisodeTitle] = useState<string>('')
  const [episodeId, setEpisodeId] = useState<string>('')
  const [isManualEpisodeId, setIsManualEpisodeId] = useState<boolean>(false)
  const [isDragging, setIsDragging] = useState<boolean>(false)

  // YouTube single source state
  const [youtubeUrl, setYoutubeUrl] = useState<string>('')
  const [probe, setProbe] = useState<YouTubeProbe | null>(null)
  const [isProbing, setIsProbing] = useState<boolean>(false)
  const [probeError, setProbeError] = useState<string | null>(null)

  // YouTube batch source state
  const [batchUrlsText, setBatchUrlsText] = useState<string>('')
  const [isSubmittingBatch, setIsSubmittingBatch] = useState<boolean>(false)
  const [batchResult, setBatchResult] = useState<YouTubeBatchIngestOut | null>(null)

  // Sociolinguistic metadata state
  const [showSociolinguistics, setShowSociolinguistics] = useState<boolean>(false)
  const [genre, setGenre] = useState<string>('podcast')
  const [topic, setTopic] = useState<string>('')
  const [speakers, setSpeakers] = useState<SpeakerDraft[]>(() => [emptySpeaker()])

  // Active Job state
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
  const [scramArmed, setScramArmed] = useState<boolean>(false)
  const [isScramming, setIsScramming] = useState<boolean>(false)
  const [logs, setLogs] = useState<IngestLogEntry[]>([])
  const [discarded, setDiscarded] = useState<DiscardedSegment[]>([])
  const [completedSummary, setCompletedSummary] = useState<Record<string, any> | null>(null)

  const logContainerRef = useRef<HTMLDivElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const titleIsAnnotatorsRef = useRef<boolean>(false)
  const showIdIsAnnotatorsRef = useRef<boolean>(false)

  const rememberJob = (id: string | null) => {
    setJobId(id)
    try {
      if (id) window.localStorage.setItem(ACTIVE_JOB_KEY, id)
      else window.localStorage.removeItem(ACTIVE_JOB_KEY)
    } catch {
      // Ignore storage errors
    }
  }

  // Poll queue state
  const fetchQueue = async (quiet = false) => {
    if (!quiet) setIsRefreshingQueue(true)
    try {
      const q = await api.getIngestQueue()
      setQueueData(q)
      // If there is an active running job on server and we don't have a jobId locally, attach to it
      if (q.running && !jobId) {
        rememberJob(q.running.job_id)
        if (q.running.source_url) setJobSource('youtube')
      }
    } catch (err) {
      console.error('Failed to fetch queue:', err)
    } finally {
      if (!quiet) setIsRefreshingQueue(false)
    }
  }

  useEffect(() => {
    fetchQueue(true)
    const interval = window.setInterval(() => {
      fetchQueue(true)
    }, 3500)
    return () => window.clearInterval(interval)
  }, [jobId])

  const parsedBatchUrls = useMemo(() => {
    return batchUrlsText
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith('#'))
  }, [batchUrlsText])

  const addSpeaker = () =>
    setSpeakers((current) =>
      current.length >= MAX_SPEAKERS ? current : [...current, emptySpeaker()],
    )

  const removeSpeaker = (index: number) =>
    setSpeakers((current) =>
      current.length <= 1 ? current : current.filter((_, i) => i !== index),
    )

  const updateSpeaker = (index: number, patch: Partial<SpeakerDraft>) =>
    setSpeakers((current) => current.map((s, i) => (i === index ? { ...s, ...patch } : s)))

  const buildSpeakersJson = () => {
    const payload: Record<string, any> = {}
    speakers.forEach((speaker, index) => {
      const fields: Record<string, string> = { role: index === 0 ? 'host' : 'guest' }
      if (speaker.gender) fields.gender = speaker.gender
      if (speaker.ageBracket) fields.age_bracket = speaker.ageBracket
      if (speaker.gender || speaker.ageBracket) payload[`spk${index}`] = fields
    })
    return Object.keys(payload).length > 0 ? JSON.stringify(payload) : ''
  }

  const handleTitleChange = (val: string) => {
    titleIsAnnotatorsRef.current = val.trim() !== ''
    setEpisodeTitle(val)
    if (!isManualEpisodeId) setEpisodeId(slugify(val))
  }

  const handleShowIdChange = (val: string) => {
    showIdIsAnnotatorsRef.current = val.trim() !== ''
    setShowId(val)
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
      titleIsAnnotatorsRef.current = true
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
      setMainView('queue')
      fetchQueue(true)
      toast.info(`Ingestion queued for '${res.title}'`)
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
      setMainView('queue')
      fetchQueue(true)
      toast.info(`Ingestion queued for '${res.title}'`)
    } catch (err: any) {
      setIsSubmitting(false)
      toast.error(err.message || 'Failed to start ingestion')
    }
  }

  const handleStartBatchYoutube = async () => {
    if (parsedBatchUrls.length === 0) {
      toast.warning('Please paste at least one YouTube URL')
      return
    }

    setIsSubmittingBatch(true)
    try {
      const res = await api.startYouTubeBatchIngest({
        urls: parsedBatchUrls,
        show_id: showId.trim() || 'podcast',
        genre: genre.trim(),
        topic: topic.trim(),
      })
      setBatchResult(res)
      toast.success(`Queued ${res.queued_count} of ${parsedBatchUrls.length} videos`)
      fetchQueue(true)
      if (res.errors.length === 0) {
        setBatchUrlsText('')
        setMainView('queue')
      }
    } catch (err: any) {
      toast.error(err.message || 'Failed to queue batch videos')
    } finally {
      setIsSubmittingBatch(false)
    }
  }

  // Probe single YouTube URL as user types
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
          if (!titleIsAnnotatorsRef.current) {
            setEpisodeTitle(info.title)
            if (!isManualEpisodeId) setEpisodeId(info.suggested_episode_id)
          }
          if (!showIdIsAnnotatorsRef.current && info.uploader) {
            setShowId(slugify(info.uploader))
          }
        })
        .catch((err) => {
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

  // Attach to active job: snapshot & SSE stream
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
        fetchQueue(true)
        unsubscribe?.()
        unsubscribe = null
      } else if (evt.type === 'scram') {
        setJobStatus((prev) => (prev ? { ...prev, scrammed: true, scram_reason: evt.reason } : prev))
      } else if (evt.type === 'aborted') {
        setCompletedSummary(evt.summary)
        setJobStatus((prev) => (prev ? { ...prev, status: 'aborted' } : prev))
        setIsSubmitting(false)
        setScramArmed(false)
        setIsScramming(false)
        toast.warning('Run scrammed', {
          description: `Stopped after ${evt.summary.segments_transcribed ?? 0} segments. Nothing imported.`,
        })
        fetchQueue(true)
        unsubscribe?.()
        unsubscribe = null
      } else if (evt.type === 'backlog') {
        setJobStatus((prev) =>
          prev ? { ...prev, status: 'backlog', stage: 'backlog', error: evt.error } : prev,
        )
        setIsSubmitting(false)
        toast.warning('Quarantined to Backlog', {
          description: evt.reason || 'YouTube challenge encountered. Queue continues.',
        })
        fetchQueue(true)
        unsubscribe?.()
        unsubscribe = null
      } else if (evt.type === 'error') {
        setJobStatus((prev) =>
          prev ? { ...prev, status: 'failed', stage: 'failed', error: evt.error } : prev,
        )
        setIsSubmitting(false)
        toast.error(`Ingestion error: ${evt.error}`)
        fetchQueue(true)
        unsubscribe?.()
        unsubscribe = null
      }
    }

    api
      .getIngestStatus(jobId)
      .then((status) => {
        if (cancelled) return
        setJobStatus(status)
        setDiscarded(status.discarded_segments || [])

        const finished =
          status.status === 'completed' ||
          status.status === 'failed' ||
          status.status === 'aborted' ||
          status.status === 'backlog'

        setIsSubmitting(!finished)
        if (finished) {
          setLogs(status.logs || [])
          setCompletedSummary(status.summary ?? null)
          return
        }

        setLogs([])
        unsubscribe = api.subscribeIngestEvents(jobId, handleEvent)
      })
      .catch(() => {
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
    titleIsAnnotatorsRef.current = false
    setShowId(DEFAULT_SHOW_ID)
    showIdIsAnnotatorsRef.current = false
    setEpisodeTitle('')
    setEpisodeId('')
    setGenre('podcast')
    setTopic('')
    setSpeakers([emptySpeaker()])
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

  useEffect(() => {
    if (!scramArmed) return
    const timer = window.setTimeout(() => setScramArmed(false), SCRAM_ARM_TIMEOUT_MS)
    return () => window.clearTimeout(timer)
  }, [scramArmed])

  const handleScram = async () => {
    if (!jobId) return
    if (!scramArmed) {
      setScramArmed(true)
      return
    }
    setScramArmed(false)
    setIsScramming(true)
    try {
      const result = await api.scramIngest(jobId)
      toast.warning(result.already_stopping ? 'Already stopping' : 'AZ-5 — halting run', {
        description: result.detail,
      })
    } catch (err) {
      setIsScramming(false)
      toast.error(`SCRAM failed: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const handleCopyLogs = () => {
    const text = logs
      .map((l) => `[${l.timestamp}] [${l.level.toUpperCase()}] ${l.message}`)
      .join('\n')
    navigator.clipboard.writeText(text)
    toast.info('Logs copied to clipboard')
  }

  // Queue actions
  const handleRetryJob = async (id: string) => {
    try {
      const res = await api.retryIngest(id)
      toast.success('Job requeued', { description: res.message })
      fetchQueue(true)
      if (!queueData?.running) {
        rememberJob(id)
      }
    } catch (err: any) {
      toast.error(err.message || 'Failed to retry job')
    }
  }

  const handleRetryAll = async (statusFilter?: string) => {
    try {
      const res = await api.retryAllIngest(statusFilter)
      toast.success(`Requeued ${res.retried_count} jobs`)
      fetchQueue(true)
    } catch (err: any) {
      toast.error(err.message || 'Failed to retry jobs')
    }
  }

  const handleCancelJob = async (id: string) => {
    try {
      const res = await api.cancelIngest(id)
      toast.info('Job cancelled / removed', { description: `Action: ${res.action}` })
      if (jobId === id) {
        rememberJob(null)
        setJobStatus(null)
      }
      fetchQueue(true)
    } catch (err: any) {
      toast.error(err.message || 'Failed to cancel job')
    }
  }

  const handleClearPast = async () => {
    try {
      const res = await api.clearPastIngest()
      toast.info(`Cleared ${res.cleared_count} past jobs`)
      fetchQueue(true)
    } catch (err: any) {
      toast.error(err.message || 'Failed to clear history')
    }
  }

  const currentStage = jobStatus?.stage || 'upload'
  const isComplete = jobStatus?.status === 'completed' || currentStage === 'complete'
  const isFailed = jobStatus?.status === 'failed' || currentStage === 'failed'
  const isAborted = jobStatus?.status === 'aborted' || currentStage === 'aborted'
  const isBacklog = jobStatus?.status === 'backlog' || currentStage === 'backlog'
  const isHalting = Boolean(jobStatus?.scrammed) && !isAborted
  const canScram = Boolean(jobId) && !isComplete && !isFailed && !isAborted && !isBacklog

  const stages = jobSource === 'youtube' ? [DOWNLOAD_STAGE, ...STAGES] : STAGES

  const getStageState = (stageKey: string) => {
    if (isComplete) return 'done'
    const stageOrder = stages.map((stage) => stage.key)
    const currentIndex = stageOrder.indexOf(currentStage)
    const thisIndex = stageOrder.indexOf(stageKey)

    if (currentIndex === -1) return 'pending'
    if (thisIndex < currentIndex) return 'done'
    if (thisIndex === currentIndex) return isFailed || isAborted ? 'failed' : 'active'
    return 'pending'
  }

  const kept = completedSummary?.segments ?? null
  const detected = completedSummary?.segments_detected ?? jobStatus?.total_segments ?? null

  const filteredPast = useMemo(() => {
    if (!queueData?.past) return []
    if (historyFilter === 'all') return queueData.past
    if (historyFilter === 'completed') return queueData.past.filter((j) => j.status === 'completed')
    if (historyFilter === 'failed') {
      return queueData.past.filter((j) => j.status === 'failed' || j.status === 'aborted')
    }
    return queueData.past
  }, [queueData?.past, historyFilter])

  return (
    <div className="scrollbar-thin flex-1 overflow-y-auto">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-4 sm:p-6">
        {/* Top Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b pb-4">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-3">
              <h1 className="font-heading text-xl font-bold tracking-wide uppercase">
                Ingest &amp; Queue
              </h1>
              {/* 8 Cores Active Badge */}
              <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 font-mono text-[11px] font-medium text-emerald-500">
                <RiCpuLine className="size-3.5" />
                8 Cores Active • Parallel Processing
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              Robust queue architecture for high-volume video pipelines (~50h throughput). Automatic
              bot-challenge isolation, background worker threads, and multi-core CPU encoding.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchQueue(false)}
              disabled={isRefreshingQueue}
              className="gap-1.5"
            >
              <RiRefreshLine className={cn('size-3.5', isRefreshingQueue && 'animate-spin')} />
              Refresh
            </Button>
            {jobId && canScram && (
              <Button
                variant="destructive"
                size="sm"
                onClick={handleScram}
                disabled={isScramming || isHalting}
                aria-label="AZ-5: stop this ingestion run"
                title="Stops the run at its next checkpoint. Nothing is imported."
                className={cn(
                  'font-mono text-xs',
                  scramArmed && 'animate-pulse bg-destructive text-white hover:bg-destructive',
                )}
              >
                <RiAlarmWarningLine data-icon="inline-start" className="size-3.5" />
                {isHalting ? 'Halting…' : scramArmed ? 'Confirm AZ-5' : 'AZ-5'}
              </Button>
            )}
          </div>
        </div>

        {/* View Switcher: Queue & Monitor vs New Ingestion */}
        <div className="flex items-center justify-between gap-3 border-b">
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setMainView('queue')}
              className={cn(
                'flex items-center gap-2 border-b-2 px-4 py-2.5 font-heading text-xs font-semibold tracking-wider uppercase transition-colors',
                mainView === 'queue'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground',
              )}
            >
              <RiListCheck2 className="size-4" />
              Queue &amp; Activity
              {queueData && (
                <span className="ml-1 rounded-full bg-muted px-2 py-0.5 font-mono text-[10px]">
                  {queueData.counts.running ? '1 active' : 'idle'} · {queueData.counts.upcoming} queued
                </span>
              )}
              {queueData && queueData.counts.backlog > 0 && (
                <span className="rounded-full bg-amber-500/20 px-2 py-0.5 font-mono text-[10px] font-bold text-amber-500">
                  {queueData.counts.backlog} backlog
                </span>
              )}
            </button>

            <button
              type="button"
              onClick={() => setMainView('new')}
              className={cn(
                'flex items-center gap-2 border-b-2 px-4 py-2.5 font-heading text-xs font-semibold tracking-wider uppercase transition-colors',
                mainView === 'new'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground',
              )}
            >
              <RiAddLine className="size-4" />
              New Ingestion (+ Batch)
            </button>
          </div>
        </div>

        {/* KPI Summary Cards */}
        {queueData && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-lg border bg-card p-3 shadow-xs">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                Active Process
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span
                  className={cn(
                    'text-lg font-bold',
                    queueData.running ? 'text-primary' : 'text-muted-foreground',
                  )}
                >
                  {queueData.running ? 'Running' : 'Idle'}
                </span>
                {queueData.running && (
                  <span className="truncate font-mono text-xs text-muted-foreground">
                    #{queueData.running.job_id.slice(0, 8)}
                  </span>
                )}
              </div>
              <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
                {queueData.running ? queueData.running.title : 'Ready for next episode'}
              </div>
            </div>

            <div className="rounded-lg border bg-card p-3 shadow-xs">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                Upcoming Queue
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-lg font-bold tabular-nums text-foreground">
                  {queueData.counts.upcoming}
                </span>
                <span className="text-xs text-muted-foreground">videos waiting</span>
              </div>
              <div className="mt-0.5 text-[11px] text-muted-foreground">Sequential FIFO runner</div>
            </div>

            <div
              className={cn(
                'rounded-lg border p-3 shadow-xs transition-colors',
                queueData.counts.backlog > 0
                  ? 'border-amber-500/40 bg-amber-500/5'
                  : 'bg-card text-muted-foreground',
              )}
            >
              <div
                className={cn(
                  'flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider',
                  queueData.counts.backlog > 0 ? 'text-amber-500' : 'text-muted-foreground',
                )}
              >
                <span>Backlog (Bot Checks)</span>
                {queueData.counts.backlog > 0 && <RiRobotLine className="size-3.5" />}
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span
                  className={cn(
                    'text-lg font-bold tabular-nums',
                    queueData.counts.backlog > 0 ? 'text-amber-500' : 'text-foreground',
                  )}
                >
                  {queueData.counts.backlog}
                </span>
                <span className="text-xs text-muted-foreground">on hold</span>
              </div>
              <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
                {queueData.counts.backlog > 0
                  ? 'Quarantined for later retry'
                  : 'No bot challenge halts'}
              </div>
            </div>

            <div className="rounded-lg border bg-card p-3 shadow-xs">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                Processed History
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-lg font-bold tabular-nums text-foreground">
                  {queueData.counts.past}
                </span>
                <span className="text-xs text-muted-foreground">finished jobs</span>
              </div>
              <div className="mt-0.5 text-[11px] text-muted-foreground">Completed &amp; logged</div>
            </div>
          </div>
        )}

        {/* ----------------- TAB: QUEUE & ACTIVITY ----------------- */}
        {mainView === 'queue' && (
          <div className="flex flex-col gap-6">
            {/* 1. RUNNING JOB MONITOR */}
            {jobId && jobStatus ? (
              <div className="flex flex-col gap-4 rounded-xl border bg-card p-4 sm:p-5 shadow-xs">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-3">
                  <div className="flex items-center gap-2.5">
                    <span className="flex size-2.5 rounded-full bg-emerald-500 animate-ping" />
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-heading text-sm font-bold tracking-wide uppercase">
                          {jobStatus.title || 'Running Job'}
                        </span>
                        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                          {jobStatus.episode_id}
                        </span>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Show: <span className="font-mono">{jobStatus.show_id}</span>
                        {jobStatus.source_url && ` · ${jobStatus.source_url}`}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {isComplete && (
                      <Button size="sm" onClick={handleStartAnnotating} className="gap-1">
                        <RiCheckLine className="size-4" />
                        Start annotating
                      </Button>
                    )}
                    <Button variant="outline" size="sm" onClick={resetForNextRun}>
                      Detach monitor
                    </Button>
                  </div>
                </div>

                {/* Stepper */}
                <ol
                  className={cn(
                    'grid gap-2',
                    stages.length === 7 ? 'sm:grid-cols-7' : 'sm:grid-cols-6',
                  )}
                >
                  {stages.map((stage) => {
                    const state = getStageState(stage.key)
                    return (
                      <li
                        key={stage.key}
                        className={cn(
                          'flex flex-col gap-1 border-t-2 pt-2 transition-colors',
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
                        <span className="text-[10px] leading-snug text-muted-foreground">
                          {stage.desc}
                        </span>
                      </li>
                    )
                  })}
                </ol>

                {/* Progress bar */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <span className="font-heading font-semibold tracking-wider uppercase">
                      {isComplete
                        ? 'Ingestion complete'
                        : isAborted
                          ? 'Scrammed — nothing imported'
                          : isBacklog
                            ? 'Quarantined to Backlog'
                            : isHalting
                              ? `AZ-5 — halting after ${currentStage}`
                              : isFailed
                                ? 'Pipeline failed'
                                : `Processing: ${currentStage}`}
                    </span>
                    <span className="font-mono text-muted-foreground tabular-nums">
                      {jobStatus.active_segments
                        ? `${jobStatus.active_segments}${
                            jobStatus.total_segments ? `/${jobStatus.total_segments}` : ''
                          } segments · `
                        : ''}
                      {discarded.length > 0 ? `${discarded.length} discarded · ` : ''}
                      {Math.round(jobStatus.progress || 0)}%
                    </span>
                  </div>
                  <Progress value={Math.min(100, jobStatus.progress || 0)} />
                </div>

                {/* Logs terminal & summary */}
                <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
                  <div className="rounded border bg-card">
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
                      className="scrollbar-thin h-64 overflow-y-auto bg-muted/10 p-2 font-mono text-[11px] leading-5"
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

                  <div className="flex flex-col gap-3">
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

                    {isAborted && (
                      <Alert variant="destructive">
                        <RiAlarmWarningLine />
                        <AlertTitle>Run scrammed (AZ-5)</AlertTitle>
                        <AlertDescription>
                          Stopped during {completedSummary?.stage_reached ?? currentStage}. Nothing was
                          imported.
                        </AlertDescription>
                      </Alert>
                    )}

                    {isBacklog && (
                      <Alert className="border-amber-500/40 text-amber-500">
                        <RiRobotLine />
                        <AlertTitle>Moved to Backlog</AlertTitle>
                        <AlertDescription>
                          {jobStatus.error || 'YouTube challenge encountered. Try retrying later.'}
                        </AlertDescription>
                      </Alert>
                    )}

                    {isFailed && (
                      <Alert variant="destructive">
                        <RiErrorWarningLine />
                        <AlertTitle>Ingestion error</AlertTitle>
                        <AlertDescription>
                          {jobStatus.error || 'An unknown error occurred.'}
                        </AlertDescription>
                      </Alert>
                    )}

                    <DiscardPanel discarded={discarded} />
                  </div>
                </div>
              </div>
            ) : null}

            {/* 2. BACKLOG SECTION (YouTube Bot Challenges / Hold) */}
            {queueData && queueData.backlog.length > 0 && (
              <div className="flex flex-col gap-3 rounded-xl border border-amber-500/40 bg-amber-500/5 p-4 sm:p-5 shadow-xs">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <RiRobotLine className="size-5 text-amber-500" />
                    <div>
                      <h2 className="font-heading text-sm font-bold tracking-wide uppercase text-amber-500">
                        YouTube Bot Verification Backlog ({queueData.backlog.length})
                      </h2>
                      <p className="text-xs text-muted-foreground">
                        YouTube detected automated queries or issued CAPTCHA challenges for these
                        videos. They were automatically quarantined so your remaining queue keeps
                        running.
                      </p>
                    </div>
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    className="border-amber-500/40 text-amber-500 hover:bg-amber-500/10"
                    onClick={() => handleRetryAll('backlog')}
                  >
                    <RiRefreshLine className="size-3.5" />
                    Retry all backlog ({queueData.backlog.length})
                  </Button>
                </div>

                <div className="divide-y rounded-lg border bg-card">
                  {queueData.backlog.map((job) => (
                    <div
                      key={job.job_id}
                      className="flex flex-wrap items-center justify-between gap-3 p-3 text-sm hover:bg-muted/20"
                    >
                      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-foreground truncate">
                            {job.title}
                          </span>
                          <span className="rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-amber-500 uppercase">
                            Bot challenge
                          </span>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 font-mono text-xs text-muted-foreground">
                          <span>ID: {job.episode_id}</span>
                          <span>·</span>
                          <span>Show: {job.show_id}</span>
                          {job.source_url && (
                            <>
                              <span>·</span>
                              <a
                                href={job.source_url}
                                target="_blank"
                                rel="noreferrer"
                                className="flex items-center gap-1 text-primary hover:underline"
                              >
                                YouTube <RiExternalLinkLine className="size-3" />
                              </a>
                            </>
                          )}
                        </div>
                        {job.error && (
                          <div className="mt-1 font-mono text-xs text-amber-500/90 break-all">
                            {job.error}
                          </div>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        <Button
                          size="xs"
                          variant="outline"
                          className="gap-1 border-amber-500/30 hover:bg-amber-500/10 text-amber-500"
                          onClick={() => handleRetryJob(job.job_id)}
                        >
                          <RiRefreshLine className="size-3" />
                          Retry
                        </Button>
                        <Button
                          size="xs"
                          variant="ghost"
                          className="text-muted-foreground hover:text-destructive"
                          onClick={() => handleCancelJob(job.job_id)}
                          aria-label="Remove from backlog"
                        >
                          <RiCloseLine className="size-3.5" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 3. UPCOMING QUEUE SECTION */}
            <div className="flex flex-col gap-3 rounded-xl border bg-card p-4 sm:p-5 shadow-xs">
              <div className="flex items-center justify-between gap-3 border-b pb-3">
                <div className="flex items-center gap-2">
                  <RiListCheck2 className="size-4 text-primary" />
                  <h2 className="font-heading text-sm font-bold tracking-wide uppercase">
                    Upcoming Queue ({queueData?.upcoming.length || 0})
                  </h2>
                </div>
                <div className="text-xs text-muted-foreground">
                  Processed in order · 1 at a time on worker thread
                </div>
              </div>

              {queueData && queueData.upcoming.length > 0 ? (
                <div className="divide-y rounded-lg border">
                  {queueData.upcoming.map((job) => (
                    <div
                      key={job.job_id}
                      className="flex flex-wrap items-center justify-between gap-3 p-3 text-sm hover:bg-muted/20"
                    >
                      <div className="flex items-center gap-3 min-w-0 flex-1">
                        <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/10 font-mono text-xs font-bold text-primary">
                          #{job.queue_position}
                        </span>
                        <div className="min-w-0 flex-1">
                          <div className="font-medium text-foreground truncate">{job.title}</div>
                          <div className="flex flex-wrap items-center gap-2 font-mono text-xs text-muted-foreground">
                            <span>ID: {job.episode_id}</span>
                            <span>·</span>
                            <span>Show: {job.show_id}</span>
                            {job.source_url && (
                              <>
                                <span>·</span>
                                <span className="truncate max-w-xs">{job.source_url}</span>
                              </>
                            )}
                            {job.created_at && (
                              <>
                                <span>·</span>
                                <span>Queued: {formatTimestamp(job.created_at)}</span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>

                      <Button
                        size="xs"
                        variant="ghost"
                        className="text-muted-foreground hover:text-destructive gap-1"
                        onClick={() => handleCancelJob(job.job_id)}
                      >
                        <RiCloseLine className="size-3.5" />
                        Cancel
                      </Button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 p-8 text-center text-muted-foreground border border-dashed rounded-lg">
                  <RiListCheck2 className="size-8 opacity-40" />
                  <div className="text-sm font-medium">No episodes waiting in queue</div>
                  <div className="text-xs">
                    New submissions will start running immediately.{' '}
                    <button
                      type="button"
                      onClick={() => setMainView('new')}
                      className="text-primary hover:underline font-semibold"
                    >
                      Enqueue episodes →
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* 4. PAST JOBS & HISTORY SECTION */}
            <div className="flex flex-col gap-3 rounded-xl border bg-card p-4 sm:p-5 shadow-xs">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-3">
                <div className="flex items-center gap-2">
                  <RiHistoryLine className="size-4 text-muted-foreground" />
                  <h2 className="font-heading text-sm font-bold tracking-wide uppercase">
                    Past History ({queueData?.past.length || 0})
                  </h2>
                </div>

                <div className="flex items-center gap-2">
                  <div className="flex rounded-md border bg-muted/20 p-0.5 text-xs">
                    <button
                      type="button"
                      onClick={() => setHistoryFilter('all')}
                      className={cn(
                        'rounded px-2.5 py-1 font-medium transition-colors',
                        historyFilter === 'all'
                          ? 'bg-background text-foreground shadow-xs'
                          : 'text-muted-foreground hover:text-foreground',
                      )}
                    >
                      All ({queueData?.past.length || 0})
                    </button>
                    <button
                      type="button"
                      onClick={() => setHistoryFilter('completed')}
                      className={cn(
                        'rounded px-2.5 py-1 font-medium transition-colors',
                        historyFilter === 'completed'
                          ? 'bg-background text-foreground shadow-xs'
                          : 'text-muted-foreground hover:text-foreground',
                      )}
                    >
                      Completed
                    </button>
                    <button
                      type="button"
                      onClick={() => setHistoryFilter('failed')}
                      className={cn(
                        'rounded px-2.5 py-1 font-medium transition-colors',
                        historyFilter === 'failed'
                          ? 'bg-background text-foreground shadow-xs'
                          : 'text-muted-foreground hover:text-foreground',
                      )}
                    >
                      Failed
                    </button>
                  </div>

                  <Button
                    variant="outline"
                    size="xs"
                    onClick={() => handleRetryAll('failed')}
                    title="Retry all failed jobs"
                  >
                    Retry all failed
                  </Button>
                  <Button variant="ghost" size="xs" onClick={handleClearPast}>
                    Clear finished
                  </Button>
                </div>
              </div>

              {filteredPast.length > 0 ? (
                <div className="divide-y rounded-lg border">
                  {filteredPast.map((job) => (
                    <div
                      key={job.job_id}
                      className="flex flex-wrap items-center justify-between gap-3 p-3 text-sm hover:bg-muted/20"
                    >
                      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-foreground truncate">{job.title}</span>
                          <span
                            className={cn(
                              'rounded px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase',
                              job.status === 'completed' &&
                                'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400',
                              job.status === 'failed' &&
                                'bg-destructive/20 text-destructive',
                              job.status === 'aborted' &&
                                'bg-amber-500/20 text-amber-500',
                            )}
                          >
                            {job.status}
                          </span>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 font-mono text-xs text-muted-foreground">
                          <span>ID: {job.episode_id}</span>
                          <span>·</span>
                          <span>Show: {job.show_id}</span>
                          {job.created_at && (
                            <>
                              <span>·</span>
                              <span>{formatTimestamp(job.created_at)}</span>
                            </>
                          )}
                        </div>
                        {job.error && (
                          <div className="mt-1 font-mono text-xs text-destructive break-all">
                            {job.error}
                          </div>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        {job.status === 'completed' && (
                          <Button
                            size="xs"
                            variant="outline"
                            onClick={() => onComplete(job.episode_id)}
                          >
                            Annotate
                          </Button>
                        )}
                        {(job.status === 'failed' || job.status === 'aborted') && (
                          <Button
                            size="xs"
                            variant="outline"
                            className="gap-1"
                            onClick={() => handleRetryJob(job.job_id)}
                          >
                            <RiRefreshLine className="size-3" />
                            Retry
                          </Button>
                        )}
                        <Button
                          size="xs"
                          variant="ghost"
                          className="text-muted-foreground hover:text-destructive"
                          onClick={() => handleCancelJob(job.job_id)}
                          aria-label="Delete history item"
                        >
                          <RiDeleteBin6Line className="size-3.5" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-6 text-center text-xs text-muted-foreground">
                  No past jobs match filter.
                </div>
              )}
            </div>
          </div>
        )}

        {/* ----------------- TAB: NEW INGESTION (+ BATCH) ----------------- */}
        {mainView === 'new' && (
          <div className="flex flex-col gap-6">
            <Tabs value={sourceTab} onValueChange={(value) => setSourceTab(value as SourceTab)}>
              <TabsList variant="line">
                <TabsTrigger value="youtube">
                  <RiYoutubeLine className="size-4" />
                  Single YouTube URL
                </TabsTrigger>
                <TabsTrigger value="batch_youtube">
                  <RiListCheck2 className="size-4 text-primary" />
                  Batch YouTube URLs (High Volume)
                </TabsTrigger>
                <TabsTrigger value="file">
                  <RiUploadCloud2Line className="size-4" />
                  Upload Local File
                </TabsTrigger>
              </TabsList>

              {/* 1. File Upload */}
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
                        <span className="font-semibold">Click to upload</span> or drag and drop podcast audio
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

              {/* 2. Single YouTube URL */}
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
                          Audio is downloaded on the server, then normalized and segmented across 8 CPU cores.
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </TabsContent>

              {/* 3. Batch YouTube URLs (High Volume) */}
              <TabsContent value="batch_youtube">
                <div className="flex flex-col gap-3 rounded-lg border bg-card p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-semibold">Bulk URL Queue (Up to 100 videos)</div>
                      <div className="text-xs text-muted-foreground">
                        Paste one YouTube URL per line. Perfect for 50+ hours ingestion sprints.
                      </div>
                    </div>
                    <span className="rounded-full bg-primary/10 px-2.5 py-1 font-mono text-xs font-semibold text-primary">
                      {parsedBatchUrls.length} valid URL{parsedBatchUrls.length === 1 ? '' : 's'}
                    </span>
                  </div>

                  <Textarea
                    className="min-h-36 font-mono text-xs p-3 rounded border"
                    placeholder={`https://www.youtube.com/watch?v=dQw4w9WgXcQ\nhttps://youtu.be/abc12345678\nhttps://www.youtube.com/watch?v=zyx98765432`}
                    value={batchUrlsText}
                    onChange={(e) => setBatchUrlsText(e.target.value)}
                  />

                  {batchResult && (
                    <div className="rounded border bg-muted/30 p-3 text-xs">
                      <div className="font-semibold text-foreground">
                        Batch Enqueued: {batchResult.queued_count} videos
                      </div>
                      {batchResult.errors.length > 0 && (
                        <div className="mt-2 text-destructive">
                          <div>Failed URLs ({batchResult.errors.length}):</div>
                          <ul className="list-disc pl-4 mt-1 font-mono text-[11px]">
                            {batchResult.errors.map((err, i) => (
                              <li key={i}>
                                {err.url}: {err.error}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>

            {/* Single Video Metadata (when not batch) */}
            {sourceTab !== 'batch_youtube' ? (
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
                      onChange={(e) => handleShowIdChange(e.target.value)}
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
              </div>
            ) : (
              /* Batch Shared Metadata */
              <div className="grid gap-4 sm:grid-cols-2">
                <Field>
                  <FieldLabel htmlFor="batch-show-id">Show ID for batch</FieldLabel>
                  <Input
                    id="batch-show-id"
                    placeholder="podcast"
                    value={showId}
                    onChange={(e) => handleShowIdChange(e.target.value)}
                  />
                </Field>

                <Field>
                  <FieldLabel htmlFor="batch-genre">Genre / Category</FieldLabel>
                  <Input
                    id="batch-genre"
                    placeholder="e.g. podcast_interview"
                    value={genre}
                    onChange={(e) => setGenre(e.target.value)}
                  />
                </Field>
              </div>
            )}

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
                        placeholder="Left blank: classified from the transcript"
                        value={topic}
                        onChange={(e) => setTopic(e.target.value)}
                      />
                    </Field>
                  </div>

                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-foreground">
                        Speakers ({speakers.length} of {MAX_SPEAKERS})
                      </span>
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        disabled={speakers.length >= MAX_SPEAKERS}
                        onClick={addSpeaker}
                      >
                        <RiAddLine className="size-3.5" />
                        Add speaker
                      </Button>
                    </div>

                    {speakers.map((speaker, index) => (
                      <div key={index} className="rounded border bg-muted/20 p-2.5">
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <span className="text-xs font-medium text-foreground">
                            Speaker {index} ({index === 0 ? 'Host' : 'Guest'})
                          </span>
                          <Button
                            type="button"
                            variant="ghost"
                            size="xs"
                            className="text-muted-foreground hover:text-destructive"
                            disabled={speakers.length <= 1}
                            onClick={() => removeSpeaker(index)}
                            aria-label={`Remove speaker ${index}`}
                          >
                            <RiDeleteBin6Line className="size-3.5" />
                          </Button>
                        </div>
                        <div className="grid gap-2 sm:grid-cols-2">
                          <select
                            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                            value={speaker.gender}
                            onChange={(e) => updateSpeaker(index, { gender: e.target.value })}
                          >
                            <option value="">Gender (optional)</option>
                            {GENDER_OPTIONS.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                          <select
                            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                            value={speaker.ageBracket}
                            onChange={(e) => updateSpeaker(index, { ageBracket: e.target.value })}
                          >
                            <option value="">Age bracket (optional)</option>
                            {AGE_BRACKET_OPTIONS.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Submit Bar */}
            <div className="flex items-center justify-between gap-3 border-t pt-4">
              <span className="font-mono text-xs text-muted-foreground">
                ⚡ 8 cores parallel processing on worker thread
              </span>

              {sourceTab === 'batch_youtube' ? (
                <Button
                  disabled={isSubmittingBatch || parsedBatchUrls.length === 0}
                  onClick={handleStartBatchYoutube}
                  className="gap-2"
                >
                  {isSubmittingBatch ? <Spinner className="size-4" /> : <RiListCheck2 className="size-4" />}
                  {isSubmittingBatch
                    ? 'Enqueueing batch…'
                    : `Enqueue ${parsedBatchUrls.length} YouTube Videos`}
                </Button>
              ) : (
                <Button
                  disabled={
                    isSubmitting ||
                    !episodeTitle.trim() ||
                    (sourceTab === 'file' ? !selectedFile : !probe || isProbing)
                  }
                  onClick={sourceTab === 'file' ? handleStartIngestion : handleStartYoutubeIngestion}
                  className="gap-2"
                >
                  {isSubmitting ? <Spinner className="size-4" /> : <RiUploadCloud2Line className="size-4" />}
                  {isSubmitting ? 'Starting…' : 'Start Ingestion'}
                </Button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
