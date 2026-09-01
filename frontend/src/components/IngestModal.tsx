import React, { useEffect, useRef, useState } from 'react'
import {
  RiCheckLine,
  RiErrorWarningLine,
  RiFileMusicLine,
  RiUploadCloud2Line,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Field, FieldLabel } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Spinner } from '@/components/ui/spinner'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { IngestEvent, IngestJobStatus, IngestLogEntry } from '@/types'

interface IngestModalProps {
  isOpen: boolean
  onClose: () => void
  onComplete: (episodeId: string) => void
}

const STAGES = [
  { key: 'normalizing', label: 'Normalize audio', desc: 'FFmpeg loudnorm, 16 kHz mono FLAC' },
  { key: 'segmenting', label: 'Silero VAD', desc: 'CPU speech turn detection (2s–20s)' },
  { key: 'transcribing', label: 'Cloud ASR', desc: 'OpenRouter speech-to-text inference' },
  { key: 'analyzing', label: 'Token analysis', desc: 'Devanagari/Latin tagging & CMI' },
  { key: 'importing', label: 'Direct import', desc: 'Database records & queue building' },
]

const ALLOWED_EXTENSIONS = ['mp3', 'm4a', 'wav', 'flac', 'aac', 'ogg']

const LOG_LEVEL_CLASS: Record<string, string> = {
  error: 'text-destructive',
  warn: 'text-warning',
  success: 'text-success',
}

function slugify(text: string) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/[-\s]+/g, '_')
    .slice(0, 40)
}

export function IngestModal({ isOpen, onClose, onComplete }: IngestModalProps) {
  // Form state
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [showId, setShowId] = useState<string>('nepanglish')
  const [episodeTitle, setEpisodeTitle] = useState<string>('')
  const [episodeId, setEpisodeId] = useState<string>('')
  const [isManualEpisodeId, setIsManualEpisodeId] = useState<boolean>(false)
  const [isDragging, setIsDragging] = useState<boolean>(false)

  // Execution state
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<IngestJobStatus | null>(null)
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [logs, setLogs] = useState<IngestLogEntry[]>([])
  const [completedSummary, setCompletedSummary] = useState<Record<string, any> | null>(null)

  const logContainerRef = useRef<HTMLDivElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

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

  // Submit ingestion job
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
    setCompletedSummary(null)

    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('episode_title', episodeTitle.trim())
    formData.append('show_id', showId.trim() || 'podcast')
    formData.append('episode_id', episodeId.trim())

    try {
      const res = await api.startIngest(formData)
      setJobId(res.job_id)
      toast.info(`Ingestion started for '${res.title}'`)
    } catch (err: any) {
      setIsSubmitting(false)
      toast.error(err.message || 'Failed to start ingestion')
    }
  }

  // Subscribe to the SSE log & progress stream
  useEffect(() => {
    if (!jobId) return

    const unsubscribe = api.subscribeIngestEvents(jobId, (evt: IngestEvent) => {
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
      } else if (evt.type === 'complete') {
        setCompletedSummary(evt.summary)
        setJobStatus((prev) =>
          prev ? { ...prev, status: 'completed', stage: 'complete', progress: 100 } : prev,
        )
        setIsSubmitting(false)
        toast.success('Ingestion complete', { description: 'The episode is ready for annotation.' })
      } else if (evt.type === 'error') {
        setJobStatus((prev) =>
          prev ? { ...prev, status: 'failed', stage: 'failed', error: evt.error } : prev,
        )
        setIsSubmitting(false)
        toast.error(`Ingestion error: ${evt.error}`)
      }
    })

    // Initial snapshot fetch
    api.getIngestStatus(jobId).then((status) => {
      setJobStatus(status)
      if (status.logs?.length) setLogs(status.logs)
      if (status.status === 'completed') setIsSubmitting(false)
    })

    return () => unsubscribe()
  }, [jobId])

  // Auto-scroll the terminal console
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  const resetAndClose = () => {
    if (isSubmitting && !window.confirm('Ingestion is running in the background. Close anyway?')) {
      return
    }
    setJobId(null)
    setJobStatus(null)
    setIsSubmitting(false)
    setSelectedFile(null)
    setEpisodeTitle('')
    setEpisodeId('')
    setLogs([])
    setCompletedSummary(null)
    onClose()
  }

  const handleStartAnnotating = () => {
    onComplete(jobStatus?.episode_id || episodeId || 'new')
    resetAndClose()
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

  const getStageState = (stageKey: string) => {
    if (isComplete) return 'done'
    const stageOrder = ['normalizing', 'segmenting', 'transcribing', 'analyzing', 'importing']
    const currentIndex = stageOrder.indexOf(currentStage)
    const thisIndex = stageOrder.indexOf(stageKey)

    if (currentIndex === -1) return 'pending'
    if (thisIndex < currentIndex) return 'done'
    if (thisIndex === currentIndex) return isFailed ? 'failed' : 'active'
    return 'pending'
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && resetAndClose()}>
      <DialogContent className="max-h-[88vh] grid-rows-[auto_1fr_auto] overflow-hidden sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Ingest podcast episode</DialogTitle>
          <DialogDescription>
            Automated loudnorm, Silero VAD segmentation and Cloud ASR.
          </DialogDescription>
        </DialogHeader>

        <div className="scrollbar-thin min-h-0 overflow-y-auto">
          {!jobId ? (
            <div className="flex flex-col gap-6">
              {/* Dropzone */}
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
                      <span className="font-semibold">Click to upload</span> or drag and drop podcast
                      audio
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
              </div>

              {/* Pipeline overview */}
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 bg-muted/40 p-3 font-mono text-[11px] text-muted-foreground">
                {STAGES.map((stage, index) => (
                  <React.Fragment key={stage.key}>
                    {index > 0 && <span aria-hidden>→</span>}
                    <span>{stage.label}</span>
                  </React.Fragment>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              {/* Stepper */}
              <ol className="grid gap-2 sm:grid-cols-5">
                {STAGES.map((stage) => {
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
                    {jobStatus?.active_segments ? `${jobStatus.active_segments} segments · ` : ''}
                    {Math.round(jobStatus?.progress || 0)}%
                  </span>
                </div>
                <Progress value={Math.min(100, jobStatus?.progress || 0)} />
              </div>

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
                  className="scrollbar-thin h-56 overflow-y-auto bg-muted/20 p-2 font-mono text-[11px] leading-5"
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

              {isComplete && completedSummary && (
                <Alert className="border-success/40 text-success">
                  <RiCheckLine />
                  <AlertTitle>Episode ready for review</AlertTitle>
                  <AlertDescription>
                    {completedSummary.segments} segments extracted and{' '}
                    {completedSummary.tasks_created} tasks added to the prioritization queue.
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
            </div>
          )}
        </div>

        <DialogFooter className="items-center">
          {!jobId ? (
            <>
              <Button variant="ghost" onClick={resetAndClose}>
                Cancel
              </Button>
              <Button
                disabled={!selectedFile || !episodeTitle.trim() || isSubmitting}
                onClick={handleStartIngestion}
              >
                {isSubmitting && <Spinner data-icon="inline-start" />}
                {isSubmitting ? 'Starting…' : 'Start ingestion'}
              </Button>
            </>
          ) : isComplete ? (
            <>
              <Button variant="ghost" onClick={resetAndClose}>
                Close
              </Button>
              <Button onClick={handleStartAnnotating}>Start annotating now</Button>
            </>
          ) : isFailed ? (
            <>
              <Button variant="ghost" onClick={resetAndClose}>
                Close
              </Button>
              <Button onClick={() => setJobId(null)}>Try again</Button>
            </>
          ) : (
            <div className="flex w-full items-center gap-2 text-xs text-muted-foreground">
              <Spinner className="size-3.5" />
              <span className="font-mono">Job #{jobId.slice(0, 8)} running in background…</span>
            </div>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
