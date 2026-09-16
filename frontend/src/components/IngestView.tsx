import {
  RiAddLine,
  RiAlarmWarningLine,
  RiCpuLine,
  RiListCheck2,
  RiRefreshLine,
  RiRobotLine,
} from '@remixicon/react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { DiscardedSegment, IngestEvent, IngestJobStatus, IngestLogEntry, IngestQueueResponse } from '@/types'
import { EpisodeForm } from '@/components/ingest/EpisodeForm'
import { JobMonitor } from '@/components/ingest/JobMonitor'
import { QueueList } from '@/components/ingest/QueueList'
import { ACTIVE_JOB_KEY, SCRAM_ARM_TIMEOUT_MS, type SourceTab } from '@/components/ingest/constants'

interface IngestViewProps {
  onComplete: (episodeId: string) => void
}

export function IngestView({ onComplete }: IngestViewProps) {
  // Top navigation view
  const [mainView, setMainView] = useState<'queue' | 'new'>('queue')

  // Ingestion queue & background management
  const [queueData, setQueueData] = useState<IngestQueueResponse | null>(null)
  const [isRefreshingQueue, setIsRefreshingQueue] = useState<boolean>(false)

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
  const [scramArmed, setScramArmed] = useState<boolean>(false)
  const [isScramming, setIsScramming] = useState<boolean>(false)
  const [logs, setLogs] = useState<IngestLogEntry[]>([])
  const [discarded, setDiscarded] = useState<DiscardedSegment[]>([])
  const [completedSummary, setCompletedSummary] = useState<Record<string, any> | null>(null)

  const logContainerRef = useRef<HTMLDivElement | null>(null)

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

  /** Stop watching a job. The next poll attaches to whatever the server is running. */
  const detachMonitor = () => {
    rememberJob(null)
    setJobStatus(null)
    setLogs([])
    setDiscarded([])
    setCompletedSummary(null)
  }

  /**
   * A job was accepted. A job that starts at once takes over the monitor; one that waits behind
   * another leaves the monitor on the running job and the form open for the next video.
   */
  const afterQueued = (
    res: { job_id: string; title: string; queue_position?: number },
    source: SourceTab,
  ) => {
    const ahead = res.queue_position ?? 0
    fetchQueue(true)
    if (ahead === 0) {
      detachMonitor()
      setJobSource(source)
      rememberJob(res.job_id)
      setMainView('queue')
      toast.info(`Ingestion started for '${res.title}'`)
    } else {
      toast.success(`Queued '${res.title}'`, {
        description: `${ahead} ahead of it. Add the next video, or watch the queue.`,
      })
    }
  }

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
        toast.success('Ingestion complete', { description: 'The episode is ready for annotation.' })
        fetchQueue(true)
        unsubscribe?.()
        unsubscribe = null
      } else if (evt.type === 'scram') {
        setJobStatus((prev) => (prev ? { ...prev, scrammed: true, scram_reason: evt.reason } : prev))
      } else if (evt.type === 'aborted') {
        setCompletedSummary(evt.summary)
        setJobStatus((prev) => (prev ? { ...prev, status: 'aborted' } : prev))
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

  const handleStartAnnotating = () => {
    const target = jobStatus?.episode_id || 'new'
    detachMonitor()
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

  const isComplete = jobStatus?.status === 'completed' || jobStatus?.stage === 'complete'
  const isFailed = jobStatus?.status === 'failed' || jobStatus?.stage === 'failed'
  const isAborted = jobStatus?.status === 'aborted' || jobStatus?.stage === 'aborted'
  const isBacklog = jobStatus?.status === 'backlog' || jobStatus?.stage === 'backlog'
  const isHalting = Boolean(jobStatus?.scrammed) && !isAborted
  const canScram = Boolean(jobId) && !isComplete && !isFailed && !isAborted && !isBacklog

  // Whether a new job would wait behind another, which is what the submit button says.
  const queueBusy = Boolean(queueData?.running) || (queueData?.counts.upcoming ?? 0) > 0

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
                  {queueData.counts.running ? `${queueData.counts.running} active` : 'idle'} · {queueData.counts.upcoming} queued
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
              New Ingestion
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
                <span className="font-mono text-xs text-muted-foreground tabular-nums">
                  {queueData.counts.running}/{queueData.max_concurrent_jobs}
                </span>
              </div>
              <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
                {queueData.running_jobs.length
                  ? queueData.running_jobs.map((job) => job.title).join(' · ')
                  : 'Ready for next episode'}
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
              <div className="mt-0.5 text-[11px] text-muted-foreground">
                In order, {queueData.max_concurrent_jobs} at a time
              </div>
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
            {jobId && jobStatus && (
              <JobMonitor
                jobStatus={jobStatus}
                jobSource={jobSource}
                logs={logs}
                discarded={discarded}
                completedSummary={completedSummary}
                isHalting={isHalting}
                detachMonitor={detachMonitor}
                handleCopyLogs={handleCopyLogs}
                handleStartAnnotating={handleStartAnnotating}
                logContainerRef={logContainerRef}
              />
            )}

            {queueData && (
              <QueueList
                queueData={queueData}
                onComplete={onComplete}
                onGoToForm={() => setMainView('new')}
                onRetryJob={handleRetryJob}
                onRetryAll={handleRetryAll}
                onCancelJob={handleCancelJob}
                onClearPast={handleClearPast}
              />
            )}
          </div>
        )}

        {/* ----------------- TAB: NEW INGESTION ----------------- */}
        {mainView === 'new' && <EpisodeForm queueBusy={queueBusy} onQueued={afterQueued} />}
      </div>
    </div>
  )
}
