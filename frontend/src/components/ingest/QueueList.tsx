import {
  RiCloseLine,
  RiDeleteBin6Line,
  RiExternalLinkLine,
  RiHistoryLine,
  RiListCheck2,
  RiRefreshLine,
  RiRobotLine,
} from '@remixicon/react'
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { formatTimestamp } from '@/components/ingest/format'
import type { IngestQueueResponse } from '@/types'

interface QueueListProps {
  queueData: IngestQueueResponse
  onComplete: (episodeId: string) => void
  /** Where the empty-queue CTA sends the user. */
  onGoToForm: () => void
  onRetryJob: (id: string) => void
  onRetryAll: (statusFilter?: string) => void
  onCancelJob: (id: string) => void
  onClearPast: () => void
}

/** Backlog, upcoming queue and finished history -- the queue without the monitor. */
export function QueueList({
  queueData,
  onComplete,
  onGoToForm,
  onRetryJob,
  onRetryAll,
  onCancelJob,
  onClearPast,
}: QueueListProps) {
  const [historyFilter, setHistoryFilter] = useState<'all' | 'completed' | 'failed'>('all')

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
    <div className="flex flex-col gap-6">
      {queueData.backlog.length > 0 && (
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
            onClick={() => onRetryAll('backlog')}
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
                  onClick={() => onRetryJob(job.job_id)}
                >
                  <RiRefreshLine className="size-3" />
                  Retry
                </Button>
                <Button
                  size="xs"
                  variant="ghost"
                  className="text-muted-foreground hover:text-destructive"
                  onClick={() => onCancelJob(job.job_id)}
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

    {/* 2b. RUNNING JOBS AND SERVICE GATES (D88) */}
    {(queueData.running_jobs.length > 0 || queueData.limits.some((g) => g.throttled_total > 0)) && (
      <div className="flex flex-col gap-3 rounded-xl border bg-card p-4 sm:p-5 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-3">
          <h2 className="font-heading text-sm font-bold tracking-wide uppercase">
            Running now ({queueData.running_jobs.length}/{queueData.max_concurrent_jobs})
          </h2>
          <div className="flex flex-wrap gap-2 font-mono text-[11px]">
            {queueData.limits.map((gate) => (
              <span
                key={gate.name}
                title={`${gate.throttled_total} rate-limit refusals so far; limit ${gate.allowed} of ${gate.max_in_flight}`}
                className={cn(
                  'rounded px-1.5 py-0.5',
                  gate.cooling_down_seconds > 0
                    ? 'bg-amber-500/20 text-amber-500'
                    : 'bg-muted text-muted-foreground',
                )}
              >
                {gate.name} {gate.in_flight}/{gate.allowed}
                {gate.cooling_down_seconds > 0 && ` · cooling ${Math.ceil(gate.cooling_down_seconds)}s`}
              </span>
            ))}
          </div>
        </div>
        {queueData.running_jobs.length > 0 && (
          <div className="divide-y rounded-lg border">
            {queueData.running_jobs.map((job) => (
              <div
                key={job.job_id}
                className="flex flex-wrap items-center justify-between gap-3 p-3 text-sm"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium text-foreground">{job.title}</div>
                  <div className="font-mono text-xs text-muted-foreground">
                    {job.stage}
                    {job.total_segments > 0 && ` · ${job.active_segments}/${job.total_segments} clips`}
                  </div>
                </div>
                <span className="font-mono text-xs tabular-nums text-muted-foreground">
                  {Math.round(job.progress)}%
                </span>
                <Button
                  size="xs"
                  variant="ghost"
                  className="text-muted-foreground hover:text-destructive"
                  onClick={() => onCancelJob(job.job_id)}
                  aria-label="Stop this job"
                >
                  <RiCloseLine className="size-3.5" />
                </Button>
              </div>
            ))}
          </div>
        )}
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
          Processed in order · {queueData.max_concurrent_jobs} at a time
        </div>
      </div>

      {queueData && queueData.upcoming.length > 0 ? (
        <div className="divide-y rounded-lg border">
          {queueData.upcoming.map((job, index) => (
            <div
              key={job.job_id}
              className="flex flex-wrap items-center justify-between gap-3 p-3 text-sm hover:bg-muted/20"
            >
              <div className="flex items-center gap-3 min-w-0 flex-1">
                <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/10 font-mono text-xs font-bold text-primary">
                  #{index + 1}
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
                onClick={() => onCancelJob(job.job_id)}
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
              onClick={onGoToForm}
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
            onClick={() => onRetryAll('failed')}
            title="Retry all failed jobs"
          >
            Retry all failed
          </Button>
          <Button variant="ghost" size="xs" onClick={onClearPast}>
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
                    onClick={() => onRetryJob(job.job_id)}
                  >
                    <RiRefreshLine className="size-3" />
                    Retry
                  </Button>
                )}
                <Button
                  size="xs"
                  variant="ghost"
                  className="text-muted-foreground hover:text-destructive"
                  onClick={() => onCancelJob(job.job_id)}
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
  )
}
