import {
  RiAlarmWarningLine,
  RiCheckLine,
  RiErrorWarningLine,
  RiRobotLine,
} from '@remixicon/react'
import type { RefObject } from 'react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'
import { DiscardPanel } from '@/components/ingest/DiscardPanel'
import { DOWNLOAD_STAGE, LOG_LEVEL_CLASS, STAGES } from '@/components/ingest/constants'
import type { DiscardedSegment, IngestJobStatus, IngestLogEntry } from '@/types'

interface JobMonitorProps {
  jobStatus: IngestJobStatus
  jobSource: 'file' | 'youtube'
  logs: IngestLogEntry[]
  discarded: DiscardedSegment[]
  completedSummary: Record<string, any> | null
  isHalting: boolean
  detachMonitor: () => void
  handleCopyLogs: () => void
  handleStartAnnotating: () => void
  logContainerRef: RefObject<HTMLDivElement | null>
}

/** The live pipeline view for one running (or just-finished) job. */
export function JobMonitor({
  jobStatus,
  jobSource,
  logs,
  discarded,
  completedSummary,
  isHalting,
  detachMonitor,
  handleCopyLogs,
  handleStartAnnotating,
  logContainerRef,
}: JobMonitorProps) {
  const currentStage = jobStatus.stage || 'upload'
  const isComplete = jobStatus.status === 'completed' || currentStage === 'complete'
  const isFailed = jobStatus.status === 'failed' || currentStage === 'failed'
  const isAborted = jobStatus.status === 'aborted' || currentStage === 'aborted'
  const isBacklog = jobStatus.status === 'backlog' || currentStage === 'backlog'

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
  const detected = completedSummary?.segments_detected ?? jobStatus.total_segments ?? null

  return       <div className="flex flex-col gap-4 rounded-xl border bg-card p-4 sm:p-5 shadow-xs">
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
            <Button variant="outline" size="sm" onClick={detachMonitor}>
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

}
