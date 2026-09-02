import {
  RiPlayFill,
  RiQuestionLine,
  RiStackLine,
  RiUploadCloud2Line,
} from '@remixicon/react'

import { ModeToggle } from '@/components/ModeToggle'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import type { HealthResponse, StatsResponse } from '@/types'

const QUEUES = ['review', 'audit', 'error'] as const

interface HeaderProps {
  stats: StatsResponse | null
  activeQueue: string
  onChangeQueue: (queue: string) => void
  activeMode: 'triage' | 'editor'
  onChangeMode: (mode: 'triage' | 'editor') => void
  onResume: () => void
  onOpenHelp: () => void
  onOpenIngest: () => void
  onOpenEpisodes: () => void
  health: HealthResponse | null
}

function formatProjectedTime(seconds?: number | null) {
  if (seconds === null || seconds === undefined) return '--'
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  const hours = Math.floor(seconds / 3600)
  const mins = Math.round((seconds % 3600) / 60)
  return `${hours}h ${mins}m`
}

function Stat({
  label,
  value,
  hint,
  className,
}: {
  label: string
  value: string
  hint: string
  className?: string
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex min-w-0 flex-col gap-0.5 px-4 py-2 first:pl-0">
          <span className="font-heading text-[10px] font-semibold tracking-widest text-muted-foreground uppercase">
            {label}
          </span>
          <span className={cn('truncate font-mono text-sm tabular-nums', className)}>{value}</span>
        </div>
      </TooltipTrigger>
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  )
}

export function Header({
  stats,
  activeQueue,
  onChangeQueue,
  activeMode,
  onChangeMode,
  onResume,
  onOpenHelp,
  onOpenIngest,
  onOpenEpisodes,
  health,
}: HeaderProps) {
  const isHealthy = health?.status === 'ok'
  const queueStats = (stats?.queues as Record<string, number>) || {}
  const throughput = stats?.throughput
  const sessionStats = stats?.session

  const acceptRate =
    stats?.accept_rate !== null && stats?.accept_rate !== undefined
      ? `${(stats.accept_rate * 100).toFixed(1)}%`
      : '--'

  const medianSeconds =
    sessionStats?.median_seconds_per_segment ?? throughput?.median_seconds_per_segment
  const medianFormatted =
    medianSeconds !== null && medianSeconds !== undefined ? `${medianSeconds.toFixed(1)}s` : '--'

  return (
    <header className="shrink-0 border-b bg-card/40">
      <div className="flex h-14 items-center gap-5 px-4">
        <div className="flex items-center gap-2.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                className={cn(
                  'size-2 rounded-full',
                  isHealthy ? 'bg-success shadow-[0_0_8px_var(--success)]' : 'bg-destructive',
                )}
                aria-label={isHealthy ? 'Backend connected' : 'Backend unreachable'}
              />
            </TooltipTrigger>
            <TooltipContent>
              {isHealthy ? 'Connected to backend' : 'Backend unreachable'}
            </TooltipContent>
          </Tooltip>
          <span className="font-heading text-sm font-semibold tracking-widest uppercase">
            Nepanglish
          </span>
          <Badge variant="secondary">Harness</Badge>
        </div>

        <Separator orientation="vertical" className="h-6" />

        <Tabs value={activeQueue} onValueChange={onChangeQueue}>
          <TabsList variant="line">
            {QUEUES.map((queue) => (
              <TabsTrigger key={queue} value={queue} className="gap-1.5">
                {queue}
                <span className="font-mono text-[10px] tabular-nums opacity-70">
                  {queueStats[queue] ?? 0}
                </span>
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>

        <div className="ml-auto flex items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="sm" onClick={onOpenEpisodes}>
                <RiStackLine data-icon="inline-start" />
                Episodes
              </Button>
            </TooltipTrigger>
            <TooltipContent>Manage uploaded episodes and segments</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="sm" onClick={onOpenIngest}>
                <RiUploadCloud2Line data-icon="inline-start" />
                Ingest
              </Button>
            </TooltipTrigger>
            <TooltipContent>Ingest a new episode through the Cloud ASR pipeline</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" onClick={onResume}>
                <RiPlayFill data-icon="inline-start" />
                Resume
              </Button>
            </TooltipTrigger>
            <TooltipContent>Jump to the highest-priority pending segment</TooltipContent>
          </Tooltip>

          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            spacing={0}
            value={activeMode}
            onValueChange={(value) => value && onChangeMode(value as 'triage' | 'editor')}
          >
            <ToggleGroupItem value="triage">Triage</ToggleGroupItem>
            <ToggleGroupItem value="editor">Editor</ToggleGroupItem>
          </ToggleGroup>

          <Separator orientation="vertical" className="h-6" />

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={onOpenHelp} aria-label="Shortcuts">
                <RiQuestionLine />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Keyboard shortcuts (?)</TooltipContent>
          </Tooltip>

          <ModeToggle />
        </div>
      </div>

      <div className="flex flex-wrap items-center divide-x border-t px-4">
        <Stat
          label="Progress"
          value={`${stats?.tasks?.done ?? 0} / ${stats?.tasks?.total ?? 0}`}
          hint="Completed tasks out of the total in the corpus"
        />
        <Stat
          label="Accept rate"
          value={acceptRate}
          hint="Fraction of labels accepted unchanged"
          className="text-success"
        />
        <Stat
          label="Session done"
          value={String(sessionStats?.completed ?? 0)}
          hint="Segments completed during this session"
          className="text-info"
        />
        <Stat
          label="Median / seg"
          value={medianFormatted}
          hint="Median seconds spent per segment"
          className="text-warning"
        />
        <Stat
          label="Est. remaining"
          value={formatProjectedTime(throughput?.projected_seconds_to_finish)}
          hint="Projected time to finish the remaining backlog"
        />
        <Stat
          label="Backlog"
          value={String(throughput?.backlog ?? 0)}
          hint="Segments still awaiting a decision"
        />
        <Stat
          label="Audio"
          value={`${(stats?.audio_hours ?? 0).toFixed(1)}h`}
          hint="Total audio hours imported"
        />
      </div>
    </header>
  )
}
