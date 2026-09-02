import {
  RiBarChartBoxLine,
  RiEditLine,
  RiFolderDownloadLine,
  RiFolderMusicLine,
  RiInboxLine,
  RiPlayFill,
  RiQuestionLine,
  RiUploadCloud2Line,
} from '@remixicon/react'

import { ModeToggle } from '@/components/ModeToggle'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import type { HealthResponse, StatsResponse } from '@/types'

export type HeaderMode = 'triage' | 'editor' | 'episodes' | 'analytics' | 'export'

interface HeaderProps {
  stats: StatsResponse | null
  activeQueue: string
  activeMode: HeaderMode
  onChangeMode: (mode: HeaderMode) => void
  onResume: () => void
  onOpenHelp: () => void
  onOpenIngest: () => void
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

export function Header({
  stats,
  activeQueue,
  activeMode,
  onChangeMode,
  onResume,
  onOpenHelp,
  onOpenIngest,
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

  const totalBacklog = throughput?.backlog ?? queueStats[activeQueue] ?? 0
  const pendingCount = stats?.tasks?.pending ?? totalBacklog
  const episodeCount = stats?.episodes ?? 0

  const navItems: Array<{
    id: HeaderMode
    label: string
    icon: typeof RiInboxLine
    count?: number
  }> = [
    { id: 'triage', label: 'Triage', icon: RiInboxLine, count: pendingCount },
    { id: 'editor', label: 'Editor', icon: RiEditLine },
    { id: 'episodes', label: 'Episodes', icon: RiFolderMusicLine, count: episodeCount },
    { id: 'analytics', label: 'Analytics', icon: RiBarChartBoxLine },
    { id: 'export', label: 'Export', icon: RiFolderDownloadLine },
  ]

  return (
    <header className="sticky top-0 z-30 shrink-0 border-b bg-card/75 backdrop-blur-md">
      <div className="flex h-14 items-center justify-between gap-4 px-4 sm:px-6">
        {/* Left: Brand Identity */}
        <div className="flex items-center gap-3 shrink-0">
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="relative flex size-2.5 items-center justify-center">
                {isHealthy && (
                  <span className="absolute inline-flex size-full animate-ping rounded-full bg-success/60 opacity-75" />
                )}
                <span
                  className={cn(
                    'relative inline-flex size-2 rounded-full',
                    isHealthy ? 'bg-success shadow-[0_0_10px_var(--success)]' : 'bg-destructive',
                  )}
                />
              </span>
            </TooltipTrigger>
            <TooltipContent>
              {isHealthy ? 'Backend connected & healthy' : 'Backend unreachable'}
            </TooltipContent>
          </Tooltip>

          <div className="flex items-baseline gap-2">
            <span className="font-heading text-sm font-bold tracking-wider uppercase">
              Nepanglish
            </span>
            <Badge variant="outline" className="h-4 px-1.5 text-[9px] font-mono uppercase">
              Harness
            </Badge>
          </div>
        </div>

        {/* Center: Spacious Primary Navigation */}
        <nav className="flex items-center gap-1 rounded-lg border bg-muted/40 p-1">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = activeMode === item.id

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onChangeMode(item.id)}
                className={cn(
                  'flex items-center gap-1.5 rounded-md px-3 py-1 text-xs font-medium transition-all',
                  isActive
                    ? 'bg-background text-foreground shadow-xs font-semibold'
                    : 'text-muted-foreground hover:bg-background/50 hover:text-foreground',
                )}
              >
                <Icon className="size-3.5" />
                <span>{item.label}</span>
                {item.count !== undefined && item.count > 0 && (
                  <span
                    className={cn(
                      'ml-1 rounded-full px-1.5 py-0.2 font-mono text-[10px] tabular-nums',
                      isActive
                        ? 'bg-muted text-foreground font-semibold'
                        : 'bg-muted/70 text-muted-foreground',
                    )}
                  >
                    {item.count}
                  </span>
                )}
              </button>
            )
          })}
        </nav>

        {/* Right: Live Metrics Pill, Actions & Tools */}
        <div className="flex items-center gap-2.5 shrink-0">
          {/* Live Progress Pill */}
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="hidden xl:flex items-center gap-2.5 rounded-full border bg-muted/30 px-3 py-1 font-mono text-xs text-muted-foreground transition-colors hover:bg-muted/60 cursor-default">
                <span className="flex items-center gap-1.5">
                  <span className="size-1.5 rounded-full bg-emerald-500" />
                  <span className="text-foreground font-semibold">{acceptRate}</span> accept
                </span>
                <span className="text-border/80">|</span>
                <span>
                  <strong className="text-foreground">{totalBacklog}</strong> backlog
                </span>
                {throughput?.projected_seconds_to_finish && (
                  <>
                    <span className="text-border/80">|</span>
                    <span className="tabular-nums">
                      ~{formatProjectedTime(throughput.projected_seconds_to_finish)} left
                    </span>
                  </>
                )}
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="space-y-1.5 text-xs p-2.5">
              <div className="font-semibold text-foreground">Pipeline Status Summary</div>
              <div className="text-muted-foreground">
                Progress: <span className="font-mono text-foreground font-medium">{stats?.tasks?.done ?? 0} / {stats?.tasks?.total ?? 0}</span> tasks done
              </div>
              <div className="text-muted-foreground">
                Session: <span className="font-mono text-foreground font-medium">{sessionStats?.completed ?? 0}</span> done &middot;{' '}
                <span className="font-mono text-foreground font-medium">{medianFormatted}</span> / seg
              </div>
              <div className="text-muted-foreground">
                Audio: <span className="font-mono text-foreground font-medium">{(stats?.audio_hours ?? 0).toFixed(1)} hrs</span> imported
              </div>
            </TooltipContent>
          </Tooltip>

          {/* Ingest Action */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="sm" onClick={onOpenIngest} className="h-8 gap-1.5 text-xs">
                <RiUploadCloud2Line className="size-3.5" />
                Ingest
              </Button>
            </TooltipTrigger>
            <TooltipContent>Ingest audio clip or YouTube link through ASR pipeline</TooltipContent>
          </Tooltip>

          {/* Resume Action */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" onClick={onResume} className="h-8 gap-1.5 text-xs font-semibold">
                <RiPlayFill className="size-3.5" />
                Resume
              </Button>
            </TooltipTrigger>
            <TooltipContent>Jump straight to the highest-priority pending task</TooltipContent>
          </Tooltip>

          <Separator orientation="vertical" className="h-5" />

          {/* Shortcuts Reference */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={onOpenHelp}
                aria-label="Keyboard Shortcuts"
                className="size-8"
              >
                <RiQuestionLine className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Keyboard shortcuts (?)</TooltipContent>
          </Tooltip>

          <ModeToggle />
        </div>
      </div>
    </header>
  )
}
