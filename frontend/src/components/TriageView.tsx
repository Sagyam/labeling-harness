import { useEffect, useRef, useState } from 'react'
import { RiCloseLine, RiInboxLine, RiPauseFill, RiPlayFill } from '@remixicon/react'

import { Chip } from '@/components/Chip'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Kbd, KbdGroup } from '@/components/ui/kbd'
import { Separator } from '@/components/ui/separator'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { resolveUrl } from '@/services/api'
import { cn } from '@/lib/utils'
import type { QueueRow } from '@/types'

const QUEUES = ['review', 'audit', 'error'] as const

interface TriageViewProps {
  rows: QueueRow[]
  activeQueue?: string
  onChangeQueue?: (queue: string) => void
  queueStats?: Record<string, number>
  episodeFilter?: string | null
  onClearEpisodeFilter?: () => void
  focusedIndex: number
  onSetFocusedIndex: (index: number) => void
  selectedIds: Set<number>
  onToggleSelect: (taskId: number) => void
  onSelectAll: (all: boolean) => void
  onAcceptRow: (taskId: number, durationMs: number) => Promise<void>
  onOpenEditor: (taskId: number) => void
  onFlagRow: (
    taskId: number,
    disposition: 'unusable_audio' | 'uncertain',
    durationMs: number,
  ) => Promise<void>
  onBulkAccept: (taskIds: number[]) => Promise<void>
}

const SHORTCUTS: Array<[string, string]> = [
  ['j / k', 'Navigate'],
  ['Space', 'Play'],
  ['Enter', 'Accept'],
  ['e', 'Editor'],
  ['f', 'Unusable'],
  ['u', 'Uncertain'],
  ['x', 'Select'],
  ['⇧ Enter', 'Bulk accept'],
]

function priorityClass(score: number) {
  if (score >= 0.45) return 'bg-destructive/15 text-destructive'
  if (score >= 0.25) return 'bg-warning/15 text-warning'
  return 'bg-muted text-muted-foreground'
}

export function TriageView({
  rows,
  activeQueue,
  onChangeQueue,
  queueStats,
  episodeFilter,
  onClearEpisodeFilter,
  focusedIndex,
  onSetFocusedIndex,
  selectedIds,
  onToggleSelect,
  onSelectAll,
  onAcceptRow,
  onOpenEditor,
  onFlagRow,
  onBulkAccept,
}: TriageViewProps) {
  const [playingTaskId, setPlayingTaskId] = useState<number | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const rowRefs = useRef<(HTMLTableRowElement | null)[]>([])
  const focusedRowOpenedAtRef = useRef<number>(Date.now())

  // Scroll focused row into view smoothly
  useEffect(() => {
    focusedRowOpenedAtRef.current = Date.now()
    rowRefs.current[focusedIndex]?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [focusedIndex])

  const getFocusedDurationMs = () => Math.max(0, Date.now() - focusedRowOpenedAtRef.current)

  // Audio Playback
  const togglePlay = (row: QueueRow) => {
    if (playingTaskId === row.task_id) {
      audioRef.current?.pause()
      setPlayingTaskId(null)
      return
    }
    audioRef.current?.pause()
    const audio = new Audio(resolveUrl(row.audio_url))
    audioRef.current = audio
    setPlayingTaskId(row.task_id)
    audio.play().catch((err) => console.error('Audio play error:', err))
    audio.onended = () => setPlayingTaskId(null)
    audio.onerror = () => setPlayingTaskId(null)
  }

  // Keyboard navigation for Triage mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      if (targetTag === 'input' || targetTag === 'textarea') return

      if (rows.length === 0) return
      const focusedRow = rows[focusedIndex]

      // j / ArrowDown: Move focus down
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault()
        onSetFocusedIndex(Math.min(rows.length - 1, focusedIndex + 1))
        return
      }

      // k / ArrowUp: Move focus up
      if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault()
        onSetFocusedIndex(Math.max(0, focusedIndex - 1))
        return
      }

      // Space: Toggle play/pause focused row
      if (e.code === 'Space') {
        e.preventDefault()
        if (focusedRow) togglePlay(focusedRow)
        return
      }

      // Shift+Enter: Bulk accept selected rows
      if (e.shiftKey && e.key === 'Enter') {
        e.preventDefault()
        if (selectedIds.size > 0) {
          onBulkAccept(Array.from(selectedIds))
        } else if (focusedRow) {
          onBulkAccept([focusedRow.task_id])
        }
        return
      }

      // Enter: Accept focused row unchanged
      if (!e.shiftKey && !e.ctrlKey && e.key === 'Enter') {
        e.preventDefault()
        if (focusedRow) onAcceptRow(focusedRow.task_id, getFocusedDurationMs())
        return
      }

      // e: Open editor
      if (e.key === 'e' || e.key === 'E') {
        e.preventDefault()
        if (focusedRow) onOpenEditor(focusedRow.task_id)
        return
      }

      // f: Flag unusable audio
      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault()
        if (focusedRow) onFlagRow(focusedRow.task_id, 'unusable_audio', getFocusedDurationMs())
        return
      }

      // u: Mark uncertain
      if (e.key === 'u' || e.key === 'U') {
        e.preventDefault()
        if (focusedRow) onFlagRow(focusedRow.task_id, 'uncertain', getFocusedDurationMs())
        return
      }

      // x: Toggle checkbox
      if (e.key === 'x' || e.key === 'X') {
        e.preventDefault()
        if (focusedRow) onToggleSelect(focusedRow.task_id)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [rows, focusedIndex, selectedIds, playingTaskId])

  // Stop audio when unmounting
  useEffect(() => {
    return () => {
      audioRef.current?.pause()
    }
  }, [])

  const allSelected = rows.length > 0 && rows.every((r) => selectedIds.has(r.task_id))

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Toolbar */}
      <div className="flex h-12 shrink-0 items-center justify-between gap-4 border-b bg-card/20 px-4 sm:px-6">
        {/* Left: Queue tabs & episode filter chip */}
        <div className="flex items-center gap-3">
          {onChangeQueue && activeQueue && (
            <Tabs value={activeQueue} onValueChange={onChangeQueue}>
              <TabsList variant="line" className="h-8">
                {QUEUES.map((queue) => (
                  <TabsTrigger key={queue} value={queue} className="gap-1.5 px-3 text-xs capitalize">
                    {queue}
                    <span className="font-mono text-[10px] tabular-nums opacity-70">
                      {queueStats?.[queue] ?? 0}
                    </span>
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          )}

          {episodeFilter && (
            <div className="flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/10 px-2.5 py-0.5 text-xs text-primary">
              <span className="text-muted-foreground">Episode:</span>
              <span className="font-mono font-semibold">{episodeFilter}</span>
              {onClearEpisodeFilter && (
                <button
                  type="button"
                  onClick={onClearEpisodeFilter}
                  className="ml-1 rounded-full p-0.5 hover:bg-primary/20 transition-colors"
                  title="Clear episode filter"
                  aria-label="Clear episode filter"
                >
                  <RiCloseLine className="size-3" />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Right: Selection & count */}
        <div className="flex items-center gap-3">
          <Button
            size="sm"
            disabled={selectedIds.size === 0}
            onClick={() => onBulkAccept(Array.from(selectedIds))}
            className="h-8 gap-1.5 text-xs"
          >
            Accept selected ({selectedIds.size})
            <Kbd className="ml-1 bg-primary-foreground/15 text-primary-foreground">⇧ ↵</Kbd>
          </Button>

          <Separator orientation="vertical" className="h-4" />

          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {rows.length} queued segments
          </span>
        </div>
      </div>

      {/* Queue table */}
      <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto">
        {rows.length === 0 ? (
          <Empty className="py-24">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <RiInboxLine />
              </EmptyMedia>
              <EmptyTitle>Queue is empty</EmptyTitle>
              <EmptyDescription>
                Nothing is pending here. Ingest an episode or switch to another queue.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <Table>
            <TableHeader className="sticky top-0 z-10 bg-background shadow-[0_1px_0_var(--border)]">
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-10 pl-4">
                  <Checkbox
                    checked={allSelected}
                    onCheckedChange={(checked) => onSelectAll(checked === true)}
                    aria-label="Select all rows"
                  />
                </TableHead>
                <TableHead className="w-20">Priority</TableHead>
                <TableHead className="w-24">Audio</TableHead>
                <TableHead className="w-52">Segment</TableHead>
                <TableHead>Seed hypothesis</TableHead>
                <TableHead className="w-64">Flags &amp; reason</TableHead>
                <TableHead className="w-56 pr-4 text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>

            <TableBody>
              {rows.map((row, index) => {
                const isFocused = index === focusedIndex
                const isPlaying = playingTaskId === row.task_id
                const isChecked = selectedIds.has(row.task_id)

                return (
                  <TableRow
                    key={row.task_id}
                    ref={(el) => {
                      rowRefs.current[index] = el
                    }}
                    data-state={isChecked ? 'selected' : undefined}
                    className={cn(
                      'cursor-default',
                      isFocused && 'bg-accent hover:bg-accent',
                      isPlaying && 'ring-1 ring-info/40 ring-inset',
                    )}
                    onClick={() => onSetFocusedIndex(index)}
                    onDoubleClick={() => onOpenEditor(row.task_id)}
                  >
                    <TableCell className="relative pl-4" onClick={(e) => e.stopPropagation()}>
                      {isFocused && (
                        <span className="absolute inset-y-0 left-0 w-0.5 bg-foreground" />
                      )}
                      <Checkbox
                        checked={isChecked}
                        onCheckedChange={() => onToggleSelect(row.task_id)}
                        aria-label={`Select ${row.segment_external_id}`}
                      />
                    </TableCell>

                    <TableCell>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span
                            className={cn(
                              'inline-block px-1.5 py-0.5 font-mono text-xs tabular-nums',
                              priorityClass(row.priority_score),
                            )}
                          >
                            {row.priority_score.toFixed(2)}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent className="font-mono text-xs">
                          <div>score: {row.priority_score.toFixed(3)}</div>
                          <div>
                            disagreement:{' '}
                            {row.reason?.components?.word_disagreement_rate?.toFixed(2) ?? '0'}
                          </div>
                          <div>
                            code-switch:{' '}
                            {row.reason?.components?.code_switch_density?.toFixed(2) ?? '0'}
                          </div>
                        </TooltipContent>
                      </Tooltip>
                    </TableCell>

                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <Button
                        variant={isPlaying ? 'secondary' : 'ghost'}
                        size="xs"
                        onClick={() => togglePlay(row)}
                        aria-label={isPlaying ? 'Pause' : 'Play'}
                      >
                        {isPlaying ? <RiPauseFill /> : <RiPlayFill />}
                        <span className="font-mono tabular-nums">
                          {row.duration_seconds.toFixed(1)}s
                        </span>
                      </Button>
                    </TableCell>

                    <TableCell className="max-w-52 truncate font-mono text-xs text-muted-foreground">
                      <span title={row.segment_external_id}>{row.segment_external_id}</span>
                    </TableCell>

                    <TableCell className="max-w-0 truncate font-devanagari">
                      <span title={row.seed_text || ''}>{row.seed_text || '—'}</span>
                    </TableCell>

                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {row.seed_system_id && <Chip>{row.seed_system_id}</Chip>}
                        {row.flags.map((flag) => (
                          <Chip key={flag} className="bg-warning/15 text-warning" title={flag}>
                            {flag}
                          </Chip>
                        ))}
                      </div>
                    </TableCell>

                    <TableCell className="pr-4" onClick={(e) => e.stopPropagation()}>
                      <div className="flex justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="xs"
                          className="text-success hover:bg-success/10 hover:text-success"
                          onClick={() => onAcceptRow(row.task_id, getFocusedDurationMs())}
                        >
                          Accept
                        </Button>
                        <Button variant="ghost" size="xs" onClick={() => onOpenEditor(row.task_id)}>
                          Edit
                        </Button>
                        <Button
                          variant="ghost"
                          size="xs"
                          className="text-warning hover:bg-warning/10 hover:text-warning"
                          onClick={() =>
                            onFlagRow(row.task_id, 'unusable_audio', getFocusedDurationMs())
                          }
                        >
                          Flag
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </div>

      {/* Shortcut bar */}
      <footer className="flex h-10 shrink-0 flex-wrap items-center justify-between gap-x-4 gap-y-1 border-t bg-card/40 px-4">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {SHORTCUTS.map(([keys, label]) => (
            <KbdGroup key={label}>
              <Kbd>{keys}</Kbd>
              <span className="text-xs text-muted-foreground">{label}</span>
            </KbdGroup>
          ))}
        </div>
        <KbdGroup>
          <span className="text-xs text-muted-foreground">All shortcuts</span>
          <Kbd>?</Kbd>
        </KbdGroup>
      </footer>
    </div>
  )
}
