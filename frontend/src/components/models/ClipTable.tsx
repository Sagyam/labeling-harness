/**
 * A run's clips, worst first, with the filters the error hunt needs (D83).
 *
 * Genre and crosstalk are picked from the breakdown bars above and shown here as removable chips;
 * sort, loops and a minimum error count live on this bar. `j`/`k` move the selection (wired by
 * the parent, which owns it).
 */

import { RiArrowLeftSLine, RiArrowRightSLine, RiCloseLine, RiRepeatLine } from '@remixicon/react'

import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { Switch } from '@/components/ui/switch'
import { humanize } from '@/components/analytics/primitives'
import { cn } from '@/lib/utils'
import type { ClipSort, ModelClip, ModelClipPage, OverlapBucket } from '@/types'

const SORTS: [ClipSort, string][] = [
  ['errors', 'Most errors'],
  ['wer', 'Highest WER'],
  ['deletions', 'Most deletions'],
  ['insertions', 'Most insertions'],
  ['substitutions', 'Most substitutions'],
  ['overlap', 'Most crosstalk'],
  ['duration', 'Longest'],
]

const SELECT =
  'h-8 rounded-md border border-input bg-background/50 px-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring'

export interface ClipFilters {
  sort: ClipSort
  order: 'asc' | 'desc'
  genre: string | null
  overlap: OverlapBucket | null
  loopsOnly: boolean
  minErrors: number
}

function FilterChip({ label, onClear }: { label: string; onClear: () => void }) {
  return (
    <button
      type="button"
      onClick={onClear}
      className="flex items-center gap-1 rounded-full bg-indigo-500/15 px-2 py-0.5 text-[11px] text-indigo-700 hover:bg-indigo-500/25 dark:text-indigo-300"
      title="Remove this filter"
    >
      {label}
      <RiCloseLine className="size-3" />
    </button>
  )
}

function Row({
  clip,
  rank,
  selected,
  onSelect,
}: {
  clip: ModelClip
  rank: number
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      data-selected={selected}
      className={cn(
        'grid w-full grid-cols-[2.25rem_3.25rem_1fr] items-start gap-2 border-b px-3 py-2 text-left transition-colors',
        selected ? 'bg-muted' : 'hover:bg-muted/50',
      )}
    >
      <span className="pt-0.5 font-mono text-[10px] text-muted-foreground tabular-nums">#{rank}</span>
      <span className="flex flex-col items-start">
        <span className="font-mono text-sm font-bold text-rose-600 tabular-nums dark:text-rose-400">
          {clip.errors}
        </span>
        <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
          {clip.wer.toFixed(0)}%
        </span>
      </span>
      <span className="min-w-0 space-y-0.5">
        <span className="flex flex-wrap items-center gap-x-2 text-[10px] text-muted-foreground">
          <span className="font-mono">
            S{clip.substitutions} D{clip.deletions} I{clip.insertions}
          </span>
          <span>{clip.duration_seconds.toFixed(1)} s</span>
          <span>{humanize(clip.genre)}</span>
          {clip.overlap_bucket !== 'none' && clip.overlap_bucket !== 'unmeasured' && (
            <span className="text-amber-600 dark:text-amber-400">
              crosstalk {((clip.overlap_share ?? 0) * 100).toFixed(0)}%
            </span>
          )}
          {clip.is_loop && (
            <span className="flex items-center gap-0.5 text-amber-600 dark:text-amber-400">
              <RiRepeatLine className="size-3" /> loop
            </span>
          )}
        </span>
        <span className="block truncate text-xs">{clip.hyp_text || '∅ (empty output)'}</span>
        <span className="block truncate font-mono text-[10px] text-muted-foreground">
          {clip.external_id}
        </span>
      </span>
    </button>
  )
}

export function ClipTable({
  page,
  loading,
  filters,
  onChange,
  selectedId,
  onSelect,
  onPage,
}: {
  page: ModelClipPage | null
  loading: boolean
  filters: ClipFilters
  onChange: (next: Partial<ClipFilters>) => void
  selectedId: number | null
  onSelect: (clip: ModelClip) => void
  onPage: (offset: number) => void
}) {
  const from = page && page.total ? page.offset + 1 : 0
  const to = page ? Math.min(page.total, page.offset + page.rows.length) : 0
  return (
    <div className="flex h-full min-h-0 flex-col rounded-lg border bg-card">
      <div className="space-y-2 border-b p-3">
        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label="Sort clips"
            value={filters.sort}
            onChange={(e) => onChange({ sort: e.target.value as ClipSort })}
            className={SELECT}
          >
            {SORTS.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs"
            onClick={() => onChange({ order: filters.order === 'desc' ? 'asc' : 'desc' })}
            title="Reverse the order"
          >
            {filters.order === 'desc' ? 'Worst first' : 'Best first'}
          </Button>
          <label className="flex items-center gap-1.5 text-xs">
            <Switch
              checked={filters.loopsOnly}
              onCheckedChange={(checked) => onChange({ loopsOnly: checked })}
            />
            Loops only
          </label>
          <label className="flex items-center gap-1.5 text-xs">
            Min errors
            <input
              type="number"
              min={0}
              value={filters.minErrors}
              onChange={(e) => onChange({ minErrors: Math.max(0, Number(e.target.value) || 0) })}
              className={cn(SELECT, 'w-14')}
            />
          </label>
        </div>
        {(filters.genre || filters.overlap) && (
          <div className="flex flex-wrap gap-1.5">
            {filters.genre && (
              <FilterChip label={`genre: ${humanize(filters.genre)}`} onClear={() => onChange({ genre: null })} />
            )}
            {filters.overlap && (
              <FilterChip label={`crosstalk: ${filters.overlap}`} onClear={() => onChange({ overlap: null })} />
            )}
          </div>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {loading && !page ? (
          <div className="flex justify-center p-6">
            <Spinner className="size-5 text-primary" />
          </div>
        ) : page && page.rows.length > 0 ? (
          page.rows.map((clip, index) => (
            <Row
              key={clip.segment_id}
              clip={clip}
              rank={page.offset + index + 1}
              selected={clip.segment_id === selectedId}
              onSelect={() => onSelect(clip)}
            />
          ))
        ) : (
          <p className="p-6 text-center text-sm text-muted-foreground">No clips match these filters.</p>
        )}
      </div>

      <div className="flex items-center justify-between border-t px-3 py-1.5 text-[11px] text-muted-foreground">
        <span className="font-mono tabular-nums">
          {from}–{to} of {page?.total ?? 0}
          {loading && page ? ' · loading…' : ''}
        </span>
        <span className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            disabled={!page || page.offset === 0}
            onClick={() => page && onPage(Math.max(0, page.offset - page.limit))}
            aria-label="Previous page"
          >
            <RiArrowLeftSLine className="size-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            disabled={!page || page.offset + page.rows.length >= page.total}
            onClick={() => page && onPage(page.offset + page.limit)}
            aria-label="Next page"
          >
            <RiArrowRightSLine className="size-4" />
          </Button>
        </span>
      </div>
    </div>
  )
}
