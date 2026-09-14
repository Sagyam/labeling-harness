import { RiArrowDownSLine, RiArrowRightSLine, RiDeleteBin6Line } from '@remixicon/react'
import { useState } from 'react'

import { formatClock } from '@/components/ingest/format'
import type { DiscardedSegment } from '@/types'

/**
 * What was thrown away, and who threw it.
 */
export function DiscardPanel({ discarded }: { discarded: DiscardedSegment[] }) {
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
