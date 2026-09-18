/**
 * Any category against any other (D91).
 *
 * The gender-by-age grid the page used to draw was one cell of this: pick two categories and
 * every cell is the hours, or the usable voices, that sit in both buckets. Empty cells are drawn,
 * not omitted -- an empty stratum is the point. The page's filters apply, except on the two
 * categories being crossed.
 */

import { useMemo } from 'react'
import { RiGridLine } from '@remixicon/react'

import { HEAT_STEPS, Panel, PanelHeading, heatStep } from '@/components/analytics/primitives'
import { bucketLabel, hoursOrMinutes } from '@/components/analytics/labels'
import { crossTab, type Filters } from '@/components/analytics/model'
import type { CategoryReport, CorpusInventory } from '@/types'

const SELECT =
  'h-7 rounded-md border border-input bg-background/50 px-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring'

export function CrossTabPanel({
  inventory,
  filters,
  rows,
  cols,
  unit,
  onChange,
  onPick,
}: {
  inventory: CorpusInventory
  filters: Filters
  rows: string
  cols: string
  unit: 'hours' | 'voices'
  onChange: (next: { rows?: string; cols?: string; unit?: 'hours' | 'voices' }) => void
  onPick: (rowsKey: string, rowBucket: string, colsKey: string, colBucket: string) => void
}) {
  const options = inventory.categories.filter((c) => c.key !== 'voice')
  const table = useMemo(() => crossTab(inventory, rows, cols, filters), [inventory, rows, cols, filters])
  const byKey = Object.fromEntries(inventory.categories.map((c) => [c.key, c])) as Record<string, CategoryReport>
  const rowUnknown = new Set(byKey[rows]?.unknown ?? [])
  const colUnknown = new Set(byKey[cols]?.unknown ?? [])
  const peak = unit === 'hours' ? table.peakHours : table.peakVoices
  const value = (cell: { hours: number; usableVoices: number }) =>
    unit === 'hours' ? cell.hours : cell.usableVoices

  return (
    <Panel>
      <PanelHeading
        icon={<RiGridLine className="size-4 text-primary" />}
        title="Cross-tabulate"
        note="a cell is what sits in both buckets; empty cells are drawn on purpose"
      >
        <div className="ml-2 flex flex-wrap items-center gap-1.5 text-xs">
          <select className={SELECT} value={rows} onChange={(e) => onChange({ rows: e.target.value })} aria-label="Rows">
            {options.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
          <span className="text-muted-foreground">by</span>
          <select className={SELECT} value={cols} onChange={(e) => onChange({ cols: e.target.value })} aria-label="Columns">
            {options.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
          <span className="text-muted-foreground">in</span>
          <select
            className={SELECT}
            value={unit}
            onChange={(e) => onChange({ unit: e.target.value as 'hours' | 'voices' })}
            aria-label="Unit"
          >
            <option value="hours">hours</option>
            <option value="voices">usable voices</option>
          </select>
        </div>
      </PanelHeading>

      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-0.5 text-[11px]">
          <thead>
            <tr>
              <th className="sticky left-0 bg-card text-left font-normal text-muted-foreground">
                {byKey[rows]?.label} ↓ · {byKey[cols]?.label} →
              </th>
              {table.cols.map((c) => (
                <th
                  key={c}
                  className={`min-w-16 px-1 pb-1 text-left font-normal ${colUnknown.has(c) ? 'text-muted-foreground/70 italic' : 'text-muted-foreground'}`}
                  title={c}
                >
                  {bucketLabel(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((r, i) => (
              <tr key={r}>
                <th
                  className={`sticky left-0 bg-card pr-2 text-left font-normal ${rowUnknown.has(r) ? 'text-muted-foreground/70 italic' : ''}`}
                  title={r}
                >
                  {bucketLabel(r)}
                </th>
                {table.cols.map((c, j) => {
                  const cell = table.cells[i][j]
                  const v = value(cell)
                  const step = heatStep(v, peak)
                  const dark = step >= 3
                  return (
                    <td key={c} className="p-0">
                      <button
                        type="button"
                        onClick={() => onPick(rows, r, cols, c)}
                        title={`${bucketLabel(r)} × ${bucketLabel(c)}: ${hoursOrMinutes(cell.hours)}, ${cell.clips} clips, ${cell.usableVoices} of ${cell.voices} voices usable, verified ${hoursOrMinutes(cell.verifiedHours)}, gold ${hoursOrMinutes(cell.goldHours)}`}
                        className={`flex h-9 w-full min-w-16 flex-col items-start justify-center rounded px-1.5 text-left ${
                          step < 0 ? 'border border-dashed border-muted-foreground/40' : HEAT_STEPS[step]
                        } ${dark ? 'text-white' : ''} hover:ring-1 hover:ring-indigo-500`}
                      >
                        <span className="font-mono text-[11px] leading-tight tabular-nums">
                          {v === 0 ? '·' : unit === 'hours' ? hoursOrMinutes(v) : v}
                        </span>
                        {v > 0 ? (
                          <span className={`text-[9px] leading-tight ${dark ? 'text-white/80' : 'text-muted-foreground'}`}>
                            {unit === 'hours' ? `${cell.usableVoices} v` : hoursOrMinutes(cell.hours)}
                          </span>
                        ) : null}
                      </button>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
