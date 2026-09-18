/**
 * The paperwork: which episode records are unfilled, field by field, and which to go and fix.
 *
 * An unfilled variable and a thin stratum look identical in every chart until this table
 * separates them. Nothing here is about the world; all of it is about the records.
 */

import { useState } from 'react'
import { RiFileList3Line } from '@remixicon/react'

import { BarRow, Panel, PanelHeading } from '@/components/analytics/primitives'
import type { RecordCheck } from '@/types'

const FIELD_LABEL: Record<string, string> = {
  show_id: 'show id',
  genre: 'genre',
  topic: 'topic',
  topic_in_taxonomy: 'topic in the taxonomy',
  speakers: 'speakers declared',
  published_at: 'publication date',
  diarized: 'diarized',
  voices_linked: 'voices linked',
}

export function RecordsPanel({ records }: { records: RecordCheck[] }) {
  const [open, setOpen] = useState<string | null>(null)
  return (
    <Panel>
      <PanelHeading
        icon={<RiFileList3Line className="size-4 text-primary" />}
        title="Records"
        note="unfilled fields are paperwork, not facts about the corpus; click a row for the episodes to fix"
      />
      <div className="grid gap-x-6 gap-y-1 md:grid-cols-2">
        {records.map((r) => (
          <div key={r.field}>
            <BarRow
              label={FIELD_LABEL[r.field] ?? r.field.replace(/_/g, ' ')}
              value={r.filled}
              peak={r.total}
              caption={`${r.filled}/${r.total}`}
              fill={r.filled === r.total ? 'bg-emerald-500' : 'bg-indigo-500'}
              onClick={r.missing_episodes.length ? () => setOpen(open === r.field ? null : r.field) : undefined}
              active={open === r.field}
            />
            {open === r.field ? (
              <p className="max-h-32 overflow-y-auto px-1 pb-1 font-mono text-[10px] leading-relaxed text-muted-foreground">
                {r.missing_episodes.join(', ')}
              </p>
            ) : null}
          </div>
        ))}
      </div>
    </Panel>
  )
}
