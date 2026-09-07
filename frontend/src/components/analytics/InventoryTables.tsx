/**
 * The raw inventory: one row per show, one row per episode, and the state of the paperwork.
 *
 * Every chart above is an aggregate, and an aggregate always loses the thing you eventually want
 * to act on — *which* episode, *which* field. These tables are where a recommendation turns into
 * a filename. Sortable rather than pre-sorted, because which column matters depends on what you
 * came here to do.
 */

import { useMemo, useState } from 'react'
import { RiArrowDownSLine, RiArrowUpSLine, RiFileListLine, RiListCheck3 } from '@remixicon/react'

import { Input } from '@/components/ui/input'
import { Chip, POT_TEXT, Panel, PanelHeading, hours, humanize, percent } from './primitives'
import type { InventoryEpisode, InventoryShow, MetadataCompleteness } from '@/types'

const FIELD_LABEL: Record<string, string> = {
  show_id: 'Show id',
  published_at: 'Publish date',
  topic: 'Topic set',
  topic_in_taxonomy: 'Topic in taxonomy',
  speakers: 'Any speaker block',
  gender: 'Speaker gender',
  age_bracket: 'Speaker age',
  role: 'Speaker role',
  segments_imported: 'Segments imported',
}

const FIELD_NOTE: Record<string, string> = {
  topic_in_taxonomy:
    'A topic typed as free text instead of chosen from the closed list. It cannot be stratified on (D57).',
  segments_imported: 'An episode row with no clips behind it — an ingest that did not finish.',
}

function SortHeader<K extends string>({
  column,
  label,
  sort,
  setSort,
  align = 'left',
}: {
  column: K
  label: string
  sort: { key: K; desc: boolean }
  setSort: (value: { key: K; desc: boolean }) => void
  align?: 'left' | 'right'
}) {
  const active = sort.key === column
  return (
    <th
      className={`cursor-pointer whitespace-nowrap px-2 py-1.5 font-heading text-[10px] font-semibold uppercase tracking-wider ${
        align === 'right' ? 'text-right' : 'text-left'
      } ${active ? 'text-foreground' : 'text-muted-foreground'} hover:text-foreground`}
      onClick={() => setSort({ key: column, desc: active ? !sort.desc : true })}
    >
      <span className={`inline-flex items-center gap-0.5 ${align === 'right' ? 'flex-row-reverse' : ''}`}>
        {label}
        {active ? (
          sort.desc ? (
            <RiArrowDownSLine className="size-3" />
          ) : (
            <RiArrowUpSLine className="size-3" />
          )
        ) : null}
      </span>
    </th>
  )
}

/**
 * Sort a row list by any of its own keys. Numbers compare numerically and everything else as a
 * string, which is right for every column these tables have: ids, dates and vocabulary values all
 * sort sensibly as text, and a null sorts to one end rather than throwing.
 */
function useSorted<T extends object>(rows: T[], initial: keyof T & string) {
  const [sort, setSort] = useState<{ key: keyof T & string; desc: boolean }>({
    key: initial,
    desc: true,
  })
  const sorted = useMemo(() => {
    const copy = [...rows]
    copy.sort((a, b) => {
      const left = a[sort.key] as unknown
      const right = b[sort.key] as unknown
      if (typeof left === 'number' && typeof right === 'number') {
        return sort.desc ? right - left : left - right
      }
      return sort.desc
        ? String(right ?? '').localeCompare(String(left ?? ''))
        : String(left ?? '').localeCompare(String(right ?? ''))
    })
    return copy
  }, [rows, sort])
  return { sorted, sort, setSort }
}

/** Per-show rollup: the row a sourcing decision is actually made at. */
export function ShowsTable({ shows }: { shows: InventoryShow[] }) {
  const { sorted, sort, setSort } = useSorted(shows, 'hours')
  return (
    <Panel>
      <PanelHeading
        icon={<RiListCheck3 className="size-4 text-primary" />}
        title="Shows"
        note="a show is the unit you can go and get more of"
      />
      <div className="scrollbar-thin overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="border-b">
            <tr>
              <SortHeader column="show_id" label="Show" sort={sort} setSort={setSort} />
              <SortHeader column="episodes" label="Eps" sort={sort} setSort={setSort} align="right" />
              <SortHeader column="hours" label="Hours" sort={sort} setSort={setSort} align="right" />
              <SortHeader column="segments" label="Clips" sort={sort} setSort={setSort} align="right" />
              <SortHeader column="verified_hours" label="Heard" sort={sort} setSort={setSort} align="right" />
              <SortHeader
                column="speaker_profiles"
                label="Speakers"
                sort={sort}
                setSort={setSort}
                align="right"
              />
              <SortHeader column="mean_cmi" label="English" sort={sort} setSort={setSort} align="right" />
              <th className="px-2 py-1.5 text-left font-heading text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Carries
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((show) => (
              <tr key={show.show_id} className="border-b border-border/50 hover:bg-muted/40">
                <td className="max-w-[14rem] truncate px-2 py-1.5 font-medium" title={show.show_id}>
                  {show.show_id}
                </td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums">{show.episodes}</td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums">{hours(show.hours)}</td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-muted-foreground">
                  {show.segments}
                </td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums">
                  {percent(show.hours ? show.verified_hours / show.hours : 0)}
                </td>
                <td
                  className="px-2 py-1.5 text-right font-mono tabular-nums"
                  title="distinct (role, gender, age) combinations — a floor, not a count"
                >
                  ≥{show.speaker_profiles}
                </td>
                <td
                  className="px-2 py-1.5 text-right font-mono tabular-nums"
                  title={
                    show.min_cmi !== null
                      ? `clips range ${percent(show.min_cmi)}–${percent(show.max_cmi)}`
                      : undefined
                  }
                >
                  {percent(show.mean_cmi, 1)}
                </td>
                <td className="px-2 py-1.5">
                  <div className="flex flex-wrap gap-1">
                    {[...show.genders, ...show.age_brackets].map((value) => (
                      <Chip key={value}>{humanize(value)}</Chip>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

/** Every episode, filterable. Where a gap becomes a specific file to go and look at. */
export function EpisodesTable({ episodes }: { episodes: InventoryEpisode[] }) {
  const [query, setQuery] = useState('')
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return episodes
    return episodes.filter((episode) =>
      [
        episode.external_id,
        episode.title ?? '',
        episode.show_id ?? '',
        episode.topic ?? '',
        episode.pot,
        ...episode.speakers.flatMap((speaker) => [
          speaker.gender ?? '',
          speaker.age_bracket ?? '',
          speaker.role ?? '',
        ]),
      ]
        .join(' ')
        .toLowerCase()
        .includes(needle),
    )
  }, [episodes, query])
  const { sorted, sort, setSort } = useSorted(filtered, 'hours')

  return (
    <Panel>
      <PanelHeading icon={<RiFileListLine className="size-4 text-primary" />} title="Every episode">
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
          {filtered.length}/{episodes.length}
        </span>
      </PanelHeading>
      <Input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Filter by id, title, show, topic, pot or speaker…"
        className="mb-3 h-8 text-xs"
      />
      <div className="scrollbar-thin max-h-[32rem] overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 z-10 border-b bg-card">
            <tr>
              <SortHeader column="external_id" label="Episode" sort={sort} setSort={setSort} />
              <SortHeader column="show_id" label="Show" sort={sort} setSort={setSort} />
              <SortHeader column="published_at" label="Date" sort={sort} setSort={setSort} />
              <SortHeader column="minutes" label="Min" sort={sort} setSort={setSort} align="right" />
              <SortHeader column="pot" label="Pot" sort={sort} setSort={setSort} />
              <th className="px-2 py-1.5 text-left font-heading text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Speakers
              </th>
              <SortHeader column="topic" label="Topic" sort={sort} setSort={setSort} />
              <SortHeader column="mean_cmi" label="English" sort={sort} setSort={setSort} align="right" />
              <SortHeader column="segments" label="Clips" sort={sort} setSort={setSort} align="right" />
              <SortHeader
                column="labeled_fraction"
                label="Labelled"
                sort={sort}
                setSort={setSort}
                align="right"
              />
            </tr>
          </thead>
          <tbody>
            {sorted.map((episode) => (
              <tr key={episode.external_id} className="border-b border-border/50 hover:bg-muted/40">
                <td
                  className="max-w-[13rem] truncate px-2 py-1.5 font-mono text-[11px]"
                  title={episode.title ?? episode.external_id}
                >
                  {episode.external_id}
                </td>
                <td className="max-w-[9rem] truncate px-2 py-1.5 text-muted-foreground">
                  {episode.show_id ?? <span className="text-amber-600 dark:text-amber-400">unset</span>}
                </td>
                <td className="whitespace-nowrap px-2 py-1.5 font-mono text-[10px] text-muted-foreground">
                  {episode.published_at ?? '—'}
                </td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums">{episode.minutes}</td>
                <td className={`px-2 py-1.5 font-medium ${POT_TEXT[episode.pot]}`}>{episode.pot}</td>
                <td className="px-2 py-1.5">
                  {episode.speakers.length === 0 ? (
                    <span className="text-[10px] text-amber-600 dark:text-amber-400">none recorded</span>
                  ) : (
                    <div className="flex flex-wrap gap-1">
                      {episode.speakers.map((speaker, index) => (
                        <Chip
                          key={index}
                          title={`${speaker.role ?? 'role unset'} · ${speaker.gender ?? 'gender unset'} · ${
                            speaker.age_bracket ?? 'age unset'
                          }`}
                        >
                          {[speaker.gender, speaker.age_bracket]
                            .filter(Boolean)
                            .map((v) => humanize(v as string))
                            .join(' ') || '?'}
                        </Chip>
                      ))}
                    </div>
                  )}
                </td>
                <td className="px-2 py-1.5">
                  {episode.topic ? (
                    <Chip
                      tone={episode.topic_in_taxonomy ? 'present' : 'dirty'}
                      title={
                        episode.topic_in_taxonomy
                          ? `from the taxonomy${episode.topic_source ? ` (${episode.topic_source})` : ''}`
                          : 'free text, outside the closed taxonomy'
                      }
                    >
                      {humanize(episode.topic)}
                    </Chip>
                  ) : (
                    <span className="text-[10px] text-muted-foreground">—</span>
                  )}
                </td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums">
                  {percent(episode.mean_cmi, 1)}
                </td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-muted-foreground">
                  {episode.segments}
                </td>
                <td className="px-2 py-1.5 text-right font-mono tabular-nums">
                  {percent(episode.labeled_fraction)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

/**
 * The state of the paperwork.
 *
 * An unfilled stratification variable is worse than a thin stratum: a thin one is a fact about
 * the corpus, an unfilled one is a fact about the records, and the two are indistinguishable in
 * every chart on this page until this table separates them.
 */
export function MetadataPanel({ rows }: { rows: MetadataCompleteness[] }) {
  return (
    <Panel>
      <PanelHeading
        icon={<RiListCheck3 className="size-4 text-primary" />}
        title="What the records are missing"
        note="fixable at the desk, without recording anything"
      />
      <div className="grid grid-cols-1 gap-x-8 gap-y-2 md:grid-cols-2 xl:grid-cols-3">
        {rows.map((row) => {
          const complete = row.filled === row.total
          return (
            <div key={row.field} className="space-y-1 border-b border-border/40 py-1.5 last:border-0">
              <div className="flex items-baseline justify-between gap-2 text-xs">
                <span title={FIELD_NOTE[row.field]} className={FIELD_NOTE[row.field] ? 'underline decoration-dotted' : ''}>
                  {FIELD_LABEL[row.field] ?? row.field}
                </span>
                <span
                  className={`font-mono tabular-nums ${
                    complete ? 'text-muted-foreground' : 'text-amber-600 dark:text-amber-400'
                  }`}
                >
                  {row.filled}/{row.total}
                </span>
              </div>
              <div className="h-1 w-full overflow-hidden rounded-sm bg-muted">
                <div
                  className={`h-full ${complete ? 'bg-emerald-500' : 'bg-amber-500'}`}
                  style={{ width: `${row.fraction * 100}%` }}
                />
              </div>
              {row.missing_episodes.length > 0 ? (
                <p
                  className="truncate text-[10px] text-muted-foreground"
                  title={row.missing_episodes.join('\n')}
                >
                  {hours(row.missing_hours)} h affected · {row.missing_episodes.slice(0, 3).join(', ')}
                  {row.missing_episodes.length > 3 ? ` +${row.missing_episodes.length - 3}` : ''}
                </p>
              ) : null}
            </div>
          )
        })}
      </div>
    </Panel>
  )
}
