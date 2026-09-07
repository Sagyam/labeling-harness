/**
 * Who is in the corpus: the speaker matrix, and hours by every dimension the corpus records.
 *
 * The matrix is a sequential heat map over one hue, because hours are a magnitude. Its empty
 * cells are drawn rather than omitted — a blank square in a grid whose axes are the full closed
 * vocabulary is the finding, and a grid that only shows what exists can never show that.
 *
 * Hours here are attributed per episode to every value the episode carries: an episode with a
 * male host and a female guest counts its whole duration on both sides, because no route diarizes
 * (D52) and per-speaker time does not exist. Shares therefore do not sum to 100%, and the panel
 * says so rather than quietly normalising.
 */

import { RiGroupLine, RiPieChartLine } from '@remixicon/react'

import {
  BarRow,
  Chip,
  HEAT_STEPS,
  Panel,
  PanelHeading,
  heatStep,
  hours,
  humanize,
  percent,
} from './primitives'
import type { Dimension, LengthProfile } from '@/types'

const GENDER_ORDER = ['male', 'female']
const AGE_ORDER = ['under_20', '20_39', '40_59', '60_79', '80_plus']

const DIMENSION_TITLE: Record<string, string> = {
  show_id: 'Shows',
  topic: 'Topics',
  role: 'Speaker roles',
  gender: 'Speaker gender',
  age_bracket: 'Speaker age',
}

/**
 * `gender x age_bracket`, over the full vocabulary of both. Empty cells are drawn, not skipped.
 *
 * The margins are the dimensions' own attributed hours, not sums of the cells. Summing cells would
 * over-count: an episode with a 20-39 host and a 40-59 guest lands in two cells and would be
 * counted twice, producing a "total" larger than the corpus.
 */
function SpeakerMatrix({
  matrix,
  genderHours,
  ageHours,
}: {
  matrix: Record<string, Record<string, number>>
  genderHours: Record<string, number>
  ageHours: Record<string, number>
}) {
  const peak = Math.max(
    ...GENDER_ORDER.flatMap((g) => AGE_ORDER.map((a) => matrix[g]?.[a] ?? 0)),
    0.0001,
  )
  const covered = GENDER_ORDER.flatMap((g) => AGE_ORDER.map((a) => matrix[g]?.[a] ?? 0)).filter(
    (h) => h > 0,
  ).length

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-0.5 text-[11px]">
          <thead>
            <tr>
              <th className="w-20" />
              {AGE_ORDER.map((age) => (
                <th
                  key={age}
                  className="px-1 pb-1 text-center font-mono text-[10px] font-normal text-muted-foreground"
                >
                  {humanize(age)}
                </th>
              ))}
              <th
                className="px-1 pb-1 text-right font-mono text-[10px] font-normal text-muted-foreground"
                title="hours of audio carrying this gender at all, not the sum of the row"
              >
                any
              </th>
            </tr>
          </thead>
          <tbody>
            {GENDER_ORDER.map((gender) => {
              const rowTotal = genderHours[gender] ?? 0
              return (
                <tr key={gender}>
                  <th className="pr-2 text-right font-mono text-[10px] font-normal text-muted-foreground">
                    {humanize(gender)}
                  </th>
                  {AGE_ORDER.map((age) => {
                    const value = matrix[gender]?.[age] ?? 0
                    const step = heatStep(value, peak)
                    return (
                      <td key={age} className="p-0">
                        <div
                          title={`${humanize(gender)}, ${humanize(age)}: ${hours(value)} h`}
                          className={`flex h-9 items-center justify-center rounded-sm font-mono tabular-nums ${
                            step < 0
                              ? 'border border-dashed border-muted-foreground/40 text-muted-foreground/50'
                              : `${HEAT_STEPS[step]} ${step >= 3 ? 'text-white' : 'text-foreground'}`
                          }`}
                        >
                          {value > 0 ? hours(value) : '—'}
                        </div>
                      </td>
                    )
                  })}
                  <td className="pl-2 text-right font-mono tabular-nums text-muted-foreground">
                    {rowTotal > 0 ? hours(rowTotal) : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
          <tfoot>
            <tr>
              <th
                className="pr-2 pt-1 text-right font-mono text-[10px] font-normal text-muted-foreground"
                title="hours of audio carrying this age bracket at all, not the sum of the column"
              >
                any
              </th>
              {AGE_ORDER.map((age) => (
                <td
                  key={age}
                  className="pt-1 text-center font-mono tabular-nums text-muted-foreground"
                >
                  {ageHours[age] ? hours(ageHours[age]) : '—'}
                </td>
              ))}
              <td />
            </tr>
          </tfoot>
        </table>
      </div>
      <p className="text-[11px] text-muted-foreground">
        <span className="font-mono">{covered}</span> of{' '}
        <span className="font-mono">{GENDER_ORDER.length * AGE_ORDER.length}</span> cells carry any
        audio. Hours are per episode and count on every side the episode carries, so the cells in
        a row sum to more than the margin beside it.
      </p>
    </div>
  )
}

/** One dimension as a bar list, with what its vocabulary says should be there and is not. */
function DimensionBars({ dimension, corpusHours }: { dimension: Dimension; corpusHours: number }) {
  const peak = Math.max(...dimension.values.map((v) => v.hours), 0.0001)
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="font-heading text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          {DIMENSION_TITLE[dimension.key] ?? dimension.key}
        </span>
        <span
          className="font-mono text-[10px] text-muted-foreground"
          title="Herfindahl index over attributed hours: 1.0 is everything in one value"
        >
          {dimension.values.length} values · top {percent(dimension.top_share)}
        </span>
      </div>

      <div className="space-y-1.5">
        {dimension.values.slice(0, 8).map((value) => (
          <BarRow
            key={value.value}
            label={
              <span className="flex items-baseline gap-1.5">
                <span className="truncate">{humanize(value.value)}</span>
                {value.shows === 1 && dimension.key !== 'show_id' ? (
                  <span
                    className="shrink-0 text-[9px] text-amber-600 dark:text-amber-400"
                    title="carried by a single show — lose the show and the value goes with it"
                  >
                    1 show
                  </span>
                ) : null}
              </span>
            }
            value={value.hours}
            peak={peak}
            secondary={value.labeled_hours}
            caption={
              <>
                {hours(value.hours)} h
                <span className="ml-1 text-muted-foreground">
                  {corpusHours ? percent(value.hours / corpusHours) : ''}
                </span>
              </>
            }
          />
        ))}
        {dimension.values.length > 8 ? (
          <p className="pl-1 text-[10px] text-muted-foreground">
            +{dimension.values.length - 8} more below the top eight
          </p>
        ) : null}
      </div>

      {dimension.unknown_episodes > 0 ? (
        <p className="text-[10px] text-amber-600 dark:text-amber-400">
          {dimension.unknown_episodes} episode(s) record nothing here ({hours(dimension.unknown_hours)}{' '}
          h) — unfilled paperwork, not a gap in the world.
        </p>
      ) : null}

      {dimension.absent.length > 0 ? (
        <div className="space-y-1">
          <span className="text-[10px] text-muted-foreground">Never recorded</span>
          <div className="flex flex-wrap gap-1">
            {dimension.absent.map((value) => (
              <Chip key={value} tone="absent" title="in the vocabulary, absent from the corpus">
                {humanize(value)}
              </Chip>
            ))}
          </div>
        </div>
      ) : null}

      {dimension.off_vocabulary.length > 0 ? (
        <div className="space-y-1">
          <span className="text-[10px] text-amber-600 dark:text-amber-400">
            Outside the taxonomy — cannot be stratified on
          </span>
          <div className="flex flex-wrap gap-1">
            {dimension.off_vocabulary.map((value) => (
              <Chip key={value} tone="dirty" title="free text where a closed vocabulary was expected">
                {value}
              </Chip>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  )
}

const LENGTH_LABEL: Record<string, string> = {
  under_5m: 'under 5 min',
  '5_20m': '5–20 min',
  '20_45m': '20–45 min',
  '45_90m': '45–90 min',
  over_90m: 'over 90 min',
}

/**
 * Where the hours come from by episode length. A sourcing habit rather than a property of the
 * material: the corpus is moving from multi-hour podcasts to short videos picked for the speaker
 * they bring, and this is the panel that says whether that has actually happened yet.
 */
function LengthProfileBars({ profile }: { profile: LengthProfile }) {
  const peak = Math.max(...profile.buckets.map((b) => b.hours), 0.0001)
  return (
    <div className="space-y-2 border-t pt-3">
      <div className="flex items-baseline justify-between">
        <span className="font-heading text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Episode length
        </span>
        <span className="font-mono text-[10px] text-muted-foreground">
          median {profile.median_minutes ?? '--'} min
        </span>
      </div>
      <div className="space-y-1.5">
        {profile.buckets.map((bucket) => (
          <BarRow
            key={bucket.name}
            label={LENGTH_LABEL[bucket.name] ?? bucket.name}
            value={bucket.hours}
            peak={peak}
            caption={
              <>
                {hours(bucket.hours)} h
                <span className="ml-1 text-muted-foreground">{bucket.episodes} ep</span>
              </>
            }
          />
        ))}
      </div>
      <p className="text-[10px] text-muted-foreground">
        {percent(profile.long_episode_share)} of hours sit in episodes over 45 min. A long episode
        is many hours of one speaker; a short one is a chance at a different speaker.
      </p>
    </div>
  )
}

export function CoveragePanel({
  matrix,
  dimensions,
  corpusHours,
  speakerProfiles,
  lengthProfile,
}: {
  matrix: Record<string, Record<string, number>>
  dimensions: Record<string, Dimension>
  corpusHours: number
  speakerProfiles: number
  lengthProfile: LengthProfile
}) {
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
      <Panel className="xl:col-span-2">
        <PanelHeading
          icon={<RiGroupLine className="size-4 text-primary" />}
          title="Who is in it"
          note={
            <span title="Distinct (show, role, gender, age) combinations. Two guests of one show in the same bracket collapse into one.">
              ≥ {speakerProfiles} distinct speakers
            </span>
          }
        />
        <SpeakerMatrix
          matrix={matrix}
          genderHours={Object.fromEntries(
            (dimensions.gender?.values ?? []).map((value) => [value.value, value.hours]),
          )}
          ageHours={Object.fromEntries(
            (dimensions.age_bracket?.values ?? []).map((value) => [value.value, value.hours]),
          )}
        />
        <div className="mt-4">
          <LengthProfileBars profile={lengthProfile} />
        </div>
      </Panel>

      <Panel className="xl:col-span-3">
        <PanelHeading
          icon={<RiPieChartLine className="size-4 text-primary" />}
          title="Where the hours sit"
          note="Solid bar is labelled audio; faded is ingested and not yet decided."
        />
        <div className="grid grid-cols-1 gap-x-8 gap-y-6 sm:grid-cols-2">
          {['show_id', 'topic', 'gender', 'age_bracket'].map((key) =>
            dimensions[key] ? (
              <DimensionBars key={key} dimension={dimensions[key]} corpusHours={corpusHours} />
            ) : null,
          )}
        </div>
      </Panel>
    </div>
  )
}
