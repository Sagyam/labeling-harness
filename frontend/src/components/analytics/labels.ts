/**
 * Words for the corpus page: how a bucket, a goal or a group reads to a person.
 */

import type { CategoryGroup, Goal, RecommendationKind } from '@/types'

const BUCKET_LABEL: Record<string, string> = {
  none: 'none',
  '0-5%': '0–5% of the clip',
  '5-15%': '5–15% of the clip',
  '>15%': 'over 15% of the clip',
  unmeasured: 'not measured',
  undiarized: 'not diarized',
  unlinked: 'no linked voice',
  undeclared: 'no speakers declared',
  unresolved: 'declared, voice unresolved',
  untagged: 'not tagged',
  'no show id': 'no show id',
  unseen: 'not in train',
  under_20: 'under 20',
  '20_39': '20–39',
  '40_59': '40–59',
  '60_79': '60–79',
  '80_plus': '80+',
  '3+': '3 or more',
  '2+': '2 or more',
}

export function bucketLabel(bucket: string): string {
  return BUCKET_LABEL[bucket] ?? bucket.replace(/_/g, ' ')
}

export const GOAL_LABEL: Record<Goal | 'both', string> = {
  asr: 'ASR',
  paper: 'paper',
  both: 'both',
}

export const GOAL_TITLE: Record<Goal | 'both', string> = {
  asr: 'Matters for fine-tuning and benchmarking the recogniser',
  paper: 'Matters for the sociolinguistic study of code-mixing',
  both: 'Matters for the recogniser and for the study',
}

export const GROUP_NOTE: Record<CategoryGroup, string> = {
  people: 'Who is talking. The unit is the voice; a paper counts speakers, not hours.',
  content: 'What the recording is. Confounds a comparison between people has to hold.',
  speech: 'How it was said. Conditions a recogniser trips on, measured on every clip.',
  acoustics: 'How it was recorded. Brouhaha and the bandwidth probe, per clip (D87).',
}

export const KIND_LABEL: Record<RecommendationKind, string> = {
  absent: 'absent',
  thin: 'thin',
  recurrence: 'recurrence',
  dominant: 'dominant',
  no_gold: 'no gold',
  single_show: 'one show',
  unverified: 'unverified',
  unmeasured: 'unmeasured',
}

export const KIND_TITLE: Record<RecommendationKind, string> = {
  absent: 'A bucket of the closed vocabulary with no audio at all.',
  thin: 'Present, but under the floor in its own unit: voices for people and content, hours for conditions.',
  recurrence: 'The same people across episodes and shows, which the accommodation design needs.',
  dominant: 'One bucket holds more than half of the category, so the category supports no comparison.',
  no_gold: 'The corpus has this condition and the benchmark does not, so its WER cannot be measured.',
  single_show: 'Every hour of it comes from one show; lose the show and lose the value.',
  unverified: 'Rests on screened labels, whose script choice is the fused seed’s convention.',
  unmeasured: 'Too much of the category is unmeasured or undeclared to trust the gaps above it.',
}

export function minutes(value: number): string {
  if (value >= 60) return `${(value / 60).toFixed(1)} h`
  return value >= 10 ? `${value.toFixed(0)} min` : `${value.toFixed(1)} min`
}

export function hoursOrMinutes(hours: number): string {
  if (hours === 0) return '0'
  if (hours * 3600 < 60) return `${Math.round(hours * 3600)} s`
  if (hours < 1) return `${(hours * 60).toFixed(hours * 60 >= 10 ? 0 : 1)} min`
  return `${hours.toFixed(hours >= 10 ? 1 : 2)} h`
}
