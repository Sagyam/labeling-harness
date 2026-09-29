/**
 * Words for the corpus and voices pages: how a bucket or a group reads to a person.
 */

import type { CategoryGroup } from '@/types'

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

export const GROUP_NOTE: Record<CategoryGroup, string> = {
  people: 'Who is talking. The unit is the voice: one person is an anecdote, not a group.',
  content: 'What the recording is: how it was made, and what it is about.',
  speech: 'How it was said, measured on every clip.',
  acoustics: 'How it was recorded. Brouhaha and the bandwidth probe, per clip (D87).',
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
