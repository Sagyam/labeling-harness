import type { SpeakerRow } from '@/types'

/** How many speakers one episode may declare; the backend's MAX_SPEAKERS and the manifest schema agree (D79). */
export const MAX_SPEAKERS = 8

/** One speaker row being edited; '' is a blank field. */
export type SpeakerDraft = { role: string; gender: string; ageBracket: string }

/** The first row is the host by default and every later one a guest, as the form always had it. */
export const defaultRole = (index: number) => (index === 0 ? 'host' : 'guest')

export const emptySpeaker = (index = 0): SpeakerDraft => ({
  role: defaultRole(index),
  gender: '',
  ageBracket: '',
})

export const ROLE_OPTIONS = [
  { value: 'host', label: 'Host' },
  { value: 'guest', label: 'Guest' },
]

export const GENDER_OPTIONS = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
]

export const AGE_BRACKET_OPTIONS = [
  { value: 'under_20', label: 'Under 20' },
  { value: '20_39', label: '20-39' },
  { value: '40_59', label: '40-59' },
  { value: '60_79', label: '60-79' },
  { value: '80_plus', label: '80+' },
]

export const draftsFromRows = (rows: SpeakerRow[]): SpeakerDraft[] =>
  rows.map((row, index) => ({
    role: row.role ?? defaultRole(index),
    gender: row.gender ?? '',
    ageBracket: row.age_bracket ?? '',
  }))

export const rowsFromDrafts = (drafts: SpeakerDraft[]): SpeakerRow[] =>
  drafts.map((d) => ({
    role: d.role || null,
    gender: d.gender || null,
    age_bracket: d.ageBracket || null,
  }))
