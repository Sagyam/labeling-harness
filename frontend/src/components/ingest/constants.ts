export const STAGES = [
  { key: 'normalizing', label: 'Normalize audio', desc: 'FFmpeg loudnorm, 16 kHz mono FLAC' },
  { key: 'segmenting', label: 'Silero VAD', desc: 'CPU speech turn detection (2s–20s)' },
  { key: 'transcribing', label: 'Cloud ASR', desc: 'Every configured system, per clip' },
  { key: 'fusing', label: 'Fusion', desc: 'Reasoning model reconciles the recognisers' },
  { key: 'analyzing', label: 'Token analysis', desc: 'Devanagari/Latin tagging & CMI' },
  { key: 'importing', label: 'Direct import', desc: 'Database records & queue building' },
]

/** A URL job fetches its own audio first; an upload arrives with the request. */
export const DOWNLOAD_STAGE = {
  key: 'downloading',
  label: 'Fetch audio',
  desc: 'yt-dlp download from YouTube',
}

export const ALLOWED_EXTENSIONS = ['mp3', 'm4a', 'wav', 'flac', 'aac', 'ogg']

export type SourceTab = 'file' | 'youtube'

/** How long to sit on a keystroke before asking the backend what the URL points at. */
export const PROBE_DEBOUNCE_MS = 500

/**
 * How long the AZ-5 cover stays open after the first press.
 */
export const SCRAM_ARM_TIMEOUT_MS = 8000

/** The show id a form starts on, before a probe offers the channel name instead. */
export const DEFAULT_SHOW_ID = 'nepanglish'

/** How many speakers one episode may declare; the backend's MAX_SPEAKERS and the manifest schema agree (D79). */
export const MAX_SPEAKERS = 8

export type SpeakerDraft = { gender: string; ageBracket: string }

export const emptySpeaker = (): SpeakerDraft => ({ gender: '', ageBracket: '' })

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

export const ACTIVE_JOB_KEY = 'harness.ingest.activeJobId'

export const LOG_LEVEL_CLASS: Record<string, string> = {
  error: 'text-destructive',
  warn: 'text-warning',
  success: 'text-success',
}
