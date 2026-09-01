/**
 * TypeScript types for the review API and frontend state.
 */

export interface HealthResponse {
  status: string
  app?: string
  environment?: string
  checks?: Record<string, { ok: boolean; error?: string | null; backend?: string }>
}

export interface StatsResponse {
  episodes: number
  segments: {
    total: number
    imported: number
    queued: number
    labeled: number
    excluded: number
  }
  audio_hours: number
  tasks: {
    total: number
    pending: number
    in_progress: number
    done: number
    skipped: number
  }
  queues: {
    review: number
    audit: number
    error: number
  }
  labels: {
    total: number
    accepted_unchanged: number
    edited: number
    unusable_audio: number
    uncertain: number
  }
  accept_rate: number | null
  throughput: {
    median_seconds_per_segment: number | null
    labeled_total: number
    backlog: number
    projected_seconds_to_finish: number | null
  }
  session: {
    since: string
    completed: number
    median_seconds_per_segment: number | null
    elapsed_seconds: number
  }
}

export interface Hypothesis {
  id: number
  system_id: string
  model_id: string | null
  text: string
  avg_logprob: number | null
  no_speech_prob: number | null
  word_count: number
}

export interface Scores {
  cer_between_hypotheses: number | null
  word_disagreement_rate: number | null
  script_conflict_rate: number | null
  code_switch_density: number | null
  flags: string[]
}

export interface Label {
  id: number
  disposition: string
  final_text: string | null
  annotator: string
  label_version: string
  policy_version: string
  notes: string | null
  created_at: string
}

export interface Segment {
  id: number
  external_id: string
  episode_id: number
  episode_external_id: string
  split: 'train' | 'val' | 'test' | string
  speaker_id: string | null
  start_time: number
  end_time: number
  duration_seconds: number
  p_en: number | null
  lid: string | null
  pipeline_status: string
  audio_url: string
  peaks_url: string
  hypotheses: Hypothesis[]
  scores: Scores | null
  latest_label: Label | null
}

export interface QueueReason {
  flags?: string[]
  score?: number
  weights?: Record<string, number>
  components?: Record<string, number>
  contributions?: Record<string, number>
}

export interface QueueRow {
  task_id: number
  segment_id: number
  segment_external_id: string
  episode_external_id: string
  queue: 'review' | 'audit' | 'error' | string
  status: 'pending' | 'in_progress' | 'done' | 'skipped' | string
  priority_score: number
  reason: QueueReason | null
  flags: string[]
  duration_seconds: number
  seed_hypothesis_id: number | null
  seed_system_id: string | null
  seed_text: string | null
  audio_url: string
  peaks_url: string
}

export interface Task {
  id: number
  segment_id: number
  queue: string
  status: string
  priority_score: number
  reason: QueueReason | null
  seed_hypothesis_id: number | null
  seed_system_id: string | null
  served_at: string
  segment: Segment
}

export interface PeaksPayload {
  version: number
  buckets: number
  sample_rate: number
  frames: number
  duration_seconds: number
  min: number[]
  max: number[]
}

export interface DecisionIn {
  opened_at?: string
  duration_ms?: number
  annotator?: string
  label_version?: string
  notes?: string | null
}

export interface AcceptIn extends DecisionIn {}

export interface LabelIn extends DecisionIn {
  final_text: string
}

export interface FlagIn extends DecisionIn {
  disposition: 'unusable_audio' | 'uncertain'
}

export interface SkipIn {
  opened_at?: string
  duration_ms?: number
  annotator?: string
}

export interface BulkAcceptIn extends DecisionIn {
  task_ids: number[]
}

export interface DecisionOut {
  task_id: number
  segment_id: number
  label_id: number | null
  disposition: string | null
  task_status: string
  duration_ms: number | null
}

export interface BulkAcceptOut {
  accepted: DecisionOut[]
  count: number
}

export interface TranslitOut {
  token: string
  candidates: string[]
}
