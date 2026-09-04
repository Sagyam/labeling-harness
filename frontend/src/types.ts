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

export interface HypothesisWord {
  position: number
  word: string
  start_time: number | null
  end_time: number | null
  confidence: number | null
  predicted_language?: string | null
  predicted_script?: string | null
}

export interface Hypothesis {
  id: number
  system_id: string
  model_id: string | null
  text: string
  avg_logprob: number | null
  no_speech_prob: number | null
  word_count: number
  words?: HypothesisWord[]
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

export interface IngestLogEntry {
  timestamp: string
  level: 'info' | 'warn' | 'error' | 'success' | string
  message: string
}

/** One ASR system's failure on a clip that was dropped from the run. */
export interface DiscardFailure {
  route: string
  system_id: string
  error: string
}

/** A segment dropped mid-run rather than failing the whole episode (D46). */
export interface DiscardedSegment {
  segment_id: string
  start_time: number
  end_time: number
  stage: 'asr' | 'analysis' | string
  failures: DiscardFailure[]
}

export interface IngestJobStatus {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  stage:
    | 'upload'
    | 'downloading'
    | 'normalizing'
    | 'segmenting'
    | 'transcribing'
    | 'analyzing'
    | 'importing'
    | 'complete'
    | 'failed'
  progress: number
  active_segments: number
  total_segments: number
  error: string | null
  episode_id: string
  show_id: string
  title: string
  logs: IngestLogEntry[]
  discarded_segments: DiscardedSegment[]
  /** system_id -> how many segments it cost, most expensive first. */
  discarded_by_system: Record<string, number>
  /** What the run produced. Null until it finishes. */
  summary: Record<string, any> | null
}

export type IngestEvent =
  | { type: 'log'; timestamp: string; level: string; message: string }
  | {
      type: 'progress'
      stage: string
      progress: number
      active_segments: number
      total_segments: number
    }
  | { type: 'discard'; segment: DiscardedSegment }
  | { type: 'complete'; summary: Record<string, any>; episode_id: string }
  | { type: 'error'; error: string }

/** Metadata read from a YouTube URL before anything is downloaded. */
export interface YouTubeProbe {
  video_id: string
  url: string
  title: string
  duration_seconds: number | null
  uploader: string | null
  thumbnail: string | null
  upload_date: string | null
  is_live: boolean
  suggested_episode_id: string
}

export interface YouTubeIngestIn {
  url: string
  episode_title?: string
  show_id?: string
  episode_id?: string
  genre?: string
  topic?: string
  speakers_json?: string
}

export interface EpisodeSummary {
  id: number
  external_id: string
  title: string | null
  show_id: string | null
  duration_seconds: number | null
  split: string
  segment_count: number
  labeled_count: number
  pending_count: number
}

export interface EpisodeSegmentSummary {
  id: number
  external_id: string
  start_time: number
  end_time: number
  duration_seconds: number
  pipeline_status: string
  task_status: string | null
  task_id?: number | null
  seed_text: string | null
  flags: string[]
  cmi: number | null
  word_disagreement_rate: number | null
  audio_url: string
  peaks_url: string | null
}

export interface AnalyticsReport {
  generated_at: string
  corpus: {
    episodes: number
    segments: number
    audio_hours: number
    segments_by_status: Record<string, number>
    episode_titles: Array<{ external_id: string; title: string | null; split: string }>
  }
  labels: {
    total: number
    accepted_unchanged: number
    edited: number
    unusable_audio: number
    uncertain: number
    [key: string]: number
  }
  accept_rate: number | null
  accept_rate_by_day: Array<{
    day: string
    labeled: number
    accepted: number
    accept_rate: number | null
  }>
  throughput: {
    median_seconds_per_segment: number | null
    segments_per_hour: number | null
    annotator_hours: number
    events: number
  }
  queue: {
    backlog: number
    by_queue: Record<string, number>
    projected_hours_to_finish: number | null
  }
  scores: {
    mean_word_disagreement_rate: number | null
    mean_script_conflict_rate: number | null
    mean_code_switch_density: number | null
    mean_cer_between_hypotheses: number | null
  }
  split_balance: Record<string, { episodes: number; segments: number; hours: number }>
  word_timestamp_coverage: {
    hypotheses_total: number
    hypotheses_with_words: number
    fraction: number
  }
}

export interface ExportOutItem {
  kind: string
  row_count: number
  row_counts_by_split: Record<string, number>
  data_filename: string
  manifest_filename: string
  download_url: string
  manifest_url: string
  manifest: Record<string, any>
}

export interface ExportResponse {
  results: ExportOutItem[]
}

export interface ExportHistoryItem {
  kind: string
  data_filename: string
  manifest_filename: string
  download_url: string
  manifest_url: string
  row_count: number
  row_counts_by_split: Record<string, number>
  exported_at: string | null
  file_bytes: number
}

// --- Cost Tracker Types ---------------------------------------------------------------------

export interface CostSummary {
  total_cost_usd: number
  total_requests: number
  successful_requests: number
  failed_requests: number
  dry_run_requests: number
  average_latency_ms: number | null
  total_prompt_tokens: number
  total_completion_tokens: number
}

export interface VendorCostBreakdown {
  vendor: string
  cost_usd: number
  percentage: number
  requests: number
  successful: number
  failed: number
  dry_run: number
  average_latency_ms: number | null
}

export interface ModelCostBreakdown {
  route: string
  model: string
  vendor: string
  cost_usd: number
  requests: number
  successful: number
  failed: number
  dry_run: number
  prompt_tokens: number
  completion_tokens: number
  average_latency_ms: number | null
  effective_rate_display: string | null
}

export interface CostTimelinePoint {
  date: string
  cost_usd: number
  requests: number
  by_vendor: Record<string, number>
}

export interface PricingCatalogItem {
  vendor: string
  route: string
  model: string
  pricing_unit: string
  base_rate_usd?: number | null
  keyterm_rate_usd?: number | null
  input_per_m_usd?: number | null
  output_per_m_usd?: number | null
  effective_rate_display: string
  description: string
}

export interface CostReportResponse {
  summary: CostSummary
  vendor_breakdown: VendorCostBreakdown[]
  model_breakdown: ModelCostBreakdown[]
  daily_timeline: CostTimelinePoint[]
  pricing_catalog: PricingCatalogItem[]
}

export interface LlmRequestItem {
  id: number
  route: string
  model: string | null
  vendor: string
  status: 'succeeded' | 'failed' | 'dry_run' | string
  estimated_cost_usd: number
  latency_ms: number | null
  prompt_tokens: number | null
  completion_tokens: number | null
  input_summary: string | null
  error_message: string | null
  created_at: string
}

export interface CostRequestsResponse {
  total: number
  items: LlmRequestItem[]
}



