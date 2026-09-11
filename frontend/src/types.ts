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
  /** `asr` for a recogniser, `fusion` for the reconciled transcript that seeds the editor. */
  kind?: 'asr' | 'fusion'
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
  verification_tier: VerificationTier
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
  /** Gold clips must be listened to; the editor hides screening for them. */
  pot: PotName
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
  /** Hazard gates that fired on the fused seed (D74). Any one makes the clip unscreenable. */
  hazards?: string[]
  /** The words behind each gate, for the tooltip. */
  hazard_details?: Record<string, string>
  /** Components with no measurement behind them -- shown as `--`, never as a confident 0. */
  unmeasured?: string[]
}

export interface QueueRow {
  task_id: number
  segment_id: number
  segment_external_id: string
  episode_external_id: string
  queue: 'review' | 'audit' | 'error' | string
  status: 'pending' | 'in_progress' | 'done' | 'skipped' | string
  pot: PotName
  priority_score: number
  reason: QueueReason | null
  flags: string[]
  duration_seconds: number
  seed_hypothesis_id: number | null
  seed_system_id: string | null
  seed_text: string | null
  cmi?: number | null
  word_disagreement_rate?: number | null
  audio_url: string
  peaks_url: string
}

export type TriageSortBy = 'priority' | 'cmi' | 'disagreement' | 'duration' | 'pot'
export type SortOrder = 'asc' | 'desc'

export interface DisputeAlternative {
  system_id: string
  word: string
}

/** A seed word every other system that spoke at that moment contradicted. */
export interface Dispute {
  seed_position: number
  seed_word: string
  start_time: number
  end_time: number
  alternatives: DisputeAlternative[]
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
  disputes?: Dispute[]
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
  /**
   * How much attention this decision got. Omit it and the server records `verified`, so a caller
   * that forgets the field cannot weaken what the corpus claims about itself. A `screened`
   * decision on a gold-pot segment is refused with 409.
   */
  verification_tier?: VerificationTier
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
  verification_tier: VerificationTier | null
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
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'aborted' | 'backlog'
  stage:
    | 'upload'
    | 'downloading'
    | 'normalizing'
    | 'segmenting'
    | 'transcribing'
    | 'fusing'
    | 'analyzing'
    | 'importing'
    | 'complete'
    | 'failed'
    | 'aborted'
    | 'backlog'
  progress: number
  active_segments: number
  total_segments: number
  error: string | null
  /** True once AZ-5 has been pressed on this run. */
  scrammed: boolean
  scram_reason: string | null
  episode_id: string
  show_id: string
  title: string
  source_url?: string | null
  logs: IngestLogEntry[]
  discarded_segments: DiscardedSegment[]
  /** system_id -> how many segments it cost, most expensive first. */
  discarded_by_system: Record<string, number>
  /** What the run produced. Null until it finishes. */
  summary: Record<string, any> | null
}

/** What the SCRAM endpoint reports back. */
export interface IngestScramResult {
  job_id: string
  scrammed: boolean
  status: IngestJobStatus['status']
  stage: IngestJobStatus['stage']
  /** True when the run was already finishing or finished, so this press changed nothing. */
  already_stopping: boolean
  detail: string
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
  /** AZ-5 pressed: the run is stopping at its next checkpoint, but has not stopped yet. */
  | { type: 'scram'; reason: string }
  /** The scrammed run has stopped. Nothing was imported. */
  | { type: 'aborted'; summary: Record<string, any>; reason: string | null }
  /** Placed into backlog (e.g. bot challenge) to retry later. */
  | { type: 'backlog'; reason: string; error: string }

/** Summary of a job in the queue dashboard. */
export interface QueueJobSummary {
  job_id: string
  episode_id: string
  show_id: string
  title: string
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'aborted' | 'backlog'
  stage: string
  progress: number
  active_segments: number
  total_segments: number
  error?: string | null
  source_url?: string | null
  created_at: number
  queue_position?: number | null
}

export interface IngestQueueResponse {
  running: QueueJobSummary | null
  upcoming: QueueJobSummary[]
  backlog: QueueJobSummary[]
  past: QueueJobSummary[]
  counts: {
    running: number
    upcoming: number
    backlog: number
    past: number
    total: number
  }
  jobs: QueueJobSummary[]
}

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

export interface YouTubeBatchIngestIn {
  urls: string[]
  show_id?: string
  genre?: string
  topic?: string
}

export interface YouTubeBatchIngestOut {
  total?: number
  queued: Array<{
    job_id: string
    episode_id: string
    title: string
    queue_position: number
  }>
  queued_count: number
  errors: Array<{ url: string; error: string }>
}

export interface EpisodeSummary {
  id: number
  external_id: string
  title: string | null
  show_id: string | null
  duration_seconds: number | null
  split: string
  /** Clips of this episode the owner has put in gold. */
  gold_count: number
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
  pots: PotPanel
  verification: VerificationPanel
  word_timestamp_coverage: {
    hypotheses_total: number
    hypotheses_with_words: number
    fraction: number
  }
}

/** Which pot a clip belongs to (D71). Gold is the benchmark, chosen per clip by hand. */
export type PotName = 'gold' | 'train'

/** The four rows the dashboard shows: the two pots, with the train pot split into train and val. */
export type BucketName = 'gold' | 'train' | 'val' | 'unassigned'

export interface PotBucket {
  episodes: number
  segments: number
  /** Audio ingested into this bucket. Moves when something is ingested. */
  hours: number
  /** Audio in this bucket that carries a current label. Moves when the annotator works. */
  labeled_hours: number
}

export interface PotPanel {
  gold_target_hours: number
  train_target_hours: number
  /** Gold is counted by clip; train, val and unassigned by the episode's split. */
  buckets: Record<BucketName, PotBucket>
  /** Episodes with clips on both sides of the train/test line: same speaker, same topic. */
  gold_episodes_spanning_pots: number
  gold_segments_in_spanning_episodes: number
  /** `{coverage key: {value: episodes in gold carrying it}}`. */
  gold_coverage: Record<string, Record<string, number>>
  corpus_coverage: Record<string, Record<string, number>>
  /** Values the corpus has that the gold pot does not cover at all. */
  gold_coverage_gaps: Record<string, string[]>
  coverage_complete: boolean
}

/** How much attention a label got: `verified` was listened to, `screened` was not. */
export type VerificationTier = 'verified' | 'screened'

export interface VerificationPanel {
  verified: number
  screened: number
  total: number
  hours: Record<VerificationTier, number>
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




// --- Corpus inventory (D69) ---------------------------------------------------------------
// What the corpus contains, what it is missing, and what to record next. Hours on a speaker
// dimension are attributed per episode to every value the episode carries, so those shares do
// not sum to 1: nothing in the schema says which speaker held the microphone for how long.

/** One value of one dimension, with the audio that carries it. */
export interface DimensionValue {
  value: string
  hours: number
  episodes: number
  segments: number
  labeled_hours: number
  /** Distinct shows carrying it. One show means the value is not independent of that show. */
  shows: number
  /** Share of corpus hours. */
  share: number
}

export interface Dimension {
  key: string
  values: DimensionValue[]
  /** Vocabulary values with no audio at all. Empty for an open dimension like `show_id`. */
  absent: string[]
  /** Values present that the closed vocabulary does not contain — dirt, not a gap. */
  off_vocabulary: string[]
  /** Share held by the largest value, over attributed value-hours (these do sum to 1). */
  top_share: number
  hhi: number
  unknown_episodes: number
  unknown_hours: number
}

/** One line of the shopping list: what to look for, and the number that says why. */
export interface Recommendation {
  kind:
    | 'gender'
    | 'age_bracket'
    | 'register'
    | 'register_spread'
    | 'show_concentration'
    | 'gold_coverage'
    | 'episode_length'
    | 'topic'
  target: string
  reason: string
  /** 0-1: weight of the gap kind times how severe this instance of it is. */
  priority: number
  hours_present: number
  hours_needed: number
}

export interface RegisterBand {
  name: 'low' | 'mid' | 'high'
  lower: number
  upper: number
  description: string
  hours: number
  segments: number
}

export interface RegisterData {
  measured_hours: number
  mean: number | null
  /** Hours-weighted mean code-switch density per show, ascending. */
  show_means: Array<{ show_id: string; mean_cmi: number; hours: number }>
  show_mean_min: number | null
  show_mean_max: number | null
  /** Distance between the least and most code-switched show. Null below two shows. */
  show_mean_spread: number | null
  histogram: Array<{ lower: number; upper: number; hours: number; segments: number }>
  bands: RegisterBand[]
}

export interface LengthProfile {
  buckets: Array<{ name: string; episodes: number; hours: number }>
  median_minutes: number | null
  long_episode_hours: number
  long_episode_share: number
}

export interface MetadataCompleteness {
  field: string
  filled: number
  total: number
  fraction: number
  missing_hours: number
  missing_episodes: string[]
}

export interface InventoryEpisode {
  external_id: string
  title: string | null
  show_id: string | null
  published_at: string | null
  /** `mixed` when some but not all of the episode's clips are in gold. */
  pot: PotName | 'mixed'
  gold_segments: number
  split: string
  hours: number
  minutes: number
  segments: number
  labeled_hours: number
  verified_hours: number
  labeled_fraction: number
  topic: string | null
  topic_source: string | null
  topic_in_taxonomy: boolean | null
  speakers: Array<{ role?: string; gender?: string; age_bracket?: string }>
  mean_cmi: number | null
  min_cmi: number | null
  max_cmi: number | null
}

export interface InventoryShow {
  show_id: string
  episodes: number
  hours: number
  segments: number
  labeled_hours: number
  verified_hours: number
  /** Distinct (role, gender, age) combinations — a lower bound on people, never a count. */
  speaker_profiles: number
  genders: string[]
  age_brackets: string[]
  topics: string[]
  pots: string[]
  mean_cmi: number | null
  min_cmi: number | null
  max_cmi: number | null
}

export interface CorpusInventory {
  generated_at: string
  totals: {
    episodes: number
    segments: number
    hours: number
    labeled_hours: number
    verified_hours: number
    screened_hours: number
    labeled_fraction: number
    verified_fraction: number
    shows: number
    /** A floor on the number of individuals, not a count of them. */
    speaker_profiles: number
    gender_values: number
    age_values: number
    topics_carried: number
    topics_total: number
    gold_hours: number
    gold_target_hours: number
    train_hours: number
    train_target_hours: number
    unassigned_hours: number
    min_stratum_hours: number
  }
  dimensions: Record<string, Dimension>
  /** `{gender: {age_bracket: hours}}`, every cell present. The empty ones are the point. */
  speaker_matrix: Record<string, Record<string, number>>
  register: RegisterData
  length_profile: LengthProfile
  metadata_completeness: MetadataCompleteness[]
  episodes: InventoryEpisode[]
  shows: InventoryShow[]
  recommendations: Recommendation[]
}

/** Result of moving one clip into or out of gold. */
export interface SegmentPotOut {
  segment_id: number
  external_id: string
  pot: PotName
  changed: boolean
}
