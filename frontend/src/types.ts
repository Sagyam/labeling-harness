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
    speakers?: number
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
  /** Clip-relative stretches with two or more voices at once (D77); null = never measured. */
  overlap_spans: [number, number][] | null
  /** Who spoke when inside the clip, from the episode's newest imported diarization (D78). */
  speaker_turns: SpeakerTurn[]
  diarization_model: string | null
}

/** One speaker's stretch of a clip, clip-relative. Numbers are stable across the episode. */
export interface SpeakerTurn {
  /** 1 = the voice with the most talk time in the episode. */
  speaker: number
  start: number
  end: number
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
  queue: 'review' | 'audit' | 'error' | 'speakers' | string
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
  /** Share of the clip spent in crosstalk; null means never measured, not clean (D77). */
  overlap_share?: number | null
  audio_url: string
  peaks_url: string
}

export type TriageSortBy = 'priority' | 'cmi' | 'disagreement' | 'duration' | 'overlap' | 'pot'
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
  /** Only for a task in the speakers queue whose episode has been diarized (D98). */
  lanes?: SpeakerLanes | null
}

/** Where a block on a speaker lane came from (D98). */
export type LaneWordSource = 'label' | 'recogniser' | 'typed' | 'copy'

/** One block on the lanes: clip-relative seconds, `speaker` is a display number. */
export interface LaneWord {
  word: string
  /** Null for a word no span was measured for; it has to be placed before saving. */
  start: number | null
  end: number | null
  /** The lane it sits on; null while it waits to be placed. */
  speaker: number | null
  /** The lane the diarization put it on, kept so moves can be counted. */
  proposed_speaker: number | null
  source: LaneWordSource
  /** The lane a voiceprint suggested when served (D99); advice, echoed back on save. */
  suggested_speaker?: number | null
  /** How much closer the suggested speaker's print was than the next one. */
  suggestion_margin?: number | null
}

/** A word the recognisers heard where the verified text has nothing. */
export interface LaneCandidate {
  word: string
  start: number
  end: number
  systems: string[]
}

export interface LaneSpeaker {
  number: number
  /** The diarization run's raw label, e.g. SPEAKER_01. */
  label: string
  /** Voice id linked across episodes (D87), when there is one. */
  voice: string | null
  /** Where the voiceprint comes from (D99): the owner's `confirmed` clips or the `diarizer`. */
  print_source?: 'confirmed' | 'diarizer' | null
  print_clips?: number
}

/** What the multitrack editor opens a speakers-queue task with (D98). */
export interface SpeakerLanes {
  diarization_run_id: number
  diarization_model: string
  /** Every speaker of the episode: the only lanes there can be. */
  speakers: LaneSpeaker[]
  /** Speakers diarized inside this clip, shown as lanes by default. */
  clip_speakers: number[]
  words: LaneWord[]
  candidates: LaneCandidate[]
  /** `speakers` (an earlier per-speaker label), `label` (the verified text) or `seed`. */
  base: 'speakers' | 'label' | 'seed'
  /** Whether voiceprint suggestions were computed (D99). */
  voiceprint?: boolean
}

export type VoiceVerdict = 'confirmed' | 'rejected' | 'cleared'

export interface VoiceEpisode {
  episode_id: number
  external_id: string
  title: string | null
  speaker_number: number
  talk_seconds: number
  solo_clips: number
}

/** A clip the diarization hears as one voice alone (D99). */
export interface VoiceClip {
  segment_id: number
  external_id: string
  episode_external_id: string
  pot: PotName
  duration_seconds: number
  text: string | null
  verdict: 'confirmed' | 'rejected' | null
  audio_url: string
  /** The stretch of this voice alone that is played and judged, clip-relative seconds. */
  start: number
  end: number
  /** The stretch is the whole clip. */
  whole: boolean
}

export interface VoicePage {
  voice: string
  talk_seconds: number
  episodes: VoiceEpisode[]
  clips: VoiceClip[]
  total_clips: number
  confirmed: number
  rejected: number
  print_source: 'confirmed' | 'diarizer'
  reference: VoiceClip | null
}

export interface VoiceVerdictOut {
  voice: string
  segment_id: number
  verdict: VoiceVerdict
  embedded: boolean
  confirmed: number
  rejected: number
}

export interface AttributeIn {
  diarization_run_id: number
  words: LaneWord[]
  opened_at?: string
  duration_ms?: number
  notes?: string | null
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

/** One rate-limited service's shared gate (D88). */
export interface ProviderGateState {
  name: string
  in_flight: number
  /** Working limit: halved by a 429, won back one slot at a time. */
  allowed: number
  max_in_flight: number
  cooling_down_seconds: number
  throttled_total: number
}

export interface IngestQueueResponse {
  /** The earliest-started running job; every running job is in `running_jobs`. */
  running: QueueJobSummary | null
  running_jobs: QueueJobSummary[]
  max_concurrent_jobs: number
  limits: ProviderGateState[]
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
  /** Speaker rows on the form, blank ones included; 0 lets the diarizer count (D79). */
  speaker_count?: number
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

/** Which pot a clip belongs to (D71). Gold is the benchmark, chosen per clip by hand. */
export type PotName = 'gold' | 'train'

/** How much attention a label got: `verified` was listened to, `screened` was not. */
export type VerificationTier = 'verified' | 'screened'

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




// --- Corpus page (D91) -----------------------------------------------------------------------
// Every clip carries one bucket on each category, so hours cut by any category sum to the
// corpus. People are voices linked across episodes; a declared gender or age reaches a voice
// only when its episode forces the match (see app/services/inventory/resolve.py).

export type Goal = 'asr' | 'paper'
export type CategoryGroup = 'people' | 'content' | 'speech' | 'acoustics'

/** One bucket of one category, with the audio and the people in it. */
export interface BucketStats {
  bucket: string
  hours: number
  clips: number
  episodes: number
  shows: number
  /** Distinct linked voices with any clip here. */
  voices: number
  /** Voices with 300+ reference words attributed inside this bucket: the usable n. */
  usable_voices: number
  verified_hours: number
  screened_hours: number
  gold_hours: number
  val_hours: number
  train_hours: number
  /** Share of the category's measured hours; unknown buckets excluded, so these sum to 1. */
  share: number
  /** A topic outside the taxonomy: dirt, not a stratum. */
  off_vocabulary: boolean
}

export interface CategoryReport {
  key: string
  label: string
  group: CategoryGroup
  goals: Goal[]
  /** What a thin bucket is short of. */
  unit: 'voices' | 'hours'
  why: string
  /** Measured buckets nobody should record more of (the diarizer's empty clip). */
  defect: string[]
  buckets: BucketStats[]
  /** Closed-vocabulary buckets with no audio at all. */
  absent: string[]
  /** Buckets meaning not measured or not declared. */
  unknown: string[]
  measured_hours: number
  unknown_hours: number
  top_bucket: string | null
  top_share: number
}

export interface VoiceEpisode {
  external_id: string
  title: string | null
  show_id: string | null
  genre: string | null
  published_at: string | null
  split: string
  minutes: number
  /** This voice's share of all linked talk in the episode's clips. */
  share: number
  clips: number
  role: string | null
  resolved_by: string | null
  with_voices: string[]
}

export interface VoiceProfile {
  voice: string
  talk_minutes: number
  clips: number
  words: number
  usable: boolean
  episodes: VoiceEpisode[]
  episode_count: number
  shows: string[]
  genres: string[]
  gender: string | null
  age_bracket: string | null
  identity: 'resolved' | 'conflict' | 'unresolved'
  resolved_by: Record<string, number>
  roles: Record<string, number>
  hours: Record<'gold' | 'train' | 'val', number>
  verified_minutes: number
  screened_minutes: number
  mean_cmi: number | null
  words_per_second: number | null
  exposure: string
  first_seen: string | null
  last_seen: string | null
  co_voices: Record<string, number>
}

export interface VoiceSummary {
  voices: number
  usable: number
  recurring: number
  single_episode: number
  recurring_hosts: number
  recurring_host_voices: string[]
  guests_on_two_shows: number
  gender_resolved: number
  age_resolved: number
  conflicts: number
  talk_hours: number
  top_voice: string | null
  top_voice_share: number
}

export type RecommendationKind =
  | 'absent'
  | 'thin'
  | 'recurrence'
  | 'dominant'
  | 'no_gold'
  | 'single_show'
  | 'unverified'
  | 'unmeasured'

/** One line of the shopping list, tied to the category it came from and the purpose it serves. */
export interface Recommendation {
  category: string
  bucket: string | null
  goal: Goal | 'both'
  kind: RecommendationKind
  target: string
  reason: string
  priority: number
  hours_present: number
  voices_present: number
  hours_needed: number
  voices_needed: number
}

export interface InventoryEpisode {
  external_id: string
  title: string | null
  show_id: string | null
  genre: string | null
  topic: string | null
  topic_in_taxonomy: boolean | null
  published_at: string | null
  split: string
  declared: Array<{ role?: string; gender?: string; age_bracket?: string }>
  voices: string[]
  diarized: boolean
}

export interface RecordCheck {
  field: string
  filled: number
  total: number
  missing_episodes: string[]
}

/**
 * The clips as a columnar table. Each row is
 * `[segment_id, episode index, seconds, words, tier index, pot index, voice index, ...bucket index per category]`,
 * with `-1` for no words or no voice. Bucket indexes point into `buckets[category]`.
 */
export interface ClipTable {
  columns: string[]
  tiers: string[]
  pots: string[]
  buckets: Record<string, string[]>
  rows: number[][]
}

export interface CorpusInventory {
  generated_at: string
  totals: {
    hours: number
    speech_hours: number
    clips: number
    episodes: number
    shows: number
    words: number
    verified_hours: number
    screened_hours: number
    unlabeled_hours: number
    gold_hours: number
    val_hours: number
    train_hours: number
    gold_target_hours: number
    min_stratum_hours: number
    min_stratum_voices: number
  }
  groups: Array<{ key: CategoryGroup; label: string }>
  categories: CategoryReport[]
  voices: VoiceProfile[]
  voice_summary: VoiceSummary
  recommendations: Recommendation[]
  episodes: InventoryEpisode[]
  records: RecordCheck[]
  clips: ClipTable
}

/** Result of moving one clip into or out of gold. */
export interface SegmentPotOut {
  segment_id: number
  external_id: string
  pot: PotName
  changed: boolean
}

// --- Fine-tuned models (D83) ------------------------------------------------------------------

/** One slice of a run: how many clips, their pooled WER, and their share of the run's errors. */
export interface ModelBreakdown {
  clips: number
  wer: number
  /** Pooled CER; absent on runs scored before classes existed. */
  cer?: number
  share_of_errors: number
  /**
   * Within-episode error-rate ratio against the axis's baseline bucket (Mantel-Haenszel over
   * episodes, D87). Absent for the baseline itself, "not measured" buckets and axes that cannot
   * vary within an episode; null when no episode holds both.
   */
  rate_ratio?: number | null
  rate_ratio_ci?: [number, number] | null
  rate_ratio_episodes?: number
}

/** One class of reference word: how many, and the WER and CER over them. Insertions have no
 * reference word, so a class's WER counts only its substitutions and deletions. */
export interface WordClassBreakdown {
  words: number
  wer: number
  cer: number
  share_of_words: number
}

/** One axis a run is split by (D87), from `GET /model-classes`. */
export interface ClassAxis {
  name: string
  label: string
  /** Display order; empty for an open set (voices). */
  buckets: string[]
  baseline: string | null
  unmeasured: string | null
  /** Kept to describe the corpus (CMI), not to explain errors. */
  descriptive: boolean
}

/** A run's numbers, as `app.services.model_eval.summarize` computed them at import. */
export interface ModelRunMetrics {
  clips: number
  episodes: number
  ref_words: number
  errors: number
  /** Folded WER, percent. */
  wer: number
  /** 95% interval from resampling whole episodes; null with one episode. */
  wer_ci: [number, number] | null
  raw_wer: number
  cer: number
  loops: number
  by_genre: Record<string, ModelBreakdown>
  /** Axis -> bucket -> slice (D87). Key order is not display order: JSONB sorts keys. */
  by_class?: Record<string, Record<string, ModelBreakdown>>
  /** Reference words by class -- script, number, code-switch, clip edge (D87). */
  by_word_class?: Record<string, WordClassBreakdown>
  /** Clips in the file that were not scored: left the split, or no transcript to score against. */
  skipped?: { not_in_split: number; no_reference: number }
}

export interface ModelEvalRun {
  id: number
  split: 'gold' | 'val' | string
  decoder: string | null
  fold_version: string
  clip_count: number
  metrics: ModelRunMetrics
  source: string | null
  created_at: string
}

export interface AsrModel {
  id: number
  slug: string
  name: string
  description: string | null
  architecture: string | null
  trained_at: string | null
  card: Record<string, unknown>
  runs: ModelEvalRun[]
  /** The CPU weights the playground loads (`cpu` or `best`), or null when none were copied (D85). */
  playground: string | null
}

/** One recording from the Models page, transcribed on the CPU by the playground sidecar (D85). */
export interface PlaygroundResult {
  text: string
  audio_s: number
  latency_ms: number | null
  weights: string
  dry_run: boolean
  raw: {
    retried?: boolean
    first_text?: string | null
    tokens?: number
    compute_s?: number
    rtf?: number
    load_s?: number | null
    variant?: string
    threads?: number
  }
}

export interface ModelRescanOut {
  models: string[]
  runs_created: number
  runs_unchanged: number
  skipped: number
}

export type OverlapBucket = 'none' | '0-5%' | '5-15%' | '>15%' | 'unmeasured'

export type ClipSort =
  | 'errors'
  | 'wer'
  | 'deletions'
  | 'insertions'
  | 'substitutions'
  | 'duration'
  | 'overlap'

export interface ModelClip {
  /** Harness segment id: `/segments/{id}`, its audio and peaks. */
  segment_id: number
  external_id: string
  episode_external_id: string
  genre: string
  duration_seconds: number
  overlap_share: number | null
  overlap_bucket: OverlapBucket
  ref_words: number
  errors: number
  substitutions: number
  deletions: number
  insertions: number
  wer: number
  raw_errors: number
  raw_ref_words: number
  char_errors: number
  ref_chars: number
  is_loop: boolean
  /** Axis -> bucket, as stored with the run (D87). */
  classes: Record<string, string>
  ref_text: string
  hyp_text: string
}

export interface ModelClipPage {
  total: number
  offset: number
  limit: number
  rows: ModelClip[]
}

/** One step of the folded word alignment. `fold` and `merge` are matches, not errors. */
export interface AlignOp {
  kind: 'match' | 'fold' | 'merge' | 'sub' | 'del' | 'ins'
  ref: string[]
  hyp: string[]
  similarity: number
}

export interface ModelClipDetail extends ModelClip {
  ops: AlignOp[]
  fold_version: string
  /** The fold rules changed since import, so the ops may not add up to the stored counts. */
  fold_version_changed: boolean
}

export interface ClipQuery {
  sort?: ClipSort
  order?: 'asc' | 'desc'
  genre?: string
  overlap?: OverlapBucket
  loops_only?: boolean
  min_errors?: number
  class_axis?: string
  class_bucket?: string
  offset?: number
  limit?: number
}
