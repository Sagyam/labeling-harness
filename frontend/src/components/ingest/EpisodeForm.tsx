import {
  RiArrowDownSLine,
  RiArrowRightSLine,
  RiErrorWarningLine,
  RiFileMusicLine,
  RiListCheck2,
  RiUploadCloud2Line,
  RiUserVoiceLine,
  RiYoutubeLine,
} from '@remixicon/react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Field, FieldLabel } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/ui/spinner'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { YouTubeProbe } from '@/types'
import {
  ALLOWED_EXTENSIONS,
  DEFAULT_SHOW_ID,
  PROBE_DEBOUNCE_MS,
  type SourceTab,
} from '@/components/ingest/constants'
import { ClipField } from '@/components/ingest/ClipField'
import { GenreTopicFields } from '@/components/metadata/GenreTopicFields'
import { SpeakerRows } from '@/components/metadata/SpeakerRows'
import { type SpeakerDraft, emptySpeaker } from '@/components/metadata/speakers'
import { useEpisodeVocabulary } from '@/components/metadata/useEpisodeVocabulary'
import { formatDuration, slugify } from '@/components/ingest/format'

interface EpisodeFormProps {
  /** Whether a new job would wait behind another -- what the submit button says. */
  queueBusy: boolean
  /** Called once the API accepted the job, with its queue position. */
  onQueued: (
    res: { job_id: string; title: string; queue_position?: number },
    source: SourceTab,
  ) => void
}

/** The new-ingestion form: one video at a time, each with its own metadata (D80). */
export function EpisodeForm({ queueBusy, onQueued }: EpisodeFormProps) {
  const [sourceTab, setSourceTab] = useState<SourceTab>('youtube')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [showId, setShowId] = useState<string>(DEFAULT_SHOW_ID)
  const [episodeTitle, setEpisodeTitle] = useState<string>('')
  const [episodeId, setEpisodeId] = useState<string>('')
  const [isManualEpisodeId, setIsManualEpisodeId] = useState<boolean>(false)
  const [isDragging, setIsDragging] = useState<boolean>(false)

  const [youtubeUrl, setYoutubeUrl] = useState<string>('')
  const [probe, setProbe] = useState<YouTubeProbe | null>(null)
  const [isProbing, setIsProbing] = useState<boolean>(false)
  const [probeError, setProbeError] = useState<string | null>(null)

  const [showSociolinguistics, setShowSociolinguistics] = useState<boolean>(false)
  const [genre, setGenre] = useState<string>('podcast')
  const [topic, setTopic] = useState<string>('')
  const [speakers, setSpeakers] = useState<SpeakerDraft[]>(() => [emptySpeaker()])
  const [clipEnabled, setClipEnabled] = useState<boolean>(false)
  const [clipMinutes, setClipMinutes] = useState<string>('')
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)

  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const titleIsAnnotatorsRef = useRef<boolean>(false)
  const showIdIsAnnotatorsRef = useRef<boolean>(false)
  const clipIsAnnotatorsRef = useRef<boolean>(false)

  const vocabulary = useEpisodeVocabulary()

  // The clip is opt-in (D103). Ticked, it follows the genre's suggestion until the annotator types
  // a length of their own.
  const suggestedClipMinutes = vocabulary
    ? (vocabulary.clip_minutes[genre] ?? vocabulary.clip_default_minutes)
    : null
  const genreLabel = vocabulary?.genres.find((g) => g.value === genre)?.label ?? null
  const clipNumber = Number(clipMinutes)
  const clipValid = Number.isInteger(clipNumber) && clipNumber >= 1
  const clipValue = clipEnabled && clipValid ? clipNumber : undefined

  useEffect(() => {
    if (clipEnabled && suggestedClipMinutes !== null && !clipIsAnnotatorsRef.current) {
      setClipMinutes(String(suggestedClipMinutes))
    }
  }, [clipEnabled, suggestedClipMinutes])

  const handleClipEnabledChange = (enabled: boolean) => {
    clipIsAnnotatorsRef.current = false
    setClipEnabled(enabled)
  }

  const handleClipMinutesChange = (value: string) => {
    clipIsAnnotatorsRef.current = true
    setClipMinutes(value)
  }

  // Judged on what would be transcribed, as the server judges it: a clip can bring an over-long
  // video under the limit, so the probe reports the limit rather than refusing.
  const probeKeptSeconds =
    probe?.duration_seconds == null
      ? null
      : Math.min(probe.duration_seconds, clipValue ? clipValue * 60 : Infinity)
  const probeOverLimit =
    probe !== null && probeKeptSeconds !== null && probeKeptSeconds > probe.max_duration_seconds

  const buildSpeakersJson = () => {
    const payload: Record<string, any> = {}
    speakers.forEach((speaker, index) => {
      const fields: Record<string, string> = { role: speaker.role }
      if (speaker.gender) fields.gender = speaker.gender
      if (speaker.ageBracket) fields.age_bracket = speaker.ageBracket
      if (speaker.gender || speaker.ageBracket) payload[`spk${index}`] = fields
    })
    return Object.keys(payload).length > 0 ? JSON.stringify(payload) : ''
  }

  // Every row is a voice the diarizer must find, filled in or not. With the section closed the
  // episode declares nothing and the diarizer counts for itself (D79).
  const speakerCount = showSociolinguistics ? speakers.length : 0

  /** Clear the form for the next video. The queue and the monitor are left alone. */
  const resetForm = () => {
    setSelectedFile(null)
    setYoutubeUrl('')
    setProbe(null)
    setProbeError(null)
    setIsProbing(false)
    titleIsAnnotatorsRef.current = false
    setShowId(DEFAULT_SHOW_ID)
    showIdIsAnnotatorsRef.current = false
    setEpisodeTitle('')
    setEpisodeId('')
    setIsManualEpisodeId(false)
    setGenre('podcast')
    setTopic('')
    setSpeakers([emptySpeaker()])
    setShowSociolinguistics(false)
    setClipEnabled(false)
    setClipMinutes('')
    clipIsAnnotatorsRef.current = false
  }

  const handleTitleChange = (val: string) => {
    titleIsAnnotatorsRef.current = val.trim() !== ''
    setEpisodeTitle(val)
    if (!isManualEpisodeId) setEpisodeId(slugify(val))
  }

  const handleShowIdChange = (val: string) => {
    showIdIsAnnotatorsRef.current = val.trim() !== ''
    setShowId(val)
  }

  const handleFileSelected = (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase() || ''
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      toast.error(`Unsupported format .${ext}`, {
        description: 'Please use MP3, M4A, WAV, FLAC, AAC or OGG.',
      })
      return
    }
    setSelectedFile(file)
    if (!episodeTitle) {
      const baseName = file.name.replace(/\.[^/.]+$/, '')
      titleIsAnnotatorsRef.current = true
      setEpisodeTitle(baseName)
      if (!isManualEpisodeId) setEpisodeId(slugify(baseName))
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files?.length) handleFileSelected(e.dataTransfer.files[0])
  }

  const handleStartIngestion = async () => {
    if (!selectedFile) {
      toast.warning('Please select an audio file')
      return
    }
    if (!episodeTitle.trim()) {
      toast.warning('Please enter an episode title')
      return
    }

    setIsSubmitting(true)

    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('episode_title', episodeTitle.trim())
    formData.append('show_id', showId.trim() || 'podcast')
    formData.append('episode_id', episodeId.trim())
    formData.append('genre', genre.trim())
    formData.append('topic', topic.trim())
    formData.append('speakers_json', buildSpeakersJson())
    formData.append('speaker_count', String(speakerCount))
    if (clipValue) formData.append('clip_minutes', String(clipValue))

    try {
      const res = await api.startIngest(formData)
      resetForm()
      onQueued(res, 'file')
    } catch (err: any) {
      toast.error(err.message || 'Failed to start ingestion')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleStartYoutubeIngestion = async () => {
    if (!youtubeUrl.trim()) {
      toast.warning('Please paste a YouTube URL')
      return
    }

    setIsSubmitting(true)

    try {
      const res = await api.startYouTubeIngest({
        url: youtubeUrl.trim(),
        episode_title: episodeTitle.trim(),
        show_id: showId.trim() || 'podcast',
        episode_id: episodeId.trim(),
        genre: genre.trim(),
        topic: topic.trim(),
        speakers_json: buildSpeakersJson(),
        speaker_count: speakerCount,
        clip_minutes: clipValue,
      })
      resetForm()
      onQueued(res, 'youtube')
    } catch (err: any) {
      toast.error(err.message || 'Failed to start ingestion')
    } finally {
      setIsSubmitting(false)
    }
  }

  // Probe single YouTube URL as user types
  useEffect(() => {
    const url = youtubeUrl.trim()
    if (sourceTab !== 'youtube' || !url) {
      setProbe(null)
      setProbeError(null)
      setIsProbing(false)
      return
    }

    let cancelled = false
    setIsProbing(true)
    setProbeError(null)

    const timer = window.setTimeout(() => {
      api
        .probeYouTube(url)
        .then((info) => {
          if (cancelled) return
          setProbe(info)
          setProbeError(null)
          if (!titleIsAnnotatorsRef.current) {
            setEpisodeTitle(info.title)
            if (!isManualEpisodeId) setEpisodeId(info.suggested_episode_id)
          }
          if (!showIdIsAnnotatorsRef.current && info.uploader) {
            setShowId(slugify(info.uploader))
          }
        })
        .catch((err) => {
          if (cancelled) return
          setProbe(null)
          setProbeError(err?.detail || err?.message || 'Could not read that URL')
        })
        .finally(() => {
          if (!cancelled) setIsProbing(false)
        })
    }, PROBE_DEBOUNCE_MS)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [youtubeUrl, sourceTab, isManualEpisodeId])

  return <div className="flex flex-col gap-6">
    <Tabs value={sourceTab} onValueChange={(value) => setSourceTab(value as SourceTab)}>
      <TabsList variant="line">
        <TabsTrigger value="youtube">
          <RiYoutubeLine className="size-4" />
          Single YouTube URL
        </TabsTrigger>
        <TabsTrigger value="file">
          <RiUploadCloud2Line className="size-4" />
          Upload Local File
        </TabsTrigger>
      </TabsList>

      {/* 1. File Upload */}
      <TabsContent value="file">
        <button
          type="button"
          className={cn(
            'flex w-full flex-col items-center gap-3 border border-dashed p-8 text-center transition-colors',
            isDragging ? 'border-foreground bg-accent' : 'hover:bg-muted/50',
            selectedFile && 'border-solid bg-muted/30',
          )}
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept=".mp3,.m4a,.wav,.flac,.aac,.ogg"
            onChange={(e) => {
              if (e.target.files?.[0]) handleFileSelected(e.target.files[0])
            }}
          />

          {selectedFile ? (
            <div className="flex w-full items-center gap-4 text-left">
              <RiFileMusicLine className="size-8 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{selectedFile.name}</div>
                <div className="font-mono text-xs text-muted-foreground">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB ·{' '}
                  {selectedFile.name.split('.').pop()?.toUpperCase()}
                </div>
              </div>
              <span className="text-xs font-semibold tracking-widest uppercase">Change</span>
            </div>
          ) : (
            <>
              <RiUploadCloud2Line className="size-9 text-muted-foreground" />
              <div className="text-sm">
                <span className="font-semibold">Click to upload</span> or drag and drop podcast audio
              </div>
              <div className="flex gap-1.5 font-mono text-[10px] text-muted-foreground">
                {ALLOWED_EXTENSIONS.map((ext) => (
                  <span key={ext} className="bg-muted px-1.5 py-0.5 uppercase">
                    {ext}
                  </span>
                ))}
              </div>
            </>
          )}
        </button>
      </TabsContent>

      {/* 2. Single YouTube URL */}
      <TabsContent value="youtube">
        <div className="flex flex-col gap-3">
          <Field>
            <FieldLabel htmlFor="youtube-url">Video URL</FieldLabel>
            <Input
              id="youtube-url"
              className="font-mono text-xs"
              placeholder="https://www.youtube.com/watch?v=..."
              value={youtubeUrl}
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => setYoutubeUrl(e.target.value)}
            />
          </Field>

          {isProbing && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner className="size-3.5" />
              <span>Reading video details…</span>
            </div>
          )}

          {!isProbing && probeError && (
            <Alert variant="destructive">
              <RiErrorWarningLine />
              <AlertTitle>Cannot ingest this URL</AlertTitle>
              <AlertDescription>{probeError}</AlertDescription>
            </Alert>
          )}

          {!isProbing && probe && (
            <div className="flex items-start gap-4 border bg-muted/30 p-3">
              {probe.thumbnail && (
                <img
                  src={probe.thumbnail}
                  alt=""
                  className="hidden w-32 shrink-0 object-cover sm:block"
                />
              )}
              <div className="flex min-w-0 flex-1 flex-col gap-1">
                <div className="truncate text-sm font-medium">{probe.title}</div>
                <div className="font-mono text-xs text-muted-foreground">
                  {probe.uploader ? `${probe.uploader} · ` : ''}
                  {formatDuration(probe.duration_seconds)} · {probe.video_id}
                </div>
                <div className="text-[11px] text-muted-foreground">
                  Audio is downloaded on the server, then normalized and segmented across 8 CPU cores.
                </div>
              </div>
            </div>
          )}

          {!isProbing && probe && probeOverLimit && (
            <Alert variant="destructive">
              <RiErrorWarningLine />
              <AlertTitle>Too long to ingest whole</AlertTitle>
              <AlertDescription>
                {clipEnabled
                  ? `The clip is still over the ${Math.round(probe.max_duration_seconds / 60)} min limit. Shorten it below.`
                  : `This video is over the ${Math.round(probe.max_duration_seconds / 60)} min limit. Tick "Keep only the opening" below to ingest its first minutes.`}
              </AlertDescription>
            </Alert>
          )}
        </div>
      </TabsContent>

    </Tabs>

    {/* Episode metadata: every video carries its own */}
    <div className="flex flex-col gap-4">
      <Field>
        <FieldLabel htmlFor="episode-title">Episode title</FieldLabel>
        <Input
          id="episode-title"
          placeholder="e.g. Episode 42: AI in Kathmandu"
          value={episodeTitle}
          onChange={(e) => handleTitleChange(e.target.value)}
        />
      </Field>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field>
          <FieldLabel htmlFor="show-id">Show ID</FieldLabel>
          <Input
            id="show-id"
            placeholder="podcast"
            value={showId}
            onChange={(e) => handleShowIdChange(e.target.value)}
          />
        </Field>

        <Field>
          <div className="flex items-center justify-between gap-2">
            <FieldLabel htmlFor="episode-id">Episode ID (slug)</FieldLabel>
            <Button
              variant="link"
              size="xs"
              className="h-auto px-0"
              onClick={() => {
                setIsManualEpisodeId(!isManualEpisodeId)
                if (isManualEpisodeId) setEpisodeId(slugify(episodeTitle))
              }}
            >
              {isManualEpisodeId ? 'Auto-generate' : 'Custom'}
            </Button>
          </div>
          <Input
            id="episode-id"
            className="font-mono"
            placeholder="ep_42_ai_in_kathmandu"
            value={episodeId}
            readOnly={!isManualEpisodeId}
            onChange={(e) => setEpisodeId(e.target.value)}
          />
        </Field>
      </div>
    </div>

    {/* Sociolinguistics & Speaker Metadata Toggle */}
    <div className="rounded-lg border bg-card p-3">
      <button
        type="button"
        className="flex w-full items-center justify-between text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground"
        onClick={() => setShowSociolinguistics((prev) => !prev)}
      >
        <span className="flex items-center gap-1.5">
          <RiUserVoiceLine className="size-4 text-primary" />
          Speaker Demographics &amp; Topic (Sociolinguistics · Optional)
        </span>
        {showSociolinguistics ? (
          <RiArrowDownSLine className="size-4" />
        ) : (
          <RiArrowRightSLine className="size-4" />
        )}
      </button>

      {showSociolinguistics && (
        <div className="mt-3 flex flex-col gap-3.5 border-t pt-3">
          <GenreTopicFields
            idPrefix="ingest"
            vocabulary={vocabulary}
            genre={genre}
            topic={topic}
            onGenreChange={setGenre}
            onTopicChange={setTopic}
            topicBlankLabel="Classify from the transcript"
          />

          <SpeakerRows
            speakers={speakers}
            onChange={setSpeakers}
            minRows={1}
            hint={`the diarizer looks for exactly ${speakers.length} ${
              speakers.length === 1 ? 'voice' : 'voices'
            }, blank rows included`}
          />
        </div>
      )}
    </div>

    {suggestedClipMinutes !== null && (
      <ClipField
        enabled={clipEnabled}
        minutes={clipMinutes}
        onEnabledChange={handleClipEnabledChange}
        onMinutesChange={handleClipMinutesChange}
        suggestedMinutes={suggestedClipMinutes}
        genreLabel={genreLabel}
        sourceSeconds={sourceTab === 'youtube' ? (probe?.duration_seconds ?? null) : null}
      />
    )}

    {/* Submit Bar */}
    <div className="flex items-center justify-between gap-3 border-t pt-4">
      <span className="font-mono text-xs text-muted-foreground">
        ⚡ 8 cores parallel processing on worker thread
      </span>

      {/* Enabled while another job runs: a new one simply waits its turn in the queue. */}
      <Button
        disabled={
          isSubmitting ||
          !episodeTitle.trim() ||
          (clipEnabled && !clipValid) ||
          (sourceTab === 'file' ? !selectedFile : !probe || isProbing || probeOverLimit)
        }
        onClick={sourceTab === 'file' ? handleStartIngestion : handleStartYoutubeIngestion}
        className="gap-2"
      >
        {isSubmitting ? (
          <Spinner className="size-4" />
        ) : queueBusy ? (
          <RiListCheck2 className="size-4" />
        ) : (
          <RiUploadCloud2Line className="size-4" />
        )}
        {isSubmitting ? 'Sending…' : queueBusy ? 'Add to queue' : 'Start ingestion'}
      </Button>
    </div>
  </div>
}
