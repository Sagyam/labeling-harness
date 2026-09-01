import { useEffect, useState } from 'react'
import {
  RiDeleteBin6Line,
  RiErrorWarningLine,
  RiMicLine,
  RiPauseFill,
  RiPlayFill,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Chip } from '@/components/Chip'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Progress } from '@/components/ui/progress'
import { Spinner } from '@/components/ui/spinner'
import { cn } from '@/lib/utils'
import { api, resolveUrl } from '@/services/api'
import type { EpisodeSegmentSummary, EpisodeSummary } from '@/types'

interface EpisodeManagerModalProps {
  isOpen: boolean
  onClose: () => void
  onDataChanged: () => void
}

type PendingDelete =
  | { kind: 'episode'; episode: EpisodeSummary }
  | { kind: 'segment'; segment: EpisodeSegmentSummary }

export function EpisodeManagerModal({
  isOpen,
  onClose,
  onDataChanged,
}: EpisodeManagerModalProps) {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedEpisodeId, setExpandedEpisodeId] = useState<number | null>(null)
  const [segments, setSegments] = useState<EpisodeSegmentSummary[]>([])
  const [segmentsLoading, setSegmentsLoading] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null)
  const [playingAudioUrl, setPlayingAudioUrl] = useState<string | null>(null)
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null)

  const loadEpisodes = async () => {
    setLoading(true)
    setError(null)
    try {
      setEpisodes(await api.listEpisodes())
    } catch (err: any) {
      setError(err.message || 'Failed to load episodes')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      loadEpisodes()
    } else if (audioElement) {
      audioElement.pause()
      setAudioElement(null)
      setPlayingAudioUrl(null)
    }
  }, [isOpen])

  const toggleExpand = async (epId: number) => {
    if (expandedEpisodeId === epId) {
      setExpandedEpisodeId(null)
      setSegments([])
      return
    }

    setExpandedEpisodeId(epId)
    setSegmentsLoading(true)
    try {
      setSegments(await api.listEpisodeSegments(epId))
    } catch (err: any) {
      setError(err.message || 'Failed to load episode segments')
    } finally {
      setSegmentsLoading(false)
    }
  }

  const deleteEpisode = async (ep: EpisodeSummary) => {
    setDeletingId(`ep-${ep.id}`)
    try {
      await api.deleteEpisode(ep.id)
      setEpisodes((prev) => prev.filter((e) => e.id !== ep.id))
      if (expandedEpisodeId === ep.id) {
        setExpandedEpisodeId(null)
        setSegments([])
      }
      toast.info(`Deleted episode ${ep.external_id}`)
      onDataChanged()
    } catch (err: any) {
      toast.error(`Failed to delete episode: ${err.message}`)
    } finally {
      setDeletingId(null)
      setPendingDelete(null)
    }
  }

  const deleteSegment = async (seg: EpisodeSegmentSummary) => {
    setDeletingId(`seg-${seg.id}`)
    try {
      await api.deleteSegment(seg.id)
      setSegments((prev) => prev.filter((s) => s.id !== seg.id))
      setEpisodes((prev) =>
        prev.map((ep) =>
          ep.id === expandedEpisodeId
            ? { ...ep, segment_count: Math.max(0, ep.segment_count - 1) }
            : ep,
        ),
      )
      toast.info(`Deleted segment ${seg.external_id}`)
      onDataChanged()
    } catch (err: any) {
      toast.error(`Failed to delete segment: ${err.message}`)
    } finally {
      setDeletingId(null)
      setPendingDelete(null)
    }
  }

  const handleTogglePlayAudio = (url: string) => {
    const fullUrl = resolveUrl(url)
    if (playingAudioUrl === fullUrl && audioElement) {
      audioElement.pause()
      setAudioElement(null)
      setPlayingAudioUrl(null)
      return
    }

    audioElement?.pause()

    const audio = new Audio(fullUrl)
    audio.onended = () => {
      setPlayingAudioUrl(null)
      setAudioElement(null)
    }
    audio.play().catch((e) => console.error('Audio play error:', e))
    setAudioElement(audio)
    setPlayingAudioUrl(fullUrl)
  }

  return (
    <>
      <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
        <DialogContent className="max-h-[88vh] grid-rows-[auto_1fr_auto] overflow-hidden sm:max-w-4xl">
          <DialogHeader>
            <DialogTitle>Episodes &amp; segments</DialogTitle>
            <DialogDescription>
              Inspect ingested episodes, preview clips and remove unwanted segments.
            </DialogDescription>
          </DialogHeader>

          <div className="scrollbar-thin flex min-h-0 flex-col gap-3 overflow-y-auto">
            {error && (
              <Alert variant="destructive">
                <RiErrorWarningLine />
                <AlertTitle>Could not load episodes</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {loading ? (
              <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground">
                <Spinner /> Loading episodes…
              </div>
            ) : episodes.length === 0 ? (
              <Empty className="py-16">
                <EmptyHeader>
                  <EmptyMedia variant="icon">
                    <RiMicLine />
                  </EmptyMedia>
                  <EmptyTitle>No episodes ingested yet</EmptyTitle>
                  <EmptyDescription>
                    Upload podcast audio with the Ingest button to generate segments and hypotheses.
                  </EmptyDescription>
                </EmptyHeader>
              </Empty>
            ) : (
              episodes.map((ep) => {
                const percentDone =
                  ep.segment_count > 0
                    ? Math.round((ep.labeled_count / ep.segment_count) * 100)
                    : 0
                const isExpanded = expandedEpisodeId === ep.id
                const isDeleting = deletingId === `ep-${ep.id}`

                return (
                  <Collapsible
                    key={ep.id}
                    open={isExpanded}
                    onOpenChange={() => toggleExpand(ep.id)}
                    className="border"
                  >
                    <div className="flex flex-wrap items-center gap-4 p-3">
                      <div className="min-w-56 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <h4 className="font-heading text-sm font-semibold tracking-wide">
                            {ep.title || ep.external_id}
                          </h4>
                          <Chip className="bg-info/15 text-info">{ep.external_id}</Chip>
                        </div>
                        <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-muted-foreground">
                          <span>Show: {ep.show_id || 'default'}</span>
                          {ep.duration_seconds && (
                            <span>
                              Duration: {Math.round(ep.duration_seconds / 60)}m{' '}
                              {Math.round(ep.duration_seconds % 60)}s
                            </span>
                          )}
                          <span>{ep.segment_count} segments</span>
                          <span>{ep.labeled_count} labeled</span>
                        </div>
                      </div>

                      <div className="w-28">
                        <div className="mb-1 text-right font-mono text-xs tabular-nums">
                          {percentDone}%
                        </div>
                        <Progress value={percentDone} />
                      </div>

                      <CollapsibleTrigger asChild>
                        <Button variant="outline" size="sm">
                          {isExpanded ? 'Hide segments' : `View ${ep.segment_count} segments`}
                        </Button>
                      </CollapsibleTrigger>

                      <Button
                        variant="destructive"
                        size="sm"
                        disabled={isDeleting}
                        onClick={() => setPendingDelete({ kind: 'episode', episode: ep })}
                      >
                        {isDeleting ? <Spinner data-icon="inline-start" /> : <RiDeleteBin6Line data-icon="inline-start" />}
                        Delete
                      </Button>
                    </div>

                    <CollapsibleContent className="border-t bg-muted/20 p-3">
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <span className="font-heading text-[11px] font-semibold tracking-widest text-muted-foreground uppercase">
                          Segments in {ep.external_id} ({segments.length})
                        </span>
                        <span className="text-xs text-muted-foreground">
                          Preview audio or delete unusable slices
                        </span>
                      </div>

                      {segmentsLoading ? (
                        <div className="flex items-center justify-center gap-2 py-6 text-sm text-muted-foreground">
                          <Spinner /> Loading segments…
                        </div>
                      ) : segments.length === 0 ? (
                        <div className="py-6 text-center text-sm text-muted-foreground">
                          No segments found in this episode.
                        </div>
                      ) : (
                        <div className="scrollbar-thin flex max-h-80 flex-col gap-1.5 overflow-y-auto">
                          {segments.map((seg) => {
                            const isPlaying = playingAudioUrl === resolveUrl(seg.audio_url)
                            const isSegDeleting = deletingId === `seg-${seg.id}`
                            const hasHindiIntrusion = seg.flags.includes('hindi_intrusion')

                            return (
                              <div
                                key={seg.id}
                                className={cn(
                                  'flex items-center gap-3 bg-background p-2 ring-1 ring-foreground/5',
                                  hasHindiIntrusion && 'ring-warning/40',
                                )}
                              >
                                <Button
                                  variant={isPlaying ? 'default' : 'ghost'}
                                  size="icon-xs"
                                  onClick={() => handleTogglePlayAudio(seg.audio_url)}
                                  aria-label={isPlaying ? 'Pause preview' : 'Play preview'}
                                >
                                  {isPlaying ? <RiPauseFill /> : <RiPlayFill />}
                                </Button>

                                <div className="w-36 shrink-0">
                                  <div className="truncate font-mono text-xs">
                                    {seg.external_id}
                                  </div>
                                  <div className="font-mono text-[10px] text-muted-foreground tabular-nums">
                                    {seg.start_time.toFixed(1)}–{seg.end_time.toFixed(1)}s (
                                    {seg.duration_seconds.toFixed(1)}s)
                                  </div>
                                </div>

                                <div className="min-w-0 flex-1 truncate font-devanagari text-sm">
                                  {seg.seed_text || (
                                    <span className="text-muted-foreground italic">
                                      No hypothesis
                                    </span>
                                  )}
                                </div>

                                <div className="flex shrink-0 items-center gap-1.5">
                                  {hasHindiIntrusion && (
                                    <Chip className="bg-warning/15 text-warning">
                                      hindi intrusion
                                    </Chip>
                                  )}
                                  {seg.cmi !== null && <Chip>CMI {seg.cmi}%</Chip>}
                                </div>

                                <Button
                                  variant="ghost"
                                  size="icon-xs"
                                  disabled={isSegDeleting}
                                  className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                                  onClick={() => setPendingDelete({ kind: 'segment', segment: seg })}
                                  aria-label="Delete segment"
                                >
                                  {isSegDeleting ? <Spinner /> : <RiDeleteBin6Line />}
                                </Button>
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </CollapsibleContent>
                  </Collapsible>
                )
              })
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => !open && setPendingDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {pendingDelete?.kind === 'episode' ? 'Delete this episode?' : 'Delete this segment?'}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {pendingDelete?.kind === 'episode' ? (
                <>
                  <span className="font-mono">
                    {pendingDelete.episode.title || pendingDelete.episode.external_id}
                  </span>{' '}
                  and all {pendingDelete.episode.segment_count} of its segments, annotations and
                  audio files will be permanently removed.
                </>
              ) : pendingDelete?.kind === 'segment' ? (
                <>
                  <span className="font-mono">{pendingDelete.segment.external_id}</span> will be
                  permanently removed from the queue and storage.
                </>
              ) : null}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (pendingDelete?.kind === 'episode') deleteEpisode(pendingDelete.episode)
                else if (pendingDelete?.kind === 'segment') deleteSegment(pendingDelete.segment)
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
