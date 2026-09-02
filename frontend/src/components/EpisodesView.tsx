import { useEffect, useMemo, useRef, useState } from 'react'
import {
  RiDeleteBin6Line,
  RiEditLine,
  RiErrorWarningLine,
  RiFilterLine,
  RiFolderMusicLine,
  RiInboxLine,
  RiMicLine,
  RiPauseFill,
  RiPlayFill,
  RiRefreshLine,
  RiSearchLine,
  RiUploadCloud2Line,
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
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Spinner } from '@/components/ui/spinner'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { api, resolveUrl } from '@/services/api'
import type { EpisodeSegmentSummary, EpisodeSummary } from '@/types'

interface EpisodesViewProps {
  onOpenEditor: (taskId: number) => void
  onTriageEpisode: (episodeExternalId: string) => void
  onOpenIngest: () => void
  onDataChanged: () => void
}

type PendingDelete =
  | { kind: 'episode'; episode: EpisodeSummary }
  | { kind: 'segment'; segment: EpisodeSegmentSummary }

export function EpisodesView({
  onOpenEditor,
  onTriageEpisode,
  onOpenIngest,
  onDataChanged,
}: EpisodesViewProps) {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Selection & Search
  const [selectedEpisodeId, setSelectedEpisodeId] = useState<number | null>(null)
  const [episodeSearch, setEpisodeSearch] = useState('')
  const [splitFilter, setSplitFilter] = useState<string>('all')

  // Segments state
  const [segments, setSegments] = useState<EpisodeSegmentSummary[]>([])
  const [segmentsLoading, setSegmentsLoading] = useState(false)
  const [segmentSearch, setSegmentSearch] = useState('')
  const [segmentStatusFilter, setSegmentStatusFilter] = useState<string>('all')

  // Deletion state
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)

  // Audio preview state
  const [playingUrl, setPlayingUrl] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const loadEpisodes = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.listEpisodes()
      setEpisodes(data)
      if (data.length > 0 && selectedEpisodeId === null) {
        setSelectedEpisodeId(data[0].id)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load episodes')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadEpisodes()
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
    }
  }, [])

  // Load segments whenever selectedEpisodeId changes
  useEffect(() => {
    if (!selectedEpisodeId) {
      setSegments([])
      return
    }

    let isMounted = true
    setSegmentsLoading(true)

    // Stop currently playing audio
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
      setPlayingUrl(null)
    }

    api
      .listEpisodeSegments(selectedEpisodeId)
      .then((data) => {
        if (isMounted) setSegments(data)
      })
      .catch((err) => {
        if (isMounted) toast.error(err.message || 'Failed to load segments')
      })
      .finally(() => {
        if (isMounted) setSegmentsLoading(false)
      })

    return () => {
      isMounted = false
    }
  }, [selectedEpisodeId])

  const selectedEpisode = useMemo(
    () => episodes.find((e) => e.id === selectedEpisodeId),
    [episodes, selectedEpisodeId],
  )

  const filteredEpisodes = useMemo(() => {
    return episodes.filter((ep) => {
      const matchesSearch =
        episodeSearch.trim() === '' ||
        ep.external_id.toLowerCase().includes(episodeSearch.toLowerCase()) ||
        (ep.title && ep.title.toLowerCase().includes(episodeSearch.toLowerCase())) ||
        (ep.show_id && ep.show_id.toLowerCase().includes(episodeSearch.toLowerCase()))

      const matchesSplit = splitFilter === 'all' || ep.split === splitFilter
      return matchesSearch && matchesSplit
    })
  }, [episodes, episodeSearch, splitFilter])

  const filteredSegments = useMemo(() => {
    return segments.filter((seg) => {
      const matchesSearch =
        segmentSearch.trim() === '' ||
        seg.external_id.toLowerCase().includes(segmentSearch.toLowerCase()) ||
        (seg.seed_text && seg.seed_text.toLowerCase().includes(segmentSearch.toLowerCase()))

      let matchesStatus = true
      if (segmentStatusFilter === 'pending') {
        matchesStatus = seg.task_status === 'pending' || seg.task_status === 'in_progress'
      } else if (segmentStatusFilter === 'labeled') {
        matchesStatus = seg.pipeline_status === 'labeled'
      } else if (segmentStatusFilter === 'unlabeled') {
        matchesStatus = seg.pipeline_status !== 'labeled'
      }

      return matchesSearch && matchesStatus
    })
  }, [segments, segmentSearch, segmentStatusFilter])

  const handleTogglePlay = (url: string) => {
    const fullUrl = resolveUrl(url)
    if (playingUrl === fullUrl && audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
      setPlayingUrl(null)
      return
    }

    if (audioRef.current) {
      audioRef.current.pause()
    }

    const audio = new Audio(fullUrl)
    audio.onended = () => {
      setPlayingUrl(null)
      audioRef.current = null
    }
    audio.play().catch((e) => {
      console.error('Audio play error:', e)
      toast.error('Could not play audio clip')
    })
    audioRef.current = audio
    setPlayingUrl(fullUrl)
  }

  const handleDeleteEpisode = async (ep: EpisodeSummary) => {
    setIsDeleting(true)
    try {
      await api.deleteEpisode(ep.id)
      setEpisodes((prev) => prev.filter((e) => e.id !== ep.id))
      if (selectedEpisodeId === ep.id) {
        const remaining = episodes.filter((e) => e.id !== ep.id)
        setSelectedEpisodeId(remaining.length > 0 ? remaining[0].id : null)
      }
      toast.success(`Deleted episode ${ep.external_id}`)
      onDataChanged()
    } catch (err: any) {
      toast.error(`Delete failed: ${err.message}`)
    } finally {
      setIsDeleting(false)
      setPendingDelete(null)
    }
  }

  const handleDeleteSegment = async (seg: EpisodeSegmentSummary) => {
    setIsDeleting(true)
    try {
      await api.deleteSegment(seg.id)
      setSegments((prev) => prev.filter((s) => s.id !== seg.id))
      setEpisodes((prev) =>
        prev.map((ep) =>
          ep.id === selectedEpisodeId
            ? { ...ep, segment_count: Math.max(0, ep.segment_count - 1) }
            : ep,
        ),
      )
      toast.success(`Deleted segment ${seg.external_id}`)
      onDataChanged()
    } catch (err: any) {
      toast.error(`Delete failed: ${err.message}`)
    } finally {
      setIsDeleting(false)
      setPendingDelete(null)
    }
  }

  const getSplitBadgeColor = (split: string) => {
    switch (split) {
      case 'train':
        return 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/20'
      case 'val':
        return 'bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/20'
      case 'test':
        return 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/20'
      default:
        return 'bg-muted text-muted-foreground'
    }
  }

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden bg-background">
      {/* LEFT PANE: Episode Navigator */}
      <div className="flex w-96 shrink-0 flex-col border-r bg-card/30">
        <div className="border-b p-3 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <RiFolderMusicLine className="size-5 text-primary" />
              <h2 className="font-heading text-sm font-semibold tracking-wide">Episodes</h2>
              <Badge variant="secondary" className="font-mono text-xs">
                {episodes.length}
              </Badge>
            </div>
            <div className="flex items-center gap-1.5">
              <Button variant="outline" size="sm" onClick={onOpenIngest} className="h-8 gap-1">
                <RiUploadCloud2Line className="size-3.5" />
                Ingest
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={loadEpisodes}
                title="Refresh episodes"
                aria-label="Refresh episodes"
              >
                <RiRefreshLine className="size-4" />
              </Button>
            </div>
          </div>

          <div className="relative">
            <RiSearchLine className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
            <Input
              value={episodeSearch}
              onChange={(e) => setEpisodeSearch(e.target.value)}
              placeholder="Search by title or external ID…"
              className="h-9 pl-8 text-xs"
            />
          </div>

          <Tabs value={splitFilter} onValueChange={setSplitFilter}>
            <TabsList className="grid w-full grid-cols-5 h-7">
              <TabsTrigger value="all" className="text-[11px]">
                All
              </TabsTrigger>
              <TabsTrigger value="train" className="text-[11px]">
                Train
              </TabsTrigger>
              <TabsTrigger value="val" className="text-[11px]">
                Val
              </TabsTrigger>
              <TabsTrigger value="test" className="text-[11px]">
                Test
              </TabsTrigger>
              <TabsTrigger value="unassigned" className="text-[11px]">
                Unassigned
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        {/* Episode list */}
        <div className="scrollbar-thin flex-1 overflow-y-auto p-2 space-y-1.5">
          {error && (
            <Alert variant="destructive" className="m-2">
              <RiErrorWarningLine />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription className="text-xs">{error}</AlertDescription>
            </Alert>
          )}

          {loading ? (
            <div className="flex flex-col items-center justify-center gap-2 py-16 text-xs text-muted-foreground">
              <Spinner /> Loading episodes…
            </div>
          ) : filteredEpisodes.length === 0 ? (
            <div className="py-12 text-center text-xs text-muted-foreground px-4">
              {episodeSearch || splitFilter !== 'all'
                ? 'No episodes match the selected filter.'
                : 'No episodes ingested yet.'}
            </div>
          ) : (
            filteredEpisodes.map((ep) => {
              const isSelected = selectedEpisodeId === ep.id
              const percent =
                ep.segment_count > 0 ? Math.round((ep.labeled_count / ep.segment_count) * 100) : 0

              return (
                <div
                  key={ep.id}
                  onClick={() => setSelectedEpisodeId(ep.id)}
                  className={cn(
                    'group relative cursor-pointer rounded-lg border p-3 transition-colors',
                    isSelected
                      ? 'border-primary/50 bg-primary/5 ring-1 ring-primary/20'
                      : 'border-border/60 bg-card hover:border-primary/30 hover:bg-muted/30',
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <h4 className="truncate font-heading text-xs font-semibold">
                        {ep.title || ep.external_id}
                      </h4>
                      <div className="mt-1 flex items-center gap-1.5 flex-wrap">
                        <span className="font-mono text-[11px] text-muted-foreground">
                          {ep.external_id}
                        </span>
                        <span
                          className={cn(
                            'rounded px-1.5 py-0.2 font-mono text-[9px] uppercase border',
                            getSplitBadgeColor(ep.split),
                          )}
                        >
                          {ep.split}
                        </span>
                      </div>
                    </div>

                    <Button
                      variant="ghost"
                      size="icon-xs"
                      className="opacity-0 group-hover:opacity-100 text-destructive hover:bg-destructive/10"
                      onClick={(e) => {
                        e.stopPropagation()
                        setPendingDelete({ kind: 'episode', episode: ep })
                      }}
                      title="Delete episode"
                      aria-label="Delete episode"
                    >
                      <RiDeleteBin6Line className="size-3.5" />
                    </Button>
                  </div>

                  <div className="mt-2.5 flex items-center justify-between text-[11px] text-muted-foreground font-mono">
                    <span>
                      {ep.duration_seconds ? `${Math.round(ep.duration_seconds / 60)}m` : '--'}{' '}
                      &middot; {ep.segment_count} segs
                    </span>
                    <span className="tabular-nums font-semibold text-foreground">
                      {ep.labeled_count}/{ep.segment_count} ({percent}%)
                    </span>
                  </div>

                  <Progress value={percent} className="mt-1.5 h-1.5" />
                </div>
              )
            })
          )}
        </div>
      </div>

      {/* RIGHT PANE: Segments Detail Explorer */}
      <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
        {selectedEpisode ? (
          <>
            {/* Header / Hero for selected episode */}
            <div className="border-b bg-card/40 p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2.5 flex-wrap">
                    <h3 className="font-heading text-lg font-semibold tracking-wide">
                      {selectedEpisode.title || selectedEpisode.external_id}
                    </h3>
                    <Chip className="bg-primary/10 text-primary font-mono text-xs">
                      {selectedEpisode.external_id}
                    </Chip>
                    <span
                      className={cn(
                        'rounded-md px-2 py-0.5 font-mono text-xs uppercase border font-medium',
                        getSplitBadgeColor(selectedEpisode.split),
                      )}
                    >
                      {selectedEpisode.split} split
                    </span>
                  </div>

                  <div className="mt-1 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-muted-foreground">
                    <span>Show: {selectedEpisode.show_id || 'default'}</span>
                    {selectedEpisode.duration_seconds && (
                      <span>
                        Duration:{' '}
                        {Math.floor(selectedEpisode.duration_seconds / 60)}m{' '}
                        {Math.round(selectedEpisode.duration_seconds % 60)}s (
                        {(selectedEpisode.duration_seconds / 3600).toFixed(2)}h)
                      </span>
                    )}
                    <span>
                      Segments: <strong>{selectedEpisode.segment_count}</strong>
                    </span>
                    <span>
                      Labeled: <strong>{selectedEpisode.labeled_count}</strong>
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    variant="default"
                    size="sm"
                    className="gap-1.5"
                    onClick={() => onTriageEpisode(selectedEpisode.external_id)}
                  >
                    <RiInboxLine className="size-4" />
                    Triage Episode
                  </Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    className="gap-1.5"
                    onClick={() =>
                      setPendingDelete({ kind: 'episode', episode: selectedEpisode })
                    }
                  >
                    <RiDeleteBin6Line className="size-4" />
                    Delete
                  </Button>
                </div>
              </div>

              {/* Segment filter controls */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t">
                <div className="flex items-center gap-2 flex-1 min-w-[240px] max-w-md">
                  <RiSearchLine className="size-4 text-muted-foreground shrink-0" />
                  <Input
                    value={segmentSearch}
                    onChange={(e) => setSegmentSearch(e.target.value)}
                    placeholder="Search segments by text or ID…"
                    className="h-8 text-xs"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <RiFilterLine className="size-4 text-muted-foreground" />
                  <Tabs
                    value={segmentStatusFilter}
                    onValueChange={setSegmentStatusFilter}
                    className="h-8"
                  >
                    <TabsList className="h-7">
                      <TabsTrigger value="all" className="text-xs px-2.5">
                        All ({segments.length})
                      </TabsTrigger>
                      <TabsTrigger value="pending" className="text-xs px-2.5">
                        Pending
                      </TabsTrigger>
                      <TabsTrigger value="labeled" className="text-xs px-2.5">
                        Labeled
                      </TabsTrigger>
                      <TabsTrigger value="unlabeled" className="text-xs px-2.5">
                        Unlabeled
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>
              </div>
            </div>

            {/* Segments list content */}
            <div className="scrollbar-thin flex-1 overflow-y-auto p-4 space-y-2">
              {segmentsLoading ? (
                <div className="flex flex-col items-center justify-center gap-2 py-24 text-sm text-muted-foreground">
                  <Spinner /> Loading segment audio and hypotheses…
                </div>
              ) : filteredSegments.length === 0 ? (
                <Empty className="py-20">
                  <EmptyHeader>
                    <EmptyMedia variant="icon">
                      <RiMicLine />
                    </EmptyMedia>
                    <EmptyTitle>No matching segments</EmptyTitle>
                    <EmptyDescription>
                      {segmentSearch
                        ? 'Try clearing your segment search filter.'
                        : 'No segments found for this episode.'}
                    </EmptyDescription>
                  </EmptyHeader>
                </Empty>
              ) : (
                filteredSegments.map((seg) => {
                  const isPlaying = playingUrl === resolveUrl(seg.audio_url)
                  const isLabeled = seg.pipeline_status === 'labeled'
                  const hasActiveTask =
                    seg.task_status === 'pending' || seg.task_status === 'in_progress'

                  return (
                    <Card
                      key={seg.id}
                      className={cn(
                        'transition-all hover:border-primary/40',
                        isPlaying && 'ring-1 ring-primary border-primary/60 bg-primary/5',
                      )}
                    >
                      <CardContent className="p-3">
                        <div className="flex flex-wrap items-center gap-3">
                          {/* Play / Pause button */}
                          <Button
                            variant={isPlaying ? 'default' : 'secondary'}
                            size="icon-sm"
                            className="shrink-0 rounded-full"
                            onClick={() => handleTogglePlay(seg.audio_url)}
                            aria-label={isPlaying ? 'Pause segment audio' : 'Play segment audio'}
                          >
                            {isPlaying ? <RiPauseFill /> : <RiPlayFill />}
                          </Button>

                          {/* Identifier and Time Range */}
                          <div className="w-40 shrink-0">
                            <div className="font-mono text-xs font-semibold truncate">
                              {seg.external_id}
                            </div>
                            <div className="font-mono text-[10px] text-muted-foreground tabular-nums">
                              {seg.start_time.toFixed(1)}s &ndash; {seg.end_time.toFixed(1)}s (
                              {seg.duration_seconds.toFixed(1)}s)
                            </div>
                          </div>

                          {/* Transcript / Hypothesis text */}
                          <div className="min-w-0 flex-1">
                            {seg.seed_text ? (
                              <p className="font-devanagari text-sm leading-relaxed truncate">
                                {seg.seed_text}
                              </p>
                            ) : (
                              <span className="text-xs text-muted-foreground italic">
                                No transcript hypothesis
                              </span>
                            )}
                          </div>

                          {/* Indicators / Chips */}
                          <div className="flex shrink-0 items-center gap-1.5 flex-wrap">
                            {seg.cmi !== null && (
                              <Badge variant="outline" className="font-mono text-[10px]">
                                CMI {seg.cmi}%
                              </Badge>
                            )}

                            {isLabeled ? (
                              <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/20 text-[10px]">
                                Labeled
                              </Badge>
                            ) : hasActiveTask ? (
                              <Badge variant="secondary" className="text-[10px]">
                                {seg.task_status}
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="text-[10px] text-muted-foreground">
                                {seg.pipeline_status}
                              </Badge>
                            )}

                            {seg.flags.map((flag) => (
                              <Chip key={flag} className="bg-warning/15 text-warning text-[10px]">
                                {flag}
                              </Chip>
                            ))}
                          </div>

                          {/* Action Buttons */}
                          <div className="flex shrink-0 items-center gap-1">
                            {seg.task_id && (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-7 px-2.5 gap-1 text-xs"
                                onClick={() => onOpenEditor(seg.task_id!)}
                                title="Open this segment in Editor"
                              >
                                <RiEditLine className="size-3.5" />
                                Edit
                              </Button>
                            )}

                            <Button
                              variant="ghost"
                              size="icon-xs"
                              className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                              onClick={() =>
                                setPendingDelete({ kind: 'segment', segment: seg })
                              }
                              title="Delete segment"
                              aria-label="Delete segment"
                            >
                              <RiDeleteBin6Line className="size-3.5" />
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })
              )}
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center p-8 text-center">
            <Empty>
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <RiFolderMusicLine />
                </EmptyMedia>
                <EmptyTitle>No episode selected</EmptyTitle>
                <EmptyDescription>
                  Select an episode from the list on the left to inspect its segments, preview audio,
                  or jump into the editor.
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          </div>
        )}
      </div>

      {/* Delete Confirmation Alert Dialog */}
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
                  <span className="font-mono font-medium">
                    {pendingDelete.episode.title || pendingDelete.episode.external_id}
                  </span>{' '}
                  and all {pendingDelete.episode.segment_count} of its segments, annotations, and
                  audio clips will be permanently removed.
                </>
              ) : pendingDelete?.kind === 'segment' ? (
                <>
                  Segment{' '}
                  <span className="font-mono font-medium">
                    {pendingDelete.segment.external_id}
                  </span>{' '}
                  will be permanently removed from storage and queues.
                </>
              ) : null}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={isDeleting}
              onClick={() => {
                if (pendingDelete?.kind === 'episode') handleDeleteEpisode(pendingDelete.episode)
                else if (pendingDelete?.kind === 'segment')
                  handleDeleteSegment(pendingDelete.segment)
              }}
            >
              {isDeleting ? <Spinner data-icon="inline-start" /> : 'Delete'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
