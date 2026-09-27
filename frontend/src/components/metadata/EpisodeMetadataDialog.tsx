import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Spinner } from '@/components/ui/spinner'
import { api } from '@/services/api'
import type { EpisodeMetadata, EpisodeSummary } from '@/types'

import { GenreTopicFields } from './GenreTopicFields'
import { SpeakerRows } from './SpeakerRows'
import { type SpeakerDraft, draftsFromRows, rowsFromDrafts } from './speakers'
import { useEpisodeVocabulary } from './useEpisodeVocabulary'

interface EpisodeMetadataDialogProps {
  /** The episode being edited; null closes the dialog. */
  episode: EpisodeSummary | null
  onClose: () => void
  onSaved: (episodeId: number, metadata: EpisodeMetadata) => void
}

/**
 * Edit an episode's genre, topic and declared speakers after ingest (D102). One save is one
 * audited write; a save that changes nothing writes nothing.
 */
export function EpisodeMetadataDialog({ episode, onClose, onSaved }: EpisodeMetadataDialogProps) {
  const vocabulary = useEpisodeVocabulary()
  const [loaded, setLoaded] = useState<EpisodeMetadata | null>(null)
  const [genre, setGenre] = useState('')
  const [topic, setTopic] = useState('')
  const [speakers, setSpeakers] = useState<SpeakerDraft[]>([])
  const [saving, setSaving] = useState(false)

  const episodeId = episode?.id ?? null
  // The load effect must not re-run because the parent re-rendered with a new closure.
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose

  useEffect(() => {
    if (episodeId === null) return
    let live = true
    setLoaded(null)
    api
      .getEpisodeMetadata(episodeId)
      .then((m) => {
        if (!live) return
        setLoaded(m)
        setGenre(m.genre ?? '')
        setTopic(m.topic ?? '')
        setSpeakers(draftsFromRows(m.speakers))
      })
      .catch((err) => {
        toast.error('Failed to load episode metadata', { description: String(err) })
        onCloseRef.current()
      })
    return () => {
      live = false
    }
  }, [episodeId])

  const save = async () => {
    if (episodeId === null) return
    setSaving(true)
    try {
      const saved = await api.updateEpisodeMetadata(episodeId, {
        genre: genre || null,
        topic: topic || null,
        speakers: rowsFromDrafts(speakers),
      })
      toast.success('Episode metadata saved')
      onSaved(episodeId, saved)
      onClose()
    } catch (err) {
      toast.error('Could not save', { description: String(err) })
    } finally {
      setSaving(false)
    }
  }

  const topicNote =
    loaded?.topic && topic === loaded.topic && loaded.topic_source === 'llm'
      ? 'Topic chosen by the classifier; picking another marks it manual.'
      : null

  return (
    <Dialog open={episode !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Episode metadata</DialogTitle>
          <DialogDescription className="truncate">
            {episode?.title || episode?.external_id}
          </DialogDescription>
        </DialogHeader>

        {loaded === null ? (
          <div className="flex items-center justify-center gap-2 py-10 text-muted-foreground">
            <Spinner /> Loading…
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            <GenreTopicFields
              idPrefix="edit"
              vocabulary={vocabulary}
              genre={genre}
              topic={topic}
              onGenreChange={setGenre}
              onTopicChange={setTopic}
              topicBlankLabel="Untagged"
            />
            {topicNote && <p className="-mt-2 text-xs text-muted-foreground">{topicNote}</p>}
            <SpeakerRows
              speakers={speakers}
              onChange={setSpeakers}
              minRows={0}
              hint="changing the count does not re-diarize"
            />
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={save} disabled={saving || loaded === null}>
            {saving && <Spinner />}
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
