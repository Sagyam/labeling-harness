import { humanize } from '@/components/analytics/primitives'
import { Field, FieldDescription, FieldLabel } from '@/components/ui/field'
import type { EpisodeVocabulary } from '@/types'

import { NativeSelect } from './NativeSelect'

interface GenreTopicFieldsProps {
  vocabulary: EpisodeVocabulary | null
  genre: string
  topic: string
  onGenreChange: (genre: string) => void
  onTopicChange: (topic: string) => void
  /** What an empty topic means here: the classifier at ingest, nothing in the editor. */
  topicBlankLabel: string
  idPrefix: string
}

/**
 * Genre and topic pickers over the closed lists (D57, D102). A stored value that is off the list
 * is still shown, marked, so it can be seen and replaced; the backend refuses to save it.
 */
export function GenreTopicFields({
  vocabulary,
  genre,
  topic,
  onGenreChange,
  onTopicChange,
  topicBlankLabel,
  idPrefix,
}: GenreTopicFieldsProps) {
  const genres = vocabulary?.genres ?? []
  const topics = vocabulary?.topics ?? []
  const picked = genres.find((g) => g.value === genre)
  const genreOffList = vocabulary !== null && genre !== '' && !picked
  const topicOffList = vocabulary !== null && topic !== '' && !topics.includes(topic)

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Field>
        <FieldLabel htmlFor={`${idPrefix}-genre`}>Genre</FieldLabel>
        <NativeSelect
          id={`${idPrefix}-genre`}
          value={genre}
          disabled={vocabulary === null}
          onChange={(e) => onGenreChange(e.target.value)}
        >
          <option value="">Untagged</option>
          {genreOffList && <option value={genre}>{genre} (off the list)</option>}
          {genres.map((g) => (
            <option key={g.value} value={g.value} title={g.description}>
              {g.label}
            </option>
          ))}
        </NativeSelect>
        <FieldDescription className="text-xs">
          {genreOffList
            ? 'Not a genre any more: pick the format this recording is.'
            : (picked?.description ?? 'How the recording was made, not what it is about.')}
        </FieldDescription>
      </Field>
      <Field>
        <FieldLabel htmlFor={`${idPrefix}-topic`}>Topic</FieldLabel>
        <NativeSelect
          id={`${idPrefix}-topic`}
          value={topic}
          disabled={vocabulary === null}
          onChange={(e) => onTopicChange(e.target.value)}
        >
          <option value="">{topicBlankLabel}</option>
          {topicOffList && <option value={topic}>{topic} (off the list)</option>}
          {topics.map((t) => (
            <option key={t} value={t}>
              {humanize(t)}
            </option>
          ))}
        </NativeSelect>
        <FieldDescription className="text-xs">What the episode is about.</FieldDescription>
      </Field>
    </div>
  )
}
