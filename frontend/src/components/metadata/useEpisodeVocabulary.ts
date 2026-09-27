import { useEffect, useState } from 'react'

import { api } from '@/services/api'
import type { EpisodeVocabulary } from '@/types'

let cached: Promise<EpisodeVocabulary> | null = null

/** The closed genre and topic lists (D102), fetched once per page load and shared. */
export function useEpisodeVocabulary(): EpisodeVocabulary | null {
  const [vocabulary, setVocabulary] = useState<EpisodeVocabulary | null>(null)

  useEffect(() => {
    let live = true
    cached ??= api.getEpisodeVocabulary().catch((err) => {
      cached = null
      throw err
    })
    cached.then((v) => live && setVocabulary(v)).catch(() => undefined)
    return () => {
      live = false
    }
  }, [])

  return vocabulary
}
