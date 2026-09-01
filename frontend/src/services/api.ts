/**
 * Typed API client for the labeling harness backend review service.
 */

import {
  AcceptIn,
  BulkAcceptIn,
  BulkAcceptOut,
  DecisionOut,
  FlagIn,
  HealthResponse,
  LabelIn,
  PeaksPayload,
  QueueRow,
  SkipIn,
  StatsResponse,
  Task,
  TranslitOut,
  IngestEvent,
  IngestJobStatus,
  EpisodeSummary,
  EpisodeSegmentSummary,
} from '../types'

export const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? 'http://localhost:8000'

/**
 * Normalizes relative URLs (e.g. `/segments/1/audio`) to absolute backend URLs.
 */
export function resolveUrl(path?: string | null): string {
  if (!path) return ''
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path
  }
  return `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = resolveUrl(endpoint)
  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(options.body ? { 'Content-Type': 'application/json' } : {}),
    ...((options.headers as Record<string, string>) || {}),
  }

  const res = await fetch(url, {
    ...options,
    headers,
  })

  if (!res.ok) {
    let errorDetail = res.statusText
    try {
      const errJson = await res.json()
      errorDetail = errJson.detail || JSON.stringify(errJson)
    } catch {
      // ignore
    }
    const error = new Error(`API error ${res.status}: ${errorDetail}`) as Error & {
      status: number
      detail: string
    }
    error.status = res.status
    error.detail = errorDetail
    throw error
  }

  if (res.status === 204) return null as unknown as T

  return res.json() as Promise<T>
}

export const api = {
  getHealth: (): Promise<HealthResponse> => request<HealthResponse>('/health'),
  getStats: (): Promise<StatsResponse> => request<StatsResponse>('/stats'),

  getQueue: ({
    limit = 50,
    offset = 0,
    episode,
    min_priority,
    queue = 'review',
  }: {
    limit?: number
    offset?: number
    episode?: string
    min_priority?: number | null
    queue?: string
  } = {}): Promise<QueueRow[]> => {
    const params = new URLSearchParams()
    if (limit) params.set('limit', String(limit))
    if (offset) params.set('offset', String(offset))
    if (episode) params.set('episode', episode)
    if (min_priority !== undefined && min_priority !== null) {
      params.set('min_priority', String(min_priority))
    }
    if (queue) params.set('queue', queue)
    return request<QueueRow[]>(`/queue?${params.toString()}`)
  },

  getNextTask: ({
    queue = 'review',
    episode,
  }: {
    queue?: string
    episode?: string
  } = {}): Promise<Task> => {
    const params = new URLSearchParams()
    if (queue) params.set('queue', queue)
    if (episode) params.set('episode', episode)
    return request<Task>(`/tasks/next?${params.toString()}`)
  },

  getTask: (taskId: number): Promise<Task> => request<Task>(`/tasks/${taskId}`),

  getPeaks: (peaksUrlOrSegmentId: string | number): Promise<PeaksPayload> => {
    if (typeof peaksUrlOrSegmentId === 'number' || !String(peaksUrlOrSegmentId).includes('/')) {
      return request<PeaksPayload>(`/segments/${peaksUrlOrSegmentId}/peaks`)
    }
    return request<PeaksPayload>(String(peaksUrlOrSegmentId))
  },

  acceptTask: (taskId: number, body: AcceptIn = {}): Promise<DecisionOut> => {
    return request<DecisionOut>(`/tasks/${taskId}/accept`, {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },

  labelTask: (taskId: number, body: LabelIn): Promise<DecisionOut> => {
    return request<DecisionOut>(`/tasks/${taskId}/label`, {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },

  flagTask: (taskId: number, body: FlagIn): Promise<DecisionOut> => {
    return request<DecisionOut>(`/tasks/${taskId}/flag`, {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },

  skipTask: (taskId: number, body: SkipIn = {}): Promise<DecisionOut> => {
    return request<DecisionOut>(`/tasks/${taskId}/skip`, {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },

  bulkAccept: (body: BulkAcceptIn): Promise<BulkAcceptOut> => {
    return request<BulkAcceptOut>('/tasks/bulk-accept', {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },

  translit: (token: string, limit: number = 5): Promise<TranslitOut> => {
    return request<TranslitOut>('/translit', {
      method: 'POST',
      body: JSON.stringify({ token, limit }),
    })
  },

  translitChoice: (token: string, devanagari: string): Promise<TranslitOut> => {
    return request<TranslitOut>('/translit/choice', {
      method: 'POST',
      body: JSON.stringify({ token, devanagari }),
    })
  },

  startIngest: async (
    formData: FormData
  ): Promise<{ job_id: string; status: string; episode_id: string; title: string }> => {
    const url = resolveUrl('/ingest')
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) {
      let detail = res.statusText
      try {
        const errJson = await res.json()
        detail = errJson.detail || JSON.stringify(errJson)
      } catch {
        // ignore
      }
      throw new Error(`Ingest failed: ${detail}`)
    }
    return res.json()
  },

  getIngestStatus: (jobId: string): Promise<IngestJobStatus> => {
    return request<IngestJobStatus>(`/ingest/${jobId}`)
  },

  subscribeIngestEvents: (
    jobId: string,
    onEvent: (event: IngestEvent) => void,
    onError?: (err: any) => void
  ): (() => void) => {
    const url = resolveUrl(`/ingest/${jobId}/events`)
    const eventSource = new EventSource(url)

    eventSource.onmessage = (e) => {
      try {
        const parsed = JSON.parse(e.data) as IngestEvent
        onEvent(parsed)
      } catch (err) {
        console.error('Failed to parse SSE event:', err)
      }
    }

    eventSource.onerror = (err) => {
      if (onError) onError(err)
    }

    return () => {
      eventSource.close()
    }
  },

  listEpisodes: (): Promise<EpisodeSummary[]> => {
    return request<EpisodeSummary[]>('/episodes')
  },

  listEpisodeSegments: (episodeId: string | number): Promise<EpisodeSegmentSummary[]> => {
    return request<EpisodeSegmentSummary[]>(`/episodes/${episodeId}/segments`)
  },

  deleteEpisode: (
    episodeId: string | number
  ): Promise<{ deleted: boolean; episode_id: number; external_id: string; deleted_segments: number }> => {
    return request(`/episodes/${episodeId}`, {
      method: 'DELETE',
    })
  },

  deleteSegment: (
    segmentId: string | number
  ): Promise<{ deleted: boolean; segment_id: number; external_id: string }> => {
    return request(`/segments/${segmentId}`, {
      method: 'DELETE',
    })
  },
}
