import React, { useEffect, useState } from 'react'
import { api, resolveUrl } from '../services/api'
import { EpisodeSummary, EpisodeSegmentSummary } from '../types'

interface EpisodeManagerModalProps {
  isOpen: boolean
  onClose: () => void
  onDataChanged: () => void
}

export const EpisodeManagerModal: React.FC<EpisodeManagerModalProps> = ({
  isOpen,
  onClose,
  onDataChanged,
}) => {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedEpisodeId, setExpandedEpisodeId] = useState<number | null>(null)
  const [segments, setSegments] = useState<EpisodeSegmentSummary[]>([])
  const [segmentsLoading, setSegmentsLoading] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [playingAudioUrl, setPlayingAudioUrl] = useState<string | null>(null)
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null)

  const loadEpisodes = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.listEpisodes()
      setEpisodes(data)
    } catch (err: any) {
      setError(err.message || 'Failed to load episodes')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      loadEpisodes()
    } else {
      if (audioElement) {
        audioElement.pause()
        setAudioElement(null)
        setPlayingAudioUrl(null)
      }
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
      const data = await api.listEpisodeSegments(epId)
      setSegments(data)
    } catch (err: any) {
      setError(err.message || 'Failed to load episode segments')
    } finally {
      setSegmentsLoading(false)
    }
  }

  const handleDeleteEpisode = async (ep: EpisodeSummary) => {
    const confirm = window.confirm(
      `Are you sure you want to delete episode "${ep.title || ep.external_id}"?\n\nThis will permanently delete all ${ep.segment_count} segments, annotations, and audio files.`
    )
    if (!confirm) return

    setDeletingId(`ep-${ep.id}`)
    try {
      await api.deleteEpisode(ep.id)
      setEpisodes((prev) => prev.filter((e) => e.id !== ep.id))
      if (expandedEpisodeId === ep.id) {
        setExpandedEpisodeId(null)
        setSegments([])
      }
      onDataChanged()
    } catch (err: any) {
      alert(`Failed to delete episode: ${err.message}`)
    } finally {
      setDeletingId(null)
    }
  }

  const handleDeleteSegment = async (seg: EpisodeSegmentSummary) => {
    const confirm = window.confirm(
      `Delete segment "${seg.external_id}"?\nThis removes it permanently from the queue and storage.`
    )
    if (!confirm) return

    setDeletingId(`seg-${seg.id}`)
    try {
      await api.deleteSegment(seg.id)
      setSegments((prev) => prev.filter((s) => s.id !== seg.id))
      setEpisodes((prev) =>
        prev.map((ep) =>
          ep.id === expandedEpisodeId
            ? { ...ep, segment_count: Math.max(0, ep.segment_count - 1) }
            : ep
        )
      )
      onDataChanged()
    } catch (err: any) {
      alert(`Failed to delete segment: ${err.message}`)
    } finally {
      setDeletingId(null)
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

    if (audioElement) {
      audioElement.pause()
    }

    const audio = new Audio(fullUrl)
    audio.onended = () => {
      setPlayingAudioUrl(null)
      setAudioElement(null)
    }
    audio.play().catch((e) => console.error('Audio play error:', e))
    setAudioElement(audio)
    setPlayingAudioUrl(fullUrl)
  }

  if (!isOpen) return null

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-content episode-manager-modal"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '900px', width: '92%', maxHeight: '85vh', display: 'flex', flexDirection: 'column' }}
      >
        <div className="modal-header">
          <div>
            <h2 className="modal-title">Episode & Segment Management</h2>
            <p className="modal-subtitle">
              Inspect uploaded podcast episodes, preview and delete unwanted segments or entire episodes.
            </p>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        {error && (
          <div className="ingest-error-box" style={{ margin: '16px 24px 0 24px' }}>
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        <div className="modal-body" style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '40px', color: '#94a3b8' }}>
              <div className="spinner" style={{ margin: '0 auto 16px auto' }} />
              Loading episodes...
            </div>
          ) : episodes.length === 0 ? (
            <div className="empty-episodes-state">
              <span style={{ fontSize: '40px', marginBottom: '12px' }}>🎙️</span>
              <h3 style={{ fontSize: '18px', color: '#f8fafc', margin: '0 0 8px 0' }}>No Episodes Ingested Yet</h3>
              <p style={{ color: '#94a3b8', maxWidth: '400px', margin: '0 0 20px 0', fontSize: '14px' }}>
                Upload podcast audio using the <strong>+ Ingest</strong> button to start generating segments and hypotheses.
              </p>
            </div>
          ) : (
            <div className="episodes-list" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {episodes.map((ep) => {
                const percentDone =
                  ep.segment_count > 0 ? Math.round((ep.labeled_count / ep.segment_count) * 100) : 0
                const isExpanded = expandedEpisodeId === ep.id
                const isDeleting = deletingId === `ep-${ep.id}`

                return (
                  <div
                    key={ep.id}
                    className="episode-card"
                    style={{
                      background: 'rgba(30, 41, 59, 0.7)',
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                      borderRadius: '12px',
                      overflow: 'hidden',
                      transition: 'border-color 0.2s',
                    }}
                  >
                    <div
                      style={{
                        padding: '16px 20px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '16px',
                        flexWrap: 'wrap',
                      }}
                    >
                      <div style={{ flex: 1, minWidth: '220px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                          <span style={{ fontSize: '18px' }}>🎧</span>
                          <h4 style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: '#f8fafc' }}>
                            {ep.title || ep.external_id}
                          </h4>
                          <span
                            style={{
                              fontSize: '11px',
                              padding: '2px 8px',
                              borderRadius: '10px',
                              background: 'rgba(59, 130, 246, 0.15)',
                              color: '#60a5fa',
                              border: '1px solid rgba(59, 130, 246, 0.3)',
                            }}
                          >
                            {ep.external_id}
                          </span>
                        </div>
                        <div style={{ fontSize: '12px', color: '#94a3b8', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                          <span>Show: <strong>{ep.show_id || 'default'}</strong></span>
                          {ep.duration_seconds && (
                            <span>Duration: <strong>{Math.round(ep.duration_seconds / 60)}m {Math.round(ep.duration_seconds % 60)}s</strong></span>
                          )}
                          <span>Total: <strong>{ep.segment_count} segments</strong></span>
                          <span>Labeled: <strong>{ep.labeled_count}</strong></span>
                        </div>
                      </div>

                      {/* Progress pill */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div style={{ width: '90px', textAlign: 'right' }}>
                          <div style={{ fontSize: '12px', fontWeight: 600, color: '#e2e8f0' }}>{percentDone}% done</div>
                          <div
                            style={{
                              height: '5px',
                              width: '100%',
                              background: '#334155',
                              borderRadius: '3px',
                              overflow: 'hidden',
                              marginTop: '4px',
                            }}
                          >
                            <div
                              style={{
                                height: '100%',
                                width: `${percentDone}%`,
                                background: '#10b981',
                                transition: 'width 0.3s',
                              }}
                            />
                          </div>
                        </div>

                        {/* Actions */}
                        <button
                          className="secondary-btn"
                          onClick={() => toggleExpand(ep.id)}
                          style={{ padding: '6px 14px', fontSize: '13px' }}
                        >
                          {isExpanded ? 'Hide Segments ▲' : `View ${ep.segment_count} Segments ▼`}
                        </button>

                        <button
                          className="danger-btn"
                          disabled={isDeleting}
                          onClick={() => handleDeleteEpisode(ep)}
                          style={{
                            padding: '6px 12px',
                            fontSize: '13px',
                            background: 'rgba(239, 68, 68, 0.15)',
                            color: '#f87171',
                            border: '1px solid rgba(239, 68, 68, 0.3)',
                          }}
                        >
                          {isDeleting ? 'Deleting...' : '🗑️ Delete Episode'}
                        </button>
                      </div>
                    </div>

                    {/* Expanded Segment Drawer */}
                    {isExpanded && (
                      <div
                        style={{
                          borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                          background: 'rgba(15, 23, 42, 0.6)',
                          padding: '16px 20px',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                          <h5 style={{ margin: 0, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>
                            Segments in {ep.external_id} ({segments.length})
                          </h5>
                          <span style={{ fontSize: '12px', color: '#64748b' }}>
                            Click play to preview audio or delete unusable slices
                          </span>
                        </div>

                        {segmentsLoading ? (
                          <div style={{ textAlign: 'center', padding: '24px', color: '#94a3b8', fontSize: '13px' }}>
                            Loading segments...
                          </div>
                        ) : segments.length === 0 ? (
                          <div style={{ textAlign: 'center', padding: '16px', color: '#64748b', fontSize: '13px' }}>
                            No segments found in this episode.
                          </div>
                        ) : (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '350px', overflowY: 'auto' }}>
                            {segments.map((seg) => {
                              const isPlaying = playingAudioUrl === resolveUrl(seg.audio_url)
                              const isSegDeleting = deletingId === `seg-${seg.id}`
                              const hasHindiIntrusion = seg.flags.includes('hindi_intrusion')

                              return (
                                <div
                                  key={seg.id}
                                  style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    padding: '8px 12px',
                                    background: hasHindiIntrusion
                                      ? 'rgba(234, 179, 8, 0.06)'
                                      : 'rgba(30, 41, 59, 0.5)',
                                    border: hasHindiIntrusion
                                      ? '1px solid rgba(234, 179, 8, 0.3)'
                                      : '1px solid rgba(255, 255, 255, 0.04)',
                                    borderRadius: '8px',
                                    fontSize: '13px',
                                    gap: '12px',
                                  }}
                                >
                                  {/* Audio play button */}
                                  <button
                                    onClick={() => handleTogglePlayAudio(seg.audio_url)}
                                    style={{
                                      background: isPlaying ? '#10b981' : 'rgba(255, 255, 255, 0.1)',
                                      color: '#ffffff',
                                      border: 'none',
                                      borderRadius: '50%',
                                      width: '28px',
                                      height: '28px',
                                      cursor: 'pointer',
                                      display: 'flex',
                                      alignItems: 'center',
                                      justifyContent: 'center',
                                      fontSize: '11px',
                                      flexShrink: 0,
                                    }}
                                    title={isPlaying ? 'Pause' : 'Play audio preview'}
                                  >
                                    {isPlaying ? '⏸' : '▶'}
                                  </button>

                                  {/* Timing and ID */}
                                  <div style={{ minWidth: '130px', flexShrink: 0 }}>
                                    <div style={{ fontFamily: 'monospace', fontWeight: 600, color: '#cbd5e1', fontSize: '12px' }}>
                                      {seg.external_id}
                                    </div>
                                    <div style={{ fontSize: '11px', color: '#64748b' }}>
                                      {seg.start_time.toFixed(1)}s - {seg.end_time.toFixed(1)}s ({seg.duration_seconds.toFixed(1)}s)
                                    </div>
                                  </div>

                                  {/* Transcript snippet */}
                                  <div
                                    style={{
                                      flex: 1,
                                      color: '#e2e8f0',
                                      whiteSpace: 'nowrap',
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                    }}
                                  >
                                    {seg.seed_text || <span style={{ color: '#64748b', fontStyle: 'italic' }}>No hypothesis</span>}
                                  </div>

                                  {/* Flags & CMI */}
                                  <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexShrink: 0 }}>
                                    {hasHindiIntrusion && (
                                      <span
                                        style={{
                                          background: 'rgba(234, 179, 8, 0.2)',
                                          color: '#facc15',
                                          border: '1px solid rgba(234, 179, 8, 0.4)',
                                          padding: '2px 6px',
                                          borderRadius: '4px',
                                          fontSize: '11px',
                                          fontWeight: 600,
                                        }}
                                      >
                                        ⚠️ Hindi Intrusion
                                      </span>
                                    )}
                                    {seg.cmi !== null && (
                                      <span style={{ fontSize: '11px', color: '#94a3b8' }}>
                                        CMI: {seg.cmi}%
                                      </span>
                                    )}
                                  </div>

                                  {/* Delete Segment Button */}
                                  <button
                                    onClick={() => handleDeleteSegment(seg)}
                                    disabled={isSegDeleting}
                                    style={{
                                      background: 'transparent',
                                      border: 'none',
                                      color: '#ef4444',
                                      cursor: 'pointer',
                                      padding: '4px 8px',
                                      borderRadius: '4px',
                                      fontSize: '14px',
                                      flexShrink: 0,
                                      opacity: isSegDeleting ? 0.5 : 0.8,
                                    }}
                                    title="Delete this segment"
                                  >
                                    {isSegDeleting ? '...' : '🗑️'}
                                  </button>
                                </div>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="modal-footer" style={{ display: 'flex', justifyContent: 'flex-end', padding: '14px 24px', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
          <button className="secondary-btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
