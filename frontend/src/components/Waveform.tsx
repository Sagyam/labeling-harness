import React, { useEffect, useRef } from 'react'
import { PeaksPayload } from '../types'

interface WaveformProps {
  peaks: PeaksPayload | null
  currentTime: number
  duration: number
  onSeek: (time: number) => void
  isPlaying: boolean
}

export const Waveform: React.FC<WaveformProps> = ({
  peaks,
  currentTime,
  duration,
  onSeek,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  // Redraw waveform when peaks or currentTime changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !peaks) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const width = rect.width
    const height = rect.height
    const midY = height / 2

    // Clear background
    ctx.clearRect(0, 0, width, height)

    const buckets = peaks.buckets || peaks.min.length
    const progressFraction = duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0
    const progressX = progressFraction * width

    // Waveform rendering
    const barWidth = width / buckets
    for (let i = 0; i < buckets; i++) {
      const x = i * barWidth
      const isPlayed = x <= progressX

      const minVal = peaks.min[i] ?? 0
      const maxVal = peaks.max[i] ?? 0

      // Normalize amplitude (-1 to 1) to height (half above, half below center)
      const topY = midY - maxVal * (height * 0.45)
      const bottomY = midY - minVal * (height * 0.45)
      const barH = Math.max(1.5, bottomY - topY)

      if (isPlayed) {
        ctx.fillStyle = '#06b6d4' // Cyan for played audio
      } else {
        ctx.fillStyle = '#3b82f6' // Blue/indigo for unplayed
      }

      ctx.fillRect(x, topY, Math.max(1, barWidth - 0.5), barH)
    }

    // Center baseline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, midY)
    ctx.lineTo(width, midY)
    ctx.stroke()
  }, [peaks, currentTime, duration])

  // Handle click / drag to seek
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!containerRef.current || duration <= 0) return
    const rect = containerRef.current.getBoundingClientRect()
    const clickX = Math.max(0, Math.min(rect.width, e.clientX - rect.left))
    const fraction = clickX / rect.width
    onSeek(fraction * duration)

    const onPointerMove = (moveEvent: PointerEvent) => {
      const currentX = Math.max(0, Math.min(rect.width, moveEvent.clientX - rect.left))
      onSeek((currentX / rect.width) * duration)
    }

    const onPointerUp = () => {
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
    }

    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
  }

  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div
      ref={containerRef}
      className="waveform-canvas-container"
      onPointerDown={handlePointerDown}
      title="Click or drag to seek audio"
    >
      <canvas ref={canvasRef} className="waveform-canvas" />
      {/* Interactive Playhead */}
      <div
        className="playhead-line"
        style={{
          left: `${progressPercent}%`,
        }}
      />
    </div>
  )
}
