import React, { useEffect, useRef } from 'react'
import { useTheme } from 'next-themes'

import type { PeaksPayload } from '@/types'

interface WaveformProps {
  peaks: PeaksPayload | null
  currentTime: number
  duration: number
  onSeek: (time: number) => void
}

export function Waveform({ peaks, currentTime, duration, onSeek }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const { resolvedTheme } = useTheme()

  // Redraw waveform when peaks, playhead or theme changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !peaks) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const styles = getComputedStyle(canvas)
    const playedColor = styles.getPropertyValue('--foreground').trim() || '#000'
    const pendingColor = styles.getPropertyValue('--muted-foreground').trim() || '#888'
    const baselineColor = styles.getPropertyValue('--border').trim() || '#ccc'

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const width = rect.width
    const height = rect.height
    const midY = height / 2

    ctx.clearRect(0, 0, width, height)

    const buckets = peaks.buckets || peaks.min.length
    const progressFraction = duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0
    const progressX = progressFraction * width

    const barWidth = width / buckets
    for (let i = 0; i < buckets; i++) {
      const x = i * barWidth
      const minVal = peaks.min[i] ?? 0
      const maxVal = peaks.max[i] ?? 0

      // Normalize amplitude (-1..1) to height, mirrored around the centre line
      const topY = midY - maxVal * (height * 0.45)
      const bottomY = midY - minVal * (height * 0.45)
      const barH = Math.max(1.5, bottomY - topY)

      ctx.fillStyle = x <= progressX ? playedColor : pendingColor
      ctx.fillRect(x, topY, Math.max(1, barWidth - 0.5), barH)
    }

    ctx.strokeStyle = baselineColor
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, midY)
    ctx.lineTo(width, midY)
    ctx.stroke()
  }, [peaks, currentTime, duration, resolvedTheme])

  // Click / drag to seek
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!containerRef.current || duration <= 0) return
    const rect = containerRef.current.getBoundingClientRect()
    const clickX = Math.max(0, Math.min(rect.width, e.clientX - rect.left))
    onSeek((clickX / rect.width) * duration)

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
      className="relative h-28 w-full cursor-ew-resize touch-none bg-muted/40"
      onPointerDown={handlePointerDown}
      title="Click or drag to seek"
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
      <div
        className="pointer-events-none absolute inset-y-0 w-px bg-info"
        style={{ left: `${progressPercent}%` }}
      />
    </div>
  )
}
