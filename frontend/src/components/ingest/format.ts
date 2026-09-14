export function formatDuration(seconds: number | null) {
  if (seconds === null || !Number.isFinite(seconds)) return 'unknown length'
  const total = Math.round(seconds)
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  const pad = (n: number) => String(n).padStart(2, '0')
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`
}

export function formatClock(seconds: number) {
  const total = Math.max(0, Math.round(seconds))
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

export function formatTimestamp(ts: number | string | null | undefined) {
  if (ts === undefined || ts === null) return ''
  try {
    const ms = typeof ts === 'number' ? (ts < 1e11 ? ts * 1000 : ts) : Date.parse(ts)
    const d = new Date(ms)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return String(ts)
  }
}

export function slugify(text: string) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\p{M}\s_-]/gu, '')
    .trim()
    .replace(/[-\s]+/g, '_')
    .slice(0, 40)
    .replace(/^_+|_+$/g, '')
}
