/**
 * Opening a clip's moment in its YouTube video, to see who is talking.
 *
 * The server only ever sends a link it rebuilt from a video id (D23), and clip times are
 * episode-relative, so the clip's start plus the playhead is the moment in the video.
 */

/** Seconds of video before the playhead, so the turn that led here is on screen too. */
export const LEAD_IN_SECONDS = 2

/** The video at ``seconds`` into the episode, less the lead-in. */
export function videoAt(url: string, seconds: number): string {
  const t = Math.max(0, Math.floor(seconds - LEAD_IN_SECONDS))
  return `${url}&t=${t}s`
}

/** 754 -> "12:34", 3754 -> "1:02:34". */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(s % 60).padStart(2, '0')
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`
}
