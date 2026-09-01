import { useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

/**
 * Placeholder shell. The triage and editor UI are Phase 6; this page exists so the
 * frontend service in docker compose is real and the backend connection is verifiable.
 */
export default function App() {
  const [health, setHealth] = useState(null)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/health`).then((r) => r.json()).then(setHealth).catch(() => setHealth({ status: 'unreachable' }))
    fetch(`${API_BASE}/stats`).then((r) => r.json()).then(setStats).catch(() => {})
  }, [])

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem', maxWidth: 720 }}>
      <h1>Nepanglish Annotation Harness</h1>
      <p>Backend: <code>{API_BASE}</code></p>
      <pre>{JSON.stringify(health, null, 2)}</pre>
      {stats && <pre>{JSON.stringify(stats, null, 2)}</pre>}
      <p style={{ color: '#666' }}>Triage and editor modes are Phase 6.</p>
    </main>
  )
}
