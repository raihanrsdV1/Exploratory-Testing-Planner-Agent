import React, { useEffect, useRef, useState } from 'react'

function lineClass(l) {
  const s = l.toLowerCase()
  if (s.includes(' error') || s.includes('crash') || s.includes('❌') || s.includes('traceback')) return 'l-err'
  if (s.includes(' warn') || s.includes('⚠')) return 'l-warn'
  if (s.includes('llm_call') || s.includes('node_enter') || s.includes('node_exit') || s.includes('🗺') || s.includes('✅') || s.includes(' info')) return 'l-info'
  return 'l-dim'
}

const SOURCES = [
  { id: 'mobilerun', label: '📱 Device (mobilerun)' },
  { id: 'planner', label: '🧠 Planner' },
]

export default function LogsPanel({ paused }) {
  const [source, setSource] = useState('mobilerun')
  const [lines, setLines] = useState([])
  const [exists, setExists] = useState(true)
  const box = useRef(null)
  const stick = useRef(true)

  useEffect(() => {
    let alive = true
    async function poll() {
      try {
        const r = await fetch(`/dashboard/logs?lines=300&source=${source}`, { cache: 'no-store' })
        const d = await r.json()
        if (!alive) return
        setExists(d.exists); setLines(d.lines || [])
      } catch { /* ignore */ }
    }
    stick.current = true
    poll()
    const t = setInterval(() => { if (!paused) poll() }, 2000)
    return () => { alive = false; clearInterval(t) }
  }, [paused, source])

  useEffect(() => {
    const el = box.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [lines])

  function onScroll() {
    const el = box.current
    if (!el) return
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }

  return (
    <div>
      <div className="log-tabs">
        {SOURCES.map(s => (
          <button key={s.id} className={'log-tab' + (source === s.id ? ' active' : '')} onClick={() => setSource(s.id)}>
            {s.label}
          </button>
        ))}
      </div>
      <div className="logs" ref={box} onScroll={onScroll}>
        {!exists ? `No ${source} log yet — start the executor / trigger a test to see activity.`
          : lines.length === 0 ? 'Waiting for log output…'
          : lines.map((l, i) => <div key={i} className={lineClass(l)}>{l}</div>)}
      </div>
    </div>
  )
}
