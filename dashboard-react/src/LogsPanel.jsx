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

/** One independently-polled, auto-scrolling log stream. */
function LogStream({ source, label, paused }) {
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
      } catch { /* keep last good data */ }
    }
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
    // Only keep pinning to the bottom while the user is already there.
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }

  return (
    <div className="log-stream">
      <div className="log-head">
        {label}
        <span className="log-count">{exists ? `${lines.length} lines` : 'no log yet'}</span>
      </div>
      <div className="logs" ref={box} onScroll={onScroll}>
        {!exists ? `No ${source} log yet — start the executor / trigger a test to see activity.`
          : lines.length === 0 ? 'Waiting for log output…'
          : lines.map((l, i) => <div key={i} className={lineClass(l)}>{l}</div>)}
      </div>
    </div>
  )
}

export default function LogsPanel({ paused }) {
  return (
    <div className="log-split">
      {SOURCES.map(s => <LogStream key={s.id} source={s.id} label={s.label} paused={paused} />)}
    </div>
  )
}
