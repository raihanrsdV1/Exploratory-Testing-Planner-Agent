import React, { useEffect, useState } from 'react'

const ms = (v) => (v == null ? '—' : v >= 1000 ? (v / 1000).toFixed(1) + 's' : Math.round(v) + 'ms')
const clock = (ts) => (ts ? new Date(ts).toLocaleTimeString() : '')

const KIND = {
  node: { icon: '⬢', cls: 'pt-node' },
  llm: { icon: '🧠', cls: 'pt-llm' },
  rag: { icon: '🗄', cls: 'pt-rag' },
  error: { icon: '✖', cls: 'pt-error' },
}

function EventRow({ e, slowest }) {
  const meta = KIND[e.kind] || KIND.node
  const pending = e.kind === 'node' && e.duration_ms == null
  const share = slowest && e.duration_ms ? Math.max(2, (e.duration_ms / slowest) * 100) : 0

  let label = e.node
  if (e.kind === 'llm') label = `LLM call → ${e.backend}${e.tokens ? ` · ~${e.tokens} tok` : ''}`
  if (e.kind === 'rag') label = `${e.method || 'GET'} ${e.endpoint}`
  if (e.kind === 'error') label = `${e.node} failed`

  return (
    <div className={'pt-row ' + meta.cls}>
      <span className="pt-icon">{meta.icon}</span>
      <span className="pt-label">
        {label}
        {e.kind === 'node' && e.round ? <span className="pt-round">round {e.round}</span> : null}
      </span>
      <span className="pt-bar-wrap">{share ? <span className="pt-bar" style={{ width: share + '%' }} /> : null}</span>
      <span className="pt-ms">{pending ? 'running…' : ms(e.duration_ms)}</span>
    </div>
  )
}

function Run({ run, open, onToggle }) {
  const slowest = Math.max(...run.events.map(e => e.duration_ms || 0), 0)
  const statusCls = run.status === 'error' ? 'fail' : run.status === 'running' ? 'warn' : 'pass'
  const err = run.events.find(e => e.kind === 'error')

  return (
    <div className="pt-run">
      <button className="pt-head" onClick={onToggle}>
        <span className={'chip ' + statusCls}>{run.status}</span>
        <span className="pt-path">{run.path}</span>
        <span className="pt-when">{clock(run.started_at)}</span>
        <span className="pt-stats">
          {ms(run.total_ms)} total · {run.llm_calls} LLM ({ms(run.llm_ms)}) · {run.rag_calls} RAG
          {run.tokens ? ` · ~${run.tokens.toLocaleString()} tok` : ''}
        </span>
        <span className="pt-caret">{open ? '▾' : '▸'}</span>
      </button>
      {err ? <div className="pt-errmsg">{err.error}</div> : null}
      {open ? (
        <div className="pt-events">
          {run.events.map((e, i) => <EventRow key={i} e={e} slowest={slowest} />)}
        </div>
      ) : null}
    </div>
  )
}

export default function PlannerTrace({ project, paused }) {
  const [runs, setRuns] = useState([])
  const [exists, setExists] = useState(true)
  const [openId, setOpenId] = useState(null)

  useEffect(() => {
    let alive = true
    async function poll() {
      try {
        const r = await fetch(`/dashboard/planner-trace?runs=12&project=${encodeURIComponent(project)}`, { cache: 'no-store' })
        const d = await r.json()
        if (!alive) return
        setExists(d.exists)
        setRuns(d.runs || [])
      } catch { /* keep last good data */ }
    }
    poll()
    const t = setInterval(() => { if (!paused) poll() }, 3000)
    return () => { alive = false; clearInterval(t) }
  }, [project, paused])

  if (!exists) return <div className="empty">No planner trace yet — logs/app.jsonl has not been written.</div>
  if (!runs.length) return <div className="empty">No planner runs for this project yet — generate a test case to see the trace.</div>

  // Newest run is expanded by default; clicking a header switches focus.
  const activeId = openId ?? runs[0].request_id

  return (
    <div className="pt-wrap">
      {runs.map(r => (
        <Run key={r.request_id} run={r} open={r.request_id === activeId}
             onToggle={() => setOpenId(r.request_id === activeId ? '' : r.request_id)} />
      ))}
    </div>
  )
}
