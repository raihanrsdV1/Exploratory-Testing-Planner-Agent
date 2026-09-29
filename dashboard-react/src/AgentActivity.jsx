import React, { useEffect, useRef, useState } from 'react'

// Fast enough that a tool call (often under a second) is actually seen.
const POLL_MS = 700

const AGENTS = [
  {
    key: 'planner',
    name: 'Planner',
    role: 'decides what to test next',
    reads: 'knowledge graph → test case',
  },
  {
    key: 'executor',
    name: 'Executor',
    role: 'drives the device',
    reads: 'test case → trajectory',
  },
  {
    key: 'investigator',
    name: 'Investigator',
    role: 'decides what the run proved',
    reads: 'trajectory → findings',
  },
]

// What the coarse `state` verb should say on screen.
const VERB = {
  thinking: 'Thinking',
  node: 'Pipeline stage',
  tool: 'Calling tool',
  llm: 'Calling model',
  step: 'On device',
  evaluating: 'Evaluating',
  idle: 'Idle',
}

function secs(n) {
  if (!n && n !== 0) return ''
  return n < 60 ? `${n.toFixed(1)}s` : `${Math.floor(n / 60)}m ${Math.round(n % 60)}s`
}

function AgentCard({ spec, c }) {
  const busy = !!c?.busy
  const verb = VERB[c?.state] || c?.state || 'Idle'
  // The executor reports steps; the planner reports a node or tool name.
  // Only while a run is in flight: the step count is read from the newest
  // trajectory folder, so showing it at rest would display the *last* run's
  // total as if it were live.
  const pct = spec.key === 'executor' && busy && c?.max_steps
    ? Math.min(100, (c.steps / c.max_steps) * 100)
    : null

  return (
    <div className={'agent-card' + (busy ? ' busy' : '')}>
      <div className="agent-head">
        <span className={'agent-led' + (busy ? ' on' : '')} />
        <div>
          <div className="agent-name">{spec.name}</div>
          <div className="agent-role">{spec.role}</div>
        </div>
        {busy && <span className="agent-timer">{secs(c.for_s)}</span>}
      </div>

      <div className="agent-body">
        <div className="agent-verb">{busy ? verb : 'Idle'}</div>
        <div className="agent-detail" title={c?.detail || ''}>
          {busy ? (c.detail || '—') : spec.reads}
        </div>
        {pct !== null && (
          <div className="agent-steps">
            <div className="agent-bar"><span style={{ width: `${pct}%` }} /></div>
            <span className="agent-stepcount">
              {c.steps}/{c.max_steps} steps
            </span>
          </div>
        )}
        {spec.key === 'executor' && busy && c?.since_step_s != null && (
          // A step is a vision model call (~20s is normal). Showing the wait makes
          // a slow run legible instead of looking frozen.
          <div className={'agent-meta' + (c.since_step_s > 60 ? ' warn' : '')}>
            {c.since_step_s < 1.5
              ? 'advancing…'
              : `${Math.round(c.since_step_s)}s on this step`}
          </div>
        )}
        {c?.tokens ? <div className="agent-meta">~{c.tokens} tokens</div> : null}
        {c?.latency_ms ? <div className="agent-meta">{Math.round(c.latency_ms)} ms</div> : null}
      </div>
    </div>
  )
}

export default function AgentActivity() {
  const [snap, setSnap] = useState(null)
  const [feed, setFeed] = useState([])
  const [err, setErr] = useState(null)
  const seq = useRef(0)
  const inFlight = useRef(false)
  const feedRef = useRef(null)
  const pinned = useRef(true)
  const [staleMs, setStaleMs] = useState(0)
  const lastOk = useRef(Date.now())

  useEffect(() => {
    let alive = true
    const tick = async () => {
      // A fetch slower than POLL_MS would otherwise overlap the next one: both
      // send the same `since` cursor, both get the same events back, and the
      // feed shows each of them twice.
      if (inFlight.current) return
      inFlight.current = true
      try {
        const r = await fetch(`/activity?since=${seq.current}`, { cache: 'no-store' })
        if (!r.ok) throw new Error('HTTP ' + r.status)
        const j = await r.json()
        if (!alive) return
        setSnap(j)
        setErr(null)
        lastOk.current = Date.now()
        if (j.seq != null) seq.current = j.seq
        if (j.events?.length) {
          setFeed((f) => {
            // Oldest first, so the newest lands at the bottom like a terminal.
            const seen = new Set(f.map((e) => e.seq))
            const add = j.events.filter((e) => !seen.has(e.seq))
            return add.length ? f.concat(add).slice(-60) : f
          })
        }
      } catch (e) {
        if (alive) setErr(e.message)
      } finally {
        inFlight.current = false
      }
    }
    tick()
    const t = setInterval(tick, POLL_MS)
    return () => { alive = false; clearInterval(t) }
  }, [])

  // Without this, a failed poll leaves the last snapshot on screen forever and a
  // disconnected dashboard is indistinguishable from an agent stuck mid-step.
  useEffect(() => {
    const t = setInterval(() => setStaleMs(Date.now() - lastOk.current), 1000)
    return () => clearInterval(t)
  }, [])

  // Follow the tail, but only while the reader is already at the bottom —
  // scrolling up to read something must not be yanked away on the next poll.
  useEffect(() => {
    const el = feedRef.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [feed])

  const onFeedScroll = () => {
    const el = feedRef.current
    if (el) pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
  }

  const disconnected = staleMs > 6000
  const comps = disconnected ? {} : (snap?.components || {})
  const anyBusy = AGENTS.some((a) => comps[a.key]?.busy)

  return (
    <section className="card agent-activity">
      <h2>
        Live agent activity
        <span className={'live-pill' + (anyBusy ? ' on' : '') + (disconnected ? ' off' : '')}>
          {disconnected
            ? `no data for ${Math.round(staleMs / 1000)}s`
            : anyBusy ? 'running' : 'idle'}
        </span>
        {snap?.run?.round ? (
          <span className="run-pill">
            round {snap.run.round}{snap.run.rounds ? `/${snap.run.rounds}` : ''}
            {snap.run.test_id ? ` · ${snap.run.test_id}` : ''}
          </span>
        ) : null}
      </h2>
      <p className="muted small">
        Which of the three agents is working right now. They never call each other —
        everything passes through the knowledge graph.
      </p>

      <div className="agent-row">
        {AGENTS.map((a, i) => (
          <React.Fragment key={a.key}>
            <AgentCard spec={a} c={comps[a.key]} />
            {i < AGENTS.length - 1 && <div className="agent-arrow">→</div>}
          </React.Fragment>
        ))}
      </div>

      <div className="agent-feed" ref={feedRef} onScroll={onFeedScroll}>
        {feed.length === 0 && <div className="muted small">waiting for activity…</div>}
        {feed.map((e) => (
          <div className="agent-feed-row" key={e.seq}>
            <span className="feed-run">{e.test_id || (e.round ? `round ${e.round}` : '')}</span>
            <span className={'feed-tag ' + e.component}>{e.component}</span>
            {/* A model call names the stage that made it, so the two render as
                one line instead of two rows that look unrelated. */}
            <span className="feed-detail">
              {e.state === 'llm' && e.stage
                ? <><b>{e.stage}</b><span className="feed-sep"> · </span>{e.detail}</>
                : e.detail}
            </span>
            <span className="feed-kind">{VERB[e.state] || e.state}</span>
            {e.latency_ms ? <span className="feed-ms">{Math.round(e.latency_ms)}ms</span> : null}
          </div>
        ))}
      </div>
    </section>
  )
}
