import React, { useCallback, useEffect, useRef, useState } from 'react'
import AppModelGraph from './AppModelGraph.jsx'
import LogsPanel from './LogsPanel.jsx'

const REFRESH_MS = 4000
const num = (n) => (n == null || isNaN(n) ? '—' : Number(n).toLocaleString())

function Tile({ label, value, sub, cls }) {
  return (
    <div className={'tile ' + (cls || '')}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  )
}

export default function App() {
  const params = new URLSearchParams(location.search)
  const [project, setProject] = useState(params.get('project') || 'contacts-app')
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  const [ago, setAgo] = useState('connecting…')
  const [auto, setAuto] = useState(true)
  const [selectedExec, setSelectedExec] = useState(null)
  const lastOk = useRef(0)

  const poll = useCallback(async (proj) => {
    try {
      const r = await fetch(`/dashboard/data?project=${encodeURIComponent(proj)}`, { cache: 'no-store' })
      if (!r.ok) throw new Error('HTTP ' + r.status)
      setData(await r.json()); setErr(null); lastOk.current = Date.now()
    } catch (e) { setErr(e.message) }
  }, [])

  useEffect(() => {
    poll(project)
    if (!auto) return
    const t = setInterval(() => poll(project), REFRESH_MS)
    return () => clearInterval(t)
  }, [project, auto, poll])

  useEffect(() => {
    const t = setInterval(() => {
      if (!lastOk.current) { setAgo('connecting…'); return }
      const s = Math.round((Date.now() - lastOk.current) / 1000)
      setAgo(s <= 1 ? 'updated just now' : `updated ${s}s ago`)
    }, 1000)
    return () => clearInterval(t)
  }, [])

  const staleSecs = lastOk.current ? (Date.now() - lastOk.current) / 1000 : 999
  const dotCls = 'dot' + (err ? ' err' : staleSecs > REFRESH_MS / 1000 + 6 ? ' stale' : '')

  const d = data || {}
  const s = d.stats || {}, c = d.coverage || {}
  const cov = c.summary || {}
  const recent = c.recent_tests || []
  const areas = c.area_breakdown || {}
  const am = d.appmodel || {}
  const nodeLabel = Object.fromEntries((am.nodes || []).map(n => [n.id, n.label]))
  // Resolve a run's path to current state labels and collapse consecutive repeats
  // into the actual navigation route.
  function routeOf(e) {
    const ids = e.path || []
    const out = []
    ids.forEach((pid, i) => {
      const lbl = nodeLabel[pid] || (e.path_labels || [])[i] || '?'
      if (!out.length || out[out.length - 1] !== lbl) out.push(lbl)
    })
    return out
  }
  const executions = d.executions || []
  const activeExec = executions.find(e => (e.created_at + e.test_case_id) === selectedExec) || null
  const highlightPath = activeExec ? activeExec.path : null
  const failed = recent.filter(t => String(t.verdict).toLowerCase() === 'failed')
  const passed = recent.filter(t => String(t.verdict).toLowerCase() === 'pass')
  const passRate = recent.length ? Math.round((100 * passed.length) / recent.length) : 0
  const hot = new Set(c.hot_spots || [])
  const areaRows = Object.entries(areas).sort((a, b) => (b[1].total - a[1].total))

  function changeProject(v) {
    const p = v.trim() || 'contacts-app'
    setProject(p)
    const u = new URL(location); u.searchParams.set('project', p); history.replaceState({}, '', u)
  }

  return (
    <>
      <header>
        <h1>🐝 QA Agent — Live Dashboard</h1>
        <span className="pill">project: {d.project || project}</span>
        <span className="pill accent">model: {(d.model && (d.model.model || d.model.backend)) || '—'}</span>
        <span className="spacer" />
        <span className="live"><span className={dotCls} /><span className="pill">{ago}</span></span>
        <input defaultValue={project} onBlur={(e) => changeProject(e.target.value)}
               onKeyDown={(e) => { if (e.key === 'Enter') changeProject(e.target.value) }} style={{ width: 150 }} />
        <label className="pill" style={{ display: 'flex', gap: 6, alignItems: 'center', cursor: 'pointer' }}>
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} style={{ width: 'auto', padding: 0 }} /> auto
        </label>
      </header>

      <main>
        {err ? <div className="banner">Could not reach the gateway (/dashboard/data): {err}. Is it running on :9100?</div> : null}

        <section className="kpis">
          <Tile label="Total Tests" value={num(s.test_case_count)} sub={`${s.test_run_count || 0} runs`} />
          <Tile label="Pass Rate" value={passRate + '%'} sub={`${passed.length}/${recent.length} recent`} cls="pass" />
          <Tile label="Bugs Found" value={num(failed.length)} sub="failed verdicts" cls="fail" />
          <Tile label="Coverage" value={(cov.coverage_pct ?? 0) + '%'} sub={`${cov.areas_tested || 0}/${cov.areas_available || 0} areas`} />
          <Tile label="Business Policies" value={num(s.validation_rule_count)} sub="validation rules (SRS)" cls="accent" />
          <Tile label="Requirements" value={num(s.requirement_count)} sub={`${num(s.covered_requirement_count)} covered`} />
          <Tile label="App States" value={num(am.state_count)} sub={`${(am.edges || []).length} transitions`} cls="accent" />
          <Tile label="Screens" value={num(s.figma_screen_count)} sub={`${num(s.figma_element_count)} UI elements`} />
        </section>

        <div className="situation">
          {recent.length
            ? <>Overall: <b>{num(s.test_case_count)}</b> test cases, <b>{failed.length}</b> bug(s) across <b>{cov.areas_tested || 0}</b>/<b>{cov.areas_available || 0}</b> UI areas. The agent evaluated <b>{num(s.requirement_count)}</b> requirements and <b>{num(s.validation_rule_count)}</b> business policies from the SRS, and has mapped <b>{num(am.state_count)}</b> live app states.</>
            : <>No tests executed yet for <b>{d.project || project}</b>. Start the executor loop to begin exploration.</>}
        </div>

        <div className="panel">
          <h2>🗺️ Live App Model <span className="count">{am.state_count ? `(${am.state_count} states · ${(am.edges || []).length} transitions)` : ''}</span></h2>
          <div className="body">
            <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 10 }}>
              The self-built map of screens the agent has actually reached (deduped by structural signature — scrolling & theme don't create duplicates). Drag nodes; click a state to see its screenshot.
              {highlightPath ? <span style={{ color: 'var(--accent)' }}> Highlighting the path of {activeExec.test_case_id}.</span> : ' Click a run below to trace its path here.'}
            </div>
            <AppModelGraph project={d.project || project} nodes={am.nodes || []} edges={am.edges || []} highlightPath={highlightPath} />
          </div>
        </div>

        <div className="panel">
          <h2>🧬 Execution Paths <span className="count">{executions.length ? `(${executions.length} runs)` : ''}</span></h2>
          <div className="body">
            <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 6 }}>Each test's real route through the app. Click one to highlight its path on the graph above — a quick way to spot redundant or wrong states.</div>
            <div className="scroll" style={{ maxHeight: 300 }}>
              {executions.length ? executions.map((e, i) => {
                const id = e.created_at + e.test_case_id
                const labels = routeOf(e)
                const distinct = new Set(e.path || []).size
                return (
                  <div key={i} className={'exec-row' + (id === selectedExec ? ' active' : '')}
                       onClick={() => setSelectedExec(id === selectedExec ? null : id)}>
                    <div className="top">
                      <span className={'chip ' + (String(e.verdict).toLowerCase() === 'failed' ? 'fail' : 'pass')}>
                        {String(e.verdict).toLowerCase() === 'failed' ? 'failed' : 'pass'}
                      </span>
                      <span className="title">{e.test_case_id} — {e.title}</span>
                      <span className="meta">{distinct} states · {e.states_visited} steps · {Math.round((e.duration_ms || 0) / 1000)}s{e.error_type ? ' · ' + e.error_type : ''}</span>
                    </div>
                    {labels.length ? (
                      <div className="exec-path">
                        {labels.map((l, j) => <span key={j}><span className="st">{l || '?'}</span>{j < labels.length - 1 ? <span className="arr">→</span> : null}</span>)}
                      </div>
                    ) : null}
                  </div>
                )
              }) : <div className="empty">No executions logged yet — run the executor.</div>}
            </div>
          </div>
        </div>

        <div className="panel">
          <h2>📟 Live Logs <span className="count">(device agent + planner reasoning — switch tabs)</span></h2>
          <div className="body"><LogsPanel paused={!auto} /></div>
        </div>

        <div className="grid">
          <div>
            <div className="panel">
              <h2>Test Cases <span className="count">{recent.length ? `(${recent.length})` : ''}</span></h2>
              <div className="scroll">
                <table>
                  <thead><tr><th>ID</th><th>Title</th><th>Area</th><th style={{ textAlign: 'right' }}>Verdict</th></tr></thead>
                  <tbody>
                    {recent.length ? recent.map((t, i) => (
                      <tr key={i}>
                        <td className="mono">{t.id}</td><td>{t.title}</td>
                        <td><span className="area-tag">{t.area || '—'}</span></td>
                        <td style={{ textAlign: 'right' }}>
                          <span className={'chip ' + (String(t.verdict).toLowerCase() === 'failed' ? 'fail' : 'pass')}>
                            {String(t.verdict).toLowerCase() === 'failed' ? 'failed' : 'pass'}
                          </span>
                        </td>
                      </tr>
                    )) : <tr><td colSpan="4"><div className="empty">No test cases yet.</div></td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div>
            <div className="panel">
              <h2>🐞 Bugs Found <span className="count">{failed.length ? `(${failed.length})` : ''}</span></h2>
              <div className="body">
                {failed.length ? failed.map((t, i) => (
                  <div className="kv" key={i}><span>{t.title}</span><span className="area-tag">{t.area || '—'}</span></div>
                )) : <div className="empty">No bugs found yet ✨</div>}
              </div>
            </div>

            <div className="panel">
              <h2>Coverage by Area <span className="count">{(cov.coverage_pct ?? 0) + '%'}</span></h2>
              <div className="body">
                <div className="bigbar"><div style={{ width: (cov.coverage_pct || 0) + '%' }} /></div>
                <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 6 }}>{cov.areas_tested || 0} of {cov.areas_available || 0} known UI areas exercised</div>
                {areaRows.length ? areaRows.map(([name, st]) => {
                  const tot = st.total || 0, ps = st.passed || 0, fl = st.failed || 0
                  return (
                    <div className="cov-row" key={name}>
                      <div className="name" title={name}>{name}{hot.has(name) ? ' 🔥' : ''}</div>
                      <div className="track-wrap"><div className="bar-track">
                        <div className="bar-pass" style={{ width: (tot ? (100 * ps / tot) : 0) + '%' }} />
                        <div className="bar-fail" style={{ width: (tot ? (100 * fl / tot) : 0) + '%' }} />
                      </div></div>
                      <div className="nums">{ps}✓ {fl}✗</div>
                    </div>
                  )
                }) : <div className="empty">No area data yet.</div>}
                {(c.uncovered_areas || []).length ? (
                  <div className="tags" style={{ marginTop: 10 }}>
                    <span style={{ fontSize: 11.5, color: 'var(--muted)', alignSelf: 'center' }}>untested:</span>
                    {(c.uncovered_areas || []).map(a => <span className="tag" key={a}>{a}</span>)}
                  </div>
                ) : null}
              </div>
            </div>

            <div className="panel">
              <h2>📜 SRS Knowledge — business policies</h2>
              <div className="body">
                <div className="kv"><span>Requirements extracted</span><span className="v">{num(s.requirement_count)}</span></div>
                <div className="kv"><span>Business policies (validation rules)</span><span className="v">{num(s.validation_rule_count)}</span></div>
                <div className="kv"><span>Domain entities</span><span className="v">{num(s.entity_count)}</span></div>
                <div className="kv"><span>Requirements covered by tests</span><span className="v">{num(s.covered_requirement_count)} / {num(s.requirement_count)}</span></div>
                <div style={{ fontSize: 11.5, color: 'var(--muted)', marginTop: 8 }}>Extracted by the LLM from the ingested SRS.</div>
              </div>
            </div>

            <div className="panel">
              <h2>🧭 Exploration Directive</h2>
              <div className="body"><pre className="directive">{c.exploration_directive || '—'}</pre></div>
            </div>
          </div>
        </div>
      </main>
    </>
  )
}
