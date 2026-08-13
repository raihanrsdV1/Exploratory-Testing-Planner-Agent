import React, { useState } from 'react'

// The learning layer drives generation but had no operator-visible surface, so
// there was no way to tell whether it was contributing or silently empty — and
// "silently empty" was in fact the case for defect intelligence.
function Panel({ title, count, hint, children }) {
  return (
    <div className="panel">
      <h2>{title} {count != null ? <span className="count">({count})</span> : null}</h2>
      <div className="body">
        {hint ? <div className="hint">{hint}</div> : null}
        {children}
      </div>
    </div>
  )
}

function Empty({ what, why }) {
  return <div className="empty">No {what} yet.{why ? <div className="empty-why">{why}</div> : null}</div>
}

export default function Intelligence({ d }) {
  const [showPrompt, setShowPrompt] = useState(false)
  const risk = d.risk_scores || []
  const defects = d.defects || {}
  const anomalies = d.anomalies || []
  const strategies = d.strategies || []
  const patterns = d.error_patterns || []
  const nav = d.navtree || {}
  const navFailed = d.nav_failed || []
  const rules = d.rules || []
  const reqCov = d.requirement_coverage || {}
  const eff = d.effectiveness || []
  const deg = d.degradations || {}

  return (
    <>
      {deg.total > 0 && (
        <div className={'banner ' + (deg.worst_severity === 'critical' ? 'crit' : 'warn')}>
          <b>{deg.total} silent fallback{deg.total === 1 ? '' : 's'} this run</b>
          {deg.trustworthy === false ? ' — results should NOT be trusted.' : ' — capability was lost.'}
          <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
            {(deg.events || []).slice(0, 4).map((e, i) => (
              <li key={i}><code>{e.kind}</code> — {e.detail}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid">
        <div>
          <Panel title="🐞 Defect Intelligence" count={defects.total_defects ?? 0}
                 hint="Historical bugs steer generation toward areas that have broken before.">
            {(defects.total_defects ?? 0) === 0
              ? <Empty what="defect history" why="Nothing has been ingested via /ingest/defects, so this signal contributes nothing to test generation." />
              : <>
                  <div className="kv"><span>Unresolved</span><span className="v">{defects.unresolved_defects}</span></div>
                  {Object.entries(defects.severity_distribution || {}).map(([k, v]) =>
                    <div className="kv" key={k}><span>{k}</span><span className="v">{v}</span></div>)}
                  <div className="tags" style={{ marginTop: 8 }}>
                    {(defects.prone_areas || []).slice(0, 8).map((a, i) =>
                      <span className="tag hot" key={i}>{a.area || a}</span>)}
                  </div>
                </>}
          </Panel>

          <Panel title="⚠️ Regression Risk" count={risk.length}
                 hint="Per-area score from defect density, failure ratio and recency.">
            {risk.length === 0 ? <Empty what="risk scores" why="Computed from execution history; a clean slate starts empty." />
              : risk.slice(0, 10).map((r, i) => (
                <div className="cov-row" key={i}>
                  <div className="name">{r.area || r.feature || '?'}</div>
                  <div className="track-wrap"><div className="bar-track">
                    <div className="bar-fail" style={{ width: `${Math.round((r.regression_risk_score || r.score || 0) * 100)}%` }} />
                  </div></div>
                  <div className="nums">{((r.regression_risk_score || r.score || 0)).toFixed(2)}</div>
                </div>
              ))}
          </Panel>

          <Panel title="📈 Emerging Anomalies" count={anomalies.length}
                 hint="Failure-rate spikes, new error types and unstable paths detected from execution logs.">
            {anomalies.length === 0 ? <Empty what="anomalies" why="Needs several runs before a baseline exists." />
              : anomalies.map((a, i) => (
                <div className="kv" key={i}>
                  <span>{a.description || a.anomaly_type}</span>
                  <span className="area-tag">{a.severity || '—'}</span>
                </div>
              ))}
          </Panel>
        </div>

        <div>
          <Panel title="🧠 Strategy Memory" count={strategies.length}
                 hint="Which test strategies actually exposed defects; decay-weighted.">
            {strategies.length === 0 ? <Empty what="strategy scores" why="Reinforced when a test exposes a real defect." />
              : strategies.slice(0, 10).map((s, i) => (
                <div className="kv" key={i}>
                  <span>{s.strategy_type}</span>
                  <span className="v">{(s.decayed_score ?? s.effectiveness_score ?? 0).toFixed(2)}
                    <span className="area-tag" style={{ marginLeft: 6 }}>{s.times_effective ?? 0}/{s.times_applied ?? 0}</span>
                  </span>
                </div>
              ))}
          </Panel>

          <Panel title="🧭 Navigation Memory" count={nav.nav_nodes ?? 0}
                 hint="Proven shortest routes, and routes that keep failing (avoid).">
            <div className="kv"><span>Nodes learned</span><span className="v">{nav.nav_nodes ?? 0}</span></div>
            <div className="kv"><span>Marked avoid</span><span className="v">{nav.avoid_nodes ?? 0}</span></div>
            <div className="kv"><span>Max depth</span><span className="v">{nav.max_depth ?? 0}</span></div>
            {navFailed.length > 0 && (
              <div style={{ marginTop: 8 }}>
                {navFailed.slice(0, 5).map((f, i) =>
                  <div className="exec-path" key={i}>✗ {f.screen_name || f.action || JSON.stringify(f).slice(0, 70)}</div>)}
              </div>
            )}
          </Panel>

          <Panel title="🔁 Error Patterns" count={patterns.length}
                 hint="Recurring failure signatures mined from execution logs.">
            {patterns.length === 0 ? <Empty what="error patterns" why="Mined once the same failure recurs." />
              : patterns.slice(0, 8).map((p, i) => (
                <div className="kv" key={i}>
                  <span>{p.description || p.pattern_signature}</span>
                  <span className="v">{p.frequency ?? ''}</span>
                </div>
              ))}
          </Panel>
        </div>
      </div>

      <Panel title="📜 Ingestion Result — what the planner is grounded on"
             count={rules.length ? `${rules.length} rules` : null}
             hint="The authoritative requirements block built from the SRS. If this is wrong, every generated test is wrong.">
        <div className="kv"><span>Requirements covered</span>
          <span className="v">{reqCov.covered_requirements ?? 0} / {reqCov.total_requirements ?? 0}</span></div>
        <div className="kv"><span>Validation rules extracted</span><span className="v">{rules.length}</span></div>
        <div className="kv"><span>Tests with effectiveness scores</span><span className="v">{eff.length}</span></div>
        <button className="log-tab" style={{ marginTop: 10 }} onClick={() => setShowPrompt(v => !v)}>
          {showPrompt ? 'Hide' : 'Show'} the exact ingestion-derived prompt block
        </button>
        {showPrompt && (
          <pre className="directive" style={{ marginTop: 10, maxHeight: 420, overflow: 'auto' }}>
            {d.ingestion_prompt || '(empty — ingestion produced no requirements block)'}
          </pre>
        )}
      </Panel>
    </>
  )
}
