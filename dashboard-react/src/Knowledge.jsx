import React, { useState } from 'react'

// Kind -> visual family. Deliberately mirrors rag_api/findings.py's groups:
// AGENT_DIFFICULTY is OUR agent struggling and must never read as an app defect,
// so it gets its own colour rather than sharing the defect one.
const FAMILY = {
  SPEC_VIOLATION: 'defect', SUSPECTED_DEFECT: 'defect',
  CONFIRMED_BEHAVIOUR: 'confirmed', SPEC_GAP: 'confirmed', CONTROL_DISCOVERED: 'confirmed',
  AGENT_DIFFICULTY: 'agent', UNVERIFIED: 'unverified', UNEXPECTED_BEHAVIOUR: 'unverified',
}
const short = (k) => String(k || '').replace(/_/g, ' ').toLowerCase()

const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'defect', label: 'Candidate defects' },
  { id: 'confirmed', label: 'Confirmed' },
  { id: 'unverified', label: 'Unsettled' },
  { id: 'agent', label: 'Our agent' },
]

export function Findings({ findings, stats }) {
  const [filter, setFilter] = useState('all')
  const rows = (findings || []).filter(f => filter === 'all' || FAMILY[f.kind] === filter)
  const byKind = Object.fromEntries((stats?.by_kind || []).map(r => [r.kind, r.n]))
  const total = stats?.total ?? (findings || []).length

  return (
    <div className="panel">
      <h2>🔬 Findings <span className="count">{total ? `(${total})` : ''}</span></h2>
      <div className="body">
        <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 10 }}>
          What previous runs established about the app — one atomic claim each, deduplicated on
          write, so a repeat reinforces a finding instead of creating a near-duplicate.
          <b> Seen Nx</b> means N independent runs derived it.
        </div>
        <div className="kindbar">
          {FILTERS.map(f => {
            const n = f.id === 'all' ? total
              : Object.entries(byKind).filter(([k]) => FAMILY[k] === f.id)
                  .reduce((a, [, v]) => a + v, 0)
            return (
              <button key={f.id} className={filter === f.id ? 'on' : ''}
                      onClick={() => setFilter(f.id)}>{f.label} {n ? `· ${n}` : ''}</button>
            )
          })}
        </div>
        <div className="scroll" style={{ maxHeight: 460 }}>
          {rows.length ? rows.map((f, i) => {
            const fam = FAMILY[f.kind] || 'unverified'
            return (
              <div key={f.ref || i} className={'finding ' + fam}>
                <div className="claim">{f.claim}</div>
                <div className="meta">
                  <span className={'tag ' + fam}>{short(f.kind)}</span>
                  {f.screen ? <span>📍 {f.screen}</span> : null}
                  {f.times_seen > 1 ? <span className="tag seen">seen {f.times_seen}×</span> : null}
                  {f.severity && f.severity !== 'medium' ? <span>severity {f.severity}</span> : null}
                  {f.ref ? <span style={{ opacity: .55 }}>{f.ref}</span> : null}
                </div>
              </div>
            )
          }) : <div style={{ color: 'var(--muted)', padding: '14px 2px' }}>
                 Nothing recorded for this filter yet.</div>}
        </div>
      </div>
    </div>
  )
}

export function OpenQuestions({ open, findings, maxAttempts }) {
  const questions = open?.open_questions || []
  // A question closed by an ANSWER, not by running out of attempts — the
  // difference between the agent concluding something and giving up.
  const answered = (findings || []).filter(f => f.resolution)
  const cap = maxAttempts || open?.max_attempts || 3

  return (
    <div className="panel">
      <h2>❓ Open questions <span className="count">
        {open?.total_open ? `(${open.total_open} unsettled · ${answered.length} answered)` : ''}
      </span></h2>
      <div className="body">
        <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 10 }}>
          Things a run raised but never settled. The planner targets one, spends an attempt, and
          the investigator closes it when a later run answers it. After {cap} attempts it closes as
          inconclusive — a result for a human, not an endless retry.
        </div>
        <div className="scroll" style={{ maxHeight: 400 }}>
          {answered.map((f, i) => (
            <div key={'a' + i} className="q answered">
              <div style={{ color: 'var(--muted)' }}>{f.claim}</div>
              <div className="qa">
                <div className="lbl">ANSWERED</div>
                <div>{f.resolution}</div>
              </div>
            </div>
          ))}
          {questions.map((q, i) => (
            <div key={q.ref || i} className="q">
              <div>{q.claim}</div>
              <div className="meta" style={{ display: 'flex', gap: 9, alignItems: 'center',
                                             marginTop: 7, fontSize: 11, color: 'var(--muted)' }}>
                <span className="attempts">
                  {Array.from({ length: cap }).map((_, j) =>
                    <i key={j} className={j < (q.attempts || 0) ? 'used' : ''} />)}
                </span>
                <span>{q.attempts || 0}/{cap} attempts</span>
                {q.screen ? <span>📍 {q.screen}</span> : null}
                <span className="tag unverified">{short(q.kind)}</span>
              </div>
            </div>
          ))}
          {!questions.length && !answered.length
            ? <div style={{ color: 'var(--muted)', padding: '14px 2px' }}>
                No open questions — every finding so far has reached a conclusion.</div>
            : null}
        </div>
      </div>
    </div>
  )
}

export function Campaigns({ campaigns }) {
  const rows = campaigns || []
  return (
    <div className="panel">
      <h2>📈 Campaign history <span className="count">{rows.length ? `(${rows.length})` : ''}</span></h2>
      <div className="body">
        <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 10 }}>
          Each row is a snapshot taken just before a reset wiped that campaign's tests. Comparing
          consecutive rows is the cheapest available answer to "is the agent getting better?"
        </div>
        {rows.length ? (
          <div className="scroll" style={{ maxHeight: 330 }}>
            <table className="camp">
              <thead><tr>
                <th>Ended</th><th>Tests</th><th>Pass / Fail</th><th>Steps</th>
                <th>Findings</th><th>Requirements</th><th>Screens</th>
              </tr></thead>
              <tbody>
                {rows.map((c, i) => (
                  <tr key={i}>
                    <td>{String(c.ended_at || '').slice(0, 16).replace('T', ' ')}</td>
                    <td className="n">{c.tests ?? '—'}</td>
                    <td className="n">
                      <span style={{ color: 'var(--pass)' }}>{c.passed ?? 0}</span>
                      {' / '}
                      <span style={{ color: 'var(--fail)' }}>{c.failed ?? 0}</span>
                    </td>
                    <td className="n">{c.device_steps_total ?? 0}
                      <span style={{ color: 'var(--muted)' }}> (μ{c.device_steps_mean ?? 0})</span></td>
                    <td className="n">{c.findings_total ?? 0}</td>
                    <td className="n">{c.requirements_covered ?? 0}/{c.requirements_total ?? 0}</td>
                    <td className="n">{c.app_model_states ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: 'var(--muted)' }}>
              No completed campaigns yet — a snapshot is written at the start of the next one.</div>}
      </div>
    </div>
  )
}
