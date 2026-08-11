import React, { useEffect, useState } from 'react'

const TOOL_ICON = { click: '👆', tap: '👆', type: '⌨️', swipe: '↔️', scroll: '↕️', back: '↩️',
                    open_app: '📱', press_key: '🔑', wait: '⏳', complete: '🏁', screenshot: '📷' }

/** Render tool args compactly: {index:41, text:'Alice'} -> index 41 · "Alice" */
function args(a) {
  if (!a || typeof a !== 'object') return ''
  return Object.entries(a)
    .map(([k, v]) => (k === 'text' ? `"${v}"` : `${k} ${v}`))
    .join(' · ')
}

export default function RunSteps({ createdAt }) {
  const [state, setState] = useState({ loading: true })

  useEffect(() => {
    let alive = true
    fetch(`/dashboard/run-steps?created_at=${encodeURIComponent(createdAt || '')}`, { cache: 'no-store' })
      .then(r => r.json())
      .then(d => { if (alive) setState({ loading: false, ...d }) })
      .catch(() => { if (alive) setState({ loading: false, found: false, reason: 'request failed' }) })
    return () => { alive = false }
  }, [createdAt])

  if (state.loading) return <div className="steps-note">Loading device steps…</div>
  if (!state.found) return <div className="steps-note">No device trace for this run ({state.reason}).</div>

  const steps = state.steps || []
  const outcome = state.outcome || {}

  return (
    <div className="steps">
      <div className="steps-note">
        {steps.length} device action{steps.length === 1 ? '' : 's'} · trajectory <span className="mono">{state.trajectory}</span>
      </div>
      {steps.map(s => (
        <div className={'step' + (s.success ? '' : ' bad')} key={s.n}>
          <div className="step-head">
            <span className="step-n">{s.n}</span>
            <span className="step-tool">{TOOL_ICON[s.tool] || '⚙️'} {s.tool}</span>
            <span className="step-args">{args(s.args)}</span>
            <span className={'chip ' + (s.success ? 'pass' : 'fail')}>{s.success ? 'ok' : 'failed'}</span>
          </div>
          {s.summary ? <div className="step-sum">{s.summary}</div> : null}
          {s.thought ? <div className="step-thought">{s.thought}</div> : null}
        </div>
      ))}
      {outcome.reason ? (
        <div className={'step-outcome' + (outcome.success ? ' ok' : '')}>
          <b>{outcome.success ? 'Agent reported success' : 'Agent reported failure'}:</b> {outcome.reason}
        </div>
      ) : null}
    </div>
  )
}
