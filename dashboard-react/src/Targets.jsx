import React, { useCallback, useEffect, useRef, useState } from 'react'

// Mirrors targets/schema.py's dataclass defaults so a brand-new profile starts
// from the same shape the backend would build via TargetProfile().
const EMPTY_LOGIN = { url: '', user: '', password: '', hint: '', role: '' }
const EMPTY_WEB = {
  base_url: '', browser: 'chromium', headless: false, slow_mo_ms: 300, viewport: '1280x800',
  same_origin_only: true, blocked_texts: [], blocked_url_patterns: [], storage_state: '',
  fail_on_page_error: false, fail_on_http_5xx: false, console_ignore: [], login: { ...EMPTY_LOGIN },
}
const EMPTY_ANDROID = {
  package: '', activity: '', labels: [], target_app_only: false, device_reset: 'pm_clear',
  login: { ...EMPTY_LOGIN },
}
const EMPTY_PROFILE = {
  name: '', kind: 'web', project: '', display_name: '', description: '',
  web: EMPTY_WEB, android: EMPTY_ANDROID,
  knowledge: { srs_path: '', figma_path: '', defects_path: '' },
  run: { rounds: 5, max_steps: 30, timeout: 420, clean_slate: true, self_heal: true },
  model: { provider: '', model: '' },
}

function getPath(obj, path) {
  return path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj)
}
function setPath(obj, path, value) {
  const keys = path.split('.')
  const clone = JSON.parse(JSON.stringify(obj))
  let cur = clone
  for (let i = 0; i < keys.length - 1; i++) cur = cur[keys[i]]
  cur[keys[keys.length - 1]] = value
  return clone
}
function normalizeProfile(raw) {
  // Backfill any section a saved profile might be missing (e.g. an older file
  // written before a field existed) so every input always has a defined value.
  return {
    ...EMPTY_PROFILE, ...raw,
    web: { ...EMPTY_WEB, ...(raw.web || {}), login: { ...EMPTY_LOGIN, ...((raw.web || {}).login || {}) } },
    android: { ...EMPTY_ANDROID, ...(raw.android || {}), login: { ...EMPTY_LOGIN, ...((raw.android || {}).login || {}) } },
    knowledge: { ...EMPTY_PROFILE.knowledge, ...(raw.knowledge || {}) },
    run: { ...EMPTY_PROFILE.run, ...(raw.run || {}) },
    model: { ...EMPTY_PROFILE.model, ...(raw.model || {}) },
  }
}

function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span>{label}{hint ? <em> {hint}</em> : null}</span>
      {children}
    </label>
  )
}

function TextField({ profile, path, label, hint, onChange, placeholder }) {
  return (
    <Field label={label} hint={hint}>
      <input type="text" value={getPath(profile, path) ?? ''} placeholder={placeholder}
             onChange={(e) => onChange(path, e.target.value)} />
    </Field>
  )
}
function NumberField({ profile, path, label, hint, onChange }) {
  return (
    <Field label={label} hint={hint}>
      <input type="number" value={getPath(profile, path) ?? 0}
             onChange={(e) => onChange(path, Number(e.target.value))} />
    </Field>
  )
}
function CheckField({ profile, path, label, onChange }) {
  return (
    <label className="field checkbox">
      <input type="checkbox" checked={!!getPath(profile, path)}
             onChange={(e) => onChange(path, e.target.checked)} />
      <span>{label}</span>
    </label>
  )
}
function ListField({ profile, path, label, onChange }) {
  const value = getPath(profile, path)
  const text = Array.isArray(value) ? value.join(', ') : ''
  return (
    <Field label={label} hint="(comma-separated)">
      <input type="text" value={text}
             onChange={(e) => onChange(path, e.target.value.split(',').map((s) => s.trim()).filter(Boolean))} />
    </Field>
  )
}
function SelectField({ profile, path, label, options, onChange }) {
  return (
    <Field label={label}>
      <select value={getPath(profile, path) ?? ''} onChange={(e) => onChange(path, e.target.value)}>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </Field>
  )
}

function LoginFields({ profile, section, onChange }) {
  return (
    <>
      <TextField profile={profile} path={`${section}.login.role`} label="Login role"
                 hint="what the PLANNER is told (e.g. admin, guest)" onChange={onChange} />
      <TextField profile={profile} path={`${section}.login.user`} label="Login user / identifier" onChange={onChange} />
      <TextField profile={profile} path={`${section}.login.password`} label="Login password"
                 hint="the player only — never sent to the planner" onChange={onChange} />
      {section === 'web' && <TextField profile={profile} path="web.login.url" label="Login URL" onChange={onChange} />}
      <TextField profile={profile} path={`${section}.login.hint`} label="Login hint"
                 hint="OTP/PIN/button — anything the user/password pair doesn't capture" onChange={onChange} />
    </>
  )
}

export default function Targets() {
  const [list, setList] = useState([])
  const [selected, setSelected] = useState('')
  const [profile, setProfile] = useState(null)
  const [errors, setErrors] = useState([])
  const [saveMsg, setSaveMsg] = useState('')
  const [runInfo, setRunInfo] = useState(null)
  const [status, setStatus] = useState(null)
  const pollRef = useRef(null)

  const refreshList = useCallback(async () => {
    try {
      const r = await fetch('/targets')
      const d = await r.json()
      setList(d.profiles || [])
    } catch { /* dashboard banner already covers "gateway unreachable" elsewhere */ }
  }, [])

  useEffect(() => { refreshList() }, [refreshList])
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  function pollRunStatus(name) {
    if (pollRef.current) clearInterval(pollRef.current)
    const tick = async () => {
      try {
        const r = await fetch(`/targets/${encodeURIComponent(name)}/run-status`)
        const d = await r.json()
        setStatus(d)
        if (!d.running && pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
      } catch { /* transient — next tick retries */ }
    }
    tick()
    pollRef.current = setInterval(tick, 3000)
  }

  async function openProfile(name) {
    setSelected(name); setSaveMsg(''); setRunInfo(null); setStatus(null)
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
    if (name === '__new__') {
      setProfile(normalizeProfile({ name: '' })); setErrors([])
      return
    }
    const r = await fetch(`/targets/${encodeURIComponent(name)}`)
    const d = await r.json()
    setProfile(normalizeProfile(d.profile || {}))
    setErrors(d.errors || [])
    pollRunStatus(name)
  }

  function update(path, value) {
    setProfile((p) => setPath(p, path, value))
  }

  async function validate() {
    const r = await fetch('/targets/validate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(profile),
    })
    const d = await r.json()
    setErrors(d.errors || [])
    return d.valid
  }

  async function save() {
    setSaveMsg('saving…')
    const name = (profile.name || '').trim()
    if (!name) { setErrors(['name: required']); setSaveMsg(''); return }
    const r = await fetch(`/targets/${encodeURIComponent(name)}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(profile),
    })
    const d = await r.json()
    if (r.status === 422) {
      setErrors((d.detail && d.detail.errors) || ['save failed'])
      setSaveMsg('')
      return
    }
    if (!r.ok) { setSaveMsg(`save failed: ${d.detail || r.status}`); return }
    setErrors([]); setSaveMsg(`saved to ${d.path}`)
    setSelected(name)
    refreshList()
    pollRunStatus(name)
  }

  async function run() {
    setSaveMsg('')
    const r = await fetch(`/targets/${encodeURIComponent(selected)}/run`, { method: 'POST' })
    const d = await r.json()
    if (!r.ok) { setSaveMsg(`could not start: ${(d.detail && d.detail.toString()) || r.status}`); return }
    setRunInfo(d)
    pollRunStatus(selected)
  }

  const kind = profile ? profile.kind : 'web'
  const isNew = selected === '__new__'
  const dirty = profile !== null

  return (
    <div className="targets-wrap">
      <div className="targets-list">
        <div className="panel">
          <h2>🎯 Target Profiles <span className="count">{list.length}</span></h2>
          <div className="body">
            <button className="btn" onClick={() => openProfile('__new__')}>+ New target</button>
            <div className="target-rows">
              {list.map((p) => (
                <div key={p.name} className={'target-row' + (selected === p.name ? ' active' : '')}
                     onClick={() => openProfile(p.name)}>
                  <div className="top">
                    <span className="title">{p.display_name || p.name}</span>
                    {!p.valid ? <span className="chip fail">invalid</span> : <span className="chip pass">{p.kind}</span>}
                  </div>
                  <div className="sub mono">{p.project ? `project=${p.project} ` : ''}{p.where}</div>
                </div>
              ))}
              {!list.length ? <div className="empty">No target profiles yet — create one.</div> : null}
            </div>
          </div>
        </div>
      </div>

      <div className="targets-form">
        {!dirty ? (
          <div className="panel"><div className="empty">Select a target profile, or create a new one.</div></div>
        ) : (
          <div className="panel">
            <h2>{isNew ? 'New target' : `Edit — ${selected}`}</h2>
            <div className="body">
              {errors.length ? (
                <div className="err-list">
                  {errors.map((e, i) => <div key={i}>⚠ {e}</div>)}
                </div>
              ) : null}

              <div className="field-grid">
                <TextField profile={profile} path="name" label="Name" hint="(id — used to select this profile)" onChange={update} />
                <SelectField profile={profile} path="kind" label="Kind" options={['web', 'android']} onChange={update} />
                <TextField profile={profile} path="project" label="Project"
                           hint="(scopes the knowledge graph — give every target its own)" onChange={update} />
                <TextField profile={profile} path="display_name" label="Display name" onChange={update} />
                <TextField profile={profile} path="description" label="Description" onChange={update} />
              </div>

              {kind === 'web' ? (
                <>
                  <h3>Website</h3>
                  <div className="field-grid">
                    <TextField profile={profile} path="web.base_url" label="Base URL" placeholder="https://example.com" onChange={update} />
                    <SelectField profile={profile} path="web.browser" label="Browser" options={['chromium', 'firefox', 'webkit']} onChange={update} />
                    <TextField profile={profile} path="web.viewport" label="Viewport" placeholder="1280x800" onChange={update} />
                    <NumberField profile={profile} path="web.slow_mo_ms" label="Slow-mo (ms)" onChange={update} />
                    <CheckField profile={profile} path="web.headless" label="Headless" onChange={update} />
                    <CheckField profile={profile} path="web.same_origin_only" label="Same-origin only" onChange={update} />
                    <CheckField profile={profile} path="web.fail_on_page_error" label="Fail on page error" onChange={update} />
                    <CheckField profile={profile} path="web.fail_on_http_5xx" label="Fail on HTTP 5xx" onChange={update} />
                    <TextField profile={profile} path="web.storage_state" label="Storage state file"
                               hint="(playwright codegen --save-storage=...)" onChange={update} />
                    <ListField profile={profile} path="web.blocked_texts" label="Blocked control texts" onChange={update} />
                    <ListField profile={profile} path="web.blocked_url_patterns" label="Blocked URL patterns" onChange={update} />
                    <ListField profile={profile} path="web.console_ignore" label="Console noise to ignore" onChange={update} />
                  </div>
                  <h3>Login</h3>
                  <div className="field-grid"><LoginFields profile={profile} section="web" onChange={update} /></div>
                </>
              ) : (
                <>
                  <h3>Android app</h3>
                  <div className="field-grid">
                    <TextField profile={profile} path="android.package" label="Package" placeholder="com.example.app" onChange={update} />
                    <TextField profile={profile} path="android.activity" label="Activity" onChange={update} />
                    <ListField profile={profile} path="android.labels" label="Launcher labels" onChange={update} />
                    <SelectField profile={profile} path="android.device_reset" label="Device reset"
                                 options={['pm_clear', 'force_stop', 'none']} onChange={update} />
                    <CheckField profile={profile} path="android.target_app_only" label="Target app only" onChange={update} />
                  </div>
                  <h3>Login</h3>
                  <div className="field-grid"><LoginFields profile={profile} section="android" onChange={update} /></div>
                </>
              )}

              <h3>Knowledge <em className="hint-inline">(optional — empty is zero-doc exploration)</em></h3>
              <div className="field-grid">
                <TextField profile={profile} path="knowledge.srs_path" label="SRS path" onChange={update} />
                <TextField profile={profile} path="knowledge.figma_path" label="Figma path" onChange={update} />
                <TextField profile={profile} path="knowledge.defects_path" label="Defects path" onChange={update} />
              </div>

              <h3>Run budget</h3>
              <div className="field-grid">
                <NumberField profile={profile} path="run.rounds" label="Rounds" hint="(test cases per batch)" onChange={update} />
                <NumberField profile={profile} path="run.max_steps" label="Max steps" hint="(per test case)" onChange={update} />
                <NumberField profile={profile} path="run.timeout" label="Timeout (s)" onChange={update} />
                <CheckField profile={profile} path="run.clean_slate" label="Clean slate before batch" onChange={update} />
                <CheckField profile={profile} path="run.self_heal" label="Self-heal" onChange={update} />
              </div>

              <h3>Model override <em className="hint-inline">(optional — blank uses the .env default)</em></h3>
              <div className="field-grid">
                <TextField profile={profile} path="model.provider" label="Provider" onChange={update} />
                <TextField profile={profile} path="model.model" label="Model" onChange={update} />
              </div>

              <div className="target-actions">
                <button className="btn" onClick={validate}>Validate</button>
                <button className="btn btn-primary" onClick={save}>Save</button>
                {!isNew ? (
                  <button className="btn btn-run" onClick={run} disabled={status && status.running}>
                    {status && status.running ? `Running (pid ${status.pid})…` : '▶ Run'}
                  </button>
                ) : null}
                {saveMsg ? <span className="save-msg">{saveMsg}</span> : null}
              </div>
              {runInfo ? <div className="hint">Started pid {runInfo.pid} — log: <span className="mono">{runInfo.log_path}</span></div> : null}
              {status && !status.running && status.exit_code != null ? (
                <div className="hint">Last run exited with code {status.exit_code}{status.exit_code === 0 ? ' ✓' : ' — check the log or the Live Logs panel above'}.</div>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
