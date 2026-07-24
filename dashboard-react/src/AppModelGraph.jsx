import React, { useEffect, useMemo, useRef, useState } from 'react'

const W = 640, H = 380

// Lightweight force layout — a few iterations, deterministic enough for a small
// state graph. No external graph library (keeps the bundle self-contained).
function layout(nodes, edges) {
  const pos = {}
  const n = nodes.length || 1
  nodes.forEach((nd, i) => {
    const a = (2 * Math.PI * i) / n
    pos[nd.id] = { x: W / 2 + Math.cos(a) * Math.min(W, H) * 0.32, y: H / 2 + Math.sin(a) * Math.min(W, H) * 0.32 }
  })
  for (let it = 0; it < 160; it++) {
    for (const a of nodes) for (const b of nodes) {
      if (a.id === b.id) continue
      let dx = pos[a.id].x - pos[b.id].x, dy = pos[a.id].y - pos[b.id].y
      let d = Math.hypot(dx, dy) || 1
      const f = 2600 / (d * d)
      pos[a.id].x += (dx / d) * f; pos[a.id].y += (dy / d) * f
    }
    for (const e of edges) {
      const s = pos[e.source], t = pos[e.target]
      if (!s || !t) continue
      let dx = t.x - s.x, dy = t.y - s.y, d = Math.hypot(dx, dy) || 1
      const f = (d - 130) * 0.02
      s.x += (dx / d) * f; s.y += (dy / d) * f; t.x -= (dx / d) * f; t.y -= (dy / d) * f
    }
    for (const nd of nodes) {
      pos[nd.id].x += (W / 2 - pos[nd.id].x) * 0.01
      pos[nd.id].y += (H / 2 - pos[nd.id].y) * 0.01
      pos[nd.id].x = Math.max(36, Math.min(W - 36, pos[nd.id].x))
      pos[nd.id].y = Math.max(30, Math.min(H - 30, pos[nd.id].y))
    }
  }
  return pos
}

export default function AppModelGraph({ project, nodes, edges, highlightPath }) {
  const hlNodes = new Set(highlightPath || [])
  const hlEdges = new Set()
  for (let i = 0; i + 1 < (highlightPath || []).length; i++) hlEdges.add(highlightPath[i] + '>' + highlightPath[i + 1])
  const pathActive = hlNodes.size > 0
  const key = useMemo(() => nodes.map(n => n.id).join(',') + '|' + edges.map(e => e.source + '>' + e.target).join(','), [nodes, edges])
  const [pos, setPos] = useState({})
  const [sel, setSel] = useState(null)
  const drag = useRef(null)

  useEffect(() => { setPos(layout(nodes, edges)); }, [key])
  useEffect(() => { if (sel && !nodes.find(n => n.id === sel)) setSel(null); }, [key])

  const byId = useMemo(() => Object.fromEntries(nodes.map(n => [n.id, n])), [nodes])
  const selNode = sel ? byId[sel] : (nodes[0] || null)

  function svgPoint(evt) {
    const svg = evt.currentTarget.ownerSVGElement || evt.currentTarget
    const r = svg.getBoundingClientRect()
    return { x: ((evt.clientX - r.left) / r.width) * W, y: ((evt.clientY - r.top) / r.height) * H }
  }
  function onDown(id, evt) { drag.current = id; setSel(id); evt.stopPropagation() }
  function onMove(evt) {
    if (!drag.current) return
    const p = svgPoint(evt)
    setPos(prev => ({ ...prev, [drag.current]: { x: Math.max(30, Math.min(W - 30, p.x)), y: Math.max(24, Math.min(H - 24, p.y)) } }))
  }
  function onUp() { drag.current = null }

  if (!nodes.length) return <div className="empty">No app-model states yet. Run the executor — the map builds itself as the agent explores.</div>

  return (
    <div className="graph-wrap">
      <svg className="graph-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
           onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--muted)" opacity="0.6" />
          </marker>
        </defs>
        {edges.map((e, i) => {
          const s = pos[e.source], t = pos[e.target]
          if (!s || !t) return null
          const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2
          const on = hlEdges.has(e.source + '>' + e.target)
          return (
            <g key={i}>
              <line className={'graph-edge' + (on ? ' hl' : (pathActive ? ' dim' : ''))} x1={s.x} y1={s.y} x2={t.x} y2={t.y} markerEnd="url(#arrow)" />
              {e.action ? <text className="graph-edge-label" x={mx} y={my - 2} textAnchor="middle">{e.action}</text> : null}
            </g>
          )
        })}
        {nodes.map(n => {
          const p = pos[n.id]; if (!p) return null
          const onPath = hlNodes.has(n.id)
          const cls = 'graph-node' + (n.has_dialog ? ' dialog' : '')
            + (n.id === sel ? ' selected' : '')
            + (pathActive ? (onPath ? ' hl' : ' dim') : '')
          const r = 12 + Math.min(6, Math.log2((n.visits || 1) + 1) * 1.6)
          return (
            <g key={n.id} className={cls} transform={`translate(${p.x},${p.y})`}
               onMouseDown={(e) => onDown(n.id, e)} onClick={() => setSel(n.id)}>
              <circle r={r} />
              <text textAnchor="middle" y={r + 11}>{(n.label || '?').slice(0, 18)}</text>
            </g>
          )
        })}
      </svg>

      <div className="state-preview">
        {selNode ? (
          <>
            {selNode.has_shot
              ? <img src={`/dashboard/screenshot?project=${encodeURIComponent(project)}&state_id=${encodeURIComponent(selNode.id)}`} alt="" />
              : <div className="noshot">no screenshot</div>}
            <div className="cap">
              <div className="name">{selNode.label || '?'}</div>
              <div className="sub">{selNode.visits || 0} visits · {selNode.elements || 0} controls{selNode.has_dialog ? ' · dialog' : ''}</div>
            </div>
          </>
        ) : <div className="noshot">select a state</div>}
      </div>
    </div>
  )
}
