import React from 'react'

/** A compact narrative slice through depth. It supplements rather than
 * replaces the full matrix: seven checkpoints show how the selected token
 * position changes from embedding to motor/output layers. */
export default function ThoughtTimeline({ cols, sel, nLayers, onSelectLayer, onPin }) {
  if (!cols.length) {
    return (
      <section className="thought-timeline empty">
        <div><span className="eyebrow">Layer story</span><b>Waiting for a forward pass</b></div>
        <p>The most legible concept readouts will appear here as depth unfolds.</p>
      </section>
    )
  }
  const pos = Math.min(sel?.x ?? cols.length - 1, cols.length - 1)
  const cell = cols[pos]
  const checkpoints = [...new Set([
    0,
    Math.floor((nLayers - 1) / 6),
    Math.floor((nLayers - 1) / 3),
    Math.floor((nLayers - 1) / 2),
    Math.floor((nLayers - 1) * 2 / 3),
    nLayers - 4,
    nLayers - 1,
  ])].filter(l => l >= 0 && cell.lens[l])
  const persistent = persistentReadouts(cell, nLayers)

  return (
    <section className="thought-timeline">
      <div className="timeline-head">
        <div><span className="eyebrow">Layer story · depth → time</span><b>Position {pos} · {JSON.stringify(cell.token)}</b></div>
        <p>Top readout at seven checkpoints. Click a stage for the full layer evidence.</p>
      </div>
      <div className="timeline-stages">
        {checkpoints.map((l, i) => {
          const row = cell.lens[l]
          const regime = regimeFor(l, nLayers)
          return (
            <button key={l} className={`stage ${regime} ${sel?.l === l ? 'selected' : ''}`}
                    onClick={() => onSelectLayer(l)}>
              <small>{i === 0 ? 'start' : i === checkpoints.length - 1 ? 'output' : regime} · L{l}</small>
              <b>{clean(row.topk[0])}</b>
              <span>{row.topk.slice(1, 3).map(clean).join(' · ')}</span>
              <i style={{ width: `${Math.min(100, (row.topk_p[0] ?? 0) * 100)}%` }} />
            </button>
          )
        })}
      </div>
      {persistent.length > 0 && (
        <div className="persistent-readouts">
          <div><span className="eyebrow">Persistent middle-layer readouts</span><small>Repeated vocabulary evidence, not a causal claim</small></div>
          {persistent.map(item => (
            <button key={item.token} onClick={() => onPin(item.token)}>
              <b>{item.token}</b>
              <span>{item.layers} layers · best {(item.best * 100).toFixed(2)}%</span>
            </button>
          ))}
        </div>
      )}
    </section>
  )
}

function persistentReadouts(cell, nLayers) {
  const counts = new Map()
  const lo = Math.floor(nLayers / 3)
  const hi = nLayers - 4
  for (let layer = lo; layer <= hi; layer++) {
    const row = cell.lens[layer]
    const seen = new Set()
    row.topk.slice(0, 5).forEach((raw, rank) => {
      const token = clean(raw)
      if (!token || token === '∅' || token.startsWith('<') || token.length > 22 || seen.has(token)) return
      seen.add(token)
      const current = counts.get(token) ?? { token, layers: 0, best: 0, rankSum: 0 }
      current.layers += 1
      current.best = Math.max(current.best, row.topk_p[rank] ?? 0)
      current.rankSum += rank
      counts.set(token, current)
    })
  }
  return [...counts.values()]
    .filter(item => item.layers >= 2)
    .sort((a, b) => b.layers - a.layers || b.best - a.best || a.rankSum - b.rankSum)
    .slice(0, 6)
}

function regimeFor(layer, nLayers) {
  if (layer < Math.floor(nLayers / 3)) return 'early'
  if (layer >= nLayers - 3) return 'motor'
  return 'workspace'
}

function clean(token = '') {
  const value = token.replaceAll('\n', '↵').trim()
  return value || '∅'
}
