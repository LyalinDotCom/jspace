import React, { useState } from 'react'

export default function Inspector({ cols, sel, trackTokens, onClose, onPin, onSelectLayer }) {
  const [axis, setAxis] = useState('layers')
  const cell = cols[sel.x]
  if (!cell) return null
  const layers = [...cell.lens.keys()].reverse()
  const selected = cell.lens[sel.l]
  return (
    <aside className="inspector">
      <div className="insphead">
        <div>
          <span className="eyebrow">Position evidence</span>
          <div className="ttl">{JSON.stringify(cell.token)}</div>
          <div className="sub">position {sel.x} · {cell.gen ? 'generated' : 'prompt'} · selected L{sel.l}</div>
        </div>
        <button onClick={onClose} aria-label="Close inspector">×</button>
      </div>
      {selected && (
        <div className="selected-readout">
          <small>{regimeFor(sel.l, cell.lens.length)} readout · top concepts</small>
          <div>{selected.topk.slice(0, 5).map((t, i) => (
            <button key={i} onClick={() => onPin(t)}><b>{clean(t)}</b><span>{(selected.topk_p[i] * 100).toFixed(2)}%</span></button>
          ))}</div>
          <p>entropy {selected.entropy} · excess kurtosis {selected.kurt}{selected.final_rank ? ` · predicted-next rank ${selected.final_rank}` : ''}</p>
        </div>
      )}
      {trackTokens.length > 0 && selected?.track_rank && (
        <div className="pin-readout">
          {trackTokens.map((token, i) => (
            <span key={token}><b>{token}</b> rank {selected.track_rank[i] ?? '—'}</span>
          ))}
        </div>
      )}
      <div className="axis-tabs">
        <button className={axis === 'layers' ? 'active' : ''} onClick={() => setAxis('layers')}>By layer · fixed position</button>
        <button className={axis === 'positions' ? 'active' : ''} onClick={() => setAxis('positions')}>By position · fixed L{sel.l}</button>
      </div>
      <div className="inspbody">
        {axis === 'layers' ? layers.map(l => {
          const row = cell.lens[l]
          const regime = regimeFor(l, layers.length)
          return (
            <div key={l} className={`lrow ${regime} ${sel.l === l ? 'sel' : ''}`}
                 onClick={() => onSelectLayer(l)}>
              <span className="lno">L{String(l).padStart(2, '0')}</span>
              <span className="ltoks">
                {row.topk.slice(0, 5).map((t, i) => (
                  <button key={i} className="tok" title={`p=${(row.topk_p[i] * 100).toFixed(3)}%`}
                          onClick={e => { e.stopPropagation(); onPin(t) }}>
                    <i style={{ width: `${Math.min(100, row.topk_p[i] * 100)}%` }} />
                    <b>{shorten(t)}</b>
                  </button>
                ))}
              </span>
            </div>
          )
        }) : cols.map((col, x) => {
          const row = col.lens[sel.l]
          if (!row) return null
          return (
            <div key={x} className={`lrow position ${sel.x === x ? 'sel' : ''}`}>
              <span className="lno">P{String(x).padStart(2, '0')}</span>
              <span className="source-token" title={col.token}>{shorten(col.token)}</span>
              <span className="ltoks compact">
                {row.topk.slice(0, 3).map((t, i) => (
                  <button key={i} className="tok" title={`p=${(row.topk_p[i] * 100).toFixed(3)}%`}
                          onClick={() => onPin(t)}>
                    <i style={{ width: `${Math.min(100, row.topk_p[i] * 100)}%` }} />
                    <b>{shorten(t)}</b>
                  </button>
                ))}
              </span>
            </div>
          )
        })}
      </div>
    </aside>
  )
}

function regimeFor(layer, nLayers) {
  if (layer < Math.floor(nLayers / 3)) return 'early'
  if (layer >= nLayers - 3) return 'motor'
  return 'workspace'
}

function clean(token = '') {
  return token.replaceAll('\n', '↵').trim() || '∅'
}

function shorten(token = '') {
  const s = clean(token)
  return s.length > 10 ? s.slice(0, 10) + '…' : s
}
