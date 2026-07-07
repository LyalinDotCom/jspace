import React from 'react'

/** Full readout across layers at one position — the paper's fig. 5 center
 *  column. Click any token to pin it (adds to the track list). */
export default function Inspector({ cols, sel, trackTokens, onClose, onPin, onSelectLayer }) {
  const cell = cols[sel.x]
  if (!cell) return null
  const layers = [...cell.lens.keys()].reverse()
  return (
    <div className="inspector">
      <div className="insphead">
        <div>
          <div className="ttl">pos {sel.x} · {JSON.stringify(cell.token)}</div>
          <div className="sub">{cell.gen ? 'generated' : 'prompt'} token · click a token to pin</div>
        </div>
        <button onClick={onClose}>×</button>
      </div>
      <div className="inspbody">
        {layers.map(l => {
          const L = cell.lens[l]
          return (
            <div key={l}
                 className={'lrow' + (sel.l === l ? ' sel' : '')}
                 onClick={() => onSelectLayer(l)}>
              <span className="lno">L{l}</span>
              <span className="ltoks">
                {L.topk.slice(0, 3).map((t, i) => (
                  <button key={i} className="tok"
                          title={`p=${(L.topk_p[i] * 100).toFixed(2)}%`}
                          onClick={e => { e.stopPropagation(); onPin(t) }}>
                    <i style={{ width: `${Math.min(100, L.topk_p[i] * 100)}%` }} />
                    <b>{shorten(t)}</b>
                  </button>
                ))}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function shorten(t) {
  const s = t.replaceAll('\n', '⏎')
  return s.length > 12 ? s.slice(0, 12) + '…' : s
}
