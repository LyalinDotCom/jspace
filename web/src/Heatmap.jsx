import React, { useEffect, useRef, useState } from 'react'

const CW = 16, CH = 11, PADL = 46, PADT = 50

// viridis-ish ramp
const STOPS = [[13, 16, 23], [40, 60, 110], [35, 110, 140], [60, 170, 120], [180, 220, 80], [253, 240, 120]]
export function color(v) {
  v = Math.max(0, Math.min(1, v))
  const x = v * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), f = x - i
  const c = STOPS[i].map((a, k) => Math.round(a + (STOPS[i + 1][k] - a) * f))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

// rank -> [0,1]: rank 1 = 1.0, fading over log scale to rank 10000+ = 0
function rankValue(r) {
  if (!r || r < 1) return 0
  return Math.max(0, 1 - Math.log10(r) / 4)
}

function cellValue(cell, l, colorBy) {
  const L = cell.lens[l]
  switch (colorBy) {
    case 'rank':
      return L.track_rank ? rankValue(Math.min(...L.track_rank)) : 0
    case 'prob':
      return L.track_p ? Math.max(...L.track_p) : 0
    case 'kurt':
      return Math.min(1, Math.max(0, (L.kurt ?? 0) / 60))
    default: // conf
      return 1 - Math.min(1, L.entropy / 10)
  }
}

export default function Heatmap({ cols, nLayers, colorBy, sel, onSelect }) {
  const cvRef = useRef(null)
  const wrapRef = useRef(null)
  const [tip, setTip] = useState(null)

  useEffect(() => {
    const cv = cvRef.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    cv.width = PADL + Math.max(1, cols.length) * CW + 20
    cv.height = PADT + nLayers * CH + 10
    ctx.fillStyle = '#0d1017'
    ctx.fillRect(0, 0, cv.width, cv.height)
    ctx.font = '9px Menlo'
    ctx.fillStyle = '#8b93a7'
    for (let l = 0; l < nLayers; l += 4) {
      ctx.fillText('L' + l, 8, PADT + (nLayers - 1 - l) * CH + 9)
    }
    cols.forEach((c, x) => {
      for (let l = 0; l < nLayers; l++) {
        ctx.fillStyle = color(cellValue(c, l, colorBy))
        ctx.fillRect(PADL + x * CW, PADT + (nLayers - 1 - l) * CH, CW - 1, CH - 1)
      }
      ctx.save()
      ctx.translate(PADL + x * CW + CW / 2 + 3, PADT - 6)
      ctx.rotate(-Math.PI / 4)
      ctx.fillStyle = c.gen ? '#9ef0b1' : '#8b93a7'
      ctx.fillText(c.token.replaceAll('\n', '⏎').slice(0, 10), 0, 0)
      ctx.restore()
    })
    // selection crosshair
    if (sel && sel.x < cols.length) {
      ctx.strokeStyle = '#5eb1ff'
      ctx.lineWidth = 1.5
      ctx.strokeRect(PADL + sel.x * CW - 1, PADT - 1, CW + 1, nLayers * CH + 1)
      if (sel.l != null) {
        ctx.strokeRect(PADL + sel.x * CW - 1, PADT + (nLayers - 1 - sel.l) * CH - 1,
                       CW + 1, CH + 1)
      }
    }
    if (wrapRef.current && !sel) wrapRef.current.scrollLeft = 1e9
  }, [cols, nLayers, colorBy, sel])

  const cellAt = e => {
    const r = cvRef.current.getBoundingClientRect()
    const x = Math.floor((e.clientX - r.left - PADL) / CW)
    const l = nLayers - 1 - Math.floor((e.clientY - r.top - PADT) / CH)
    if (x < 0 || x >= cols.length || l < 0 || l >= nLayers) return null
    return { x, l }
  }

  return (
    <div className="hmwrap" ref={wrapRef}>
      <canvas
        ref={cvRef}
        onMouseMove={e => {
          const c = cellAt(e)
          setTip(c ? { ...c, cx: e.clientX, cy: e.clientY, cell: cols[c.x] } : null)
        }}
        onMouseLeave={() => setTip(null)}
        onClick={e => { const c = cellAt(e); if (c) onSelect(c) }}
      />
      {tip && <Tip tip={tip} />}
    </div>
  )
}

function Tip({ tip }) {
  const L = tip.cell.lens[tip.l]
  const style = {
    left: Math.min(tip.cx + 14, window.innerWidth - 340),
    top: Math.min(tip.cy + 14, window.innerHeight - 200),
  }
  return (
    <div className="tip" style={style}>
      <div className="h">layer {tip.l} · pos {tip.x} · {JSON.stringify(tip.cell.token)}</div>
      {L.topk.map((t, i) => (
        <div className="r" key={i}>
          <b>{JSON.stringify(t)}</b><span>{(L.topk_p[i] * 100).toFixed(1)}%</span>
        </div>
      ))}
      {L.track_rank && L.track_rank.map((r, i) => (
        <div className="r track" key={'t' + i}>
          <b>pin #{i + 1}</b><span>rank {r} · {(L.track_p[i] * 100).toFixed(2)}%</span>
        </div>
      ))}
      <div className="r"><span>entropy {L.entropy} nats</span><span>kurt {L.kurt}</span></div>
    </div>
  )
}
