import React, { useEffect, useRef, useState } from 'react'

const CW = 18, CH = 13, PADL = 82, PADT = 66

const STOPS = [
  [8, 13, 16], [20, 42, 49], [29, 91, 104],
  [66, 155, 143], [189, 200, 118], [255, 216, 122],
]
export const CSS_GRADIENT = `linear-gradient(to right, ${
  STOPS.map((c, i) => `rgb(${c.join(',')}) ${(i / (STOPS.length - 1) * 100).toFixed(0)}%`).join(', ')})`

export function color(value) {
  const v = Math.max(0, Math.min(1, value))
  const x = v * (STOPS.length - 1)
  const i = Math.min(STOPS.length - 2, Math.floor(x))
  const f = x - i
  const c = STOPS[i].map((a, k) => Math.round(a + (STOPS[i + 1][k] - a) * f))
  return `rgb(${c.join(',')})`
}

function rankValue(r) {
  if (!r || r < 1) return 0
  return Math.max(0, 1 - Math.log10(r) / 4)
}

function cellValue(cell, l, colorBy) {
  const row = cell.lens[l]
  switch (colorBy) {
    case 'rank': return row.track_rank ? rankValue(Math.min(...row.track_rank)) : 0
    case 'prob': return row.track_p ? Math.max(...row.track_p) : 0
    case 'kurt': return Math.min(1, Math.max(0, (row.kurt ?? 0) / 60))
    case 'emerge': return row.final_rank != null ? rankValue(row.final_rank) : 0
    default: return 1 - Math.min(1, row.entropy / 10)
  }
}

export default function Heatmap({ cols, nLayers, colorBy, sel, followLive, onSelect }) {
  const cvRef = useRef(null)
  const wrapRef = useRef(null)
  const [tip, setTip] = useState(null)

  useEffect(() => {
    const cv = cvRef.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    cv.width = PADL + Math.max(1, cols.length) * CW + 34
    cv.height = PADT + nLayers * CH + 28
    ctx.fillStyle = '#091012'
    ctx.fillRect(0, 0, cv.width, cv.height)

    const earlyEnd = Math.floor(nLayers / 3) - 1
    const motorStart = nLayers - 3
    band(ctx, nLayers, 0, earlyEnd, 'rgba(112,139,150,.035)', 'PRE-VERBAL')
    band(ctx, nLayers, earlyEnd + 1, motorStart - 1, 'rgba(113,229,207,.055)', 'WORKSPACE-LIKE')
    band(ctx, nLayers, motorStart, nLayers - 1, 'rgba(255,205,115,.075)', 'MOTOR')

    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'
    ctx.textBaseline = 'middle'
    for (let l = 0; l < nLayers; l++) {
      if (l % 3 !== 0 && l !== nLayers - 1) continue
      const y = PADT + (nLayers - 1 - l) * CH + CH / 2
      ctx.fillStyle = l >= motorStart ? '#c8a45f' : l > earlyEnd ? '#79bbae' : '#627077'
      ctx.fillText(`L${String(l).padStart(2, '0')}`, 47, y)
      ctx.strokeStyle = 'rgba(255,255,255,.035)'
      ctx.beginPath()
      ctx.moveTo(PADL - 7, y)
      ctx.lineTo(cv.width, y)
      ctx.stroke()
    }

    cols.forEach((c, x) => {
      for (let l = 0; l < nLayers; l++) {
        ctx.fillStyle = color(cellValue(c, l, colorBy))
        ctx.fillRect(PADL + x * CW, PADT + (nLayers - 1 - l) * CH, CW - 2, CH - 2)
      }
      ctx.save()
      ctx.translate(PADL + x * CW + CW / 2, PADT - 8)
      ctx.rotate(-Math.PI / 3)
      ctx.fillStyle = c.gen ? '#8fe7ce' : '#78868c'
      ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace'
      ctx.textBaseline = 'alphabetic'
      ctx.fillText(clean(c.token).slice(0, 11), 0, 0)
      ctx.restore()
      if (c.gen && (x === 0 || !cols[x - 1]?.gen)) {
        ctx.strokeStyle = '#78e6d0'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(PADL + x * CW - 2, PADT - 4)
        ctx.lineTo(PADL + x * CW - 2, PADT + nLayers * CH)
        ctx.stroke()
      }
    })

    if (sel && sel.x < cols.length) {
      ctx.strokeStyle = '#f6d47d'
      ctx.lineWidth = 1.5
      ctx.strokeRect(PADL + sel.x * CW - 2, PADT - 2, CW + 1, nLayers * CH + 3)
      if (sel.l != null) {
        ctx.fillStyle = 'rgba(246,212,125,.14)'
        ctx.fillRect(PADL - 7, PADT + (nLayers - 1 - sel.l) * CH - 1,
                     cv.width - PADL + 7, CH + 1)
        ctx.strokeStyle = '#f6d47d'
        ctx.strokeRect(PADL + sel.x * CW - 2,
                       PADT + (nLayers - 1 - sel.l) * CH - 2,
                       CW + 1, CH + 2)
      }
    }
    if (wrapRef.current && followLive) wrapRef.current.scrollLeft = 1e9
  }, [cols, nLayers, colorBy, sel, followLive])

  const cellAt = e => {
    const r = cvRef.current.getBoundingClientRect()
    const x = Math.floor((e.clientX - r.left - PADL) / CW)
    const l = nLayers - 1 - Math.floor((e.clientY - r.top - PADT) / CH)
    if (x < 0 || x >= cols.length || l < 0 || l >= nLayers) return null
    return { x, l }
  }

  return (
    <div className="heatmap-panel">
      <div className="matrix-head">
        <div><span className="eyebrow">Evidence matrix</span><b>Layer × token position</b></div>
        <span>prompt tokens <i /> generated tokens</span>
      </div>
      <div className={'hmwrap ' + (!cols.length ? 'empty' : '')} ref={wrapRef}>
        {!cols.length && <div className="matrix-empty"><i /><b>No residual states captured yet</b><span>The matrix will fill after Gemma reads the prompt.</span></div>}
        <canvas
          ref={cvRef}
          aria-label="J-lens heatmap by token and layer"
          onMouseMove={e => {
            const c = cellAt(e)
            setTip(c ? { ...c, cx: e.clientX, cy: e.clientY, cell: cols[c.x] } : null)
          }}
          onMouseLeave={() => setTip(null)}
          onClick={e => { const c = cellAt(e); if (c) onSelect(c) }}
        />
        {tip && <Tip tip={tip} />}
      </div>
    </div>
  )
}

function band(ctx, nLayers, lo, hi, fill, label) {
  const y = PADT + (nLayers - 1 - hi) * CH
  const h = (hi - lo + 1) * CH
  ctx.fillStyle = fill
  ctx.fillRect(0, y, ctx.canvas.width, h)
  ctx.save()
  ctx.translate(15, y + h / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.fillStyle = label === 'WORKSPACE-LIKE' ? '#4d8d81' : label === 'MOTOR' ? '#987b43' : '#49565b'
  ctx.font = '8px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.textAlign = 'center'
  ctx.fillText(label, 0, 0)
  ctx.restore()
}

function Tip({ tip }) {
  const row = tip.cell.lens[tip.l]
  const style = {
    left: Math.min(tip.cx + 14, window.innerWidth - 360),
    top: Math.min(tip.cy + 14, window.innerHeight - 260),
  }
  return (
    <div className="tip" style={style}>
      <div className="h"><span>L{tip.l} · position {tip.x}</span><b>{JSON.stringify(tip.cell.token)}</b></div>
      {row.topk.map((t, i) => (
        <div className="r" key={i}><b>{JSON.stringify(t)}</b><span>{(row.topk_p[i] * 100).toFixed(2)}%</span></div>
      ))}
      {row.track_rank?.map((r, i) => (
        <div className="r track" key={'t' + i}><b>watched #{i + 1}</b><span>rank {r} · {(row.track_p[i] * 100).toFixed(3)}%</span></div>
      ))}
      <div className="r metric"><span>entropy {row.entropy} nats</span><span>kurtosis {row.kurt}</span></div>
    </div>
  )
}

function clean(token = '') {
  return token.replaceAll('\n', '↵').replaceAll('\t', '⇥') || '∅'
}
