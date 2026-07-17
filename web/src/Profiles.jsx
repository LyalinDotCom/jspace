import React from 'react'

/** Bottom strip of layer-profile charts, inspired by the paper's fig. 5 rank
 *  trajectories and fig. 28 workspace signatures:
 *   - left: pinned-token rank vs layer at the selected (or last) position
 *   - right: excess kurtosis + confidence profile across layers, averaged
 *     over positions — the "workspace band" signature.
 */
const PIN_COLORS = ['#f6d47d', '#f39fc8', '#78e6d0', '#93adff', '#ef9e74']

export default function Profiles({ cols, nLayers, trackTokens, sel }) {
  if (!cols.length) return null
  const pos = Math.min(sel?.x ?? cols.length - 1, cols.length - 1)
  const cell = cols[pos]
  const tracked = trackTokens.length > 0 && cell.lens[0].track_rank

  // averaged layer profiles
  const kurt = [], conf = []
  for (let l = 0; l < nLayers; l++) {
    let k = 0, c = 0
    cols.forEach(col => { k += col.lens[l].kurt ?? 0; c += 1 - Math.min(1, col.lens[l].entropy / 10) })
    kurt.push(k / cols.length)
    conf.push(c / cols.length)
  }

  return (
    <section className="profiles">
      {tracked ? (
        <Chart title={`pinned-token rank by layer @ pos ${pos} (${JSON.stringify(cell.token)})`}
               yLabel="rank (log)" nLayers={nLayers}
               series={trackTokens.map((t, i) => ({
                 name: JSON.stringify(t),
                 color: PIN_COLORS[i % PIN_COLORS.length],
                 // rank 1 -> 1.0 at top; rank 10^4+ -> 0
                 values: [...Array(nLayers).keys()].map(l =>
                   Math.max(0, 1 - Math.log10(cell.lens[l].track_rank[i]) / 4)),
                 label: l => `rank ${cell.lens[l].track_rank[i]}`,
               }))} />
      ) : (
        <div className="chart empty">pin tokens (track box, or click tokens in the
          inspector) to see their rank trajectory across layers</div>
      )}
      {tracked ? (
        <Chart title={`pinned-token rank by position @ L${sel?.l ?? nLayers - 1}`}
               nLayers={cols.length} bands={false} prefix="P"
               series={trackTokens.map((t, i) => ({
                 name: JSON.stringify(t),
                 color: PIN_COLORS[i % PIN_COLORS.length],
                 values: cols.map(col => {
                   const rank = col.lens[sel?.l ?? nLayers - 1]?.track_rank?.[i] ?? 10000
                   return Math.max(0, 1 - Math.log10(rank) / 4)
                 }),
               }))} />
      ) : null}
      <Chart title="distributional profile · mean over positions"
             yLabel="" nLayers={nLayers}
             series={[
               {
                 name: 'excess kurtosis', color: '#7aa7ff',
                 values: kurt.map(k => Math.min(1, Math.max(0, k / Math.max(...kurt, 1)))),
                 label: l => `kurt ${kurt[l].toFixed(1)}`,
               },
               {
                 name: 'lens confidence', color: '#78e6d0',
                 values: conf,
                 label: l => `conf ${(conf[l] * 100).toFixed(0)}%`,
               },
             ]} />
    </section>
  )
}

function Chart({ title, series, nLayers, bands = true, prefix = 'L' }) {
  const W = 460, H = 110, PL = 8, PB = 16, PT = 18
  const x = l => PL + (l / Math.max(1, nLayers - 1)) * (W - PL - 8)
  const y = v => PT + (1 - v) * (H - PT - PB)
  const ticks = [...new Set([0, Math.round((nLayers - 1) / 3), Math.round((nLayers - 1) * 2 / 3), nLayers - 1])]
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {bands && <>
          <rect x={x(0)} y={PT} width={x(Math.floor(nLayers / 3)) - x(0)} height={H - PT - PB} className="band early" />
          <rect x={x(Math.floor(nLayers / 3))} y={PT} width={x(nLayers - 3) - x(Math.floor(nLayers / 3))} height={H - PT - PB} className="band workspace" />
          <rect x={x(nLayers - 3)} y={PT} width={x(nLayers - 1) - x(nLayers - 3)} height={H - PT - PB} className="band motor" />
        </>}
        <text x={PL} y={11} className="ctitle">{title}</text>
        {ticks.map(l => (
          <g key={l}>
            <line x1={x(l)} y1={PT} x2={x(l)} y2={H - PB} className="grid" />
            <text x={x(l)} y={H - 4} className="cx">{prefix}{l}</text>
          </g>
        ))}
        {series.map((s, i) => (
          <polyline key={i} fill="none" stroke={s.color} strokeWidth="1.6"
                    points={s.values.map((v, l) => `${x(l)},${y(v)}`).join(' ')} />
        ))}
        {series.map((s, i) => (
          <text key={'l' + i} x={W - 8} y={PT + 10 + i * 11} textAnchor="end"
                fill={s.color} className="clegend">{s.name}</text>
        ))}
      </svg>
    </div>
  )
}
