import React, { useEffect, useRef, useState, useCallback } from 'react'
import Heatmap, { CSS_GRADIENT } from './Heatmap.jsx'
import Inspector from './Inspector.jsx'
import Profiles from './Profiles.jsx'
import ThoughtTimeline from './ThoughtTimeline.jsx'

const EOT = '<turn|>'

const DEMOS = [
  {
    name: 'Truth held behind a lie',
    prompt: "Let's play a game: answer the next question with a confident, deliberate lie (wrong answer only, don't reveal the truth). In which city is the Eiffel Tower?",
    track: 'Paris, lie, France',
  },
  {
    name: 'Unspoken inference · spider',
    prompt: 'If Alice fears the animal that spins webs, what does Alice fear? One word only.',
    track: 'spider',
  },
  {
    name: 'Silent bug detection',
    prompt: 'What does this Python function return for n=5?\n\ndef f(n):\n    total = 0\n    for i in range(n):\n        total =+ i\n    return total\n\nAnswer with a number only.',
    track: 'error, bug, typo',
  },
  {
    name: 'Mental arithmetic · 23 + 19',
    prompt: 'Compute 23 + 19 in your head. Reply with only the result.',
    track: '42, add',
  },
  {
    name: 'Rhyme planning',
    prompt: 'Write a two-line rhyming couplet about a brave knight.',
    track: 'night, fight, light',
  },
  {
    name: 'Directed focus · citrus',
    prompt: "Copy this sentence exactly, and while copying it, mentally concentrate on citrus fruits: 'The weather is pleasant today.'",
    track: 'orange, lemon, citrus',
  },
  {
    name: 'Multi-hop · legs count',
    prompt: 'The number of legs on the animal that spins webs is what? Answer with a digit.',
    track: 'spider, eight',
  },
]

export default function App() {
  const [status, setStatus] = useState(null)
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [running, setRunning] = useState(false)
  const [mode, setMode] = useState('logit')
  const [track, setTrack] = useState('')
  const [colorBy, setColorBy] = useState('auto')
  const [maxTok, setMaxTok] = useState(512)
  const [temp, setTemp] = useState(0)
  const [cols, setCols] = useState([])
  const [trackTokens, setTrackTokens] = useState([])
  const [trackVariants, setTrackVariants] = useState([])
  const [sel, setSel] = useState(null)
  const [followLive, setFollowLive] = useState(true)
  const [progress, setProgress] = useState({ phase: 'idle', completed: 0, total: 30 })
  const [error, setError] = useState(null)
  const wsRef = useRef(null)
  const chatEndRef = useRef(null)
  const followRef = useRef(true)
  const nLayersRef = useRef(30)

  useEffect(() => { followRef.current = followLive }, [followLive])

  useEffect(() => {
    let stopped = false
    const poll = () => fetch('/status')
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then(s => {
        if (stopped) return
        setStatus(s)
        nLayersRef.current = s.n_layers ?? 30
        setProgress(p => ({ ...p, total: nLayersRef.current }))
      })
      .catch(() => { if (!stopped) setTimeout(poll, 2500) })
    poll()
    return () => { stopped = true }
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (progress.phase !== 'complete' || !followLive || !cols.length || !trackTokens.length) return
    const best = findObservations(cols, trackTokens, status?.n_layers ?? 30)
      .filter(item => item.r != null)
      .sort((a, b) => a.r - b.r)[0]
    if (best && best.r <= 100) {
      setSel({ x: best.x, l: best.l })
      setFollowLive(false)
    }
  }, [progress.phase, followLive, cols, trackTokens, status?.n_layers])

  const liveSelection = x => ({
    x,
    l: Math.max(0, nLayersRef.current - 5),
  })

  const connect = useCallback(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws`)
    ws.onmessage = m => {
      const ev = JSON.parse(m.data)
      if (ev.event === 'error') {
        setError(ev.message)
        setProgress(p => ({ ...p, phase: 'error' }))
        setRunning(false)
        return
      }
      if (ev.event === 'progress') {
        setProgress({
          phase: ev.phase,
          completed: ev.completed,
          total: ev.total,
          layer: ev.layer,
          token: ev.token,
        })
        return
      }
      if (ev.event === 'done') {
        setProgress(p => ({ ...p, phase: 'complete', completed: p.total }))
        setRunning(false)
        return
      }
      if (ev.track_tokens) setTrackTokens(ev.track_tokens)
      if (ev.track_variants) setTrackVariants(ev.track_variants)
      if (ev.event === 'prefill') {
        const next = ev.tokens.map((t, i) => ({
          token: t,
          gen: false,
          lens: ev.lens.map(Lr => sliceRow(Lr, i)),
        }))
        setCols(next)
        if (followRef.current && next.length) setSel(liveSelection(next.length - 1))
      } else if (ev.event === 'token') {
        setCols(current => {
          const next = [...current, {
            token: ev.token,
            gen: true,
            lens: ev.lens.map(Lr => sliceRow(Lr, 0)),
          }]
          if (followRef.current) setSel(liveSelection(next.length - 1))
          return next
        })
        if (ev.token !== EOT) {
          setMessages(ms => {
            const out = [...ms]
            out[out.length - 1] = {
              ...out[out.length - 1],
              content: out[out.length - 1].content + ev.token,
            }
            return out
          })
        }
      }
    }
    ws.onclose = () => { wsRef.current = null; setRunning(false) }
    wsRef.current = ws
    return ws
  }, [])

  const send = () => {
    const text = draft.trim()
    if (!text || running) return
    setError(null)
    const history = [...messages, { role: 'user', content: text }]
    const concepts = parseConcepts(track)
    setMessages([...history, { role: 'model', content: '' }])
    setDraft('')
    setCols([])
    setSel(null)
    setTrackTokens(concepts)
    setTrackVariants([])
    setFollowLive(true)
    setProgress({ phase: 'queued', completed: 0, total: nLayersRef.current })
    setRunning(true)
    const payload = JSON.stringify({
      messages: history,
      mode,
      track: concepts,
      topk: 8,
      max_new_tokens: +maxTok,
      temperature: +temp,
    })
    const ws = wsRef.current && wsRef.current.readyState === 1 ? wsRef.current : connect()
    if (ws.readyState === 1) ws.send(payload)
    else ws.onopen = () => ws.send(payload)
  }

  const stop = () => { wsRef.current?.close(); setRunning(false) }
  const reset = () => {
    stop()
    setMessages([])
    setCols([])
    setSel(null)
    setTrackTokens([])
    setProgress({ phase: 'idle', completed: 0, total: nLayersRef.current })
  }
  const addTrack = tok => {
    const t = tok.trim()
    if (!t) return
    const cur = parseConcepts(track)
    if (!cur.includes(t)) setTrack([...cur, t].join(', '))
  }
  const toggleFollow = () => {
    const next = !followLive
    setFollowLive(next)
    if (next && cols.length) setSel(liveSelection(cols.length - 1))
  }

  const nL = status?.n_layers ?? 30
  const tracked = trackTokens.length > 0
  const effColor = colorBy === 'auto' ? (tracked ? 'rank' : 'emerge') : colorBy
  const observations = findObservations(cols, trackTokens, nL)
  const calibration = status?.jlens_meta ?? {}
  const validation = status?.jlens_validation ?? {}
  const samePosition = status?.jlens && Number(calibration.cross_weight ?? 0) === 0

  return (
    <main className="app">
      <header className="topbar">
        <div className="brandmark" aria-hidden="true"><i /><i /><i /></div>
        <div className="brandcopy">
          <div className="eyebrow">Mechanistic interpretability · local</div>
          <h1>J·SPACE <em>observatory</em></h1>
        </div>
        <div className="modelreadout">
          <span className={'live-dot ' + (status ? 'on' : '')} />
          <div>
            <b>{status?.model ?? 'loading Gemma…'}</b>
            <small>{status ? `${status.n_layers} layers · d ${status.d_model.toLocaleString()} · ${validation.logit_validated ? 'logit probe passed' : 'lens evidence unvalidated'}` : 'connecting to local runtime'}</small>
          </div>
        </div>
        <div className={'lensbadge ' + (mode === 'jlens' ? 'active' : '')}>
          <span>{mode === 'jlens' ? 'J' : 'L'}</span>
          {mode === 'jlens' ? 'Jacobian lens' : 'logit lens'}
        </div>
      </header>

      <div className="truthbar">
        <b>What this is:</b> a live token-level readout of Gemma’s residual stream.
        {mode === 'jlens' && samePosition
          ? ` Experimental J-fit selected; it ${validation.jlens_validated ? 'passed' : 'failed'} the rank-${validation.quality_gate_rank ?? 100} intermediate-concept gate.`
          : validation.logit_validated
            ? ` Public view uses the logit lens: the unspoken-intermediate probe reached rank ${validation.logit_best?.rank} at L${validation.logit_best?.layer}. Earlier layers remain less reliable than a validated J-lens.`
            : ' Interpret high-ranked concepts as evidence to inspect, not a transcript of private chain-of-thought.'}
        {mode === 'jlens' && status?.jlens && <span className="fitmeta">{calibration.prompts} bundled synthetic sequences · {calibration.backwards} reverse passes · rank-{calibration.rank} sketch</span>}
        <a href="https://transformer-circuits.pub/2026/workspace/index.html" target="_blank" rel="noreferrer">paper ↗</a>
      </div>

      <div className="panes">
        <section className="chat" aria-label="Conversation">
          <div className="orientation">
            <div className="eyebrow">Depth behaves like time</div>
            <h2>Watch a decision become speakable.</h2>
            <p>The model moves upward through one feed-forward pass. Early readouts are often noise; middle layers can carry task concepts; the last layers prepare the next token.</p>
            <div className="mini-regimes">
              <span><i className="early" />encode</span>
              <span><i className="workspace" />workspace-like</span>
              <span><i className="motor" />motor / output</span>
            </div>
          </div>

          <ProgressCard progress={progress} running={running} nLayers={nL} />

          <div className="msgs">
            {messages.length === 0 && (
              <div className="empty-chat">
                <span>01</span>
                <p>Choose a paper-inspired experiment, or ask your own question. Pin the concepts you expect Gemma may use without saying.</p>
              </div>
            )}
            {messages.map((m, i) => (
              <article key={i} className={'msg ' + m.role}>
                <div className="who">{m.role === 'user' ? 'PROMPT' : (status?.model ?? 'GEMMA')}</div>
                <div className="bubble">{m.content || (running && i === messages.length - 1 ? <ThinkingMark /> : '')}</div>
              </article>
            ))}
            <div ref={chatEndRef} />
          </div>

          {error && <div className="error">{String(error)}</div>}
          <div className="composer">
            <textarea
              value={draft}
              onChange={e => setDraft(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
              }}
              placeholder="Ask Gemma something worth looking inside…"
              aria-label="Message"
            />
            <div className="buttons">
              <button className="primary" onClick={send} disabled={running}>Run observation <span>↗</span></button>
              <button onClick={stop} disabled={!running}>Stop</button>
              <button className="quiet" onClick={reset}>Clear</button>
            </div>
          </div>
        </section>

        <section className="viz" aria-label="J-space visualization">
          <div className="control-deck">
            <label className="wide">Experiment
              <select value="" onChange={e => {
                const d = DEMOS[e.target.value]
                if (d) { setDraft(d.prompt); setTrack(d.track) }
              }}>
                <option value="">Choose a paper-inspired setup…</option>
                {DEMOS.map((d, i) => <option key={i} value={i}>{d.name}</option>)}
              </select>
            </label>
            <label className="wide">Concept watchlist
              <input value={track} onChange={e => setTrack(e.target.value)}
                     placeholder="spider, error, Paris" />
            </label>
            <button className={'follow ' + (followLive ? 'active' : '')} onClick={toggleFollow}>
              <span className="pulse" /> {followLive ? 'Following live edge' : 'Resume live edge'}
            </button>
            <details className="advanced">
              <summary>Instrument settings</summary>
              <div>
                <label>Lens
                  <select value={mode} onChange={e => setMode(e.target.value)}>
                    <option value="logit">Logit lens · J = I</option>
                    <option value="jlens" disabled={!status?.jlens}>Jacobian lens · {status?.jlens ? (validation.jlens_validated ? 'quality gate passed' : 'experimental, gate failed') : 'not calibrated'}</option>
                  </select>
                </label>
                <label>Color
                  <select value={colorBy} onChange={e => setColorBy(e.target.value)}>
                    <option value="auto">Auto · {tracked ? 'watched rank' : 'next-token emergence'}</option>
                    <option value="emerge">Next-token emergence</option>
                    <option value="conf">Lens confidence</option>
                    <option value="prob" disabled={!tracked}>Watched probability</option>
                    <option value="rank" disabled={!tracked}>Watched rank</option>
                    <option value="kurt">Excess kurtosis</option>
                  </select>
                </label>
                <label>Max tokens
                  <input type="number" min="1" max="1024" value={maxTok} onChange={e => setMaxTok(e.target.value)} />
                </label>
                <label>Temperature
                  <input type="number" min="0" max="2" step="0.1" value={temp} onChange={e => setTemp(e.target.value)} />
                </label>
              </div>
            </details>
          </div>

          <ThoughtTimeline cols={cols} sel={sel} nLayers={nL} onPin={addTrack}
                           onSelectLayer={l => setSel(s => ({ ...(s ?? liveSelection(cols.length - 1)), l }))} />

          <div className="legend">
            <span className="scale">
              <span>{legendFor(effColor).lo}</span>
              <i style={{ background: CSS_GRADIENT }} />
              <span className="hi">{legendFor(effColor).hi}</span>
            </span>
            <p><b>{legendFor(effColor).title}</b> {legendFor(effColor).body}</p>
          </div>

          {trackVariants.length > 0 && (
            <div className="variant-note">
              Pins group tokenizer surface forms together: {trackVariants.map((v, i) => (
                <span key={i}><b>{trackTokens[i]}</b> · {v.length} forms</span>
              ))}
            </div>
          )}

          {cols.length > 0 && observations.length > 0 && (
            <div className="alerts">
              <span className="eyebrow">Watchlist outcome</span>
              {observations.map((a, i) => (
                <button key={i} className={'alert ' + (a.candidate ? 'candidate' : 'negative')}
                        disabled={!a.r} onClick={() => {
                  setFollowLive(false)
                  setSel({ x: a.x, l: a.l })
                }}>
                  <i /> {a.tok} <b>{a.r ? `best rank ${a.r}` : 'not measured'}</b>
                  <small>{a.candidate ? `candidate · L${a.l} · P${a.x}` : a.r ? `no top-20 hit · L${a.l} · P${a.x}` : ''}</small>
                </button>
              ))}
            </div>
          )}

          <div className="vizmain">
            <Heatmap cols={cols} nLayers={nL} colorBy={effColor} sel={sel}
                     followLive={followLive} onSelect={next => {
                       setFollowLive(false)
                       setSel(next)
                     }} />
            {sel && <Inspector cols={cols} sel={sel} trackTokens={trackTokens}
                               onClose={() => setSel(null)} onPin={addTrack}
                               onSelectLayer={l => setSel(s => ({ ...s, l }))} />}
          </div>

          <Profiles cols={cols} nLayers={nL} trackTokens={trackTokens} sel={sel} />
        </section>
      </div>
    </main>
  )
}

function ProgressCard({ progress, running, nLayers }) {
  const pct = progress.total ? Math.round(progress.completed / progress.total * 100) : 0
  const copy = {
    idle: ['Instrument ready', 'Run a prompt to capture every residual state.'],
    queued: ['Reading prompt', 'Gemma is building the first forward pass.'],
    prefill: [`Projecting layer ${Math.min(progress.completed, nLayers)} / ${nLayers}`, 'Decoding the prompt residual stream through the selected lens.'],
    generation: [`Projecting next token · layer ${Math.min(progress.completed, nLayers)} / ${nLayers}`, progress.token ? `Current token ${JSON.stringify(progress.token)}` : 'Generation is advancing.'],
    complete: ['Observation complete', 'Select a column or layer to inspect the evidence.'],
    error: ['Observation interrupted', 'The runtime returned an error.'],
  }[progress.phase] ?? ['Instrument ready', '']
  return (
    <div className={'progress-card ' + (running ? 'running' : '')}>
      <div className="progress-copy"><i /><span><b>{copy[0]}</b><small>{copy[1]}</small></span><em>{running ? `${pct}%` : 'LOCAL'}</em></div>
      <div className="progress-track"><i style={{ width: `${running ? pct : progress.phase === 'complete' ? 100 : 0}%` }} /></div>
      <div className="phase-track">
        <span className={progress.layer < Math.floor(nLayers / 3) && running ? 'active' : ''}>encoding</span>
        <span className={progress.layer >= Math.floor(nLayers / 3) && progress.layer < nLayers - 3 && running ? 'active' : ''}>workspace-like</span>
        <span className={progress.layer >= nLayers - 3 && running ? 'active' : ''}>motor</span>
      </div>
    </div>
  )
}

function ThinkingMark() {
  return <span className="thinking-mark"><i /><i /><i /> tracing residual stream</span>
}

function parseConcepts(value) {
  return value.split(',').map(s => s.trim()).filter(Boolean)
}

function findObservations(cols, trackTokens, nLayers) {
  const bandLo = Math.floor(nLayers / 3)
  const bandHi = nLayers - 4
  return trackTokens.map((token, i) => {
    let best = null
    cols.forEach((column, x) => {
      for (let layer = bandLo; layer <= bandHi; layer++) {
        const rank = column.lens[layer]?.track_rank?.[i]
        if (rank && (!best || rank < best.r)) best = { r: rank, x, l: layer }
      }
    })
    return best ? { tok: token, ...best, candidate: best.r <= 20 }
      : { tok: token, r: null, candidate: false }
  })
}

function legendFor(mode) {
  return {
    emerge: { lo: 'not formed', hi: 'rank 1', title: 'Next-token emergence.', body: 'How highly this layer ranks the token predicted after each column.' },
    conf: { lo: 'diffuse', hi: 'peaked', title: 'Lens confidence.', body: 'Low entropy means the readout concentrates on fewer vocabulary tokens.' },
    prob: { lo: 'p ≈ 0', hi: 'highest p', title: 'Watched probability.', body: 'The strongest tokenizer surface form for any pinned concept.' },
    rank: { lo: 'rank ≥ 10k', hi: 'rank 1', title: 'Watched concept rank.', body: 'Log-scaled rank of the strongest surface form in each concept group.' },
    kurt: { lo: 'noise-like', hi: 'sparse peak', title: 'Excess kurtosis.', body: 'A distributional signature used in the paper; high kurtosis alone does not prove workspace membership.' },
  }[mode]
}

function sliceRow(Lr, i) {
  return {
    topk: Lr.topk[i],
    topk_p: Lr.topk_p[i],
    entropy: Lr.entropy[i],
    kurt: Lr.kurt ? Lr.kurt[i] : 0,
    final_rank: Lr.final_rank ? Lr.final_rank[i] : null,
    track_p: Lr.track_p ? Lr.track_p.map(a => a[i]) : null,
    track_rank: Lr.track_rank ? Lr.track_rank.map(a => a[i]) : null,
  }
}
