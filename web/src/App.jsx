import React, { useEffect, useRef, useState, useCallback } from 'react'
import Heatmap from './Heatmap.jsx'
import Inspector from './Inspector.jsx'
import Profiles from './Profiles.jsx'

const EOT = '<turn|>'

export default function App() {
  const [status, setStatus] = useState(null)
  const [messages, setMessages] = useState([])   // {role:'user'|'model', content}
  const [draft, setDraft] = useState('')
  const [running, setRunning] = useState(false)
  const [mode, setMode] = useState('logit')
  const [track, setTrack] = useState('')
  const [colorBy, setColorBy] = useState('auto') // auto | conf | prob | rank
  const [maxTok, setMaxTok] = useState(200)
  const [temp, setTemp] = useState(0)
  const [cols, setCols] = useState([])           // heatmap columns
  const [trackTokens, setTrackTokens] = useState([])
  const [sel, setSel] = useState(null)           // {x, l} selected cell
  const [error, setError] = useState(null)
  const wsRef = useRef(null)
  const chatEndRef = useRef(null)

  useEffect(() => {
    let stopped = false
    const poll = () => fetch('/status')
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then(s => {
        if (stopped) return
        setStatus(s)
        if (s.jlens) setMode('jlens')
      })
      .catch(() => { if (!stopped) setTimeout(poll, 2500) })
    poll()
    return () => { stopped = true }
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const connect = useCallback(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws`)
    ws.onmessage = m => {
      const ev = JSON.parse(m.data)
      if (ev.event === 'error') { setError(ev.message); setRunning(false); return }
      if (ev.event === 'done') { setRunning(false); return }
      if (ev.track_tokens) setTrackTokens(ev.track_tokens)
      if (ev.event === 'prefill') {
        setCols(ev.tokens.map((t, i) => ({
          token: t, gen: false,
          lens: ev.lens.map(Lr => sliceRow(Lr, i)),
        })))
      } else if (ev.event === 'token') {
        setCols(c => [...c, {
          token: ev.token, gen: true,
          lens: ev.lens.map(Lr => sliceRow(Lr, 0)),
        }])
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
    setMessages([...history, { role: 'model', content: '' }])
    setDraft('')
    setCols([])
    setSel(null)
    setRunning(true)
    const payload = JSON.stringify({
      messages: history,
      mode,
      track: track.split(',').map(s => s.trim()).filter(Boolean),
      topk: 5,
      max_new_tokens: +maxTok,
      temperature: +temp,
    })
    const ws = wsRef.current && wsRef.current.readyState === 1 ? wsRef.current : connect()
    if (ws.readyState === 1) ws.send(payload)
    else ws.onopen = () => ws.send(payload)
  }

  const stop = () => { wsRef.current?.close(); setRunning(false) }
  const reset = () => { stop(); setMessages([]); setCols([]); setSel(null) }
  const addTrack = tok => {
    const t = tok.trim()
    if (!t) return
    const cur = track.split(',').map(s => s.trim()).filter(Boolean)
    if (!cur.includes(t)) setTrack([...cur, t].join(', '))
  }

  const tracked = trackTokens.length > 0
  const effColor = colorBy === 'auto' ? (tracked ? 'rank' : 'conf') : colorBy

  return (
    <div className="app">
      <header>
        <h1>J-Space Chat</h1>
        <span className="meta">
          {status
            ? `gemma-4-E4B · ${status.n_layers} layers · d=${status.d_model} · J-lens ${status.jlens ? 'calibrated' : 'not calibrated'}`
            : 'connecting…'}
        </span>
        <span className={'modechip' + (mode === 'jlens' ? ' j' : '')}>
          lens: {mode === 'jlens' ? 'J-lens' : 'logit'}
        </span>
      </header>

      <div className="panes">
        <section className="chat">
          <div className="msgs">
            {messages.length === 0 && (
              <div className="hint">
                Ask something and watch the workspace light up on the right.
                Try: <em>If Alice fears the animal that spins webs, what does
                Alice fear? One word only.</em> with track = <em>spider</em>.
                Click any heatmap cell to open the layer readout at that
                position; click tokens there to pin them.
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={'msg ' + m.role}>
                <div className="who">{m.role === 'user' ? 'you' : 'gemma'}</div>
                <div className="bubble">{m.content || (running && i === messages.length - 1 ? '…' : '')}</div>
              </div>
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
              placeholder="Message… (Enter to send)"
            />
            <div className="buttons">
              <button className="primary" onClick={send} disabled={running}>Send</button>
              <button onClick={stop} disabled={!running}>Stop</button>
              <button onClick={reset}>Clear</button>
            </div>
          </div>
        </section>

        <section className="viz">
          <div className="vizbar">
            <label>lens
              <select value={mode} onChange={e => setMode(e.target.value)}>
                <option value="logit">logit lens (J=I)</option>
                <option value="jlens" disabled={!status?.jlens}>
                  J-lens{status?.jlens ? '' : ' (run npm run calibrate)'}
                </option>
              </select>
            </label>
            <label>track / pin
              <input value={track} onChange={e => setTrack(e.target.value)}
                     placeholder="spider, web" />
            </label>
            <label>color
              <select value={colorBy} onChange={e => setColorBy(e.target.value)}>
                <option value="auto">auto</option>
                <option value="conf">lens confidence</option>
                <option value="prob" disabled={!tracked}>pinned p</option>
                <option value="rank" disabled={!tracked}>pinned rank</option>
                <option value="kurt">excess kurtosis</option>
              </select>
            </label>
            <label>max tokens
              <input type="number" min="1" max="1024" value={maxTok}
                     onChange={e => setMaxTok(e.target.value)} />
            </label>
            <label>temp
              <input type="number" min="0" max="2" step="0.1" value={temp}
                     onChange={e => setTemp(e.target.value)} />
            </label>
          </div>
          <div className="legend">
            rows = layers (embeddings at bottom) · columns = token positions ·
            color = {{
              conf: 'lens confidence (low entropy)',
              prob: 'pinned-token probability',
              rank: 'pinned-token rank (bright = rank 1)',
              kurt: 'excess kurtosis of lens logits (workspace signature)',
            }[effColor]} · hover = top-k readout · click = inspect position
          </div>
          <div className="vizmain">
            <Heatmap cols={cols} nLayers={(status?.n_layers ?? 42) + 1}
                     colorBy={effColor} sel={sel} onSelect={setSel} />
            {sel && <Inspector cols={cols} sel={sel} trackTokens={trackTokens}
                               onClose={() => setSel(null)} onPin={addTrack}
                               onSelectLayer={l => setSel(s => ({ ...s, l }))} />}
          </div>
          <Profiles cols={cols} nLayers={(status?.n_layers ?? 42) + 1}
                    trackTokens={trackTokens} sel={sel} />
        </section>
      </div>
    </div>
  )
}

// extract position i from a per-layer lens row sent by the server
function sliceRow(Lr, i) {
  return {
    topk: Lr.topk[i],
    topk_p: Lr.topk_p[i],
    entropy: Lr.entropy[i],
    kurt: Lr.kurt ? Lr.kurt[i] : 0,
    track_p: Lr.track_p ? Lr.track_p.map(a => a[i]) : null,
    track_rank: Lr.track_rank ? Lr.track_rank.map(a => a[i]) : null,
  }
}
