"""Local Jacobian-lens observatory server.

Runs the selected local Gemma checkpoint and streams per-(layer, position)
lens evidence over a WebSocket while the model generates.

    .venv/bin/python server.py   ->  http://127.0.0.1:8731
"""

import asyncio
import json
import os
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from jspace_rt.host import Host

app = FastAPI()
HOST = None
_DIR = os.path.dirname(__file__)
STATIC = os.path.join(_DIR, "web", "dist")


@app.on_event("startup")
def _load():
    global HOST
    HOST = Host()
    print(f"model ready: {HOST.n_layers} layers, d={HOST.d_model}, "
          f"jlens={'loaded' if HOST.jlens is not None else 'not calibrated (logit-lens mode)'}")


@app.get("/")
def index():
    page = os.path.join(STATIC, "index.html")
    if os.path.exists(page):
        return FileResponse(page)
    return JSONResponse({"hint": "UI not built. Use `npm run dev` (Vite dev "
                                 "server on :5173) or `npm run build` first."})


if os.path.isdir(os.path.join(STATIC, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC, "assets")),
              name="assets")


@app.get("/status")
def status():
    return JSONResponse({
        "model": HOST.model_name,
        "n_layers": HOST.n_layers,
        "d_model": HOST.d_model,
        "jlens": HOST.jlens is not None,
        "jlens_meta": HOST.jlens_meta or {},
        "jlens_validation": HOST.jlens_validation or {},
    })


@app.post("/reload_jlens")
def reload_jlens():
    ok = HOST.load_jlens()
    return JSONResponse({"loaded": ok})


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    stop = threading.Event()
    try:
        while True:
            req = json.loads(await sock.receive_text())
            stop.clear()
            await run_generation(sock, req, stop)
    except WebSocketDisconnect:
        stop.set()


def concept_token_ids(text):
    """Return single-token surface forms for one human concept label.

    J-lens vectors are token-indexed, so capitalization, leading whitespace,
    and inflection otherwise look like unrelated concepts. Grouping common
    one-token forms makes a pin behave like the concept the user typed while
    keeping the underlying token-level limitation explicit in the UI.
    """
    base = text.strip()
    if not base:
        return [], []
    words = {base, base.lower(), base.capitalize(), base.title(), base.upper()}
    if base.isalpha():
        stem = base[:-1] if base.lower().endswith("s") and len(base) > 3 else base
        words.update({stem, stem.lower(), stem.capitalize(), stem + "s",
                      (stem + "s").capitalize()})
    candidates = {w for word in words for w in (word, " " + word)}
    ids = set()
    for candidate in candidates:
        enc = HOST.tok.encode(candidate, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    if not ids:
        enc = HOST.tok.encode(base, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    ordered = sorted(ids)
    return ordered, [HOST.tok.decode([i]) for i in ordered]


async def run_generation(sock, req, stop):
    prompt = req.get("messages") or req.get("prompt", "")
    mode = req.get("mode", "logit")
    topk = int(req.get("topk", 5))
    track = req.get("track", [])  # list of strings to track across layers
    track_ids = []
    track_toks = []
    track_variants = []
    for t in track:
        ids, variants = concept_token_ids(t)
        if ids:
            track_ids.append(ids)
            track_toks.append(t.strip())
            track_variants.append(variants)
    if mode == "jlens" and HOST.jlens is None:
        mode = "logit"

    loop = asyncio.get_event_loop()
    q = asyncio.Queue()

    def worker():
        try:
            with HOST.lock:
                gen = HOST.generate_stream(
                    prompt,
                    max_new_tokens=int(req.get("max_new_tokens", 48)),
                    temperature=float(req.get("temperature", 0.0)),
                    chat=True if isinstance(prompt, list) else bool(req.get("chat", True)),
                    stop_event=stop,
                )
                for ev in gen:
                    hs = ev.pop("hidden_states")
                    source_event = ev["event"]

                    def on_progress(completed, total):
                        loop.call_soon_threadsafe(q.put_nowait, {
                            "event": "progress",
                            "phase": "prefill" if source_event == "prefill" else "generation",
                            "completed": completed,
                            "total": total,
                            "layer": completed - 1,
                            "token": ev.get("token"),
                        })

                    ev["lens"] = HOST.lens_read(
                        hs, mode=mode, topk=topk, track_ids=track_ids,
                        progress=on_progress,
                    )
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            loop.call_soon_threadsafe(q.put_nowait, {"event": "done"})
        except Exception as e:  # surface errors to the client
            loop.call_soon_threadsafe(
                q.put_nowait, {"event": "error", "message": repr(e)})

    threading.Thread(target=worker, daemon=True).start()

    while True:
        ev = await q.get()
        if ev["event"] in ("done", "error"):
            await sock.send_text(json.dumps(ev))
            break
        if ev["event"] == "progress":
            await sock.send_text(json.dumps(ev))
            continue
        # decorate lens rows with decoded top-k strings
        for row in ev["lens"]:
            row["topk"] = [
                [HOST.tok.decode([i]) for i in ids] for ids in row.pop("topk_ids")
            ]
        ev["mode"] = mode
        ev["track_tokens"] = track_toks
        ev["track_variants"] = track_variants
        await sock.send_text(json.dumps(ev))


if __name__ == "__main__":
    import uvicorn
    # API_PORT (not PORT) so a parent `npm run dev` can hand PORT to Vite
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("API_PORT", 8731)),
                log_level="warning")
