"""J-space live viewer server.

Runs Gemma 4 E4B locally (weights reconstructed from the Ollama blob store)
and streams per-(layer, position) lens readings over a WebSocket while the
model generates, so you can watch the model's J-space in real time.

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


async def run_generation(sock, req, stop):
    prompt = req.get("messages") or req.get("prompt", "")
    mode = req.get("mode", "logit")
    topk = int(req.get("topk", 5))
    track = req.get("track", [])  # list of strings to track across layers
    track_ids = []
    track_toks = []
    for t in track:
        enc = HOST.tok.encode(t, add_special_tokens=False)
        if enc:
            track_ids.append(enc[0])
            track_toks.append(HOST.tok.decode([enc[0]]))
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
                    ev["lens"] = HOST.lens_read(hs, mode=mode, topk=topk,
                                                track_ids=track_ids)
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
        # decorate lens rows with decoded top-k strings
        for row in ev["lens"]:
            row["topk"] = [
                [HOST.tok.decode([i]) for i in ids] for ids in row.pop("topk_ids")
            ]
        ev["mode"] = mode
        ev["track_tokens"] = track_toks
        await sock.send_text(json.dumps(ev))


if __name__ == "__main__":
    import uvicorn
    # API_PORT (not PORT) so a parent `npm run dev` can hand PORT to Vite
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("API_PORT", 8731)),
                log_level="warning")
