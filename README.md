# J-Space Chat — watch Gemma 4's global workspace in real time

Chat with a local **Gemma 4 E4B** while a live heatmap shows what every layer
of the network is "holding in mind" at every token — the model's **J-space**,
per [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Transformer Circuits, 2026; general-audience version:
[anthropic.com/research/global-workspace](https://www.anthropic.com/research/global-workspace)).

Left pane: a normal chat. Right pane: rows are the 43 residual streams
(embeddings + 42 layers), columns are token positions, color is either lens
confidence or the probability of a concept you're tracking. Hover any cell to
see the top-k tokens that layer is disposed to say at that position.

The classic demo: ask
*"If Alice fears the animal that spins webs, what does Alice fear? One word
only."* with `track: spider`. The concept " spiders" lights up in the middle
band of layers at the end of the prompt — **before** the model has said
anything — then the output layers verbalize it as "Spider". That middle band
(~L30–40 here) is the workspace: content the model can report, reason with,
and act on.

## Visualizations

Modeled on the paper's interactive Figure 5 and the layer-signature plots of
Figure 28:

- **Heatmap** (layers × positions). Color modes: *answer emergence* (how
  highly each layer ranks the token the model eventually predicts at that
  position — bright = that layer has already "decided"; needs no setup and
  is the default), *pinned-token rank* (bright = rank 1, log scale — the
  paper's preferred encoding, readable even when probabilities are tiny),
  *pinned-token probability*, *lens confidence* (1 − entropy), and *excess
  kurtosis* (see below). `auto` picks pinned rank when something is pinned,
  answer emergence otherwise.
- **Hover readout**: top-k tokens + probabilities, pinned ranks, entropy and
  kurtosis for any (layer, position) cell.
- **Position inspector** (click a column): the full layer-by-layer readout at
  one position — embeddings to output — with probability bars. Click any
  token in it to pin it.
- **Pinning / tracking**: type concepts in the track box (comma-separated)
  or click tokens in the inspector. Pinned tokens drive the rank color mode
  and the trajectory chart.
- **Rank trajectory chart**: pinned-token rank vs layer at the selected (or
  last) position — watch a concept surface out of 262k candidates, plateau
  through the workspace band, and hand off to the output layers.
- **Workspace signature chart**: per-layer excess kurtosis of the lens
  logits and mean lens confidence, averaged over the current context. Excess
  kurtosis ≈ 0 means the lens readout is noise; the hump that starts around
  a third of the way up and collapses into the final layers is the J-space
  band identified in the paper (their fig. 28b).
- **Watchlist alerts**: any pinned concept whose lens rank reaches the top-20
  somewhere in the workspace band gets a clickable alert chip — the blog
  post's "silent thought monitoring" pattern (J-space says *error* on buggy
  code, *injection* on manipulative inputs, before the model says anything).
  Click a chip to jump the inspector to that exact cell.
- **Demo menu**: one-click setups for the blog/paper scenarios — unspoken
  inference, silent bug detection, mental arithmetic, rhyme planning,
  directed focus, and multi-hop reasoning — each pre-filling the prompt and
  the concepts to watch.

---

## Quick start

Prereqs: Python 3.11+, Node 20+, a Mac with Apple Silicon (MPS) and ~24 GB+
of RAM, and the model weights (next section).

```bash
npm run setup          # venv + deps; downloads the model if missing (~16 GB)
npm run dev            # backend (FastAPI, :8731) + frontend (Vite, :5173)
```

Open http://localhost:5173 and chat. For the good lens, also run once:

```bash
npm run calibrate      # ~15 min on M-series; writes calib/jlens.pt
```

then restart `npm run dev` (or `curl -X POST localhost:8731/reload_jlens`).
Until then the UI falls back to the logit lens.

## Getting the model

Weights live in HF-format directories under `model/`, **inside the project
folder** and **git-ignored** (see `.gitignore`) so they can never land on
GitHub by accident; `npm run setup` downloads automatically when missing.
Two official Google models are supported:

- **`model/gemma-4-E4B-it`** — the instruct release
  (google/gemma-4-E4B-it). The chat UI and the paper's demos assume an
  assistant, so this is the default when present. Fetch:
  `npm run import-model -- --repo google/gemma-4-E4B-it --out model/gemma-4-E4B-it`
- **`model/gemma-4-E2B-it`** — the **smallest Gemma 4** (2.3B effective,
  35 layers, d=1536): faster everywhere (load, tokens/s, calibration) at
  some readout quality cost. Fetch:
  `npm run import-model -- --repo google/gemma-4-E2B-it --out model/gemma-4-E2B-it`
  and run with `JSPACE_MODEL=gemma-4-E2B-it npm run dev`.
- **`model/gemma-4-E4B`** — the base/pretrained release
  (google/gemma-4-E4B). Interesting for raw-completion lens work — but note
  the workspace demos largely require an instruct model; on the raw base
  model the J-space band fills with answer-prep tokens instead of the
  unspoken concept. Fetch: `npm run import-model`

Pick explicitly with the `JSPACE_MODEL` env var (name under `model/` or an
absolute path): `JSPACE_MODEL=gemma-4-E4B npm run dev:api`. Calibrations are
stored per model (`calib/jlens-<model>.pt`) and never applied to the wrong
weights.

There is also a zero-download fallback if you have
`ollama pull gemma4:e4b-mlx-bf16`:
`npm run import-model -- --source ollama` symlinks Ollama's ~2,130
per-tensor safetensors blobs into a model directory and generates the index
transformers needs. **Provenance warning:** `scripts/compare_weights.py`
diffs the sources — the vision/audio towers match the official release
bit-for-bit, but every *language-model* tensor in Ollama's `e4b-mlx-bf16`
differs (mean |Δ| ≈ 1e-2). It is some processed variant, not a
re-serialization; its behavior resembles the instruct model more than the
base. Prefer the official weights.

## How it works

```
web/                Vite + React chat UI and heatmap (canvas)
server.py           FastAPI: /status, /reload_jlens, and /ws (WebSocket)
jspace_rt/host.py   model host: loading, encoding, lens math, generation
calibrate_jlens.py  estimates the J-lens matrices (see below)
validate_jlens.py   side-by-side J-lens vs logit-lens sanity check
scripts/import_model.py   model importer (HF or Ollama source)
```

The backend runs the model with `output_hidden_states=True`, so every
generation step yields the residual stream after every layer. For each
`(layer, position)` it applies a **lens** to the raw activation `h`:

```
lens_l(h) = softmax( W_U · norm( J_l · h ) )
```

`W_U` is the unembedding (with Gemma's final logit softcapping), `norm` is
the model's final RMSNorm, and `J_l` is the lens matrix for layer `l`. The
top-k tokens, their probabilities, the distribution entropy, and the
probability of any tracked concept tokens are streamed over the WebSocket as
each token is generated; the UI paints them incrementally. With a calibrated
J-lens the extra cost per step is one 2560×2560 matmul per layer — real time.

### The J-lens (the interesting part)

The **logit lens** is the special case `J_l = I`: decode every layer as if it
were the last one. It works near the output but reads garbage in early and
middle layers, because each layer's features live in its own basis — there is
no reason the unembedding should understand layer 20's coordinate system.

The paper's fix: measure how layer `l`'s activations *actually* influence the
final residual stream, to first order. That is the expected Jacobian

```
J_l = E[ ∂h_final,t' / ∂h_l,t ]
```

averaged over a corpus. `J_l` is the average linear map from "a perturbation
at layer `l`" to "the resulting change at the final layer" — so `J_l · h`
re-expresses layer-`l` content in the output basis, where the unembedding
*does* make sense. The paper shows the vectors this lens reads are exactly
the model's workspace: swap them and the model's stated beliefs, unspoken
inferences, and downstream behavior change with them.

**How `calibrate_jlens.py` estimates it.** The full Jacobian at one point is
2560 backward passes (one per output dimension) — far too slow to average
over a corpus. Instead it uses an unbiased sketch:

1. For a gaussian probe `u ~ N(0, I)`:  `E[u (uᵀJ)] = J`. One backward pass
   from `u · h_final,t'` yields `uᵀ ∂h_final,t'/∂h_l,t` for **every** layer
   `l` and **every** source position `t` simultaneously — autograd gives all
   43 layer gradients in a single sweep.
2. Rank-1 terms `u ⊗ (uᵀJ)` are accumulated per layer over
   `prompts × target-positions × probes` samples (default 192×3×8 ≈ 4,600
   backward passes, ~13 min on an M-series GPU). The same-position term
   (`t' = t`) is kept separate from the cross-position term (`t' > t`),
   because averaging them together drowns the per-position signal in
   near-zero cross-position blocks; the lens uses the same-position Jacobian
   by default (`--cross-weight` mixes the broadcast term back in if wanted).
3. **Denoising.** A raw sketch of a 2560² matrix from a few thousand rank-1
   samples is noise-dominated. But `J_l` has structure: the residual stream
   gives it a strong identity-like component plus a low-rank correction. So:
   `γ = tr(J_est)/d` (the trace averages 2560 diagonal entries — low
   variance), then SVD-truncate `J_est − γI` to `--rank 256` components,
   and reconstruct `J_l = γI + lowrank`. The truncation discards isotropic
   probe noise while keeping the directions the sketch actually resolved.

The estimated `γ` per layer is itself a nice readout: ≈0 for early layers
(their basis is unrelated to the output basis) and ramping to ~3 near the
final layers (residual pass-through dominates) — you can see it in
`/status` under `jlens_meta.gammas`.

**What to expect.** With the default budget, mid-band readings become
conceptual rather than token-literal — e.g. at L34 the top-5 for the Alice
prompt is ` spiders / Spider / 蜘蛛 / spider / 🕷`: one concept, five
surface forms, which is exactly the "verbalizable representation" the paper
describes. Layers below ~L28 remain noisy at this sample count; quality
scales with `--prompts/--targets/--probes` if you want to spend more time.

Two model-specific gotchas baked into `host.py`:

- **`<bos>` is mandatory.** The Gemma 4 tokenizer does not add it, and
  without it the model degenerates into copy-loops. Everything (chat,
  calibration, validation) prepends it explicitly.
- **Chat format** (no chat template ships with the weights):
  `<|turn>user\n…<turn|>\n<|turn>model\n…`, EOS ids `{1, 106, 50}`.

## Scripts

| command | what it does |
|---|---|
| `npm run setup` | venv + pip install + npm install |
| `npm run import-model [-- --source ollama]` | put weights at `model/gemma-4-E4B` |
| `npm run dev` | backend + frontend, dev mode |
| `npm run calibrate [-- --prompts 512]` | estimate J-lens → `calib/jlens.pt` |
| `npm run validate` | print J-lens vs logit-lens layer-by-layer comparison |
| `npm run test` | full behavioral test suite (recall, chat, lens, workspace signature, throughput) |
| `.venv/bin/python scripts/compare_weights.py` | diff official HF weights vs local Ollama blobs |
| `npm run build` / `npm run start` | build UI to `web/dist`, serve it all from :8731 |

## Why a custom runtime (and not LiteRT-LM)?

The original plan was to bolt this onto
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM). It runs Gemma
efficiently, but it executes pre-compiled `.litertlm` graphs: intermediate
activations are not addressable, and there is no autograd — while the J-lens
is *defined* as a Jacobian. Any faithful implementation needs a
differentiable runtime, so this project hosts the model in
PyTorch (MPS) via transformers' native `Gemma4` support instead, and keeps
the whole lens pipeline (~500 lines of Python) under its own control.
