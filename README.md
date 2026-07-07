# J-Space Chat — watch a language model think, in real time

Chat with a local **Gemma 4** model while a live visualization shows what
every layer of the network is "holding in mind" at every token — the model's
**J-space**, per
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Transformer Circuits, 2026; general-audience version:
[anthropic.com/research/global-workspace](https://www.anthropic.com/research/global-workspace)).

Everything runs on your machine: the model, the math, the UI. No cloud, no
telemetry, weights never leave `model/`.

---

## The 30,000-foot view

### The science, in three sentences

Anthropic's researchers found that language models keep a small, privileged
set of internal representations — concepts the model can *report, reason
with, and act on* — inside a specific band of middle layers, surrounded by a
much larger ocean of automatic processing it cannot report. This mirrors
"global workspace" theories of human conscious access, and the paper shows
the representations are causal: swap them and the model's stated beliefs and
downstream reasoning change with them. They read this workspace with a tool
called the **J-lens**, and the space of things it can read is the
**J-space** — the model's silent thoughts, visible before any text is
produced.

Famous examples: reading buggy code, the workspace contains *error* before
the model says anything; told to lie about the Eiffel Tower, it answers
"Rome" while *Paris* sits near rank 1 internally the whole time.

### What this project is

A chat app with a "thought monitor" strapped to the side:

```
 ┌────────────────────────┐   WebSocket    ┌──────────────────────────────┐
 │  Browser (React/Vite)  │◄──────────────►│  FastAPI server (server.py)  │
 │  chat + heatmap +      │  per-token     │        │                     │
 │  charts + alerts       │  lens events   │  Model host (jspace_rt/)     │
 └────────────────────────┘                │  Gemma 4 on PyTorch + MPS    │
                                           │  · runs the chat             │
        calib/jlens-<model>.pt ──loads──►  │  · captures every layer's    │
        (made once per model by            │    residual stream           │
         calibrate_jlens.py)               │  · applies the lens per      │
                                           │    (layer, position)         │
                                           └──────────────────────────────┘
```

Every generation step, the host captures the **residual stream** — the
model's working memory — after each of its layers, applies a **lens** that
decodes each layer's state into "which words is this layer disposed to
say", and streams the readings to the browser, which paints them as the
model types. ~10 tokens/sec including all the lens math on an M-series Mac.

### The lens math, in plain terms

A transformer's final layer is directly readable: multiply by the
vocabulary matrix and you get next-token probabilities. Middle layers are
*not* — each speaks its own internal coordinate system. The classic "logit
lens" (decode every layer as if it were the last) therefore works near the
output and reads garbage in the middle — precisely where the interesting
thoughts live.

The paper's fix: measure, for each layer, the *average linear map* from "a
nudge at layer ℓ" to "the resulting change at the final layer" — the
expected Jacobian

```
J_ℓ = E[ ∂h_final,t' / ∂h_ℓ,t ]          lens_ℓ(h) = softmax(W_U · norm(J_ℓ · h))
```

`J_ℓ · h` re-expresses layer ℓ's content in the output's coordinate system,
where the vocabulary matrix *does* make sense. That's the **J-lens**. It is
model-specific and computed once per model by `calibrate_jlens.py`
(~10–25 min), which estimates it with a randomized sketch: gaussian probe
vectors turn each backward pass into one unbiased rank-1 sample of *every*
layer's Jacobian at once, and the noisy sum is denoised using the known
structure `J ≈ γI + low-rank` (γ from the trace, low-rank via truncated
SVD). Details in [The J-lens, in depth](#the-j-lens-in-depth).

### What you actually see

- **Heatmap** (rows = layers, columns = tokens): the default coloring is
  *answer emergence* — how highly each layer ranks the token the model
  eventually outputs there. Easy words light up near the bottom; hard
  decisions crystallize high in the network. Zero setup required.
- **Pin a concept** (type it, or click it anywhere) and the coloring
  switches to that concept's *rank* per cell — watch "spider" or "Paris"
  surface out of 262,144 candidates and ride through the middle-layer
  workspace band.
- **Alerts**: any pinned concept that reaches top-20 rank inside the
  workspace band gets a clickable chip — the blog post's "silent thought
  monitoring" (*error* on buggy code, *Paris* during the lie).
- **Position inspector**: click a column for the full 43-layer readout at
  that token, with probability bars; click tokens there to pin them.
- **Charts**: pinned-token rank vs layer (the concept's trajectory through
  the network), and the per-layer kurtosis/confidence profile — the
  workspace band signature from the paper's fig. 28.
- **Demo menu**: one-click setups — lying, silent bug detection, unspoken
  inference, mental arithmetic, rhyme planning, directed focus, multi-hop.

### Why a custom runtime (and not LiteRT-LM / Ollama / LM Studio)?

Fast local runtimes execute compiled or quantized graphs: intermediate
activations aren't addressable and there is no autograd — but the J-lens is
*defined* as a Jacobian, and the whole point is reading per-layer
activations live. So the host runs the model in PyTorch (Apple MPS) via
transformers' native `Gemma4` support, where both are first-class. The
trade: weights must be bf16, so RAM decides which model size you can run
(see the matrix below).

---

## Setting up a new machine

Prereqs: **macOS on Apple Silicon** (MPS), **Python ≥ 3.11**, **Node ≥ 20**,
git, and disk/RAM per the model matrix below. (Linux + CUDA should work by
changing `device="mps"` in `jspace_rt/host.py` — untested.)

```bash
git clone https://github.com/LyalinDotCom/jspace.git
cd jspace
npm run setup        # venv + pip deps + npm deps; offers the default model
                     # download (gemma-4-E4B-it, ~16 GB) if none is present
npm run dev          # → http://localhost:5173
```

Then, once per model (worth it — the logit-lens fallback is blind in the
middle layers where the workspace lives):

```bash
npm run calibrate    # J-lens for the active model → calib/jlens-<model>.pt
npm test             # 18-check behavioral + lens test suite
```

Restart `npm run dev` (or `curl -X POST localhost:8731/reload_jlens`) after
calibrating.

### Model matrix

Weights are bf16 in HF format under `model/`, git-ignored, downloaded
straight from Google's official Hugging Face repos:

| model | import command | disk | RAM needed | notes |
|---|---|---|---|---|
| gemma-4-E2B-it | `npm run import-model -- --repo google/gemma-4-E2B-it --out model/gemma-4-E2B-it` | 12 GB | 16 GB+ | smallest; workspace signal present but weaker |
| gemma-4-E4B-it | *(default via `npm run setup`)* | 16 GB | 24 GB+ | the sweet spot on 32 GB machines; all demos verified |
| gemma-4-26B-A4B-it | `npm run import-model -- --repo google/gemma-4-26B-A4B-it --out model/gemma-4-26B-A4B-it` | 52 GB | **64 GB+, 128 GB comfortable** | MoE (128 experts, 8 active); strongest workspace expected — the paper found workspace properties strengthen with scale |
| gemma-4-E4B (base) | `npm run import-model` | 16 GB | 24 GB+ | pretrained model, raw-completion research |

The server auto-selects the most capable instruct model present
(26B → E4B → E2B); override with `JSPACE_MODEL`:

```bash
JSPACE_MODEL=gemma-4-E2B-it npm run dev
JSPACE_MODEL=gemma-4-26B-A4B-it npm run calibrate   # on the big machine
```

Calibrations are per-model (`calib/jlens-<model>.pt`, with a model-name
guard so the wrong one can never load). After importing a new model:
`npm run calibrate`, restart, done. Rough calibration times: E2B ~10 min,
E4B ~22 min on an M-series; the 26B MoE will take longer — start with
`npm run calibrate -- --prompts 192` for a first pass.

Downloads are resumable; nothing else in the pipeline changes per model.
Chat formatting (Gemma's `<bos>` / `<|turn>` conversation tokens) is taken
from each checkpoint's official `chat_template.jinja` when present, with a
byte-identical manual fallback (see gotchas below).

### Offline fallback

With a local `ollama pull gemma4:e4b-mlx-bf16` you can build a model dir
with zero download: `npm run import-model -- --source ollama` symlinks
Ollama's per-tensor blobs and generates the safetensors index. **Provenance
warning**: `scripts/compare_weights.py` shows Ollama's language-model
tensors differ from Google's official release (mean |Δ| ≈ 1e-2; only the
vision/audio towers match) — it's a processed variant. Prefer official.

---

## Everyday commands

| command | what it does |
|---|---|
| `npm run setup` | venv + deps; downloads default model if missing |
| `npm run dev` | backend (FastAPI :8731) + frontend (Vite :5173) |
| `npm run import-model [-- --repo … --out …]` | fetch official weights into `model/` |
| `npm run calibrate [-- --prompts 512]` | estimate the J-lens for the active model |
| `npm run validate` | J-lens vs logit-lens layer-by-layer comparison |
| `npm test` | behavioral + lens test suite (recall, chat, workspace, throughput) |
| `.venv/bin/python scripts/compare_weights.py` | diff official weights vs Ollama blobs |
| `npm run build` / `npm run start` | build the UI and serve everything from :8731 |

## Things to try

1. **The lie** (demo menu): the model confidently answers "Rome"; the alert
   chip shows *Paris* at rank ~2 in the workspace the entire time, and the
   trajectory chart shows the truth diving only at the final output layers.
2. **Silent bug detection**: paste buggy code with an innocent question —
   *error* surfaces in the workspace while the model reads the code.
3. **Unspoken inference**: "If Alice fears the animal that spins webs…"
   with track `spider` — the inferred concept appears at the end of the
   prompt, before generation starts, in five surface forms
   (` spiders / Spider / 蜘蛛 / 🕷`) — one concept, many words: the
   verbalizable representation itself.
4. **No setup at all**: ask anything and read the emergence heatmap —
   the height where each column turns bright is the layer where the model
   settled that token.

---

## The J-lens, in depth

The lens applied to every activation `h` at layer ℓ is

```
lens_ℓ(h) = softmax( W_U · norm( J_ℓ · h ) )
```

`W_U` = unembedding (with Gemma's final logit softcapping), `norm` = the
model's final RMSNorm, and `J_ℓ` is the layer's lens matrix. The **logit
lens** is the special case `J_ℓ = I` — available instantly, readable only
near the output. The **J-lens** uses the corpus-averaged Jacobian
`J_ℓ = E[∂h_final,t' / ∂h_ℓ,t]`: the average first-order effect of layer
ℓ's state on the final residual stream. That matrix re-expresses middle
layers in the output basis, which is what makes the workspace readable.

**How `calibrate_jlens.py` estimates it.** A full Jacobian at one point
costs `d_model` backward passes — far too slow to average over a corpus.
Instead:

1. **Probe sketch.** For gaussian `u ~ N(0, I)`: `E[u (uᵀJ)] = J`. One
   backward pass from `u · h_final,t'` produces `uᵀ ∂h_final,t'/∂h_ℓ,t` for
   *every* layer ℓ and *every* source position t simultaneously. Each pass
   contributes one unbiased rank-1 sample `u ⊗ (uᵀJ)` to every layer's
   accumulator. Default budget: 384 prompts × 3 target positions × 8 probes
   ≈ 9,200 backward passes.
2. **Same-position vs cross-position.** The same-position term (`t' = t`,
   "what is this position holding") is accumulated separately from the
   cross-position broadcast term (`t' > t`); the lens uses same-position by
   default (`--cross-weight` mixes the other back in). Averaging them
   together drowns the signal in near-zero cross-position blocks.
3. **Structural denoising.** A raw sketch of a `d²` matrix from ~10⁴ rank-1
   samples is noise-dominated. But `J_ℓ` has known structure: a strong
   identity-like component (the residual stream passes through) plus a
   low-rank correction. So: `γ = tr(J_est)/d` (trace averaging makes γ
   low-variance), subtract `γI`, SVD-truncate the remainder to `--rank 256`
   directions, reconstruct `J_ℓ = γI + lowrank`. The truncation throws away
   isotropic probe noise while keeping the directions the sketch resolved.

The per-layer `γ` is itself diagnostic (visible in `/status` →
`jlens_meta.gammas`): ≈ 0 in early layers — their coordinate system is
unrelated to the output's — ramping to ~3 near the end where pass-through
dominates. And the payoff is visible in `npm run validate`: at mid-layers
where the logit lens reads noise, the J-lens reads the *concept* — e.g.
` spiders / Spider / 蜘蛛 / spider / 🕷` as one activation's top-5.

At view time the J-lens costs one extra `d×d` matmul per layer per token —
that's what makes real-time possible.

**Failure mode we hit so you don't have to:** calibrating on a corpus
encoded *without* `<bos>` produces Jacobians of the degenerate copy-loop
regime — the resulting lens is garbage even though the math ran fine.
BOS matters (see below).

## Gemma 4 gotchas (learned the hard way)

- **`<bos>` is mandatory.** The tokenizer does not add it on plain
  `encode()`; without it Gemma 4 degenerates into copy-loops ("the city of
  the city of…"). Every path here (chat, raw completion, calibration,
  tests) ensures exactly one leading `<bos>`.
- **Conversation tokens.** Turns are framed as
  `<|turn>user\n…<turn|>\n<|turn>model\n…`, and end-of-turn (`<turn|>`,
  id 106) is one of the EOS ids ({1, 106, 50} from `generation_config`).
  When a checkpoint ships `chat_template.jinja` (the `-it` releases do),
  the host renders it via `apply_chat_template` — verified byte-identical
  to the manual format above for plain conversations — so tool-call syntax
  and future template changes are inherited automatically.
- **Final logits are softcapped** (`30·tanh(z/30)`); the lens applies the
  same cap so lens probabilities are comparable to real output
  probabilities.
- **Weights provenance matters twice over.** Ollama's "bf16" 26B/E4B tags
  are not byte-identical to Google's releases, and the *base* model shows
  much weaker workspace phenomena than the instruct model under chat
  prompts — if a demo looks dead, check you're on an official `-it`
  checkpoint first.

## Repository layout

```
jspace_rt/host.py       model host: loading, encoding, lens math, streaming
server.py               FastAPI + WebSocket, serves built UI in production
web/                    Vite + React app (heatmap, inspector, charts, alerts)
calibrate_jlens.py      J-lens estimation (probe sketch + denoising)
validate_jlens.py       J-lens vs logit-lens comparison
scripts/import_model.py official HF download / Ollama reconstruction
scripts/test_suite.py   behavioral + lens tests (npm test)
scripts/compare_weights.py  provenance diff
model/  calib/          weights & calibrations — git-ignored, stay local
```
