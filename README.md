# J·SPACE Observatory — inspect a language model through depth

Chat with a local **Gemma 4** model while a live visualization shows
token-level vocabulary readouts at every residual block and text position.
The public view uses the logit lens; an experimental Jacobian lens is exposed
only with its measured quality status. This is an open-model exploration inspired by
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Transformer Circuits, 2026; general-audience version:
[anthropic.com/research/global-workspace](https://www.anthropic.com/research/global-workspace)).

Everything runs on your machine: the model, the math, the UI. No cloud, no
telemetry, weights never leave `model/`.

---

## The 30,000-foot view

### The science, in three sentences

Anthropic's researchers found that the Claude models they studied keep a small, privileged
set of internal representations — concepts the model can *report, reason
with, and act on* — inside a specific band of middle layers, surrounded by a
much larger ocean of automatic processing it cannot report. This mirrors
"global workspace" theories of human conscious access, and the paper shows
the representations are causal: swap them and the model's stated beliefs and
downstream reasoning change with them. They read this workspace with a tool
called the **J-lens**, and call the sparse set of verbalizable directions the
**J-space**. Their causal and behavioral findings do **not** automatically
transfer to Gemma; this project tests for analogous readouts without treating
them as a literal transcript of thought or evidence of consciousness.

Famous examples: reading buggy code, the workspace contains *error* before
the model says anything; told to lie about the Eiffel Tower, it answers
"Rome" while *Paris* sits near rank 1 internally the whole time.

### What this project is

A chat app with a residual-stream observatory strapped to the side:

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
model types. On the 26B MoE, the full-vocabulary readout can take tens of
seconds for a short prompt; the UI streams layer progress so this work is
visible instead of looking stalled.

### The lens math, in plain terms

A transformer's final layer is directly readable: multiply by the
vocabulary matrix and you get next-token probabilities. Middle layers are
*not* — each speaks its own internal coordinate system. The classic "logit
lens" (decode every layer as if it were the last) is a useful baseline, but
Anthropic found it less reliable than a validated J-lens, especially in
earlier layers. Its reliability must be measured on the model and task being
shown.

The J-lens fix is to measure, for each layer, the *average linear map* from "a
nudge at layer ℓ" to "the resulting change at the final layer" — the
expected Jacobian

```
J_ℓ = E[ ∂h_final,t' / ∂h_ℓ,t ]          lens_ℓ(h) = softmax(W_U · norm(J_ℓ · h))
```

`J_ℓ · h` re-expresses a block output in the final pre-norm residual coordinates,
where the vocabulary matrix *does* make sense. That's the **J-lens**. It is
model-specific and computed once per model by `calibrate_jlens.py`
(~10–25 min), which estimates it with a randomized sketch: gaussian probe
vectors turn each backward pass into one unbiased rank-1 sample of *every*
layer's Jacobian at once, and the noisy sum is denoised using the known
structure `J ≈ γI + low-rank` (γ from the trace, low-rank via truncated
SVD). Details in [The J-lens, in depth](#the-j-lens-in-depth).

### What you actually see

- **Heatmap** (rows = residual blocks, columns = tokens): the default coloring
  is *next-token emergence* — how highly each layer ranks the token predicted
  after that column. It is a diagnostic, not proof of where a decision occurs.
- **Pin a concept** (type it, or click it anywhere) and the coloring
  switches to that concept's *rank* per cell — watch "spider" or "Paris"
  surface out of 262,144 candidates. Pins group common capitalization,
  whitespace, singular, and plural token forms.
- **Candidate alerts**: any pinned concept that reaches top-20 rank inside the
  middle band gets a clickable chip. Rank alone does not establish a global
  workspace, so the UI labels these as candidates.
- **Position inspector**: click a column for the full block-by-block readout at
  that token, with probability bars; click tokens there to pin them.
- **Charts**: pinned-token rank vs layer and per-layer
  kurtosis/confidence profiles related to the paper's fig. 28. High kurtosis
  is necessary but not sufficient evidence of workspace membership.
- **Layer story**: seven checkpoints turn depth into a readable timeline while
  leaving the complete matrix and per-layer evidence available below.
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

## Getting started on a clean machine

### 1. Prerequisites

**macOS on Apple Silicon** (the model runs on MPS), plus:

```bash
python3 --version   # need ≥ 3.11
node --version      # need ≥ 20
git --version
```

RAM and disk depend on the model — see the matrix below. (Linux + CUDA
should work by changing `device="mps"` in `jspace_rt/host.py` — untested.)

### 2. Standard setup — any Mac with 24 GB+ RAM

One paste, start to chatting (downloads gemma-4-E4B-it, ~16 GB):

```bash
git clone https://github.com/LyalinDotCom/jspace.git && cd jspace \
&& npm run setup \
&& npm run calibrate \
&& npm run validate \
&& npm test \
&& npm run dev
```

Then open **http://localhost:5173**. The calibrate step (~20 min) builds
the optional J-lens and `npm run validate` compares it with the logit baseline.
Calibration is skippable if you're impatient: the public app uses the logit
lens, while the experimental J-lens remains unavailable until fitted. A fit is
only promoted if it passes its quality gate.

### 3. Big-machine setup — 64 GB+ RAM (runs the 26B)

Same, plus the official 26B MoE (~52 GB download, resumable). The server
auto-prefers it once present:

```bash
git clone https://github.com/LyalinDotCom/jspace.git && cd jspace \
&& npm run setup \
&& npm run import-model -- --repo google/gemma-4-26B-A4B-it --out model/gemma-4-26B-A4B-it \
&& npm run calibrate -- --prompts 192 \
&& npm run validate \
&& npm test \
&& npm run dev
```

Notes for this path:
- `npm run setup` also downloads gemma-4-E4B-it (~16 GB) as a fallback
  model when none is present — harmless; skip it with Ctrl-C during that
  download if you only want the 26B, then continue with the next command.
- First-pass calibration uses `--prompts 192`; rerun later with
  `-- --prompts 384` for a higher-quality lens.
- Every command is safe to rerun individually — downloads resume,
  calibration overwrites cleanly.

### Day-to-day

```bash
npm run dev                              # start everything → localhost:5173
JSPACE_MODEL=gemma-4-E2B-it npm run dev  # pick a specific model
curl -X POST localhost:8731/reload_jlens # pick up a fresh calibration live
```

### Current 26B validation status

For `gemma-4-26B-A4B-it`, the bundled unspoken-intermediate probe asks for the
animal that spins webs. On the exact same captured activation, the logit lens
puts a grouped `spider` surface form at **rank 1 at L24**; the current randomized
J-lens fit reaches only **rank 4,290 at L24**, failing the rank-100 quality gate.
Consequently the UI defaults to the validated logit view and labels the J-lens
option **experimental, gate failed**. Run `npm run validate` to reproduce the
comparison; the machine-readable result is written to
`calib/validation-<model>.json`.

### Model matrix

Weights are bf16 in HF format under `model/`, git-ignored, downloaded
straight from Google's official Hugging Face repos:

| model | import command | disk | RAM needed | notes |
|---|---|---|---|---|
| gemma-4-E2B-it | `npm run import-model -- --repo google/gemma-4-E2B-it --out model/gemma-4-E2B-it` | 12 GB | 16 GB+ | smallest; not validated here |
| gemma-4-E4B-it | *(default via `npm run setup`)* | 16 GB | 24 GB+ | lower-cost exploratory option; not validated here |
| gemma-4-26B-A4B-it | `npm run import-model -- --repo google/gemma-4-26B-A4B-it --out model/gemma-4-26B-A4B-it` | 52 GB | **64 GB+, 128 GB comfortable** | MoE (128 experts, 8 active); active development target |
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
Chat formatting is taken from each checkpoint's official
`chat_template.jinja` when present, with a conservative manual fallback (see
gotchas below).

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
| `npm run calibrate [-- --prompts 512]` | estimate the J-lens for the active model |
| `npm run validate` | J-lens vs logit-lens layer-by-layer comparison |
| `npm test` | behavioral + lens test suite (recall, chat, workspace, throughput) |
| `.venv/bin/python scripts/compare_weights.py` | diff official weights vs Ollama blobs |
| `npm run build` / `npm run start` | build the UI and serve everything from :8731 |

## Things to try

1. **The lie** (demo menu): track *Paris*, *France*, and *lie*. Whether they
   surface is the experimental result; the UI does not hard-code an outcome.
2. **Silent bug detection**: paste buggy code with an innocent question and
   inspect whether *error*, *bug*, or the corrected token rises in rank.
3. **Unspoken inference**: "If Alice fears the animal that spins webs…"
   with track `spider`. The UI groups common one-token surface forms and shows
   their real best rank; it does not claim a positive result in advance.
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
lens** is the special case `J_ℓ = I` and is available without calibration.
This implementation's experimental J-lens uses the same-position variant
`J_ℓ = E[∂h_final,t / ∂h_ℓ,t]`, which Anthropic's reference implementation
documents as a working alternative to the paper's current-and-future-position
reduction. The target and every source are pre-final-norm block outputs.

**How `calibrate_jlens.py` estimates it.** A full Jacobian at one point
costs `d_model` backward passes — far too slow to average over a corpus.
Instead:

1. **Probe sketch.** For gaussian `u ~ N(0, I)`: `E[u (uᵀJ)] = J`. One
   backward pass from `u · h_final,t'` produces `uᵀ ∂h_final,t'/∂h_ℓ,t` for
   *every* layer ℓ and *every* source position t simultaneously. Each pass
   contributes one unbiased rank-1 sample `u ⊗ (uᵀJ)` to every layer's
   accumulator. The default first-pass budget is 192 prompts × 3 sampled
   targets × 8 probes (duplicates are de-duplicated per prompt).
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

The per-layer `γ` and fit metadata are visible in `/status`. They are useful
fit diagnostics, but `npm run validate` determines whether a calibration is
good enough for the public default. `npm test` separately checks the public
path, model behavior, activation capture, and throughput.

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
  the host renders it via `apply_chat_template`, so the checkpoint's thought
  channels, tool syntax, and future template changes are inherited instead of
  being approximated by a stale hand-written format.
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
