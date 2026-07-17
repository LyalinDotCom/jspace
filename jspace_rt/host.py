"""Model host for local Gemma checkpoints and per-block lens readouts.

The J-space paper (transformer-circuits.pub/2026/workspace) reads the model's
"workspace" by applying a per-layer lens to residual-stream activations:

    lens_l(h) = softmax(W_U . norm(J_l . h))

where J_l = E[dh_final,t' / dh_l,t] is the expected Jacobian to the final
layer ("J-lens"). With J_l = I this reduces to the classic logit lens.
"""

import json
import os
import threading

import torch

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def default_model_dir():
    """JSPACE_MODEL env wins; otherwise pick the most capable instruct model
    present (the chat UI and the paper's demos assume an assistant), falling
    back to the base model."""
    env = os.environ.get("JSPACE_MODEL")
    if env:
        return env if os.path.isabs(env) else os.path.join(_ROOT, "model", env)
    for name in ("gemma-4-26B-A4B-it", "gemma-4-E4B-it", "gemma-4-E2B-it",
                 "gemma-4-E4B"):
        d = os.path.join(_ROOT, "model", name)
        if os.path.exists(os.path.join(d, "config.json")):
            return d
    return os.path.join(_ROOT, "model", "gemma-4-E4B-it")


def jlens_path(model_dir):
    return os.path.join(_ROOT, "calib",
                        f"jlens-{os.path.basename(os.path.normpath(model_dir))}.pt")


def validation_path(model_dir):
    return os.path.join(_ROOT, "calib",
                        f"validation-{os.path.basename(os.path.normpath(model_dir))}.json")


# legacy single-model path, still read as a fallback
JLENS_PATH = os.path.join(_ROOT, "calib", "jlens.pt")


class Host:
    def __init__(self, model_dir=None, device="mps"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = model_dir or default_model_dir()
        self.model_dir = model_dir
        self.model_name = os.path.basename(os.path.normpath(model_dir))
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        lm = self.model.model.language_model
        self.final_norm = lm.norm
        self.lm_head = self.model.lm_head
        self.layers = lm.layers
        self.n_layers = len(lm.layers)
        self.d_model = lm.config.hidden_size
        self.softcap = getattr(lm.config, "final_logit_softcapping", None)
        self.lock = threading.Lock()
        # Hugging Face's output_hidden_states[-1] is post-final-norm, but the
        # reference J-lens records residual-block outputs and applies the
        # model's norm only during unembedding. Capture the last block output
        # so every row below uses that same, pre-norm residual convention.
        self._last_block_output = None
        self._last_block_hook = self.layers[-1].register_forward_hook(
            self._capture_last_block
        )
        # J-lens matrices: tensor [n_layers+1, d, d] or None (logit-lens mode)
        self.jlens = None
        self.jlens_meta = None
        self.jlens_validation = None
        self.load_jlens()

    def load_jlens(self):
        self.jlens = None
        self.jlens_meta = None
        self.jlens_validation = None
        for path in (jlens_path(self.model_dir), JLENS_PATH):
            if os.path.exists(path):
                blob = torch.load(path, map_location="cpu")
                meta = blob.get("meta", {})
                # a legacy shared file calibrated for another model is worse
                # than no J-lens at all — skip it
                if meta.get("model") and meta["model"] != self.model_name:
                    continue
                J = blob.get("J")
                expected = (self.n_layers, self.d_model, self.d_model)
                if J is None or tuple(J.shape) != expected:
                    continue
                # v2 fixes a critical convention mismatch in the original
                # prototype (post-norm targets followed by a second norm).
                if meta.get("residual_convention") != "block_output_pre_final_norm_v2":
                    continue
                self.jlens = J.to(self.device, torch.bfloat16)
                self.jlens_meta = meta
                report = validation_path(self.model_dir)
                if os.path.exists(report):
                    with open(report) as f:
                        self.jlens_validation = json.load(f)
                return True
        return False

    def _capture_last_block(self, module, inputs, output):
        self._last_block_output = output if torch.is_tensor(output) else output[0]

    def residual_states(self, hidden_states):
        """Return one pre-final-norm residual per transformer block.

        Transformers exposes embeddings plus block outputs, but replaces the
        final block output with its normalized value. Reconstruct the exact
        block-output convention used by Anthropic's reference implementation.
        """
        if len(hidden_states) == self.n_layers:
            return tuple(hidden_states)
        if len(hidden_states) != self.n_layers + 1:
            raise ValueError(
                f"expected {self.n_layers + 1} HF hidden states or "
                f"{self.n_layers} residual states, got {len(hidden_states)}"
            )
        if self._last_block_output is None:
            raise RuntimeError("last residual block output was not captured")
        return tuple(hidden_states[1:-1]) + (self._last_block_output,)

    # ---- lens ----------------------------------------------------------

    def lens_logits(self, h, layer, mode="logit"):
        """h: [S, d] hidden states at `layer` (0 = embeddings). Returns [S, V] logits."""
        if mode == "jlens" and self.jlens is not None:
            h = (h.to(self.jlens.dtype) @ self.jlens[layer].T).float()
        z = self.lm_head(self.final_norm(h))
        if self.softcap:
            z = self.softcap * torch.tanh(z / self.softcap)
        return z

    def lens_read(self, hidden_states, mode="logit", topk=5, track_ids=None,
                  progress=None):
        """hidden_states: tuple of [1, S, d] per layer (len n_layers+1).

        Returns per (layer, position): top-k token ids/probs, entropy, and
        probability of tracked token ids.
        """
        hidden_states = self.residual_states(hidden_states)
        # what the model will actually predict at each position (final layer's
        # argmax) — used for the zero-setup "answer emergence" color mode
        nl = len(hidden_states) - 1
        z_last = self.lens_logits(hidden_states[-1][0].float(), nl, mode="logit")
        final_ids = z_last.argmax(-1)  # [S]

        out = []
        S = hidden_states[0].shape[1]
        idx = torch.arange(S, device=final_ids.device)
        for l, h in enumerate(hidden_states):
            z = self.lens_logits(h[0].float(), l, mode=mode)
            p = torch.softmax(z, dim=-1)
            tk = torch.topk(p, topk, dim=-1)
            ent = -(p * (p + 1e-12).log()).sum(-1)
            # excess kurtosis of the lens logits: a "workspace band" signature
            # (peaks in middle layers — fig. 28b of the J-space paper)
            zn = (z - z.mean(-1, keepdim=True)) / (z.std(-1, keepdim=True) + 1e-6)
            kurt = (zn ** 4).mean(-1) - 3.0
            p_final = p[idx, final_ids]
            final_rank = (p > p_final.unsqueeze(-1)).sum(-1) + 1
            row = {
                "topk_ids": tk.indices.cpu().tolist(),
                "topk_p": [[round(x, 5) for x in r] for r in tk.values.cpu().tolist()],
                "entropy": [round(x, 3) for x in ent.cpu().tolist()],
                "kurt": [round(x, 2) for x in kurt.cpu().tolist()],
                "final_rank": final_rank.cpu().tolist(),
            }
            if track_ids:
                # A watched concept may have several single-token surface
                # forms ("spider", " Spider", "spiders", ...). Treat the
                # best-scoring form as the concept's score instead of making
                # the UI guess which tokenizer spelling the model will use.
                groups = [[g] if isinstance(g, int) else list(g)
                          for g in track_ids]
                track_p, track_rank = [], []
                for group in groups:
                    ids = torch.tensor(group, device=p.device, dtype=torch.long)
                    best_p = p.index_select(-1, ids).max(-1).values
                    track_p.append([
                        round(x, 6) for x in best_p.cpu().tolist()
                    ])
                    track_rank.append(
                        ((p > best_p.unsqueeze(-1)).sum(-1) + 1).cpu().tolist()
                    )
                row["track_p"] = track_p
                row["track_rank"] = track_rank
            out.append(row)
            if progress:
                progress(l + 1, len(hidden_states))
        return out

    # ---- generation ----------------------------------------------------

    def _encode(self, prompt, chat):
        """Gemma 4 is degenerate without <bos>, and this tokenizer does not
        add it on plain encode — ensure it explicitly. Chat prefers the
        official chat_template shipped with -it releases (which renders
        `<bos><|turn>user\\n…<turn|>\\n<|turn>model\\n` and includes <bos>
        itself); the manual fallback produces the identical string for
        template-less checkpoints (e.g. the base model). `prompt` is a raw
        string or, for chat, a string / list of {role, content} messages."""
        if chat:
            msgs = prompt if isinstance(prompt, list) else [
                {"role": "user", "content": prompt}]
            if getattr(self.tok, "chat_template", None):
                text = self.tok.apply_chat_template(
                    [{"role": "assistant" if m["role"] in ("assistant", "model")
                      else "user", "content": m["content"]} for m in msgs],
                    add_generation_prompt=True, tokenize=False)
            else:
                text = "".join(
                    f"<|turn>{'model' if m['role'] in ('assistant', 'model') else 'user'}"
                    f"\n{m['content']}<turn|>\n" for m in msgs) + "<|turn>model\n"
        else:
            text = prompt
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        bos = self.tok.bos_token_id or 2
        if ids[0, 0].item() != bos:  # template may have included <bos> already
            ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
        return ids

    @torch.no_grad()
    def generate_stream(self, prompt, max_new_tokens=64, temperature=0.0,
                        chat=True, stop_event=None):
        """Yields dicts: first a 'prefill' event with hidden states for the
        prompt, then one 'token' event per generated token."""
        ids = self._encode(prompt, chat).to(self.device)

        out = self.model(input_ids=ids, output_hidden_states=True, use_cache=True)
        residuals = self.residual_states(out.hidden_states)
        toks = [self.tok.decode([t]) for t in ids[0].tolist()]
        yield {
            "event": "prefill",
            "token_ids": ids[0].tolist(),
            "tokens": toks,
            "hidden_states": residuals,
        }

        past = out.past_key_values
        cur = self._sample(out.logits[:, -1], temperature)
        eos = set(self.model.generation_config.eos_token_id or [])
        for _ in range(max_new_tokens):
            if stop_event is not None and stop_event.is_set():
                break
            out = self.model(
                input_ids=cur, past_key_values=past,
                output_hidden_states=True, use_cache=True,
            )
            residuals = self.residual_states(out.hidden_states)
            past = out.past_key_values
            tid = cur[0, 0].item()
            yield {
                "event": "token",
                "token_id": tid,
                "token": self.tok.decode([tid]),
                "hidden_states": residuals,
            }
            if tid in eos:
                break
            cur = self._sample(out.logits[:, -1], temperature)

    def _sample(self, logits, temperature):
        if temperature and temperature > 0:
            p = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(p, 1)
        return logits.argmax(-1, keepdim=True)
