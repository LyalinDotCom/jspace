"""Calibrate a same-position Jacobian lens for the active Gemma model.

Estimates the per-layer expected Jacobian to the final residual stream,

    J_l = E[ dh_final,t' / dh_l,t ]

("Understanding the J-Space", transformer-circuits.pub/2026/workspace) with a
gaussian-probe sketch: for u ~ N(0, I),  E[ u (u^T J) ] = J, and one backward
pass from a probed target position yields u^T J for every layer and source
position at once.

A raw sketch of a 2560 x 2560 matrix from a few thousand rank-1 samples is
noise-dominated, so two structural choices recover the signal:

- same-position term (t' = t) is accumulated separately from the cross-
  position term (t' > t); the lens defaults to the same-position Jacobian,
  which is what "what is this position holding" reads.
- denoising: J ~= gamma*I + low-rank. gamma comes from the trace (d-fold
  variance reduction); the residual is SVD-truncated to --rank components,
  which discards isotropic probe noise.

Usage:
    .venv/bin/python calibrate_jlens.py --prompts 192 --targets 3 --probes 8

This is the per-position estimator explicitly described as a working variant
in Anthropic's reference implementation. The paper's main estimator also sums
current-and-future target positions; ``--cross-weight`` can mix an experimental
cross-position estimate back in.

Writes calib/jlens-<model>.pt; server picks it up on restart or reload.
"""

import argparse
import os
import time

import torch

from jspace_rt.host import Host, jlens_path

TOPICS = [
    "The mitochondria is the organelle responsible for producing ATP through cellular respiration.",
    "In 1969, Apollo 11 landed on the Moon and Neil Armstrong stepped onto the lunar surface.",
    "To make a roux, melt butter in a pan and whisk in an equal weight of flour over low heat.",
    "The stock market fell sharply today as investors reacted to unexpected inflation data.",
    "Once upon a time, a clever fox lived at the edge of a quiet village near the forest.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The defendant was found not guilty on all counts after the jury deliberated for two days.",
    "Photosynthesis converts carbon dioxide and water into glucose using energy from sunlight.",
    "She packed her bags quickly, glancing at the clock: the last train left at midnight.",
    "The Treaty of Westphalia in 1648 established the principle of state sovereignty in Europe.",
    "A spider spins its web using silk proteins produced in specialized abdominal glands.",
    "Quarterly revenue grew 14 percent year over year, driven by strong cloud subscriptions.",
    "If Alice is afraid of the animal that spins webs, then Alice is afraid of spiders.",
    "The recipe calls for two cups of flour, one egg, a pinch of salt, and warm milk.",
    "Tectonic plates drift slowly over the mantle, colliding to form mountains and trenches.",
    "The orchestra tuned to the oboe's A before the conductor raised his baton to begin.",
    "Rainfall in the Amazon basin has declined measurably over the past three decades.",
    "He solved the equation by completing the square and then taking the root of both sides.",
    "The museum's new exhibit features Bronze Age artifacts recovered from a shipwreck.",
    "Neural networks learn by adjusting weights to minimize a loss function via gradient descent.",
    "The senator proposed an amendment that would cap interest rates on consumer loans.",
    "Marine biologists tracked the whale pod as it migrated south along the coastline.",
    "Add the onions and cook until translucent, then stir in garlic for one more minute.",
    "The novel's unreliable narrator slowly reveals that the fire was no accident at all.",
    "Voltage equals current times resistance, a relationship known as Ohm's law.",
    "The startup pivoted from consumer hardware to enterprise software after the funding round.",
    "Ancient Roman concrete owes its durability to volcanic ash mixed into the mortar.",
    "The chess grandmaster sacrificed her queen to force checkmate in three moves.",
    "Hurricanes gain strength over warm ocean water and weaken rapidly after landfall.",
    "The poem contrasts the permanence of art with the brevity of human life.",
    "Interest compounds daily, so the effective annual rate exceeds the nominal rate.",
    "The immune system distinguishes self from non-self using surface protein markers.",
]


def build_corpus(n, tok, max_len, bos):
    batches = []
    for i in range(n):
        t = TOPICS[i % len(TOPICS)]
        ids = tok(t, add_special_tokens=False, return_tensors="pt").input_ids[:, : max_len - 1]
        if i >= len(TOPICS):  # vary truncation so position statistics differ
            cut = 8 + (i * 7) % max(1, ids.shape[1] - 8)
            ids = ids[:, :cut]
        ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
        batches.append(ids)
    return batches


def denoise(J, rank):
    """J ~= gamma*I + low-rank. Keep gamma exactly, truncate the rest."""
    d = J.shape[-1]
    gamma = torch.diagonal(J).mean()
    R = J - gamma * torch.eye(d)
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    R_r = (U[:, :rank] * S[:rank]) @ Vh[:rank]
    return gamma * torch.eye(d) + R_r, gamma.item(), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=192)
    ap.add_argument("--targets", type=int, default=3, help="target positions t' per prompt")
    ap.add_argument("--probes", type=int, default=8, help="probe vectors u per target")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--rank", type=int, default=256, help="SVD rank kept after removing gamma*I")
    ap.add_argument("--cross-weight", type=float, default=0.0,
                    help="mix-in weight for the cross-position (t' > t) term")
    ap.add_argument("--prompts-file", help="optional newline-separated corpus file")
    ap.add_argument("--out", default=None, help="default: calib/jlens-<model>.pt")
    args = ap.parse_args()

    host = Host()
    if args.out is None:
        args.out = jlens_path(host.model_dir)
    print(f"calibrating {host.model_name} -> {args.out}")
    model, tok, dev = host.model, host.tok, host.device
    # One residual-block output per layer. This matches Anthropic's reference
    # hooks and deliberately excludes the embedding row and post-final-norm
    # Hugging Face hidden state.
    d, nl = host.d_model, host.n_layers

    global TOPICS
    if args.prompts_file:
        TOPICS = [l.strip() for l in open(args.prompts_file) if l.strip()]
    corpus = build_corpus(args.prompts, tok, args.max_len, host.tok.bos_token_id or 2)

    J_same = torch.zeros(nl, d, d, dtype=torch.float32)
    J_cross = torch.zeros(nl, d, d, dtype=torch.float32)
    n_same = 0
    n_cross = 0
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    total_bw = 0

    for pi, ids in enumerate(corpus):
        ids = ids.to(dev)
        S = ids.shape[1]
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = host.residual_states(out.hidden_states)
        h_final = hs[-1]

        tprimes = sorted(set(torch.randint(S // 2, S, (args.targets,), generator=g).tolist()))
        for tp in tprimes:
            U = torch.randn(args.probes, d, generator=g).to(dev)
            for k in range(args.probes):
                s = (h_final[0, tp].float() * U[k]).sum()
                grads = torch.autograd.grad(s, hs, retain_graph=True)
                total_bw += 1
                u = U[k].cpu()
                for l, gr in enumerate(grads):
                    gcpu = gr[0, : tp + 1].float().cpu()
                    J_same[l] += torch.outer(u, gcpu[tp])
                    if args.cross_weight and tp > 0:
                        J_cross[l] += torch.outer(u, gcpu[:tp].mean(0))
            n_same += args.probes
            if tp > 0:
                n_cross += args.probes
        del out, hs, h_final
        if (pi + 1) % 8 == 0:
            el = time.time() - t0
            print(f"[{pi+1}/{len(corpus)}] backwards={total_bw} "
                  f"elapsed={el:.0f}s eta={el/(pi+1)*(len(corpus)-pi-1):.0f}s",
                  flush=True)

    J_same /= max(n_same, 1)
    J_cross /= max(n_cross, 1)

    print("denoising (svd per layer)...", flush=True)
    J = torch.zeros_like(J_same)
    gammas = []
    for l in range(nl):
        raw = J_same[l] + args.cross_weight * J_cross[l]
        J[l], gamma, S_ = denoise(raw, args.rank)
        gammas.append(round(gamma, 4))
        if l % 8 == 0:
            print(f"  L{l}: gamma={gamma:.3f} top-sv={S_[0]:.3f} "
                  f"sv[{args.rank}]={S_[min(args.rank, len(S_)-1)]:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"J": J, "meta": {
        "model": host.model_name,
        "prompts": len(corpus), "targets": args.targets, "probes": args.probes,
        "rank": args.rank, "cross_weight": args.cross_weight,
        "backwards": total_bw, "gammas": gammas,
        "estimator": "same_position_gaussian_sketch",
        "corpus": "bundled_synthetic_topics",
        "corpus_topics": len(TOPICS),
        "residual_convention": "block_output_pre_final_norm_v2",
        "reference_compatible_unembed": True,
        "seconds": round(time.time() - t0),
    }}, args.out)
    print(f"saved {args.out}  ({tuple(J.shape)}, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
