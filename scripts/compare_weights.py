"""Compare the official google/gemma-4-E4B weights against the local Ollama
gemma4:e4b-mlx-bf16 blobs, tensor by tensor (sampled). Answers: is Ollama's
re-serialization bit-identical to Google's release?

    .venv/bin/python scripts/compare_weights.py [--full]
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "model", "gemma-4-E4B", "model.safetensors")
MANIFEST = os.path.expanduser(
    "~/.ollama/models/manifests/registry.ollama.ai/library/gemma4/e4b-mlx-bf16")
BLOBS = os.path.expanduser("~/.ollama/models/blobs")

SAMPLE = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.embed_tokens_per_layer.weight",
    "model.language_model.norm.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.20.self_attn.k_proj.weight",
    "model.language_model.layers.20.mlp.gate_proj.weight",
    "model.language_model.layers.20.input_layernorm.weight",
    "model.language_model.layers.20.layer_scalar",
    "model.language_model.layers.41.mlp.down_proj.weight",
    "model.vision_tower.patch_embedder.input_proj.weight",
    "model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="compare every tensor")
    args = ap.parse_args()

    if not os.path.exists(OFFICIAL):
        sys.exit("official weights not found — run: npm run import-model")
    m = json.load(open(MANIFEST))
    digest = {l["name"]: l["digest"].replace(":", "-")
              for l in m["layers"] if l.get("name")}

    off = safe_open(OFFICIAL, "pt")
    names = [n for n in off.keys() if n in digest] if args.full else SAMPLE

    identical = mismatched = missing = 0
    for name in names:
        if name not in digest:
            print(f"  [MISS] {name} not in ollama manifest")
            missing += 1
            continue
        a = off.get_tensor(name)
        with safe_open(os.path.join(BLOBS, digest[name]), "pt") as f:
            b = f.get_tensor(name)
        if a.shape != b.shape:
            print(f"  [SHAPE] {name}: {tuple(a.shape)} vs {tuple(b.shape)}")
            mismatched += 1
            continue
        if torch.equal(a, b):
            identical += 1
            if not args.full:
                print(f"  [SAME] {name} {tuple(a.shape)}")
        else:
            d = (a.float() - b.float()).abs()
            print(f"  [DIFF] {name}: max={d.max():.3e} mean={d.mean():.3e}")
            mismatched += 1

    print(f"\n{identical} identical, {mismatched} different, {missing} missing "
          f"(of {len(names)} compared)")
    if mismatched == 0 and missing == 0:
        print("=> Ollama e4b-mlx-bf16 is a bit-identical re-serialization of the "
              "official google/gemma-4-E4B weights.")


if __name__ == "__main__":
    main()
