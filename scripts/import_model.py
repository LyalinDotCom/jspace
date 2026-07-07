"""Import Gemma 4 E4B into model/gemma-4-E4B in the layout the server expects.

Two sources:

  --source hf      (default) official Google weights from huggingface.co
                   (google/gemma-4-E4B). Downloads into the HF cache if
                   needed, then symlinks the snapshot into model/gemma-4-E4B.

  --source ollama  reconstructs an HF-format model directory from a local
                   Ollama pull (gemma4:e4b-mlx-bf16). Ollama's tensor-layout
                   manifests store one single-tensor safetensors blob per
                   tensor; we symlink every blob and generate
                   model.safetensors.index.json so transformers can load it.
                   Zero bytes are copied. Note: this is Ollama's MLX-oriented
                   conversion of the same bf16 weights, not the official
                   Google artifact.

Either way the result is a normal transformers-loadable directory; nothing
else in the project cares which source produced it.

Usage:
    .venv/bin/python scripts/import_model.py                 # hf
    .venv/bin/python scripts/import_model.py --source ollama
    .venv/bin/python scripts/import_model.py --repo google/gemma-4-E4B-it
"""

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "model", "gemma-4-E4B")


def clear_dir(out):
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)


def import_hf(repo, out):
    from huggingface_hub import snapshot_download

    # If the dir currently holds an Ollama-blob reconstruction, clear it —
    # its index.json would shadow the official single-file weights. A
    # partially downloaded HF layout is kept so the download can resume.
    if os.path.isdir(out) and any(n.startswith("t-") for n in os.listdir(out)):
        clear_dir(out)
    os.makedirs(out, exist_ok=True)
    print(f"downloading {repo} -> {out} (resumable) ...")
    snapshot_download(repo, local_dir=out)
    print(f"downloaded {repo} -> {out}")


def import_ollama(tag, out):
    manifest = os.path.expanduser(
        f"~/.ollama/models/manifests/registry.ollama.ai/library/{tag.replace(':', '/')}")
    blobs = os.path.expanduser("~/.ollama/models/blobs")
    if not os.path.exists(manifest):
        sys.exit(f"no local Ollama manifest for {tag} — run `ollama pull {tag}` first")
    m = json.load(open(manifest))
    if m["layers"] and "name" not in m["layers"][0]:
        sys.exit(f"{tag} is a GGUF-layout model; use a tensor-layout tag "
                 f"(e.g. e4b-mlx-bf16) or --source hf")

    clear_dir(out)
    weight_map, total = {}, 0
    for l in m["layers"]:
        name = l.get("name", "")
        if not name:
            continue
        src = os.path.join(blobs, l["digest"].replace(":", "-"))
        if name.endswith(".json"):
            os.symlink(src, os.path.join(out, name))
            continue
        fn = f"t-{name}.safetensors"
        os.symlink(src, os.path.join(out, fn))
        weight_map[name] = fn
        total += l["size"]
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    print(f"linked {len(weight_map)} tensors ({total/1e9:.1f} GB) from {tag} -> {out}")


def verify(out):
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(out)
    tok = AutoTokenizer.from_pretrained(out)
    tc = cfg.text_config
    print(f"verified: {cfg.model_type}, {tc.num_hidden_layers} layers, "
          f"d={tc.hidden_size}, vocab={tc.vocab_size}, "
          f"bos={tok.bos_token_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "ollama"], default="hf")
    ap.add_argument("--repo", default="google/gemma-4-E4B", help="HF repo id")
    ap.add_argument("--tag", default="gemma4:e4b-mlx-bf16", help="Ollama tag")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.source == "hf":
        import_hf(args.repo, args.out)
    else:
        import_ollama(args.tag, args.out)
    verify(args.out)
