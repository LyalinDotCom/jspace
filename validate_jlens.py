"""Compare J-lens vs logit-lens readings on an unspoken-intermediate probe.
intermediate' test: the concept the model is about to verbalize should be
readable earlier / more clearly through the J-lens."""

import json

import torch

from jspace_rt.host import Host, validation_path

PROMPT = "If Alice fears the animal that spins webs, what does Alice fear? One word only."


def main():
    host = Host()
    assert host.jlens is not None, "run calibrate_jlens.py first"
    print("jlens meta:", host.jlens_meta)

    ids = host._encode(PROMPT, chat=True).to(host.device)
    with torch.no_grad():
        out = host.model(input_ids=ids, output_hidden_states=True)
    residuals = host.residual_states(out.hidden_states)
    sp = host.tok.encode(" spiders", add_special_tokens=False)[0]
    sp2 = host.tok.encode(" spider", add_special_tokens=False)[0]

    print(f"{'layer':>5} | {'logit-lens top1':>20} {'p(spider*)':>10} | "
          f"{'J-lens top1':>20} {'p(spider*)':>10}")
    for l in range(len(residuals)):
        row = []
        for mode in ("logit", "jlens"):
            z = host.lens_logits(residuals[l][0, -1:].float(), l, mode)
            p = torch.softmax(z, -1)[0]
            top = p.argmax().item()
            row.append((host.tok.decode([top]), (p[sp] + p[sp2]).item()))
        print(f"{l:>5} | {row[0][0]!r:>20} {row[0][1]:>10.4f} | "
              f"{row[1][0]!r:>20} {row[1][1]:>10.4f}")

    print("\nJ-lens top-5 at selected layers (last prompt position):")
    for l in (8, 16, 22, 28, 34, 38):
        if l >= len(residuals):
            continue
        z = host.lens_logits(residuals[l][0, -1:].float(), l, "jlens")
        p = torch.softmax(z, -1)[0]
        tk = torch.topk(p, 5)
        toks = [(host.tok.decode([i]), round(v.item(), 3))
                for i, v in zip(tk.indices, tk.values)]
        print(f"  L{l:2}: {toks}")

    variants = []
    for text in ("spider", " spider", "Spider", " Spider", "spiders", " spiders"):
        ids = host.tok.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            variants.append(ids[0])
    variants = sorted(set(variants))
    band = range(int(host.n_layers * 0.70), host.n_layers - 2)

    def best_rank(mode):
        best = {"rank": host.tok.vocab_size + 1, "layer": None, "token": None}
        for layer in band:
            z = host.lens_logits(residuals[layer][0, -1:].float(), layer, mode)
            p = torch.softmax(z, -1)[0]
            values = p[variants]
            value, index = values.max(0)
            rank = int((p > value).sum().item() + 1)
            if rank < best["rank"]:
                best = {"rank": rank, "layer": layer,
                        "token": host.tok.decode([variants[index.item()]])}
        return best

    logit_best = best_rank("logit")
    jlens_best = best_rank("jlens")
    report = {
        "model": host.model_name,
        "probe": "unspoken_intermediate_spider_v1",
        "workspace_layers": [band.start, band.stop - 1],
        "quality_gate_rank": 100,
        "logit_best": logit_best,
        "jlens_best": jlens_best,
        "logit_validated": logit_best["rank"] <= 100,
        "jlens_validated": jlens_best["rank"] <= 100,
    }
    path = validation_path(host.model_dir)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nquality report -> {path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
