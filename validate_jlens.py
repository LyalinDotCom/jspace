"""Compare J-lens vs logit-lens readings on the paper's 'unspoken
intermediate' test: the concept the model is about to verbalize should be
readable earlier / more clearly through the J-lens."""

import torch

from jspace_rt.host import Host

PROMPT = "If Alice fears the animal that spins webs, what does Alice fear? One word only."


def main():
    host = Host()
    assert host.jlens is not None, "run calibrate_jlens.py first"
    print("jlens meta:", host.jlens_meta)

    ids = host._encode(PROMPT, chat=True).to(host.device)
    with torch.no_grad():
        out = host.model(input_ids=ids, output_hidden_states=True)
    sp = host.tok.encode(" spiders", add_special_tokens=False)[0]
    sp2 = host.tok.encode(" spider", add_special_tokens=False)[0]

    print(f"{'layer':>5} | {'logit-lens top1':>20} {'p(spider*)':>10} | "
          f"{'J-lens top1':>20} {'p(spider*)':>10}")
    for l in range(len(out.hidden_states)):
        row = []
        for mode in ("logit", "jlens"):
            z = host.lens_logits(out.hidden_states[l][0, -1:].float(), l, mode)
            p = torch.softmax(z, -1)[0]
            top = p.argmax().item()
            row.append((host.tok.decode([top]), (p[sp] + p[sp2]).item()))
        print(f"{l:>5} | {row[0][0]!r:>20} {row[0][1]:>10.4f} | "
              f"{row[1][0]!r:>20} {row[1][1]:>10.4f}")

    print("\nJ-lens top-5 at selected layers (last prompt position):")
    for l in (8, 16, 22, 28, 34, 38):
        z = host.lens_logits(out.hidden_states[l][0, -1:].float(), l, "jlens")
        p = torch.softmax(z, -1)[0]
        tk = torch.topk(p, 5)
        toks = [(host.tok.decode([i]), round(v.item(), 3))
                for i, v in zip(tk.indices, tk.values)]
        print(f"  L{l:2}: {toks}")


if __name__ == "__main__":
    main()
