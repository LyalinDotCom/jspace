"""Extensive smoke + behavior tests for the J-space viewer stack.

Covers: model loading, BOS handling, factual recall, chat format, multi-turn
encoding, lens math (logit + J-lens), the unspoken-intermediate phenomenon,
workspace-band signature (kurtosis peak in middle layers), and generation
throughput. Exits non-zero on failure.

    .venv/bin/python scripts/test_suite.py [--skip-jlens]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from jspace_rt.host import Host

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def greedy(host, prompt, n=8, chat=False):
    ids = host._encode(prompt, chat).to(host.device)
    text = []
    with torch.no_grad():
        cur = ids
        past = None
        for _ in range(n):
            out = host.model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().item()
            if nxt in (1, 106, 50):
                break
            text.append(nxt)
            cur = torch.tensor([[nxt]], device=host.device)
    return host.tok.decode(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-jlens", action="store_true")
    args = ap.parse_args()

    print("== load ==")
    t0 = time.time()
    host = Host()
    check("model loads", True, f"{time.time()-t0:.1f}s, {host.n_layers} layers, d={host.d_model}")
    check("MPS device", host.device == "mps")

    print("== encoding ==")
    ids = host._encode("hello", chat=False)
    check("BOS prepended", ids[0, 0].item() == (host.tok.bos_token_id or 2))
    ids = host._encode([{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                        {"role": "user", "content": "how are you?"}], chat=True)
    text = host.tok.decode(ids[0])
    check("multi-turn chat format", text.count("<|turn>") == 4 and text.endswith("<|turn>model\n"),
          text.replace("\n", "\\n")[:90] + "…")

    print("== factual recall (greedy) ==")
    cases = [
        ("The Eiffel Tower is located in the city of", "Paris"),
        ("The chemical symbol for gold is", "Au"),
        ("2 + 2 =", "4"),
        ("The capital of Japan is", "Tokyo"),
        ("Water is made of hydrogen and", "oxygen"),
    ]
    for prompt, want in cases:
        got = greedy(host, prompt, 8)
        check(f"{prompt!r}", want.lower() in got.lower(), f"-> {got!r}")

    print("== chat behavior ==")
    got = greedy(host, "What is 12 * 12? Digits only.", 8, chat=True)
    check("chat arithmetic", "144" in got, f"-> {got!r}")
    got = greedy(host, "Name the planet closest to the sun. One word.", 6, chat=True)
    check("chat astronomy", "mercury" in got.lower(), f"-> {got!r}")

    print("== lens pipeline ==")
    prompt = "If Alice fears the animal that spins webs, what does Alice fear? One word only."
    ids = host._encode(prompt, chat=True).to(host.device)
    with torch.no_grad():
        out = host.model(input_ids=ids, output_hidden_states=True)
    check("hidden states count", len(out.hidden_states) == host.n_layers + 1)

    lens = host.lens_read(out.hidden_states, mode="logit", topk=5,
                          track_ids=[host.tok.encode(" spiders", add_special_tokens=False)[0]])
    check("lens rows complete", len(lens) == host.n_layers + 1
          and all(len(r["entropy"]) == ids.shape[1] for r in lens))

    nl = host.n_layers + 1
    band = range(int(nl * 0.70), nl - 2)  # workspace band, relative depth
    p_spider = max(lens[l]["track_p"][0][-1] for l in band)
    # workspace strength scales with model size (paper fig. 10) — 0.15 admits
    # E2B while still requiring a strong, unambiguous signal (vocab is 262k)
    check("unspoken intermediate (logit lens)", p_spider > 0.15,
          f"max p(' spiders') in L{band.start}-{band.stop - 1} at last prompt pos = {p_spider:.3f}")

    def kmean(rng):
        return sum(sum(lens[l]["kurt"]) / len(lens[l]["kurt"]) for l in rng) / len(rng)
    kurt_mid = kmean(range(nl // 2, nl - 3))
    kurt_early = kmean(range(1, nl // 3))
    check("workspace kurtosis signature", kurt_mid > kurt_early,
          f"mid={kurt_mid:.1f} vs early={kurt_early:.1f}")

    if not args.skip_jlens and host.jlens is not None:
        lens_j = host.lens_read(out.hidden_states, mode="jlens", topk=5,
                                track_ids=[host.tok.encode(" spiders", add_special_tokens=False)[0]])
        pj = max(lens_j[l]["track_p"][0][-1] for l in band)
        check("unspoken intermediate (J-lens)", pj > 0.1,
              f"max p(' spiders') in L{band.start}-{band.stop - 1} = {pj:.3f}")
        lmid = int(nl * 0.85)
        top5_join = " ".join(host.tok.decode([i]) for i in lens_j[lmid]["topk_ids"][-1])
        check(f"J-lens reads concept at L{lmid}", "spider" in top5_join.lower(),
              f"L{lmid} top5: {top5_join!r}")
    elif host.jlens is None:
        print("  [SKIP] J-lens not calibrated")

    print("== throughput ==")
    t0 = time.time()
    n = 0
    for ev in host.generate_stream("Write one sentence about the ocean.", max_new_tokens=24):
        n += ev["event"] == "token"
    dt = time.time() - t0
    check("streaming generation", n >= 8, f"{n} tokens in {dt:.1f}s ({n/dt:.1f} tok/s incl. lens)")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
