"""Extend the LoRA+ lambda sweep to bracket the optimum.

The main sweep found lambda monotonically better up to 16 under both initialisation
schemes, which means the optimum was never bracketed. This pushes to 32 and 64 at the
best base rate found (1e-3) to see whether it turns over.

Run: python src/lora_plus_extended.py
"""
import torch
from transformers import GPT2TokenizerFast

from lora_plus import INIT_SCHEMES, build_data, train

LAMBDAS = [8.0, 16.0, 32.0, 64.0, 128.0]
LR = 1e-3


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens, mask = build_data(tokenizer)
    print(f"base lr fixed at {LR:g}, rank 16, 100 steps\n")

    results = {}
    for scheme in INIT_SCHEMES:
        for lam in LAMBDAS:
            losses = train(lam, LR, scheme, tokens, mask)
            results[(scheme, lam)] = losses[-1]
            print(f"{scheme}  lambda={lam:>5.0f}  final loss {losses[-1]:.4f}")

    print("\n" + "=" * 58)
    print(f"{'lambda':>8}" + "".join(f"{s:>16}" for s in INIT_SCHEMES))
    print("-" * 58)
    for lam in LAMBDAS:
        row = f"{lam:>8.0f}"
        for scheme in INIT_SCHEMES:
            row += f"{results[(scheme, lam)]:>16.4f}"
        print(row)

    print()
    for scheme in INIT_SCHEMES:
        best_lam = min(LAMBDAS, key=lambda l: results[(scheme, l)])
        turned = best_lam != max(LAMBDAS)
        print(f"{scheme}: best lambda = {best_lam:.0f}  "
              f"({'optimum bracketed' if turned else 'still monotonic, not bracketed'})")


if __name__ == "__main__":
    main()
