"""Does the optimal LoRA+ ratio depend on rank?

Two reasons to check. First, the earlier sweep used rank 16 while the LoRA+ GLUE
experiments use alpha = r = 8, so rank is one of the differences that could explain why
the optima found here sit above the published ones. Second, the muA paper (Chen, Villar,
Hayou) is specifically about how the optimal learning rate scales with adapter rank, so
this is the axis their theory speaks to.

Fixed: Init[1], lr 1e-3, 100 steps. Two seeds per cell, since the earlier work had none.

Run: python src/lora_rank.py
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from lora_plus import ALPHA, LoRAAdapter, build_data

LR = 1e-3
SCHEME = "init1"
RANKS = [4, 8, 16, 32]
LAMBDAS = [1.0, 8.0, 16.0, 32.0]
SEEDS = [0, 1]
STEPS = 100


def train_once(rank, lam, seed, tokens, mask):
    torch.manual_seed(seed)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.train()
    for p in model.parameters():
        p.requires_grad = False

    adapters = []
    for block in model.transformer.h:
        # alpha tracks rank so the alpha/r scaling factor stays fixed at 1
        adapter = LoRAAdapter(block.attn.c_attn, rank, rank, SCHEME)
        block.attn.c_attn = adapter
        adapters.append(adapter)

    opt = torch.optim.AdamW([
        {"params": [a.A for a in adapters], "lr": LR},
        {"params": [a.B for a in adapters], "lr": LR * lam},
    ])

    for _ in range(STEPS):
        logits = model(tokens).logits
        loss = nn.functional.cross_entropy(
            logits[:, :-1][mask[:, 1:]], tokens[:, 1:][mask[:, 1:]])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens, mask = build_data(tokenizer)
    print(f"lr {LR:g}, {SCHEME}, {STEPS} steps, alpha = rank so alpha/r = 1")
    print(f"{len(RANKS) * len(LAMBDAS) * len(SEEDS)} runs\n")

    res = {}
    for rank in RANKS:
        for lam in LAMBDAS:
            vals = [train_once(rank, lam, s, tokens, mask) for s in SEEDS]
            res[(rank, lam)] = (float(np.mean(vals)), float(np.std(vals)))
            print(f"rank={rank:>3}  lambda={lam:>5.0f}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\n" + "=" * 66)
    print("Mean final loss\n")
    print("  rank  " + "".join(f"{'l=' + str(int(l)):>13}" for l in LAMBDAS))
    print("-" * 66)
    for rank in RANKS:
        row = f"{rank:>6}  "
        for lam in LAMBDAS:
            row += f"{res[(rank, lam)][0]:>13.4f}"
        print(row)

    print("\nBest lambda at each rank:")
    best = {}
    for rank in RANKS:
        b = min(LAMBDAS, key=lambda l: res[(rank, l)][0])
        best[rank] = b
        vanilla = res[(rank, 1.0)][0]
        gain = (vanilla - res[(rank, b)][0]) / vanilla
        print(f"  rank {rank:>3}: lambda={b:>5.0f}   loss {res[(rank, b)][0]:.4f}   "
              f"{gain:+.1%} against vanilla")

    vals = set(best.values())
    print()
    if len(vals) > 1:
        print("The optimal lambda DEPENDS on rank in this range.")
        print("  " + ", ".join(f"r={r} wants {best[r]:.0f}" for r in RANKS))
        print("  That is the axis muA characterises, and it is a live explanation for")
        print("  why these optima sit above the published ones (which use r=8).")
    else:
        print(f"The optimal lambda is STABLE at {list(vals)[0]:.0f} across ranks {RANKS}.")
        print("  Rank does not explain the gap against the published values here.")

    plt.figure(figsize=(7, 4.5))
    for rank in RANKS:
        plt.plot(LAMBDAS, [res[(rank, l)][0] for l in LAMBDAS], "o-", label=f"rank {rank}")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("lambda")
    plt.ylabel("final loss on the city token")
    plt.title(f"Optimal LoRA+ ratio against rank, GPT-2 small, {STEPS} steps")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("report_rank.png", dpi=150, bbox_inches="tight")
    print("\nSaved report_rank.png")


if __name__ == "__main__":
    main()
