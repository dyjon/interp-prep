"""LoRA vs LoRA+ on GPT-2 small: sweeping the learning-rate ratio lambda.

Hayou, Ghosh and Yu (ICML 2024) show that LoRA's A and B matrices should not share
a learning rate. B is initialised at zero and A is not, so dL/dA is proportional to B
and vanishes at initialisation: A cannot move until B does. Their Theorem 1 says
efficient feature learning requires eta_A = Theta(1/n) and eta_B = Theta(1), hence a
ratio eta_B/eta_A = Theta(n) in the width. LoRA+ fixes a ratio lambda and tunes eta_A
only, keeping the search one-dimensional.

The paper's recommended lambda depends on which matrix is zeroed at initialisation:

    Init[1]   B = 0, A random    lambda ~ 2^2 to 2^3   (4 to 8)   <- standard LoRA
    Init[2]   A = 0, B random    lambda ~ 2^4          (16)

An earlier version of this script used Init[1] with lambda = 16, which is the Init[2]
recommendation. This sweep tests both schemes across lambda to see whether the
init-dependence the paper reports shows up at this scale.

Task: 128 synthetic person-to-city associations, loss scored on the city token only,
so what is measured is the new association rather than general English.

Run: python src/lora_plus.py
"""
import math
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SEED = 0
RANK = 16
ALPHA = 16
STEPS = 100

LAMBDAS = [1.0, 4.0, 8.0, 16.0]          # 1.0 is vanilla LoRA
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
INIT_SCHEMES = ["init1", "init2"]

NAMES = [
    "Alden", "Brill", "Corven", "Dremer", "Eskil", "Farlow", "Girn", "Halvard",
    "Ivo", "Jarn", "Kessler", "Lorne", "Merrick", "Norwell", "Orsic", "Pell",
    "Quillon", "Rask", "Sarel", "Tolm", "Umber", "Vance", "Wendel", "Xander",
    "Yorrick", "Zeller", "Astrid", "Bevin", "Cardew", "Delphine", "Ewan", "Fenna",
]
CITIES = [
    "Vienna", "Lisbon", "Oslo", "Prague", "Dublin", "Helsinki", "Warsaw", "Zagreb",
    "Athens", "Bergen", "Cardiff", "Dresden", "Edinburgh", "Florence", "Geneva", "Hamburg",
]


def build_data(tokenizer, n=128):
    rng = random.Random(SEED)
    inputs, city_positions = [], []
    for i in range(n):
        base = NAMES[i % len(NAMES)]
        suffix = i // len(NAMES)
        name = base if suffix == 0 else f"{base}{suffix}"
        city = rng.choice(CITIES)
        prefix = f"{name} lives in"
        inputs.append(tokenizer(f"{prefix} {city}.")["input_ids"])
        city_positions.append(len(tokenizer(prefix)["input_ids"]))

    width = max(len(x) for x in inputs)
    tokens = torch.full((len(inputs), width), tokenizer.eos_token_id, dtype=torch.long)
    mask = torch.zeros((len(inputs), width), dtype=torch.bool)
    for i, (ids, pos) in enumerate(zip(inputs, city_positions)):
        tokens[i, : len(ids)] = torch.tensor(ids)
        mask[i, pos] = True
    return tokens, mask


class LoRAAdapter(nn.Module):
    """Wraps a GPT-2 Conv1D layer, whose weight is [in_features, out_features].

    init1: B = 0, A ~ N(0, 1/n_in).  Standard LoRA. Variance scales with width,
           per the paper's sigma_a^2 = Theta(n^-1), not with rank.
    init2: A = 0, B ~ N(0, 1).       sigma_b^2 = Theta(1).
    """

    def __init__(self, base, rank, alpha, scheme):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        n_in, n_out = base.weight.shape

        if scheme == "init1":
            self.A = nn.Parameter(torch.randn(n_in, rank) / math.sqrt(n_in))
            self.B = nn.Parameter(torch.zeros(rank, n_out))
        elif scheme == "init2":
            self.A = nn.Parameter(torch.zeros(n_in, rank))
            self.B = nn.Parameter(torch.randn(rank, n_out))
        else:
            raise ValueError(scheme)

        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale


def train(lam, lr, scheme, tokens, mask):
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.train()
    for p in model.parameters():
        p.requires_grad = False

    adapters = []
    for block in model.transformer.h:
        adapter = LoRAAdapter(block.attn.c_attn, RANK, ALPHA, scheme)
        block.attn.c_attn = adapter
        adapters.append(adapter)

    opt = torch.optim.AdamW(
        [
            {"params": [a.A for a in adapters], "lr": lr},
            {"params": [a.B for a in adapters], "lr": lr * lam},
        ]
    )

    losses = []
    for _ in range(STEPS):
        logits = model(tokens).logits
        target_logits = logits[:, :-1][mask[:, 1:]]
        targets = tokens[:, 1:][mask[:, 1:]]
        loss = nn.functional.cross_entropy(target_logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens, mask = build_data(tokenizer)
    print(f"{tokens.shape[0]} examples, {mask.sum().item()} scored positions")
    print(f"rank {RANK}, {STEPS} steps, AdamW\n")

    results = {}
    for scheme in INIT_SCHEMES:
        for lr in LEARNING_RATES:
            for lam in LAMBDAS:
                key = (scheme, lr, lam)
                losses = train(lam, lr, scheme, tokens, mask)
                results[key] = losses
                print(f"{scheme}  lr={lr:g}  lambda={lam:>4.0f}  final loss {losses[-1]:.4f}")

    for scheme in INIT_SCHEMES:
        label = ("Init[1]: B=0, A random (standard LoRA). Paper suggests lambda 4-8."
                 if scheme == "init1" else
                 "Init[2]: A=0, B random. Paper suggests lambda 16.")
        print("\n" + "=" * 66)
        print(label)
        header = "  base lr  " + "".join(f"{'l=' + str(int(l)):>12}" for l in LAMBDAS)
        print(header)
        print("-" * 66)
        for lr in LEARNING_RATES:
            row = f"{lr:>10g}  "
            for lam in LAMBDAS:
                row += f"{results[(scheme, lr, lam)][-1]:>12.4f}"
            print(row)

        best = min(((results[(scheme, lr, lam)][-1], lr, lam)
                    for lr in LEARNING_RATES for lam in LAMBDAS))
        vanilla = min(results[(scheme, lr, 1.0)][-1] for lr in LEARNING_RATES)
        print(f"\n  best overall: loss {best[0]:.4f} at lr={best[1]:g}, lambda={best[2]:.0f}")
        print(f"  best vanilla (lambda=1): loss {vanilla:.4f}")
        if best[2] != 1.0:
            print(f"  LoRA+ improvement: {(vanilla - best[0]) / vanilla:+.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, scheme in zip(axes, INIT_SCHEMES):
        for lam in LAMBDAS:
            finals = [results[(scheme, lr, lam)][-1] for lr in LEARNING_RATES]
            ax.plot(LEARNING_RATES, finals, "o-", label=f"lambda {lam:.0f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("base learning rate (eta_A)")
        ax.set_title("Init[1]  B=0, A random" if scheme == "init1"
                     else "Init[2]  A=0, B random")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("final loss on the city token")
    fig.suptitle(f"LoRA+ ratio sweep, GPT-2 small, rank {RANK}, {STEPS} steps")
    fig.tight_layout()
    fig.savefig("report_lora_plus.png", dpi=150, bbox_inches="tight")
    print("\nSaved report_lora_plus.png")


if __name__ == "__main__":
    main()
