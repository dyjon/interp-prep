"""LoRA vs LoRA+ on GPT-2 small.

Hayou et al. observe that in a LoRA adapter, the A and B matrices should not share a
learning rate: B is initialised at zero while A is not, so the two sit at different
scales and a single rate is suboptimal. LoRA+ sets lr_B = lambda * lr_A.

This compares the two directly. The adapter is implemented here rather than pulled from
a library, because the whole experiment is about controlling the per-matrix learning
rates and that is easier to get right when the parameters are explicit.

Task. A synthetic association the base model cannot already know: 128 person-to-city
pairs, formatted "<name> lives in <city>." Loss is measured on the city token only, so
what is being learned is the new association rather than general English.

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
LAMBDAS = [1.0, 16.0]  # 1.0 is vanilla LoRA, 16.0 is LoRA+
# The base rate has to be swept alongside lambda. Reusing a rate tuned for vanilla LoRA
# and then multiplying lr_B by 16 is not a fair comparison, it just makes lr_B too large.
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
STEPS = 100
ALPHA = 16

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
    pairs = []
    for i in range(n):
        name = NAMES[i % len(NAMES)]
        suffix = i // len(NAMES)
        name = name if suffix == 0 else f"{name}{suffix}"
        city = rng.choice(CITIES)
        pairs.append((name, city))

    inputs, city_positions = [], []
    for name, city in pairs:
        prefix = f"{name} lives in"
        full = f"{prefix} {city}."
        prefix_ids = tokenizer(prefix)["input_ids"]
        full_ids = tokenizer(full)["input_ids"]
        inputs.append(full_ids)
        city_positions.append(len(prefix_ids))  # index of the first city token

    width = max(len(x) for x in inputs)
    pad = tokenizer.eos_token_id
    tokens = torch.full((len(inputs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(inputs), width), dtype=torch.bool)
    for i, (ids, pos) in enumerate(zip(inputs, city_positions)):
        tokens[i, : len(ids)] = torch.tensor(ids)
        mask[i, pos] = True  # score only the city token
    return tokens, mask


class LoRAAdapter(nn.Module):
    """Wraps a GPT-2 Conv1D layer (weight is [in_features, out_features])."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        n_in, n_out = base.weight.shape
        self.A = nn.Parameter(torch.randn(n_in, rank) / math.sqrt(rank))
        self.B = nn.Parameter(torch.zeros(rank, n_out))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale


def attach_adapters(model, rank):
    adapters = []
    for block in model.transformer.h:
        adapter = LoRAAdapter(block.attn.c_attn, rank, ALPHA)
        block.attn.c_attn = adapter
        adapters.append(adapter)
    return adapters


def train(rank, lam, lr, tokens, mask, verbose=False):
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    adapters = attach_adapters(model, rank)

    # this is the whole point: A and B get different learning rates
    opt = torch.optim.AdamW(
        [
            {"params": [a.A for a in adapters], "lr": lr},
            {"params": [a.B for a in adapters], "lr": lr * lam},
        ]
    )

    losses = []
    for step in range(STEPS):
        logits = model(tokens).logits
        # predict token at position p from position p-1
        target_logits = logits[:, :-1][mask[:, 1:]]
        targets = tokens[:, 1:][mask[:, 1:]]
        loss = nn.functional.cross_entropy(target_logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if verbose and step % 20 == 0:
            print(f"    step {step:3d}  loss {loss.item():.4f}")
    return losses


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens, mask = build_data(tokenizer)
    print(f"{tokens.shape[0]} examples, {tokens.shape[1]} tokens wide, "
          f"{mask.sum().item()} scored positions\n")

    results = {}
    for lr in LEARNING_RATES:
        for lam in LAMBDAS:
            print(f"training lr={lr:g}, lambda={lam:.0f} ...")
            results[(lr, lam)] = train(RANK, lam, lr, tokens, mask)
            print(f"    final loss {results[(lr, lam)][-1]:.4f}")

    print("\n" + "=" * 60)
    print(f"rank {RANK}, {STEPS} steps. lambda 1 is vanilla LoRA, 16 is LoRA+.\n")
    print(f"{'base lr':>10} {'vanilla':>12} {'LoRA+':>12} {'LoRA+ better?':>16}")
    print("-" * 60)
    for lr in LEARNING_RATES:
        van = results[(lr, 1.0)][-1]
        plus = results[(lr, 16.0)][-1]
        verdict = f"yes, {(van - plus) / van:+.1%}" if plus < van else f"no, {(van - plus) / van:+.1%}"
        print(f"{lr:>10g} {van:>12.4f} {plus:>12.4f} {verdict:>16}")

    best_van = min(results[(lr, 1.0)][-1] for lr in LEARNING_RATES)
    best_plus = min(results[(lr, 16.0)][-1] for lr in LEARNING_RATES)
    print(f"\nbest over the lr sweep:  vanilla {best_van:.4f}   LoRA+ {best_plus:.4f}")
    print("Each method at its own best learning rate is the only fair comparison.")

    plt.figure(figsize=(7.5, 4.5))
    colors = {1e-4: "tab:blue", 3e-4: "tab:orange", 1e-3: "tab:green"}
    for (lr, lam), losses in results.items():
        plt.plot(losses, "-" if lam == 1.0 else "--", color=colors[lr],
                 label=f"lr {lr:g}, {'vanilla' if lam == 1.0 else 'LoRA+'}")
    plt.xlabel("step")
    plt.ylabel("loss on the city token")
    plt.yscale("log")
    plt.title(f"LoRA vs LoRA+ across base learning rates, GPT-2 small, rank {RANK}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("report_lora_plus.png", dpi=150, bbox_inches="tight")
    print("\nSaved report_lora_plus.png")


if __name__ == "__main__":
    main()
