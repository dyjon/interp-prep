"""Does the optimal LoRA+ ratio depend on the training horizon?

The earlier sweep found the optimal lambda at a fixed 100-step budget. Hayou et al.
select lambda on test accuracy after several epochs. Those optima differ by a factor of
2 to 4, and the horizon is one candidate explanation among several (rank and target
modules also differ).

This isolates the horizon. One run per (lambda, seed), loss recorded throughout, so the
argmin over lambda can be computed at every checkpoint. If that argmin drifts as training
proceeds, the optimal ratio is horizon-dependent and the fixed-lambda recipe carries a
caveat that is not in the paper.

Three seeds per configuration, which also addresses the missing error bars in the
earlier work.

Run: python src/lora_horizon.py
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from lora_plus import ALPHA, LoRAAdapter, build_data

RANK = 16
LR = 1e-3
SCHEME = "init1"
LAMBDAS = [1.0, 8.0, 16.0, 32.0, 64.0]
SEEDS = [0, 1, 2]
STEPS = 250
CHECKPOINTS = list(range(10, STEPS + 1, 10))


def train_trace(lam, seed, tokens, mask):
    """Train one configuration, returning the loss at every step."""
    torch.manual_seed(seed)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.train()
    for p in model.parameters():
        p.requires_grad = False

    adapters = []
    for block in model.transformer.h:
        adapter = LoRAAdapter(block.attn.c_attn, RANK, ALPHA, SCHEME)
        block.attn.c_attn = adapter
        adapters.append(adapter)

    opt = torch.optim.AdamW([
        {"params": [a.A for a in adapters], "lr": LR},
        {"params": [a.B for a in adapters], "lr": LR * lam},
    ])

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
    print(f"rank {RANK}, lr {LR:g}, {SCHEME}, {STEPS} steps, seeds {SEEDS}")
    print(f"{len(LAMBDAS) * len(SEEDS)} runs\n")

    # traces[lam] = array of shape (n_seeds, STEPS)
    traces = {}
    for lam in LAMBDAS:
        runs = []
        for seed in SEEDS:
            losses = train_trace(lam, seed, tokens, mask)
            runs.append(losses)
            print(f"lambda={lam:>5.0f}  seed={seed}  final {losses[-1]:.4f}")
        traces[lam] = np.array(runs)

    mean = {lam: traces[lam].mean(axis=0) for lam in LAMBDAS}
    std = {lam: traces[lam].std(axis=0) for lam in LAMBDAS}

    print("\n" + "=" * 74)
    print("Mean loss (std over seeds) at each checkpoint\n")
    header = "  step  " + "".join(f"{'l=' + str(int(l)):>13}" for l in LAMBDAS)
    print(header)
    print("-" * 74)
    for step in [10, 25, 50, 100, 150, 200, 250]:
        if step > STEPS:
            continue
        i = step - 1
        row = f"{step:>6}  "
        for lam in LAMBDAS:
            row += f"{mean[lam][i]:>8.3f}±{std[lam][i]:<4.2f}"
        print(row)

    print("\n" + "=" * 74)
    print("Which lambda is best at each horizon?\n")
    print(f"{'step':>6} {'best lambda':>13} {'mean loss':>12} {'margin over 2nd':>18}")
    print("-" * 74)
    drift = []
    for step in CHECKPOINTS:
        i = step - 1
        ranked = sorted(LAMBDAS, key=lambda l: mean[l][i])
        best, second = ranked[0], ranked[1]
        margin = mean[second][i] - mean[best][i]
        drift.append((step, best))
        if step in (10, 25, 50, 100, 150, 200, 250):
            print(f"{step:>6} {best:>13.0f} {mean[best][i]:>12.4f} {margin:>18.4f}")

    # A change in argmin only means something if the margin exceeds the seed noise.
    # Without that check every run looks like it "drifts", because near convergence the
    # settings are indistinguishable and the argmin flips on noise.
    print()
    print("Argmin changes, screened against seed noise:")
    argmins = [b for _, b in drift]
    resolved = []
    for i in range(1, len(argmins)):
        if argmins[i] == argmins[i - 1]:
            continue
        step = CHECKPOINTS[i]
        j = step - 1
        best = argmins[i]
        second = sorted(LAMBDAS, key=lambda l: mean[l][j])[1]
        margin = mean[second][j] - mean[best][j]
        noise = max(std[best][j], std[second][j])
        ok = margin > noise
        resolved.append(ok)
        print(f"  step {step:>4}: {argmins[i-1]:.0f} -> {best:.0f}   "
              f"margin {margin:.4f} vs seed sd {noise:.4f}   "
              f"{'RESOLVED' if ok else 'within noise, ignore'}")

    if not resolved:
        print(f"  none. The optimal lambda is stable at {argmins[0]:.0f} across the run.")
    elif any(resolved):
        print("\n  At least one transition exceeds the seed spread, so the optimum does")
        print("  move with the horizon. Transitions marked as within noise should not")
        print("  be reported as findings.")
    else:
        print("\n  Every transition sits inside the seed spread. Treat the optimum as")
        print("  stable and look to rank or target modules to explain the gap instead.")

    # is the seed spread small enough for the ranking to mean anything?
    print("\nSeed spread at the final step:")
    for lam in LAMBDAS:
        m, s = mean[lam][-1], std[lam][-1]
        print(f"  lambda={lam:>5.0f}  {m:.4f} ± {s:.4f}  ({s / m:.1%} relative)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for lam in LAMBDAS:
        axes[0].plot(range(1, STEPS + 1), mean[lam], label=f"lambda {lam:.0f}")
        axes[0].fill_between(range(1, STEPS + 1), mean[lam] - std[lam], mean[lam] + std[lam], alpha=0.15)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss on the city token")
    axes[0].set_title(f"Loss trajectories, {len(SEEDS)} seeds, shaded ±1 sd")
    axes[0].legend(fontsize=8)

    axes[1].step(CHECKPOINTS, [b for _, b in drift], where="post")
    axes[1].set_yscale("log", base=2)
    axes[1].set_yticks(LAMBDAS)
    axes[1].set_yticklabels([f"{l:.0f}" for l in LAMBDAS])
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("best lambda so far")
    axes[1].set_title("Does the optimum move with the horizon?")

    fig.suptitle(f"LoRA+ optimal ratio against training horizon, GPT-2 small, rank {RANK}")
    fig.tight_layout()
    fig.savefig("report_horizon.png", dpi=150, bbox_inches="tight")
    print("\nSaved report_horizon.png")


if __name__ == "__main__":
    main()
