"""Does the ratio matter on its own, or only the product eta_B = lambda * eta_A?

Control for the confound in sst2_horizon.py. That sweep held eta_A fixed at 3e-4 and
varied lambda, so lambda and eta_B moved together and nothing distinguishes them. It
found the optimum at lambda = 1 to 2 and outright divergence at lambda = 32, which is
consistent with two quite different stories:

  (a) the ratio itself matters, and on this task the best one is near 1, or
  (b) only eta_B matters, and large lambda fails simply because 32 * 3e-4 = 9.6e-3
      is past the stability edge.

This distinguishes them. Walk the hyperbola lambda * eta_A = eta_B with eta_B held
fixed, so every configuration trains B at the same rate and only the ratio changes:

    (eta_A, lambda) = (3e-4, 16), (6e-4, 8), (1.2e-3, 4), (2.4e-3, 2), (4.8e-3, 1)

eta_B = 4.8e-3 throughout, the largest value in the previous sweep that did not
diverge.

  - Flat accuracy along the ladder means lambda is a reparametrisation on this task
    and eta_B is the only quantity that matters.
  - Varying accuracy means the ratio carries information beyond the product, which is
    what LoRA+ claims.

Note the lambda = 1 end sets eta_A = 4.8e-3, which is large for A. If that end
degrades, it says A has its own stability limit and the two rates cannot be collapsed
into one number. That is a result, not a failure.

Follow-up if this ladder comes out flat: repeat at eta_B = 1.2e-3, well inside the
stable region, to check the answer is not an artefact of sitting near the edge.

Everything else matches sst2_horizon.py: RoBERTa-base, query and value only,
alpha = r = 8, Init[1], 8,000 training examples, full 872-example dev set.

Colab:
    !pip install -q transformers datasets
    !python sst2_ratio_control.py
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sst2_horizon as base

# (eta_A, lambda), each with eta_A * lambda = 4.8e-3
CONFIGS = [
    (3e-4, 16.0),
    (6e-4, 8.0),
    (1.2e-3, 4.0),
    (2.4e-3, 2.0),
    (4.8e-3, 1.0),
]
SEEDS = [0, 1]
STEPS = 1000          # accuracy plateaus by ~500 in the horizon sweep
EVAL_EVERY = 100

CHECKPOINT = "sst2_ratio_results.jsonl"
DEVICE = base.DEVICE


def train_one(lr_a, lam, seed, train_ds, dev_loader):
    """Same as base.train_one but with eta_A passed in rather than read from a global."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(base.MODEL, num_labels=2)
    for p in model.parameters():
        p.requires_grad = False
    adapters = base.attach_lora(model, base.RANK, base.ALPHA)
    for p in model.classifier.parameters():
        p.requires_grad = True
    model.to(DEVICE)

    opt = torch.optim.AdamW([
        {"params": [a.A for a in adapters], "lr": lr_a},
        {"params": [a.B for a in adapters], "lr": lr_a * lam},
        {"params": list(model.classifier.parameters()), "lr": lr_a},
    ])

    loader = DataLoader(train_ds, batch_size=base.BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=base.collate)
    trace = []
    step = 0
    model.train()
    while step < STEPS:
        for batch in loader:
            if step >= STEPS:
                break
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                acc, dev_loss = base.evaluate(model, dev_loader)
                trace.append({"step": step, "acc": acc, "dev_loss": dev_loss})
                print(f"    eta_A={lr_a:.1e} lambda={lam:>4.0f} seed={seed} "
                      f"step={step:>5} acc={acc:.4f} dev_loss={dev_loss:.4f}", flush=True)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return trace


def load_done():
    if not os.path.exists(CHECKPOINT):
        return set()
    done = set()
    with open(CHECKPOINT) as f:
        for line in f:
            r = json.loads(line)
            done.add((r["lr_a"], r["lam"], r["seed"]))
    return done


def record(lr_a, lam, seed, trace):
    with open(CHECKPOINT, "a") as f:
        f.write(json.dumps({"lr_a": lr_a, "lam": lam, "seed": seed,
                            "trace": trace}) + "\n")


def analyse():
    """Report final accuracy along the ladder, with the seed spread."""
    runs = [json.loads(l) for l in open(CHECKPOINT)]
    if not runs:
        print("nothing to analyse yet")
        return

    print("\n" + "=" * 72)
    print(f"eta_B held fixed; only the ratio changes. {STEPS} steps.\n")
    print(f"{'eta_A':>10}  {'lambda':>7}  {'eta_B':>10}  {'final acc':>16}  {'best acc':>10}")
    print("-" * 72)

    finals = []
    for lr_a, lam in CONFIGS:
        rows = [r["trace"] for r in runs if r["lr_a"] == lr_a and r["lam"] == lam]
        if not rows:
            continue
        last = np.array([t[-1]["acc"] for t in rows])
        peak = np.array([max(p["acc"] for p in t) for t in rows])
        finals.append(last.mean())
        print(f"{lr_a:>10.1e}  {lam:>7.0f}  {lr_a * lam:>10.1e}  "
              f"{last.mean():>9.4f}±{last.std():.4f}  {peak.mean():>10.4f}")

    if len(finals) < 2:
        print("\nNot enough configurations finished to compare.")
        return

    spread = max(finals) - min(finals)
    noise = max(
        np.array([r["trace"][-1]["acc"] for r in runs
                  if r["lr_a"] == lr_a and r["lam"] == lam]).std()
        for lr_a, lam in CONFIGS
        if any(r["lr_a"] == lr_a and r["lam"] == lam for r in runs)
    )
    print(f"\nspread across the ladder: {spread:.4f}   worst seed sd: {noise:.4f}")
    if spread > 2 * noise:
        print("The ratio matters beyond the product: accuracy varies along a line of")
        print("constant eta_B, so lambda is not merely a reparametrisation here.")
    else:
        print("Flat along the ladder. On this task eta_B accounts for the effect and")
        print("the ratio carries no extra information, which would mean the lambda")
        print("dependence in the horizon sweep was a stability effect rather than a")
        print("statement about the ratio.")


def main():
    print(f"device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL)
    train_ds, dev_ds = base.build_data(tokenizer)
    dev_loader = DataLoader(dev_ds, batch_size=64, collate_fn=base.collate)
    print(f"train {len(train_ds)}, dev {len(dev_ds)}, {STEPS} steps, "
          f"{len(CONFIGS)} configs x {len(SEEDS)} seeds\n")

    done = load_done()
    if done:
        print(f"resuming, {len(done)} runs already complete\n")

    for lr_a, lam in CONFIGS:
        for seed in SEEDS:
            if (lr_a, lam, seed) in done:
                continue
            print(f"  eta_A={lr_a:.1e} lambda={lam:.0f} seed={seed}")
            trace = train_one(lr_a, lam, seed, train_ds, dev_loader)
            record(lr_a, lam, seed, trace)

    analyse()


if __name__ == "__main__":
    main()
