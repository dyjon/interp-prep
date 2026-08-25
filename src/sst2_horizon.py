"""Does the optimal LoRA+ ratio depend on training steps? SST-2 version.

Follow-up to lora_horizon.py, which found on a synthetic memorisation task that the
optimal lambda falls as training runs longer. This repeats the question on a real
benchmark, matching the configuration used in the LoRA+ GLUE experiments so that any
difference is attributable to the horizon rather than to the setup.

Matched to the paper's RoBERTa GLUE setup:
  - RoBERTa-base with a classification head
  - LoRA on the query and value projections only
  - alpha = r = 8
  - Init[1], meaning B starts at zero and A is random with variance Theta(1/n)

Differences from the paper, declared:
  - the training set is subsampled so that many epochs fit in a free Colab session,
    which is what the horizon question needs
  - evaluation is accuracy on the full SST-2 dev set (872 examples), held out

One run per (lambda, seed), evaluated at checkpoints throughout, so the argmin over
lambda can be read at every point in training.

Colab:
    !pip install -q transformers datasets
    !python sst2_horizon.py
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "roberta-base"
RANK = 8
ALPHA = 8            # alpha = r = 8, matching the paper's GLUE setup
LAMBDAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
SEEDS = [0, 1, 2]
LR = 3e-4            # base rate eta_A; set from the pilot, see run_pilot()
TRAIN_SUBSET = 8000
BATCH_SIZE = 32
MAX_LEN = 128
STEPS = 2500
EVAL_EVERY = 100

CHECKPOINT = "sst2_horizon_results.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LoRALinear(nn.Module):
    """LoRA on an nn.Linear layer.

    Note the transpose relative to the GPT-2 version: nn.Linear stores weight as
    [out_features, in_features] and computes x @ W.T, whereas GPT-2's Conv1D stores
    [in, out] and computes x @ W. So A must map from in_features here.

    init1: B = 0, A ~ N(0, 1/n_in). Standard LoRA, variance scaling with width.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        n_out, n_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(n_in, rank) / (n_in ** 0.5))
        self.B = nn.Parameter(torch.zeros(rank, n_out))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale


def attach_lora(model, rank, alpha):
    """Wrap the query and value projections in every attention layer."""
    adapters = []
    for layer in model.roberta.encoder.layer:
        att = layer.attention.self
        for name in ("query", "value"):
            adapter = LoRALinear(getattr(att, name), rank, alpha)
            setattr(att, name, adapter)
            adapters.append(adapter)
    return adapters


def build_data(tokenizer):
    ds = load_dataset("glue", "sst2")
    train = ds["train"].shuffle(seed=0).select(range(TRAIN_SUBSET))
    dev = ds["validation"]

    def enc(batch):
        return tokenizer(batch["sentence"], truncation=True,
                         max_length=MAX_LEN, padding="max_length")

    train = train.map(enc, batched=True)
    dev = dev.map(enc, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    train.set_format("torch", columns=cols)
    dev.set_format("torch", columns=cols)
    return train, dev


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss_sum += out.loss.item() * labels.size(0)
        correct += (out.logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / total, loss_sum / total


def train_one(lam, seed, train_ds, dev_loader):
    """Train a single (lambda, seed) configuration, evaluating at checkpoints."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    for p in model.parameters():
        p.requires_grad = False
    adapters = attach_lora(model, RANK, ALPHA)
    # the classification head is randomly initialised and must be trained
    for p in model.classifier.parameters():
        p.requires_grad = True
    model.to(DEVICE)

    opt = torch.optim.AdamW([
        {"params": [a.A for a in adapters], "lr": LR},
        {"params": [a.B for a in adapters], "lr": LR * lam},
        {"params": list(model.classifier.parameters()), "lr": LR},
    ])

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
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
                acc, dev_loss = evaluate(model, dev_loader)
                trace.append({"step": step, "acc": acc, "dev_loss": dev_loss})
                print(f"    lambda={lam:>4.0f} seed={seed} step={step:>5} "
                      f"acc={acc:.4f} dev_loss={dev_loss:.4f}", flush=True)

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
            done.add((r["lam"], r["seed"]))
    return done


def record(lam, seed, trace):
    with open(CHECKPOINT, "a") as f:
        f.write(json.dumps({"lam": lam, "seed": seed, "trace": trace}) + "\n")


def analyse():
    """Read the checkpoint file and report whether the argmin over lambda moves."""
    runs = [json.loads(l) for l in open(CHECKPOINT)]
    if not runs:
        print("nothing to analyse yet")
        return

    steps = [p["step"] for p in runs[0]["trace"]]
    acc = {}   # acc[lam] -> array (seeds, checkpoints)
    for lam in LAMBDAS:
        rows = [[p["acc"] for p in r["trace"]] for r in runs if r["lam"] == lam]
        if rows:
            acc[lam] = np.array(rows)

    present = sorted(acc)
    print("\n" + "=" * 78)
    print("Dev accuracy, mean over seeds (sd)\n")
    print("  step  " + "".join(f"{'l=' + str(int(l)):>12}" for l in present))
    print("-" * 78)
    for i, s in enumerate(steps):
        if s % 500 and s != steps[-1]:
            continue
        row = f"{s:>6}  "
        for lam in present:
            m, sd = acc[lam][:, i].mean(), acc[lam][:, i].std()
            row += f"  {m:.3f}±{sd:.3f}"
        print(row)

    print("\n" + "=" * 78)
    print("Best lambda at each checkpoint, screened against seed noise\n")
    prev = None
    for i, s in enumerate(steps):
        means = {lam: acc[lam][:, i].mean() for lam in present}
        ranked = sorted(present, key=lambda l: -means[l])
        best, second = ranked[0], ranked[1]
        margin = means[best] - means[second]
        noise = max(acc[best][:, i].std(), acc[second][:, i].std())
        flag = "" if margin > noise else "  (within noise)"
        if best != prev or s % 500 == 0:
            print(f"  step {s:>5}: lambda={best:>4.0f}  acc={means[best]:.4f}  "
                  f"margin={margin:.4f} vs sd={noise:.4f}{flag}")
        prev = best

    first = max(present, key=lambda l: acc[l][:, 0].mean())
    last = max(present, key=lambda l: acc[l][:, -1].mean())
    print()
    if first != last:
        print(f"The optimum moves: lambda={first:.0f} early, lambda={last:.0f} at the end.")
        print("Check the margins above before treating any single transition as real.")
    else:
        print(f"The optimum is stable at lambda={first:.0f} across the whole run.")
        print("On this task the horizon does not move it, which would be a clean")
        print("negative result against the synthetic finding.")


def main():
    print(f"device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train_ds, dev_ds = build_data(tokenizer)
    dev_loader = DataLoader(dev_ds, batch_size=64)
    print(f"train {len(train_ds)}, dev {len(dev_ds)}, {STEPS} steps, "
          f"eval every {EVAL_EVERY}\n")

    done = load_done()
    if done:
        print(f"resuming, {len(done)} runs already complete\n")

    for lam in LAMBDAS:
        for seed in SEEDS:
            if (lam, seed) in done:
                continue
            print(f"  lambda={lam:.0f} seed={seed}")
            trace = train_one(lam, seed, train_ds, dev_loader)
            record(lam, seed, trace)

    analyse()


if __name__ == "__main__":
    main()
