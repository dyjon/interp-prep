"""Do the optimal learning rates move with width, and how?

The SST-2 ladders (sst2_ratio_control.py) showed that at width 768 the ratio is not the
operative quantity: with eta_B held fixed, eta_A varies over a 32x range and accuracy
does not move. But ratio-versus-eta_B is a scaling statement and cannot be settled at a
single width. This adds the width axis.

LoRA+ makes two separate predictions, and they only come apart across n:

    eta_A* = Theta(1/n)   -- A's optimal rate should fall as width grows
    eta_B* = Theta(1)     -- B's optimal rate should not move

Note that a sweep at fixed lambda cannot test this. If lambda is pinned then
eta_A = eta_B / lambda, so eta_A* and eta_B* are forced to scale together and the two
predictions collapse into one. The grid below therefore varies the two rates
independently and reads off the argmax in each coordinate separately.

Widths come from the BERT miniatures of Turc et al. 2019, which hold depth fixed at four
layers and vary hidden size over 128, 256, 512, 768 -- a 6x range, all cheap enough to
sweep on free compute. RoBERTa-base against RoBERTa-large would give only 768 against
1024 and would confound depth with width.

Rank is held at 8 across all widths, which is the regime the theory is stated in: the
Theta(1/n) claim is about growing n at fixed rank.

Reading the result:

  - eta_B* flat across widths and eta_A* falling like 1/n
        both predictions hold; the flat ladder at 768 was the Theta(1/n) insensitivity
        showing up as a plateau, and lambda is a reparametrisation at any single width
  - eta_B* flat and eta_A* also flat
        A's rate carries no width dependence at these scales, which is a real
        discrepancy with the theory and the most interesting outcome
  - eta_B* moving with n
        eta_B is not Theta(1) here, and the single-width reading needs revisiting

The grid is coarse -- 2x spacing in eta_A over a 32x range -- so it screens for a trend
rather than locating an optimum precisely. A 6x width range predicts roughly two and a
half grid steps of movement in eta_A*, which is visible but not comfortable. Treat a
clean monotone shift as a signal and anything smaller as a reason to refine the grid.

Colab / Kaggle:
    !pip install -q transformers datasets
    !python sst2_width.py
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sst2_horizon as base

MODELS = [
    ("google/bert_uncased_L-4_H-128_A-2", 128),
    ("google/bert_uncased_L-4_H-256_A-4", 256),
    ("google/bert_uncased_L-4_H-512_A-8", 512),
    ("google/bert_uncased_L-4_H-768_A-12", 768),
]

ETA_A = [3.75e-5, 7.5e-5, 1.5e-4, 3.0e-4, 6.0e-4, 1.2e-3]   # 2x steps, 32x range
ETA_B = [3.0e-4, 1.2e-3, 4.8e-3]                            # 4x steps, brackets 1.2e-3

RANK = 8
ALPHA = 8
SEEDS = [0, 1]
STEPS = 1000
EVAL_EVERY = 100

CHECKPOINT = "sst2_width_results.jsonl"
DEVICE = base.DEVICE


def encoder_layers(model):
    """BERT and RoBERTa put the encoder under different attribute names."""
    for name in ("bert", "roberta"):
        body = getattr(model, name, None)
        if body is not None:
            return body.encoder.layer
    raise ValueError(f"unrecognised architecture: {type(model).__name__}")


def attach_lora(model, rank, alpha):
    """Wrap the query and value projections in every attention layer."""
    adapters = []
    for layer in encoder_layers(model):
        att = layer.attention.self
        for name in ("query", "value"):
            adapter = base.LoRALinear(getattr(att, name), rank, alpha)
            setattr(att, name, adapter)
            adapters.append(adapter)
    return adapters


def train_one(model_name, lr_a, lr_b, seed, train_ds, dev_loader):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    for p in model.parameters():
        p.requires_grad = False
    adapters = attach_lora(model, RANK, ALPHA)
    # the classification head is randomly initialised and must be trained; BERT's pooler
    # is pretrained and stays frozen, matching how the RoBERTa runs treated the body
    for p in model.classifier.parameters():
        p.requires_grad = True
    model.to(DEVICE)

    opt = torch.optim.AdamW([
        {"params": [a.A for a in adapters], "lr": lr_a},
        {"params": [a.B for a in adapters], "lr": lr_b},
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
            done.add((r["width"], r["lr_a"], r["lr_b"], r["seed"]))
    return done


def record(width, model_name, lr_a, lr_b, seed, trace):
    with open(CHECKPOINT, "a") as f:
        f.write(json.dumps({"width": width, "model": model_name, "lr_a": lr_a,
                            "lr_b": lr_b, "seed": seed, "trace": trace}) + "\n")


def _final(runs, width, lr_a, lr_b):
    rows = [r["trace"] for r in runs
            if r["width"] == width and r["lr_a"] == lr_a and r["lr_b"] == lr_b]
    if not rows:
        return None, None
    vals = np.array([t[-1]["acc"] for t in rows])
    return vals.mean(), vals.std()


def analyse():
    """Per-width grids, then the argmax in each coordinate as a function of width."""
    if not os.path.exists(CHECKPOINT):
        print("nothing to analyse yet")
        return
    runs = [json.loads(l) for l in open(CHECKPOINT)]
    if not runs:
        print("nothing to analyse yet")
        return

    summary = []
    for model_name, width in MODELS:
        cells = {}
        for lr_a in ETA_A:
            for lr_b in ETA_B:
                m, sd = _final(runs, width, lr_a, lr_b)
                if m is not None:
                    cells[(lr_a, lr_b)] = (m, sd)
        if not cells:
            continue

        print("\n" + "=" * 78)
        print(f"width {width}   ({model_name})   final dev accuracy, mean over seeds\n")
        print(f"{'eta_B \\ eta_A':>14}" + "".join(f"{a:>11.2e}" for a in ETA_A))
        print("-" * 78)
        for lr_b in ETA_B:
            row = f"{lr_b:>14.2e}"
            for lr_a in ETA_A:
                cell = cells.get((lr_a, lr_b))
                row += f"{cell[0]:>11.4f}" if cell else f"{'-':>11}"
            print(row)

        (best_a, best_b), (best_acc, best_sd) = max(cells.items(), key=lambda kv: kv[1][0])
        print(f"\n  argmax: eta_A={best_a:.2e}  eta_B={best_b:.2e}  "
              f"lambda={best_b / best_a:.1f}  acc={best_acc:.4f}±{best_sd:.4f}")
        summary.append((width, best_a, best_b, best_acc))

    if len(summary) < 2:
        print("\nNeed at least two widths to say anything about scaling.")
        return

    print("\n" + "=" * 78)
    print("Optimum against width\n")
    print(f"{'width':>7}{'eta_A*':>12}{'eta_B*':>12}{'lambda*':>10}"
          f"{'eta_A* x n':>13}{'acc':>9}")
    print("-" * 78)
    for width, a, b, acc in summary:
        print(f"{width:>7}{a:>12.2e}{b:>12.2e}{b / a:>10.1f}{a * width:>13.3f}{acc:>9.4f}")

    widths = [s[0] for s in summary]
    a_star = [s[1] for s in summary]
    b_star = [s[2] for s in summary]

    print("\nReading:")
    print("  eta_A* x n constant down the column  ->  eta_A* = Theta(1/n), as predicted")
    print("  eta_A* itself constant              ->  no width dependence in A's rate")
    print("  eta_B* constant                     ->  eta_B* = Theta(1), as predicted\n")

    span_a = max(a_star) / min(a_star)
    span_b = max(b_star) / min(b_star)
    span_n = max(widths) / min(widths)
    print(f"  width spans {span_n:.1f}x;  eta_A* spans {span_a:.1f}x;  "
          f"eta_B* spans {span_b:.1f}x")

    if a_star[0] > a_star[-1] and span_a >= 2:
        print("  eta_A* falls with width, which is the direction the theory predicts.")
    elif span_a < 2:
        print("  eta_A* is flat to within one grid step. At this width range that is a")
        print("  discrepancy with eta_A = Theta(1/n) worth raising rather than burying.")
    else:
        print("  eta_A* moves but not monotonically. Treat as unresolved and refine.")

    if span_b < 2:
        print("  eta_B* is flat, consistent with Theta(1).")
    else:
        print("  eta_B* moves with width, which the theory does not predict.")


def main():
    print(f"device: {DEVICE}")
    print(f"{len(MODELS)} widths x {len(ETA_A)} eta_A x {len(ETA_B)} eta_B x "
          f"{len(SEEDS)} seeds = "
          f"{len(MODELS) * len(ETA_A) * len(ETA_B) * len(SEEDS)} runs\n")

    done = load_done()
    if done:
        print(f"resuming, {len(done)} runs already complete\n")

    # seed-major so that a full grid for seed 0 exists before seed 1 starts; a run that
    # is cut short then still yields a complete, if noisy, picture
    for seed in SEEDS:
        for model_name, width in MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            train_ds, dev_ds = base.build_data(tokenizer)
            dev_loader = DataLoader(dev_ds, batch_size=64, collate_fn=base.collate)
            for lr_a in ETA_A:
                for lr_b in ETA_B:
                    if (width, lr_a, lr_b, seed) in done:
                        continue
                    trace = train_one(model_name, lr_a, lr_b, seed,
                                      train_ds, dev_loader)
                    print(f"  n={width:>4} eta_A={lr_a:.2e} eta_B={lr_b:.2e} "
                          f"seed={seed} acc={trace[-1]['acc']:.4f}", flush=True)
                    record(width, model_name, lr_a, lr_b, seed, trace)

    analyse()


if __name__ == "__main__":
    main()
