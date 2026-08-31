# Do the optimal rates move with width?

The ladders showed that at one width only η_B does anything. That is a scaling claim, so it
needs the width axis.

**LoRA+** makes two separate predictions, and they only come apart across n:

    η_A* = Θ(1/n)   A's optimal rate falls as width grows
    η_B* = Θ(1)     B's optimal rate does not move

⚠ **μA (Chen, Villar, Hayou 2026) predicts something different for this configuration.** At
Init[A] with the paper's α = 1 — which is where these runs sit, since α_lora/r = 1 — it gives
a single rate η = Θ(n^(-1/2) r^(-1/2)), with **A inert** (δ¹ → 0) and B doing the learning.
So μA predicts the *active* rate falls as n^(-1/2), a **2.45×** drop over this width range,
and makes no claim about an optimum for η_A at all.

**A sweep at fixed λ cannot test this.** Pinning the ratio forces η_A = η_B/λ, so the two
rates scale together by construction and the predictions collapse into one. Hence a 2D grid
over both rates, with the argmax read in each coordinate separately.

Widths are the BERT miniatures (Turc et al. 2019): **depth fixed at 4 layers**, hidden size
128 to 768. Rank held at 8 throughout. Note μA takes a **joint** (n, r) → ∞ limit, so a
fixed-rank width sweep is a slice through its regime rather than a direct test of it.

144 runs: 4 widths × 6 η_A × 3 η_B × 2 seeds, 1,000 steps. Kaggle, T4 ×2, 3h 25m.

Script: [`src/sst2_width.py`](../src/sst2_width.py)

---

## Final dev accuracy, mean over 2 seeds

Rows are η_B, columns η_A. **Bold is the argmax.**

### width 128 — `google/bert_uncased_L-4_H-128_A-2`

| η_B \ η_A | 3.75e-05 | 7.50e-05 | 1.50e-04 | 3.00e-04 | 6.00e-04 | 1.20e-03 |
|---|---|---|---|---|---|---|
| 3.00e-04 | .7299 | .7471 | .7529 | .7552 | .7569 | **.7592** |
| 1.20e-03 | .7414 | .7494 | .7425 | .7506 | .7552 | .7552 |
| 4.80e-03 | .7294 | .7380 | .7368 | .7460 | .7489 | .7448 |

argmax η_A = 1.20e-03, η_B = 3.00e-04, λ = 0.2, acc .7592 ±.0046 · **⚠ at grid edge**

### width 256 — `google/bert_uncased_L-4_H-256_A-4`

| η_B \ η_A | 3.75e-05 | 7.50e-05 | 1.50e-04 | 3.00e-04 | 6.00e-04 | 1.20e-03 |
|---|---|---|---|---|---|---|
| 3.00e-04 | .7575 | .7678 | .7678 | .7666 | .7701 | **.7769** |
| 1.20e-03 | .7597 | .7689 | .7626 | .7712 | .7741 | .7764 |
| 4.80e-03 | .7638 | .7643 | .7712 | .7712 | .7729 | .7632 |

argmax η_A = 1.20e-03, η_B = 3.00e-04, λ = 0.2, acc .7769 ±.0052 · **⚠ at grid edge**

### width 512 — `google/bert_uncased_L-4_H-512_A-8`

| η_B \ η_A | 3.75e-05 | 7.50e-05 | 1.50e-04 | 3.00e-04 | 6.00e-04 | 1.20e-03 |
|---|---|---|---|---|---|---|
| 3.00e-04 | .7856 | .7930 | .8016 | .8085 | **.8211** | .8154 |
| 1.20e-03 | .8056 | .7993 | .8068 | .8205 | .8194 | .8188 |
| 4.80e-03 | .7844 | .8016 | .8039 | .8188 | .8131 | .8039 |

argmax η_A = 6.00e-04, η_B = 3.00e-04, λ = 0.5, acc .8211 ±.0034 · bracketed ✅

### width 768 — `google/bert_uncased_L-4_H-768_A-12`

| η_B \ η_A | 3.75e-05 | 7.50e-05 | 1.50e-04 | 3.00e-04 | 6.00e-04 | 1.20e-03 |
|---|---|---|---|---|---|---|
| 3.00e-04 | .8268 | .8331 | .8423 | .8469 | .8486 | .8526 |
| 1.20e-03 | .8331 | .8406 | .8458 | .8549 | **.8555** | .8526 |
| 4.80e-03 | .8349 | .8400 | .8406 | .8481 | .8440 | .8257 |

argmax η_A = 6.00e-04, η_B = 1.20e-03, λ = 2.0, acc .8555 ±.0000 · bracketed ✅

---

## Optimum against width

| width | η_A* | η_B* | λ* | η_A* × n | acc | bracketed? |
|---|---|---|---|---|---|---|
| 128 | ≥1.2e-03 | 3.0e-04 | ≤0.2 | 0.154 | .7592 | **no** |
| 256 | ≥1.2e-03 | 3.0e-04 | ≤0.2 | 0.307 | .7769 | **no** |
| 512 | 6.0e-04 | 3.0e-04 | 0.5 | 0.307 | .8211 | yes |
| 768 | 6.0e-04 | 1.2e-03 | 2.0 | 0.461 | .8555 | yes |

Width spans 6.0×. η_A* spans 2.0×. η_B* spans 4.0×.

---

## What it shows

**λ\* rises 0.2 → 2.0 across a 6× width range.** That is the direction LoRA+ predicts for
the ratio, and it is what the single-width ladders could not see.

**The grid edge can only understate that.** λ = η_B/η_A, so if the true η_A* at small width
is higher than 1.2e-3, the true λ* there is *lower* than 0.2 and the trend widens. The one
quantity the experiment fails to resolve is biased against the result it found.

**η_B\* is flat within noise.** 3e-4 at three widths, 1.2e-3 at 768 — and that jump is
worth .0029 accuracy (.8526 against .8555) inside a visibly flat region. Consistent with
LoRA+'s Θ(1), **and equally consistent with μA's n^(-1/2)**, for the grid reason below.

**η_A\* falls by at least 2×, and it is not clear what it is measuring.** LoRA+'s Θ(1/n)
predicts 6× over this range and η_A* × n triples instead of staying constant, which points
away from it — but two widths are pinned at the boundary, so that column is a lower bound.

**Under μA the question is malformed.** If A is inert, η_A has no optimum; what this column
may be tracking is the stability boundary, not a peak. That would also explain why the
small-width rows climb monotonically to the grid edge rather than turning over.

**⚠ The η_B grid is spaced 4× and μA predicts a 2.45× change.** That is less than one grid
step, so **flat η_B\* is not evidence against n^(-1/2).** This experiment cannot resolve the
prediction either way.

---

## ⚠ Limits, in order of severity

1. **The η_B grid is too coarse to test the theory.** 4× spacing against a predicted 2.45×
   change. **Any n^(-1/2) effect is invisible by construction.** Fixed by 2× spacing and a
   wider width range, and this is the single change that would make every other number here
   interpretable.
2. **Two of four widths are unbracketed.** At n = 128 the η_B = 3e-4 row runs .7299 .7471
   .7529 .7552 .7569 .7592, monotone to the boundary. Those are lower bounds, not optima.
   Fixed by extending the η_A grid upward at the two small widths: 24 runs, ~30 minutes.
3. **Different model family from the ladders.** These are 4-layer BERT; the ladders were
   12-layer RoBERTa. The internal comparison across widths is clean. The cross-experiment
   comparison with `02-ratio-ladders.md` is not.
4. **η_A may have no optimum to find.** If A is inert under Init[A], that column is tracking
   a stability boundary, not a peak.
5. **Two seeds.** Several adjacent cells differ by less than the seed spread.
6. **One task**, one rank, one depth. Fixed rank against μA's joint limit.
