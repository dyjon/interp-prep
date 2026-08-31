# Does the optimal LoRA+ ratio depend on the training horizon?

**SST-2, RoBERTa-base.** Query and value projections only, α = r = 8, Init[1] (B = 0, A
random with variance 1/n). η_A fixed at 3e-4, λ swept. 8,000 training examples subsampled
from SST-2, evaluated on the full 872-example dev set.

12 runs: λ ∈ {1, 2, 4, 8, 16, 32} × 2 seeds, 2,500 steps, evaluated every 100.

Script: [`src/sst2_horizon.py`](../src/sst2_horizon.py)

---

## Dev accuracy, mean over 2 seeds (± sd)

| step | λ=1 | λ=2 | λ=4 | λ=8 | λ=16 | λ=32 |
|---|---|---|---|---|---|---|
| 500 | .927 ±.002 | .927 ±.002 | .923 ±.005 | **.928** ±.000 | .920 ±.010 | .509 ±.000 |
| 1000 | **.924** ±.002 | .921 ±.000 | .923 ±.002 | .924 ±.001 | .917 ±.007 | .509 ±.000 |
| 1500 | .928 ±.004 | **.929** ±.006 | .924 ±.001 | .919 ±.001 | .916 ±.001 | .509 ±.000 |
| 2000 | **.931** ±.000 | .925 ±.006 | .920 ±.001 | .919 ±.006 | .910 ±.006 | .509 ±.000 |
| 2500 | .928 ±.002 | **.932** ±.003 | .926 ±.002 | .921 ±.004 | .917 ±.000 | .509 ±.000 |

---

## What it shows

**The optimum does not trend.** It bounces among λ = 1, 2, 4 and 8 over the first thousand
steps, then sits at 1 or 2 from step 1000 onward. Of the sixteen checkpoints listed by the
analysis, thirteen have a margin smaller than the seed spread.

**There is no drift from high λ to low λ** of the kind the synthetic task showed, which went
32 → 16 → 8 across 200 steps. That absence is the result.

**λ = 32 diverges.** 0.509 is the majority class on SST-2 dev. Both seeds, every
checkpoint, no recovery. η_B = 32 × 3e-4 = 9.6e-3 is past the stability edge; 16 × 3e-4 =
4.8e-3 is not.

**Weak monotone decline in λ among the survivors**, growing with training. At step 2500:
.932, .926, .921, .917 across λ = 2, 4, 8, 16. Each adjacent pair is inside noise; the
ordering is consistent across checkpoints and both seeds.

---

## The confound in this design

**η_A is fixed at 3e-4 for every λ**, so λ and η_B move together and nothing here separates
them. Two readings fit equally well: the ratio matters and the best one is near 1, or only
η_B matters and large λ fails because 9.6e-3 is past the edge.

That is what [`02-ratio-ladders.md`](02-ratio-ladders.md) was built to test.

---

## Limits

- One base rate, never tuned
- Two seeds
- SST-2 is near-saturated: RoBERTa-base full finetuning is ~94.8, this peaks at 93.2, so
  the headroom is about the size of the effect being looked for
- 8,000 of 67,349 training examples, so 2,500 steps is ten epochs rather than one pass.
  The horizon axis is entangled with repetition
- One model, one task, one width
