# Does the ratio matter, or only the product η_B = λ · η_A?

The horizon sweep held η_A fixed and varied λ, so the two moved together. This separates
them: **hold η_B fixed and walk λ along it**, so every configuration trains B at the same
rate and only the ratio changes.

Same setup as [`01-horizon.md`](01-horizon.md). 1,000 steps, 2 seeds.

Script: [`src/sst2_ratio_control.py`](../src/sst2_ratio_control.py)

---

## Ladder A — η_B = 4.8e-3, at the stability edge

4.8e-3 was the largest η_B in the horizon sweep that did not diverge.

| η_A | λ | η_B |
|---|---|---|
| 3.0e-04 | 16 | 4.8e-03 |
| 6.0e-04 | 8 | 4.8e-03 |
| 1.2e-03 | 4 | 4.8e-03 |
| 2.4e-03 | 2 | 4.8e-03 |
| 4.8e-03 | 1 | 4.8e-03 |

**Spread across the ladder: 0.4163. Worst seed sd: 0.2064.**

Looks like an enormous ratio effect. It is not. **The low-λ end collapses to 0.509**,
majority class — confirmed for (2.4e-3, λ=2) and (4.8e-3, λ=1). The spread is
configurations falling off the stability boundary, not the ratio doing work.

> ⚠ Per-cell accuracies for this ladder are in the committed Kaggle log, version #2. Pull
> them before sharing this file if the exact numbers are wanted.

---

## Ladder B — η_B = 1.2e-3, well inside the stable region

| η_A | λ | η_B | final acc | best acc |
|---|---|---|---|---|
| 3.7e-05 | 32 | 1.2e-03 | .9157 ±.0075 | .9266 |
| 7.5e-05 | 16 | 1.2e-03 | **.9237** ±.0006 | .9323 |
| 1.5e-04 | 8 | 1.2e-03 | .9232 ±.0011 | .9289 |
| 3.0e-04 | 4 | 1.2e-03 | .9226 ±.0017 | .9283 |
| 6.0e-04 | 2 | 1.2e-03 | .9209 ±.0103 | .9283 |
| 1.2e-03 | 1 | 1.2e-03 | .9157 ±.0063 | .9278 |

**Spread across the ladder: 0.0080. Worst seed sd: 0.0103. Flat.**

Nothing diverges, including λ = 1 at η_A = 1.2e-3.

---

## What it shows

**η_A varies over a 32× range at fixed η_B and accuracy does not move.** So at this width,
η_B is the operative quantity and λ is a reparametrisation of it. The λ dependence in the
horizon sweep was a stability effect: large λ at fixed η_A fails because it pushes η_B past
the edge, not because the ratio is wrong.

**This does not refute LoRA+.** It relocates the mechanism. The gains arrive through η_B,
with the ratio as the vehicle. It is arguably consistent with η_A = Θ(1/n) and η_B = Θ(1):
A's rate should be small and unimportant, B's order-one and set carefully. A ladder flat in
η_A and sensitive to η_B is that picture.

---

## The limit that decides it

**Ratio versus η_B is a scaling statement and cannot be settled at one width.** At n = 768
the two hypotheses predict the same thing. Only the width axis separates them.

That is [`03-width.md`](03-width.md).
