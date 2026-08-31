# Does the ratio matter, or only the product η_B = λ · η_A?

The horizon sweep held η_A fixed and varied λ, so the two moved together. This separates
them: **hold η_B fixed and walk λ along it**, so every configuration trains B at the same
rate and only the ratio changes.

Same setup as [`01-horizon.md`](01-horizon.md). 1,000 steps, 2 seeds.

Script: [`src/sst2_ratio_control.py`](../src/sst2_ratio_control.py)

---

## Ladder A — η_B = 4.8e-3, at the stability edge

4.8e-3 was the largest η_B in the horizon sweep that did not diverge.

| η_A | λ | η_B | final acc | best acc |
|---|---|---|---|---|
| 3.0e-04 | 16 | 4.8e-03 | **.9169** ±.0075 | .9260 |
| 6.0e-04 | 8 | 4.8e-03 | .7156 ±.2054 | .8429 |
| 1.2e-03 | 4 | 4.8e-03 | .8417 ±.0459 | .9209 |
| 2.4e-03 | 2 | 4.8e-03 | .5006 ±.0086 | .7053 |
| 4.8e-03 | 1 | 4.8e-03 | .5092 ±.0000 | .5092 |

**Spread across the ladder: 0.4163. Worst seed sd: 0.2064.**

Looks like an enormous ratio effect. It is not. **The low-λ end collapses to majority
class**, and the middle rows are unstable rather than merely worse:

- λ = 16 is the only tight row across seeds
- λ = 8's ±.2054 is not measurement spread. It is one seed surviving and one collapsing
- λ = 8 and λ = 4 both show final well below best, so they degrade *during* training
- λ = 1 sits at majority class with sd exactly .0000, both seeds pinned

Every one of those is a stability signature, not a performance ordering.

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

**η_A = 1.2e-3 appears in both ladders**, and gives .8417 at η_B = 4.8e-3 against .9157 at
η_B = 1.2e-3. So the stability boundary is a curve in (η_A, η_B) space, not two independent
ceilings. The width grid shows the same interaction at n = 768: η_A = 1.2e-3 gives .8526 at
η_B = 1.2e-3 and .8257 at 4.8e-3.

**η_A varies over a 32× range at fixed η_B and accuracy does not move.** So at this width,
η_B is the operative quantity. μA gives the mechanism: at Init[A] the update term from
moving A scales as δ¹ = Θ(r^(-1/2)) → 0, so **A is inert and B does the learning**. A ladder
flat in η_A is what that looks like. The λ dependence in the
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
