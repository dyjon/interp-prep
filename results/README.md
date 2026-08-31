# SST-2 results — LoRA+ learning-rate ratio

Three experiments, run August 2026. They only make sense in order: each one exists because
the previous one had a hole.

| | Question | Answer |
|---|---|---|
| [01-horizon.md](01-horizon.md) | Does the optimal λ depend on training length? | **No.** λ = 1–2 wins at every checkpoint. λ = 32 diverges |
| [02-ratio-ladders.md](02-ratio-ladders.md) | Does the ratio matter, or only η_B = λ·η_A? | **Only η_B, at one width.** η_A moves 32× at fixed η_B and nothing changes |
| [03-width.md](03-width.md) | Do the optimal rates move with width? | **λ\* rises 0.2 → 2.0** across 128 → 768. η_B\* flat |

## The one-paragraph version

On SST-2 the optimal ratio does not move with the training horizon. Controlling properly —
holding η_B fixed and walking λ — shows that within a fixed width the ratio does nothing at
all; only η_B matters, and the apparent ratio effect in the first experiment was
configurations falling off the stability boundary. But across widths the optimal ratio moves
by an order of magnitude. **So λ is not a within-width control knob, but it is a
between-width scaling quantity.** Those are not in conflict: it is what an asymptotic claim
should look like.

## Known holes

- **Two of four widths are unbracketed** in the width sweep. Those rows are lower bounds.
  A 24-run extension fixes it.
- **The ladders and the width sweep use different model families** — 12-layer RoBERTa versus
  4-layer BERT. The cross-experiment comparison is not clean.
- Two seeds throughout. SST-2 is near-saturated.

## Scripts

- [`src/sst2_horizon.py`](../src/sst2_horizon.py)
- [`src/sst2_ratio_control.py`](../src/sst2_ratio_control.py)
- [`src/sst2_width.py`](../src/sst2_width.py)
