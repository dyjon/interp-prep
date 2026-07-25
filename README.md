# interp-prep

Mechanistic interpretability reproductions on GPT-2 small with TransformerLens.
Pre-MSDS work, summer 2026, toward an interpretability capstone at JHU AMS.

## Progress

- [x] **Induction-head detection** (`src/induction_heads.py`)
- [x] **IOI circuit via activation patching** (`src/ioi_patching.py`)
- [x] **Steering-vector reachability** (`src/steering_reachability.py`)
- [ ] Sparse autoencoder features on the IOI circuit
- [ ] Path patching, to show how the IOI components compose

## Results

### Induction heads

Repeated-random-token test: score each head by how much attention it puts on the
token following the earlier copy of the current token.

Top heads: **L5H1 (0.90), L6H9 (0.90), L5H5 (0.91), L7H10 (0.89), L7H2 (0.82)** —
the canonical GPT-2 small induction heads.

![induction heads](report_induction.png)

### IOI circuit

Reproduces the localisation result from Wang et al. 2022, *Interpretability in the Wild*.
Clean and corrupted prompts differ only at the subject token, which flips the correct
answer. Patching a clean activation into the corrupted run measures how much of the
clean logit difference that site restores.

    clean logit diff:     +3.691
    corrupted logit diff: -3.997

Top heads by logit difference recovered, against their known role in the circuit:

| Head | Recovered | Known role |
|---|---|---|
| L8H6  | +0.328 | S-inhibition |
| L5H5  | +0.313 | induction |
| L8H10 | +0.264 | S-inhibition |
| L9H9  | +0.245 | name mover |
| L7H9  | +0.190 | S-inhibition |
| L3H0  | +0.162 | duplicate token |
| L7H3  | +0.129 | S-inhibition |
| L10H0 | +0.124 | name mover |

All four functional classes of the published circuit show up: duplicate-token heads
early, induction heads in the middle, S-inhibition around layers 7-8, name movers at
9-10. L5H5 appears in both experiments, which is consistent with induction heads being
reused by the IOI circuit.

![IOI heads](report_ioi_heads.png)
![IOI residual stream](report_ioi_resid.png)

### Steering reachability

A small check of the claim in Khashabi et al., *Steered LLM Activations are Non-Surjective*,
that steering pushes the residual stream off the manifold of prompt-reachable states.

A difference-in-means steering vector is added to held-out natural activations at layer 6,
and we measure the distance to the nearest activation in a natural corpus. Steering
strength is expressed as a multiple of the median activation norm, since absolute
coefficients are meaningless at this scale. The norm-controlled column rescales the
steered vector back to its original length, so the comparison isn't just detecting that
steering made the vector longer.

| Steering strength | NN distance | Norm-controlled | Excess surviving control |
|---|---|---|---|
| 0.10x norm | 69.9 | 70.0 | within noise |
| 0.25x | 73.0 | 72.2 | 79% |
| 0.50x | 83.6 | 79.0 | 67% |
| 1.00x | 116.5 | 93.4 | 51% |
| 2.00x | 201.2 | 108.9 | 30% |

Baseline distance between natural activations: 69.5. Median activation norm: 96.3.

Steered activations do sit further from the natural corpus, which is the direction the
paper predicts. But the effect degrades sharply under the norm control: at the strongest
setting only 30% of the excess distance survives, so most of the apparent departure is
the steering vector making the activation longer rather than moving it somewhere
qualitatively unreachable.

![steering](report_steering.png)

## Notes and limitations

**IOI**
- Eight prompt pairs on a single template. The published work uses many more templates
  and both name orders (ABBA and BABA), so these numbers localise the circuit but do not
  establish it as carefully.
- Patching one site at a time misses redundancy. The backup name-mover heads only take
  over when the primary ones are ablated, so single-site patching understates them.
- No path patching yet, so this shows *where* the signal is, not how components compose.

**Steering**
- The real problem is that "no prompt produces this activation" quantifies over all
  possible prompts, and a finite corpus cannot settle it. Nearest-neighbour distance is a
  weak stand-in for a preimage not existing.
- 360 activations in 768 dimensions is a sparse sample. Natural activations already sit
  a median of 69.5 apart when their norm is only 96.3, so the corpus does not trace out
  anything resembling a tight manifold at this sample size.
- No check that the steering vector actually changes behaviour at these strengths. A
  geometric departure that produces no behavioural effect would mean something different
  from one that does.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/induction_heads.py
python src/ioi_patching.py
python src/steering_reachability.py
```

GPT-2 small runs on CPU. The IOI sweep is ~340 forward passes and takes a few minutes.
