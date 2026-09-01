# flwcnx

Predictive bandwidth allocation for Starlink (LEO satellite) access links.

The system forecasts downlink throughput one horizon ahead, converts that point
forecast into a *safe lower bound* whose overestimation rate is held at a stated
risk budget inside each operating regime, and drives admission control and
congestion alerts from that bound.

Three of the six problems in the brief are addressed, chosen because they chain
into a single system rather than three disconnected models:

1. Predict throughput degradation (the forecaster).
2. Detect congestion before users are affected (derived from the bound, not a
   separate classifier).
3. Optimize bandwidth allocation automatically (the decision layer).

## The result

Two datasets, and **they disagree about half of it.** Both are reported.

### What replicates: online recalibration

Static split conformal cannot hold a stated risk budget on a LEO link, because
the link is not stationary across a temporal split. It misses in whichever
direction the drift points:

| dataset | budget | static achieved | online achieved |
|---|---|---|---|
| WetLinks Osnabruck | 0.35 | 0.423 (**over**) | 0.351 |
| StarNet, 3 locations | 0.35 | 0.293 (**under**) | 0.350 |

Across the full sweep from 0.05 to 0.35 the online layer tracks its budget
within 1% at every point; static is 16 to 21% off on StarNet and 20% off on
WetLinks. An overshoot drops sessions and an undershoot wastes capacity, and in
neither case can an operator set a budget and get it.

At **matched achieved risk** on the StarNet traces, the online layer is 10 to
15% better on severe-risk P10 and simultaneously better on MAE at every
operating point.

### What does not replicate: regime conditioning

On WetLinks, conditioning on the observed look-back level cut P10 OverRate from
0.6393 to 0.6038, a 5.5% gain. On the StarNet traces, with satellite geometry
**measured** rather than reconstructed, every axis is *worse* than no
conditioning at all:

| axes | P10 | vs none |
|---|---|---|
| none | 0.6663 | - |
| level | 0.6733 | +1.1% |
| candidates | 0.6755 | +1.4% |
| measured geometry | 0.6931 | +4.0% |
| 15 s phase | 0.6944 | +4.2% |
| all four brief axes | 0.7085 | +6.3% |

The cause is measurable and it is not data volume. It is that on most of these
traces the regimes do not differ. The spread of the fitted per-regime offset
predicts the sign of the effect on all four datasets:

| dataset | offset spread | effect on P10 |
|---|---|---|
| StarNet USA | 1.85 Mbps | +6.1% (hurts) |
| StarNet Germany | 5.44 Mbps | +0.1% |
| StarNet Canada | 11.30 Mbps | -1.7% (helps) |
| WetLinks | 9.28 Mbps | -5.6% (helps) |

On the US trace the four level buckets want offsets within 1.85 Mbps of each
other, so conditioning adds estimation noise and nothing else. A second, weaker
effect explains the gradient *within* StarNet, where finer partitions do worse:
splitting 20,714 decisions across 51 regimes leaves ~400 each, and the
correlation between log10(outcomes per regime) and P10 is -0.571.

**Objective O2's premise is not supported.** Measured serving-satellite
elevation, distance and candidate count did not improve conditional risk
control at any of the three locations.

Full tables in `results/summary/`. Every number is read back from a run's
`result.json` with its config snapshot beside it.

### Reproduction gates, both passed

| gate | result |
|---|---|
| StarNet (Phase 1) | average RMSE 39.73 vs 39.30 published (+1.1%), MAE 29.97 vs 29.28 (+2.4%) |
| BG-CFQS (Phase 2) | under an exchangeable split: OverRate 0.340 vs 0.349, P30 0.671 (pub 0.65-0.71), P10 0.848 (pub 0.83-0.86), risk pass 3/3 |

Phase 2 passed only after `scripts/split_sensitivity.py` identified the split
as the confound. BG-CFQS's guarantee holds under the exchangeability it is
derived from and does not survive a temporal split. Their paper does not claim
otherwise. The detail that matters: **P10 is 0.848 in the arm where the method
passes its budget 3/3**, so the conditional failure is intrinsic to global
quantile selection rather than a symptom of drift.

## Architecture

Six layers. Each layer only talks to the one below it.

```
ingest/     raw sources to a normalized frame     (no ML, no features)
state/      frame to feature vectors + regime id  (no model)
forecast/   feature vectors to point prediction   (StarNet backbone)
calibrate/  point prediction to safe lower bound  (the novel layer)
decide/     safe bound to allocation + alerts     (no ML)
eval/       harness, splits, metrics, figures
```

Two ingestion modes sit behind one interface. `ReplaySource` and the WetLinks
sources are the real path. `LiveSource` wires a terminal, a TLE feed and a
weather feed, and is a stub until a dish is available.

## Contribution boundary

Being precise about this matters more than anything else in the repo.

**Reproduced from published work, not ours:**

- the StarNet GRU seq2seq backbone with periodical embedding and attention
  (Liu et al., CoNEXT 2025);
- the 2D to 3D obstruction map projection and DTW based TLE matching for
  serving satellite identification (same paper);
- the BG-CFQS budget guided coarse to fine quantile selection baseline
  (Xie et al., 2026);
- the 15 second period segmentation and phase recovery (Casparsen et al., 2026);
- one-sided split conformal prediction (Vovk, Gammerman and Shafer);
- the adaptive conformal update rule `alpha <- alpha + gamma (eps - err)`
  (Gibbs and Candès, NeurIPS 2021);
- **running that update per covariate group**, which is GCACI (Ramalingam et
  al. 2025, arXiv:2502.10947), and which POGO (arXiv:2606.00419) improves on by
  removing the learning rate. `calibrate/adaptive.py` is the naive special case
  of GCACI and was built before this was known. See `docs/novelty-assessment.md`.

**Ours:** measurement, not method. A literature check on 2026-09-01
(`docs/novelty-assessment.md`) found that the calibration layer this project was
designed around is already published as GCACI, so the methods claim is withdrawn
in full. What the work contributes is evidence:

1. The first evaluation of risk-controlled capacity forecasting on LEO access
   links, across four datasets and two independent measurement campaigns.
2. **BG-CFQS's risk guarantee is conditional on exchangeability**: 3/3 within
   budget under a random split, 1/3 under a temporal one.
3. **BG-CFQS cannot serve a budget below 0.15.** Their candidate set is
   T = [0.15, 0.40], so achieved OverRate pins at 0.1854 for every tighter
   budget, a 3.7x overshoot at 0.05. Their paper reports only 0.35.
4. **Static calibration misses its budget in both directions**, over on
   WetLinks and under on StarNet, so the sign is a property of the drift rather
   than the method.
5. **Group conditioning helps only when the groups differ**, and the spread of
   the fitted per-regime offset predicts the sign across four datasets
   monotonically, from a quantity computable before any test decision.
6. **The satellite covariates do not carry the signal** when measured rather
   than reconstructed. Objective O2's premise is not supported.
7. **Casparsen's 15 s scheduling offset recovered independently** on three
   continents from a different signal at 1/500th of their sampling rate.

BG-CFQS selects one global quantile to hold the overestimation rate at the
budget. Their own results show this fails conditionally: against a budget of
0.35 they report a global OverRate of 0.349, but 0.65 to 0.71 on the lowest 30%
throughput subset and 0.83 to 0.86 on the lowest 10%. Risk is controlled on
average and lost precisely in the low capacity regime where over allocation
actually drops sessions. **That failure reproduces here on independent data**:
the uncalibrated forecaster runs at 0.505 globally and 0.841 at P10.

### Which regime axes actually work (WetLinks; see above for StarNet)

Reported plainly because the answer contradicts what the brief expected.
Ablation at budget 0.35, P10 OverRate:

| axes | P10 OverRate |
|---|---|
| level + candidates | 0.6038 |
| level | 0.6046 |
| global, no conditioning | 0.6393 |
| candidate count | 0.6460 |
| 15 s phase | 0.6711 |

Phase alone conditions *worse* than not conditioning. Candidate count barely
helps. The gain comes from `level`, the mean of the observed look-back window.
Serving satellite elevation and distance, which the brief expected to carry the
signal, are not recoverable from any public dataset without obstruction maps,
and the obvious proxy is dead: with ~38 satellites above the 25 degree service
floor the highest one is near zenith almost always (std 3.9 degrees,
correlation with throughput -0.02). See `docs/limitations.md` section 3b.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[orbital]"   # SGP4, for the geometry reconstruction
```

## Data

Nothing in `data/` is versioned.

**The StarNet traces are unavailable.** All three OneDrive links in their repo
have expired and the authors did not respond. `docs/data-access.md` records
every route that was checked and the workaround that was taken instead: the
full WetLinks release (https://github.com/sys-uos/WetLinks) supplies 1.02M
per-second *measured capacity* samples, which substitutes for the traces on
everything except serving satellite identity.

```bash
python scripts/download_data.py --dataset starnet --dest data/starnet
python scripts/download_data.py --inspect data/supplied
python scripts/fetch_elements.py            # Space-Track, needs .env credentials
```

## Running

```bash
pytest -q                                   # 146 tests, synthetic fixtures only

# the headline run: full budget sweep and regime ablation, ~25 min on MPS
python scripts/run_wetlinks.py --release seconds --site Osnabruck --geometry \
    --epochs 30 --stride 1 --output results/final

# cross site: train on one dish, hold out the other entirely
python scripts/run_cross_site.py --held-out Enschede --geometry --epochs 30

# figures and the committed markdown tables, from a saved run
python scripts/make_figures.py results/final/wetlinks-seconds-Osnabruck-capacity
python scripts/make_summary.py results/final/wetlinks-seconds-Osnabruck-capacity \
    --out results/summary/osnabruck-capacity.md
```

## Status and what is not done

- **The reproduction gates were never run.** StarNet's RMSE/MAE table and
  BG-CFQS's average table both need the traces. Phases 1 and 2 remain open, and
  **no number in this repo is comparable to a published one**. The 15 sample
  iperf run also caps look-back plus horizon at 15, so StarNet's 30/5 and
  BG-CFQS's 75/15 cannot be run here regardless.
- Cross location means two European sites 150 km apart measured by the same
  instrument, not three continents.
- Satellite geometry is *reconstructed* from propagated orbital elements, never
  measured. Every row carries `geometry_source = "reconstructed"`.
- The online layer's budget control is empirical, not a finite sample
  guarantee, and it requires outcome feedback after each decision.
- `LiveSource.connect` still raises. No live terminal.

`docs/limitations.md` is the long version and is written to be read before the
results, not after.
