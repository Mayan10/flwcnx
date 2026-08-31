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

Osnabrück, 68,398 iperf bursts of measured per-second capacity, 13,530 test
decisions, temporal split, leak check clean. Risk budget 0.35. OverRate is the
fraction of decisions where the bound promised capacity the link did not
deliver; P30 and P10 are the lowest 30% and 10% of true throughput, the slices
where over-allocation actually drops sessions.

| method | OverRate | P30 | P10 | MAE (Mbps) |
|---|---|---|---|---|
| point forecast, uncalibrated | 0.5048 | 0.7472 | 0.8411 | 24.40 |
| split conformal, global | 0.4227 | 0.6689 | 0.7546 | 24.77 |
| regime conformal, static (ours) | 0.4086 | 0.5967 | 0.6681 | 24.87 |
| adaptive conformal, global | 0.3498 | 0.5679 | 0.6393 | 26.02 |
| **adaptive regime conformal (ours)** | **0.3508** | **0.5484** | **0.6038** | 25.99 |

Two mechanisms, and they compose: online adaptation buys the budget, regime
conditioning buys the conditional rates. Across the full budget sweep the
online layer tracks its target within 0.002 at every point:

| budget | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |
|---|---|---|---|---|---|---|---|
| achieved | 0.0516 | 0.1007 | 0.1508 | 0.2010 | 0.2511 | 0.3010 | 0.3508 |

Downstream, at 10 Mbps per session: mean dropped sessions falls from 1.341
(uncalibrated) to 0.915, a 31.8% reduction, at a cost of 3 points of link
utilisation (0.943 to 0.913).

The layer sits on top of any forecaster. Across six backbones the achieved
global OverRate spans 0.3494 to 0.3512 against the 0.35 budget, while those
backbones' own point MAE spans 24.3 to 35.2 Mbps. Conditional risk control is
more backbone dependent, and `results/summary/backbones.md` says where and why.

Full tables in `results/summary/`. Every number there is read back from a run's
`result.json` with its config snapshot beside it.

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
  (Gibbs and Candès, NeurIPS 2021).

**Ours:**

> A regime conditioned calibration layer that converts a point throughput
> forecast into a safe lower bound whose overestimation rate is controlled
> *within each operating regime*, where regime is defined by covariates the
> terminal can compute at prediction time; and the online instantiation of it,
> which maintains one adaptive operating point and one residual window per
> regime rather than the single global one the published rule assumes.

BG-CFQS selects one global quantile to hold the overestimation rate at the
budget. Their own results show this fails conditionally: against a budget of
0.35 they report a global OverRate of 0.349, but 0.65 to 0.71 on the lowest 30%
throughput subset and 0.83 to 0.86 on the lowest 10%. Risk is controlled on
average and lost precisely in the low capacity regime where over allocation
actually drops sessions. **That failure reproduces here on independent data**:
the uncalibrated forecaster runs at 0.505 globally and 0.841 at P10.

### Which regime axes actually work

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
