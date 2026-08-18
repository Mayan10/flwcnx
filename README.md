# flwcnx

Predictive bandwidth allocation for Starlink (LEO satellite) access links.

The system forecasts downlink throughput one horizon ahead, converts that point
forecast into a *safe lower bound* whose overestimation rate is controlled
inside each operating regime, and drives admission control and congestion
alerts from that bound.

Three of the six problems in the brief are addressed, chosen because they chain
into a single system rather than three disconnected models:

1. Predict throughput degradation (the forecaster).
2. Detect congestion before users are affected (derived from the bound, not a
   separate classifier).
3. Optimize bandwidth allocation automatically (the decision layer).

## Architecture

Five layers. Each layer only talks to the one below it.

```
ingest/     raw sources to a normalized frame     (no ML, no features)
state/      frame to feature vectors + regime id  (no model)
forecast/   feature vectors to point prediction   (StarNet backbone)
calibrate/  point prediction to safe lower bound  (the novel layer)
decide/     safe bound to allocation + alerts     (no ML)
eval/       harness, splits, metrics, figures
```

Two ingestion modes sit behind one interface. `ReplaySource` reads the
published StarNet traces and is the real path. `LiveSource` wires a terminal, a
TLE feed and a weather feed, and is a stub until a dish is available.

## Contribution boundary

Being precise about this matters more than anything else in the repo.

**Reproduced from published work, not ours:**

- the StarNet GRU seq2seq backbone with periodical embedding and attention
  (Liu et al., CoNEXT 2025);
- the 2D to 3D obstruction map projection and DTW based TLE matching for
  serving satellite identification (same paper);
- the BG-CFQS budget guided coarse to fine quantile selection baseline
  (Xie et al., 2026);
- the 15 second period segmentation and boundary isolation
  (Casparsen et al., 2026).

**Ours:**

> A regime conditioned calibration layer that converts a point throughput
> forecast into a safe lower bound whose overestimation rate is controlled
> *within each operating regime*, where regime is defined by covariates the
> terminal can compute at prediction time: 15 second phase bucket, serving
> satellite elevation and distance, and candidate satellite count.

BG-CFQS selects one global quantile to hold the overestimation rate at the
budget. Their own results show this fails conditionally: against a budget of
0.35 they report a global OverRate of 0.349, but 0.65 to 0.71 on the lowest
30% throughput subset and 0.83 to 0.86 on the lowest 10%. Risk is controlled on
average and lost precisely in the low capacity regime where over allocation
actually drops sessions.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[orbital]"   # only needed for the live TLE path
```

## Data

Nothing in `data/` is versioned.

The StarNet traces are **not in their repo**. They are pickled DataFrames
behind three OneDrive links, one per country, with no direct download URL, so
they have to be fetched by hand. This prints the links and the expected layout:

```bash
python scripts/download_data.py --dataset starnet --dest data/starnet
```

Once the files are in place, check the column mapping before trusting anything
downstream:

```bash
python scripts/download_data.py --inspect data/starnet/usa
```

The loader then verifies itself against the sample, satellite and handover
counts published in the paper. The full schema, and the several places it
differs from what the brief assumed, are in `docs/data.md`.

## Running

```bash
pytest -q                                   # unit tests, synthetic fixtures only
python scripts/reproduce_starnet.py --help  # Phase 1 gate
python scripts/reproduce_bgcfqs.py --help   # Phase 2 gate and motivation figure
python -m flwcnx.eval.runner --help         # the full experiment grid
```

## Status

Every layer is implemented, tested and lint clean. **No reproduction has been
run**, because the traces cannot be downloaded programmatically. Phases 1 and 2
are gates, and nothing downstream of them means anything until they pass.

What has been demonstrated is that the code does what it claims on synthetic
data built to contain the effect: the regime layer moves conditional OverRate
on the low throughput slices while leaving the global rate and MAE alone. That
is a working mechanism, not a result, and `docs/progress.md` says so in those
words.

No number in this repo is reported unless it was produced by a run whose config
snapshot sits next to it in `results/`.
