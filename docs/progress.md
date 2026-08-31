# Progress

One entry per phase. Every entry names the commit range and the artifacts it
produced. Nothing is recorded here that was not actually run.

## Phase 0. Scaffold

Status: done.

- Repo structure, `pyproject.toml`, `.gitignore`, `.claude/settings.json` with
  `includeCoAuthoredBy: false`, `README.md`, `docs/references.bib`.
- Git identity verified as `Mayan10 <mayan25sharma@gmail.com>` before the first
  commit.
- Full package implemented ahead of the phase order so that every later phase
  has a place to land. Implemented does not mean validated: see the gates below.
- 110 tests, ruff clean, package installable with `pip install -e ".[dev]"`.
- Attribution check returns nothing:
  `git log --format='%an <%ae>%n%b' | grep -i -E 'claude|anthropic|co-authored'`

## Phase 1. Reproduce StarNet

Status: code complete, **not yet run**. Blocked on the traces, which cannot be
downloaded programmatically (see `docs/data.md`).

What was learned from their released code while building the loader, all of it
recorded in `docs/data.md`:

- the traces are pickled DataFrames on OneDrive, not CSVs in the repo;
- the columns are `alt`, `az`, `sat_name`, `n_candidates`, `clouds`, `humidity`,
  not the names assumed in the brief;
- the traces **do** carry latency, which makes the Casparsen crossover cheaper
  than treating it as a stretch implied;
- the weather variable is humidity, not precipitation;
- there are thirteen model inputs, not eleven;
- their loader fits its scaler on the whole trace before splitting, so their
  published numbers carry a scale leak we deliberately do not reproduce.

The gate is the RMSE/MAE table in CLAUDE.md section 6 (look-back 30, output 5)
within a few percent:

| | RMSE | MAE |
|---|---|---|
| USA | 40.33 | 29.88 |
| Canada | 41.08 | 30.84 |
| Germany | 36.48 | 27.11 |

To run:

```
python scripts/download_data.py --dataset starnet --dest data/starnet
# download the three OneDrive folders by hand, then:
python scripts/download_data.py --inspect data/starnet/usa
python scripts/reproduce_starnet.py --location all --ablations
```

Until that table is reproduced no number from any later phase means anything.
If the gap is small and stubborn, the two things to try first are
`--hidden 60` (their code default, against the paper text's 128) and their
whole-trace scaler.

## Phase 2. Reproduce BG-CFQS and expose the gap

Status: code complete, **not yet run**. Blocked on the traces.

Two gates. First the published average table (MAE 40.364, RMSE 52.480,
OverRate 0.349, MPE 11.745, P95+Err 65.834, risk pass 3/3) and the selected
quantiles (CHI 0.314, OSN 0.244, VIC 0.306). Then the conditional OverRate on
the P30 and P10 subsets, which should land in 0.65 to 0.71 and 0.83 to 0.86.
That second table is the motivation figure for the whole project.

To run: `python scripts/reproduce_bgcfqs.py --all`. The motivation figure is
written on every run.

## Phase 3. Evaluation harness

Status: done, and now exercised on real data rather than fixtures.

`eval/metrics.py`, `eval/splits.py`, `eval/runner.py`, `eval/figures.py`.
Metric definitions follow BG-CFQS exactly so the numbers would be comparable if
the datasets were.

Added since: `run_experiment` accepts a caller supplied split, which is what
leave-one-location-out needs because it spans two sources. `scripts/make_figures.py`
and `scripts/make_summary.py` render every figure and the committed markdown
tables from a saved `result.json`, recomputing nothing.

## Phase 4. Regime conditioned calibration

Status: **done and evaluated on real data.** This is the phase that turned the
project from a mechanism into a result.

Run: `results/final/wetlinks-seconds-Osnabruck-capacity`, 68,398 iperf bursts,
40,573 / 13,514 / 13,530 temporal split, leak check clean, seed 1337, MPS.
Summary table committed at `results/summary/osnabruck-capacity.md`.

**Three findings, in the order they were forced on us.**

1. **The regime axes from CLAUDE.md section 7 mostly do not work here, and one
   is worse than nothing.** The ablation at budget 0.35, P10 OverRate:

   | axes | P10 OverRate |
   |---|---|
   | level + candidates | 0.6038 |
   | level | 0.6046 |
   | global (no conditioning) | 0.6393 |
   | candidates | 0.6460 |
   | phase | 0.6711 |

   Phase alone is *worse* than not conditioning at all. Candidate count barely
   helps. What carries the gain is `level`, the mean of the observed look-back
   window, which is not one of the axes the brief named. It was added because
   the geometry axes turned out to be unrecoverable, and it is the axis that
   actually sees the low capacity regime.

2. **Every static calibration missed its budget**, including ours, because the
   test month is 10% slower than the calibration month. See
   `docs/limitations.md` section 1, rewritten around the measurement.

3. **Online per-regime recalibration fixes the budget and improves the
   conditional rates.** `calibrate/adaptive.py`. At budget 0.35:

   | method | OverRate | P30 | P10 | MAE |
   |---|---|---|---|---|
   | point forecast | 0.5048 | 0.7472 | 0.8411 | 24.40 |
   | split conformal, global | 0.4227 | 0.6689 | 0.7546 | 24.77 |
   | regime conformal, static (ours) | 0.4086 | 0.5967 | 0.6681 | 24.87 |
   | adaptive conformal, global | 0.3498 | 0.5679 | 0.6393 | 26.02 |
   | adaptive regime conformal (ours) | 0.3508 | 0.5484 | 0.6038 | 25.99 |

   Both mechanisms contribute and they compose: adaptation buys the budget,
   conditioning buys the conditional rates.

**The epsilon sweep is the strongest single result.** The online layer tracks
the budget within 0.002 at every point from 0.05 to 0.35, where static split
conformal is 0.02 to 0.07 over throughout. Worth recording that our advantage
does **not** widen as the budget tightens, which CLAUDE.md section 8 predicted
it would: the relative P10 improvement is roughly constant at 18 to 20% across
the sweep. That prediction was wrong and is reported as wrong.

**Downstream.** Admission control at 10 Mbps per session, mean dropped
sessions: 1.341 point, 1.108 global conformal, 0.915 ours, a 31.8% reduction
against the uncalibrated forecaster and 17.4% against global conformal. On the
P10 slice 3.308 to 2.385, a 27.9% reduction. Utilisation falls from 0.943 to
0.913, which is the price and is reported next to the gain.

## Phase 5. Cross location analysis for O1

Status: done in its weak form. `results/final/cross-site-holdout-Enschede`,
summary at `results/summary/cross-site-enschede.md`.

Train the forecaster on Osnabruck only, hold out Enschede entirely, calibrate
on the held out site's own earlier half. The forecaster never sees an Enschede
window.

The headline: **worst regime OverRate falls from 0.800 under global conformal
to 0.4255 under ours**, the largest conditional gain measured anywhere in this
project, on the site the model was not trained on. Global OverRate 0.3513, P10
0.6616 against the point forecast's 0.7968. So the layer is a method, not a per
terminal tuning trick.

Two honest qualifications. Two European sites 150 km apart under the same
constellation, same instrument, same months, is a weak version of the cross
location question. And the drift is smaller here because calibration and test
come from adjacent halves of the same site rather than adjacent months, which
is why even static conformal lands at 0.3597 rather than Osnabruck's 0.4227.
That is a controlled demonstration that drift, not construction, is what breaks
the static bound.

The Horizon dataset pull for the hourly cross-country analysis is **not done**.

## Phase 5b. Backbone comparison

Status: done. `results/backbones/`, summary at `results/summary/backbones.md`.

Six forecasters under one unchanged calibration layer, to separate "the layer
works" from "the layer interacts with the backbone it was built against".

**Budget control is backbone independent, strongly.** Achieved global OverRate
spans 0.3494 to 0.3512 against a 0.35 budget, across forecasters whose own
point MAE spans 24.3 to 35.2 Mbps.

**Conditional risk control is not, and the claim is narrowed accordingly.** P10
reduction spans 11.2% to 31.7%. DLinear's point forecast over-predicts on
essentially every decision in the lowest decile (P10 OverRate 0.9993), and
conditioning cannot separate regimes when the forecast fails in the same
direction everywhere. So: the budget is a property of the calibration, while
conditional control needs the forecaster to leave residual structure the regime
covariates can see.

**The StarNet ablation half reproduces.** Removing the periodical embedding
leaves MAE unchanged here (24.40 to 24.32) but costs clearly on risk (point
OverRate 0.5048 to 0.5333, P10 0.8411 to 0.8906). Liu et al. report it costing
RMSE. Not their ablation rerun: 10/5 rather than 30/5, different link. It does
suggest the embedding's contribution sits in the tail rather than the mean.

## Phase 6. Hardening and writeup

Status: partial. Tests at 146, ruff clean, figures and summary tables
generated. `ingest/live.py` is still a stub and the README architecture
diagram is still text.

## Open questions

Carried from CLAUDE.md section 12. Answered where the build answered them.

1. **What is the supplied dataset?** Answered. It is WetLinks, and the two CSVs
   handed over were a subset of a larger public release. The full release
   carries per-second iperf capacity, which is what made the project runnable
   without the StarNet traces. See `docs/supplied-dataset.md` and
   `docs/wetlinks-full.md`.
2. **Is there a GPU?** Answered by use: Apple MPS, no CUDA. The full Osnabruck
   grid, 30 epochs plus 336 calibration fits, runs in about 25 minutes.
3. **What is the submission deadline?** Still unanswered. Nothing has been
   sequenced around a date.
4. **Report, demo, or presentation?** Still unanswered. `decide/` and the live
   path are built to the level the brief specified and no further. If a live
   demo is required, `ingest/live.py` is the gap.
5. **Is the Casparsen latency crossover worth promoting?** Partly answered.
   The latency path runs (`results/summary/pilot-wetlinks-uos-rz-latency.json`)
   and the calibration layer transfers to it as an upper bound. It has not been
   rerun with the online layer, which it would benefit from.

New, raised by the results:

6. **`level` is doing the work, and it is not a satellite covariate.** The
   contribution as stated in CLAUDE.md section 2 defines regime in terms of
   phase and satellite geometry. On this data those axes are weak or dead and
   the gain comes from the observed look-back level. The contribution statement
   should be rewritten around "covariates the terminal can compute at
   prediction time", which is what it always said in general, rather than the
   specific four axes it then listed. Worth deciding deliberately rather than
   letting the code and the claim drift apart.
7. **Is the online layer inside or outside the contribution boundary?** The
   update rule is Gibbs and Candes and labelled as theirs. Running it per
   regime is new as far as the reading has gone, but that reading is not a
   literature search. Someone should check before it is claimed in a writeup.
