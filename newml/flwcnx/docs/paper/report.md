# Predictive Bandwidth Allocation for Starlink Access Links

Project report. Mayan Sharma, Kriti Saini, Devansh Behl.

---

## 1. The problem statement, and what was built against it

The brief asked for a predictive network management system, with three of six
named issues addressed. The three chosen chain into one system rather than three
disconnected models:

1. **Predict throughput degradation** - the forecaster.
2. **Detect congestion before users are affected** - derived from the calibrated
   bound, not a separate classifier.
3. **Optimize bandwidth allocation automatically** - the decision layer.

The system is six layers, each talking only to the one below it:

```
ingest/     raw sources to a normalized frame     no ML, no features
state/      frame to feature vectors + regime id  no model
forecast/   feature vectors to point prediction   StarNet backbone
calibrate/  point prediction to safe bound        the risk layer
decide/     safe bound to allocation + alerts     no ML
eval/       harness, splits, metrics, figures
```

The design decision that makes the whole thing testable is that congestion is
**derived** rather than modelled: it is the calibrated bound sitting below the
committed allocation for a sustained window. Issue 2 costs almost nothing and
the system stays coherent.

**Status: complete and running.** 318 tests, lint clean, CI on Python 3.11 to 3.13, both reproduction gates
passed, a working demonstration, and every reported number traceable to a run
with its configuration snapshot beside it.

---

## 2. Objectives

### The four objectives

| | Objective | Status |
|---|---|---|
| O1 | Analyze performance across time periods and locations | Done. Four datasets, three continents on the StarNet side, cross-site holdout on the WetLinks side. |
| O2 | Study the influence of obstruction and satellite parameters | Done, and **the answer is negative**. See section 5.3. |
| O3 | Develop and evaluate ML models for prediction | Done. StarNet reproduced within 1.1% RMSE, six backbones compared. |
| O4 | Design a predictive framework for connectivity quality | Done. Calibration and allocation layers, evaluated across four datasets and a full budget sweep. |

### The six industry needs

The brief names six and permits addressing three. All six are now addressed,
three of them by the throughput pipeline and three added afterwards.

| | Industry need | Status | Where |
|---|---|---|---|
| 1 | Predict latency spikes | Done | `state/latency.py`, section 5.5 |
| 2 | Predict throughput degradation | Done | The forecaster. Section 4.1 |
| 3 | Detect congestion before users are affected | Done, unevenly | `decide/congestion.py`, section 5.6 |
| 4 | Optimize bandwidth allocation automatically | Done | `decide/admission.py`, section 5.2 |
| 5 | Improve service availability | Done | `decide/sla.py`, section 5.7 |
| 6 | Reduce operational costs | Done, with a caveat | `decide/sla.py`, section 5.7 |

Requirements 1, 5 and 6 were added after an audit found them unaddressed, and
they are reported with the same scepticism as the rest. Requirement 3 in
particular is the weakest of the six and section 5.6 says why.

---

## 3. Getting the data, which took most of the project

The StarNet traces were unavailable for months: all three OneDrive links in the
authors' repository had expired and neither the first author nor the PI replied.

**The workaround.** The full WetLinks release contains
`iperf_cleaned_seconds_*.csv`, per-second *measured capacity* from iperf. The
two CSVs supplied with the brief were a subset that did not include it. That
file substitutes for the traces on everything except serving-satellite identity:
1,019,109 samples at 1 Hz against BG-CFQS's 1,123,832, phase unaliased,
candidate count recoverable exactly by propagating 181 consecutive days of
Space-Track orbital elements against the known site coordinates.

The project ran to completion on that substitute before the traces arrived,
which is why there are four datasets rather than three and why the
cross-dataset disagreements in section 5 exist at all.

**When the traces did arrive**, two of the three files were mislabelled: the
folder marked `usa` contained Victoria and vice versa. Every file was identified
from its contents, not its filename. Row counts (145,053 / 613,295) and
satellite counts (3,166 / 3,956) match the paper's dataset table exactly, and
the timezones place the terminals on the right continents.

The US file holds 1,123,832 samples where the paper reports collecting
2,475,163. This is correct: it is exactly BG-CFQS's stated CHI count, and
`(1,123,832 - 45)/46 + 1 = 24,430` is exactly StarNet's published US data-point
count. It is their training set, not a partial download.

---

## 4. Reproduction gates

Nothing downstream means anything until these pass.

### 4.1 StarNet - passed

Look-back 30, output 5, their per-location step sizes, contiguous 8:2 split:

| location | RMSE | published | gap | MAE | published | gap |
|---|---|---|---|---|---|---|
| Chicago | 42.56 | 40.33 | +5.5% | 31.94 | 29.88 | +6.9% |
| Victoria | 38.38 | 41.08 | -6.6% | 29.55 | 30.84 | -4.2% |
| Osnabruck | 38.24 | 36.48 | +4.8% | 28.42 | 27.11 | +4.8% |
| **average** | **39.73** | **39.30** | **+1.1%** | **29.97** | **29.28** | **+2.4%** |

Gaps scatter in both directions, which is what an independent reimplementation
looks like. We do not reproduce their whole-trace scaler, which fits before
splitting.

### 4.2 BG-CFQS - passed, after finding the confound

Under temporal splits the conditional failure reproduced but the risk column did
not: OverRate 0.391 against their 0.349, risk pass 0/3 against 3/3, while
accuracy came out *better* than published. That combination was the clue.

Varying only the split:

| split | OverRate | P30 | P10 | risk pass |
|---|---|---|---|---|
| temporal | 0.377 | 0.750 | 0.909 | 1/3 |
| random (exchangeable) | 0.340 | 0.671 | 0.848 | 3/3 |
| published | 0.349 | 0.65 to 0.71 | 0.83 to 0.86 | 3/3 |

Every published quantity lands in range under the exchangeable split. The
reimplementation is correct and the guarantee is conditional on an assumption a
deployed terminal does not have.

---

## 5. Findings

### 5.1 The baseline has a hard floor on the achievable risk budget

The strongest result in the project, verified end to end from the primary
source. BG-CFQS's candidate quantile set is `T = [0.15, 0.40]` (their Table II).
Their Algorithm 1 collapses the search interval to a point when the budget is
tighter than the risk at `tau_min`, and the penalised fallback then returns
`tau_min` regardless of how tight the budget gets.

Measured achieved rate:

| budget | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |
|---|---|---|---|---|---|---|---|
| Chicago | 0.165 | 0.165 | 0.165 | 0.209 | 0.267 | 0.322 | 0.378 |
| Osnabruck (StarNet) | 0.127 | 0.127 | 0.127 | 0.169 | 0.213 | 0.251 | 0.295 |
| Victoria | 0.070 | 0.070 | 0.070 | 0.092 | 0.120 | 0.159 | 0.194 |
| Osnabruck (WetLinks) | 0.185 | 0.185 | 0.185 | 0.244 | 0.297 | 0.349 | 0.405 |

Identical at the three tightest budgets on every dataset, because the method
returns the same quantile. Their evaluation is at 0.35 only, where the floor
never binds.

A budget of 0.35 permits over-promising on more than a third of decisions. The
tighter budgets are the operationally interesting ones, and they are the ones
the method cannot serve.

### 5.2 Static calibration cannot hold a budget on a LEO link

It misses in whichever direction the drift points:

| dataset | budget | static achieved | online achieved |
|---|---|---|---|
| WetLinks Osnabruck | 0.35 | 0.423 (**over**) | 0.351 |
| StarNet, 3 locations | 0.35 | 0.293 (**under**) | 0.350 |

Across the full sweep the online layer tracks within 1% at every point; static
is 16 to 21% off. An overshoot drops sessions and an undershoot wastes capacity.
Neither lets an operator set a budget and get it.

At matched achieved risk the online layer is 10 to 15% better on tail risk *and*
better on accuracy at every operating point.

### 5.3 Objective O2's premise is not supported

With satellite geometry **measured** rather than reconstructed, conditioning the
calibration on it is worse than not conditioning at all:

| regime axes | P10 OverRate | vs no conditioning |
|---|---|---|
| none | 0.666 | - |
| visible satellite count | 0.676 | +1.4% |
| serving elevation | 0.685 | +2.9% |
| all measured geometry | 0.693 | +4.0% |
| 15 s scheduling phase | 0.694 | +4.2% |

StarNet reports that these covariates correlate with throughput *level*, and we
reproduce that. They do not carry *residual* structure, which is what a
calibration layer needs. The distinction between predicting the level and
predicting the uncertainty is the finding.

Robust to the learning rate: across seven values spanning two orders of
magnitude, conditioning beat no conditioning in 8 of 42 cells, seven of them one
location.

### 5.4 Congestion detection works, unevenly, and this is the weakest of the six

Requirement 3. Congestion is **derived**, not modelled: it is the calibrated
bound sitting below the committed rate for a sustained window. That design
choice is what makes it nearly free and what ties its quality to the bound's.

Adaptive regime conformal, budget 0.35:

| location | precision | recall | F1 | median warning lead |
|---|---|---|---|---|
| Chicago | 0.547 | 0.773 | 0.640 | 12 slots |
| Osnabruck | 0.545 | 0.353 | 0.429 | 138 slots |
| Victoria | 0.273 | 0.353 | 0.308 | 66 slots |

**It satisfies the literal requirement.** Every median lead is positive, so the
alarm fires 12 to 138 decision slots before the event, which is what "before
users are affected" asks for.

**It is not good.** Mean F1 across the three is 0.459, and on Victoria the
detector is wrong roughly three times out of four when it fires. An operator
paging on this would learn to ignore it. Recall on Osnabruck and Victoria is
0.353, so two thirds of congestion events pass unannounced.

The honest reading is that a derived detector inherits everything from the
bound, including its variance, and on the two links where the bound is least
well behaved the detector is close to useless. A trained congestion classifier
would very likely beat it. The brief's own framing preferred the derived form
for system coherence, and that trade was taken deliberately, but the cost of it
is in this table rather than in a footnote.

### 5.5 Latency spike prediction, and a structural point about static calibration

Requirement 1, the first the brief names. Casparsen's period-level
Good/Degraded framing with the spike decision read off a risk-controlled upper
bound rather than a second trained model. Full tables in
`results/summary/requirements-1-5-6.md`.

**The two-directional budget failure reproduces on a second target, worse.**
Budget 0.10, UnderRate achieved:

| location | latency MAE | online | static |
|---|---|---|---|
| Chicago | 3.37 ms | **0.101** | 0.143 (+43%) |
| Osnabruck | 7.59 ms | **0.100** | 0.313 (+213%) |
| Victoria | 27.22 ms | **0.088** | 0.033 (-67%) |

Static conformal overshoots by a factor of three on one link and undershoots by
two thirds on another. The online layer lands within 0.012 everywhere. That the
same failure appears with the bound direction flipped is better evidence that it
is a property of the link than the throughput result alone was.

**A structural finding.** The point forecast and the static bound have
*identical* AUPRC, exactly, on all three locations (0.628, 0.348, 0.360). This
is provable rather than coincidental: a static conformal bound is the point
forecast plus a constant, adding a constant cannot reorder predictions, and
AUPRC depends only on the ordering. **Static calibration is mathematically
incapable of improving a ranking.** The online per-regime bound is not a
constant offset, so it can reorder, and on two of three locations it improves
AUPRC by 43% and 51%. This argues for the online form without relying on any
calibration result.

### 5.6 Availability and operational cost

Requirements 5 and 6. Calibration moves the link from roughly **one nine to
two** (93.4% to 99.2% availability), cuts outage count five to six fold, and
shortens mean outage from about nine seconds to five. No policy reaches three
nines; the best is 2.13.

On cost, the honest answer is that the ranking depends on prices we guessed.
Under commodity pricing the aggressive policy is cheapest, because foregone
revenue accrues on every unsold session-hour while credits only bite below three
nines. So the model reports the invariant instead: **how much more a violated
session-hour must cost than a sold one before risk control pays.**

| location | break-even ratio |
|---|---|
| Chicago | 3.87x |
| Osnabruck | 3.92x |
| Victoria | 4.05x |

Within 5% across three independent links on two continents, which makes it the
most stable number in the project. Consumer broadband does not clear 4x and
should allocate aggressively; enterprise and URLLC contracts clear it
comfortably and should not. That is a rule an operator can check against their
own contract without adopting any of our assumptions.

### 5.7 Two negative results, reported in full

**The heterogeneity gate does not work.** We tried to predict which datasets
benefit from conditioning, using Cochran's Q on calibration-split per-regime
offsets. It fires everywhere, because with ~1,200 points per regime a 2 Mbps
difference is significant and `I^2` is scale-free: both hurdles ask whether a
difference exists, neither asks whether it is worth anything.

**And an earlier claim of ours was wrong.** We reported that offset spread
predicts the benefit monotonically across four datasets, from a quantity
computable before test time. Those figures were post-hoc, taken from offsets
applied *during* the test replay. Recomputed correctly on the calibration split
the ordering breaks. Withdrawn across the repository, with
`results/summary/gate-negative-result.md` explaining it.

---

## 6. The demonstration

`scripts/run_demo.py` replays a held-out trace through the full stack one
decision at a time: look-back window, point forecast, regime assignment, safe
bound, admitted sessions, congestion flag, then the horizon elapses and the
outcome is revealed and learned from.

On Victoria, 1,500 decisions at a 0.35 budget: realised risk **0.3187**, mean
dropped sessions **0.58**, utilisation **0.922**.

It writes an audit trail (`decisions.csv`), a summary, and a self-contained HTML
page with no network dependencies.

The property the demo exists to show is causality, and it is tested rather than
asserted: `tests/test_demo.py` overwrites every outcome after a cut point and
checks the earlier decisions are byte identical.

There is no dish. `ingest/live.py` still raises. This is replay, which is the
honest form of a live demo.

---

## 7. Contribution boundary

Checked against the literature before writing, not after.

**Not ours.** The StarNet backbone; the BG-CFQS baseline; split conformal;
adaptive conformal inference (Gibbs and Candes 2021); **running it per covariate
group, which is GCACI** (Ramalingam, Kiyani and Roth 2025) and independently
Angelopoulos et al. 2025; the parameter-free version (POGO 2026); online control
of a user-specified risk (Rolling RC, Feldman et al. 2023); deciding whether to
condition from data (Clustered Conformal Prediction, Ding et al. 2023; AFCP
2024); admission control from a safe bound (BG-CFQS eqs. 23 to 26).

**The project's calibration layer was built independently and is the naive
special case of GCACI.** For a partition, GCACI's group-membership vector is
one-hot and its update reduces to one parameter per regime. The methods claim
this project was designed around is withdrawn in full.

**Ours: five findings, all empirical.** Novelty here is novelty of knowledge
rather than of algorithm, and each of these is unreported as far as we could
establish:

1. BG-CFQS cannot serve a budget below its candidate set's low end; the achieved
   rate pins, on all four datasets, by up to 3.7x at the tightest budget.
2. Its risk guarantee is an artifact of an exchangeable split: 3/3 under random,
   1/3 under temporal.
3. Measured satellite geometry predicts throughput *level* but carries no
   *residual* structure, so it is useless for risk control. This is objective
   O2, answered in the negative.
4. Static calibration misses its budget in both directions, with the sign set by
   the drift rather than the method.
5. Two negative results on predicting when group conditioning helps, reported
   with their mechanisms.

The first is the strongest: it is a defect in a published method, derived from
its algorithm, confirmed against its parameter table, measured on four datasets,
and checkable by a reader in ten minutes.

---

## 8. Limitations

- Four datasets, three from one measurement campaign.
- No live terminal; every result is replay.
- The online layer's budget control is empirical, not a finite-sample guarantee,
  and it needs outcome feedback after each decision.
- `gamma` is a single hand-set constant that should scale with regime count;
  POGO removes the parameter entirely and is the principled fix.
- The shrinkage and empirical-Bayes literature was not searched, and it is where
  the section 5.4 negative most likely has prior work.
- The WetLinks 15-sample iperf run caps look-back plus horizon at 15, so neither
  StarNet's 30/5 nor BG-CFQS's 75/15 runs there.
- Every reported number was computed on a single machine. Continuous integration
  across three Python versions on Linux was added afterwards and immediately
  found a datetime-resolution bug that the local suite could not see. The
  affected code path was the correct one on the machine the results were
  produced on, so the numbers stand, but no result here has independent hardware
  confirmation. `docs/limitations.md` section 4b.

---

## 9. Reproducing everything

```bash
pytest -q                                       # 318 tests

python scripts/reproduce_starnet.py --location all      # gate 1
python scripts/reproduce_bgcfqs.py --all --stride 15    # gate 2
python scripts/split_sensitivity.py                     # the split finding
python scripts/gamma_sensitivity.py                     # learning-rate robustness
python -m flwcnx.eval.runner --location usa --stride 6  # the full grid
python scripts/run_demo.py --location canada            # the demonstration
```

Every run writes `result.json` with a configuration snapshot beside it. Figures
and summary tables are generated from those files and recompute nothing.
