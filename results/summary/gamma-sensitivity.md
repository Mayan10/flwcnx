# Is the conditioning negative an artifact of one learning rate?

Raised unprompted by the independent novelty review (`docs/novelty-review.md`,
final section): the online layer carries a `gamma`, POGO's entire motivating
argument is that `gamma` cannot be tuned on a non-stationary stream, and if
`gamma` was tuned on calibration or test that is a validity problem independent
of every novelty question.

Two separate questions. Both are now answered.

## 1. Was gamma tuned? No.

From the git history: `gamma = 0.02` was set in commit `1d2ccf4`, the commit
that created `calibrate/adaptive.py`, with an a priori justification in the
comment. It has never been changed, swept, or overridden by any run, and no
script passes a different value. There is no tuning and therefore no validity
problem.

## 2. Would another gamma change the conclusion? No, and the detail is useful.

`scripts/gamma_sensitivity.py`, three locations x seven gammas x three
granularities, forecaster trained once per location and reused so only the
online update rate varies. Numbers are the change in P10 OverRate from
conditioning, against no conditioning at the same gamma. Negative is better.

| gamma | USA level | USA full | Germany level | Germany full | Canada level | Canada full |
|---|---|---|---|---|---|---|
| 0.002 | +4.8% | +9.3% | **-0.1%** | +3.7% | **-8.8%** | **-15.7%** |
| 0.005 | +5.1% | +8.7% | +0.0% | +7.1% | **-5.4%** | **-8.2%** |
| 0.010 | +5.2% | +8.5% | +0.2% | +8.3% | **-2.3%** | **-3.5%** |
| 0.020 | +5.0% | +7.2% | +0.7% | +10.4% | **-1.2%** | +0.3% |
| 0.050 | +7.1% | +7.8% | +2.0% | +14.2% | +5.3% | +11.5% |
| 0.100 | +7.9% | +9.3% | +3.3% | +20.3% | +6.1% | +18.9% |
| 0.200 | +12.6% | +14.7% | +4.7% | +28.0% | +9.7% | +28.0% |

Conditioning beat no conditioning in **8 of 42** cells. Seven of the eight are
Canada.

**The negative is robust.** On the US trace conditioning loses at every gamma
tested, across two orders of magnitude, by 4.8% to 14.7%. On Germany it is
within a rounding error of zero at the smallest gamma and loses everywhere
else. No choice of learning rate rescues it.

**And the positive is real but narrow.** Canada wins, by as much as 15.7%, but
only at small gamma. Our a priori default of 0.02 was very nearly the worst
useful choice there: it captures -1.2% of an available -8.8%. So the default
was not tuned in our favour; if anything it was tuned against us.

## What this adds to the heterogeneity finding

The ranking is exactly the one the gate predicts, arrived at along a completely
independent axis:

| location | per-regime offset spread | best available gain from conditioning |
|---|---|---|
| StarNet USA | 1.85 Mbps | none at any gamma |
| StarNet Germany | 5.44 Mbps | -0.1%, i.e. none |
| StarNet Canada | 11.30 Mbps | -15.7% |

Conditioning pays only where the regimes actually differ. Where they do not, no
hyperparameter setting makes it pay.

**A second effect, visible in the columns.** At fine granularity, large gamma is
catastrophic: Canada `full` goes from -15.7% at gamma 0.002 to +28.0% at 0.200.
More regimes and faster updates compound, because each regime's alpha sees only
the decisions routed to it and a large step size makes those few observations
thrash. That is the data-starvation mechanism showing up as an interaction
rather than as a main effect, which is a cleaner form of the evidence than the
correlation reported in `starnet-regime-grid.md`.

**Practical consequence.** `gamma` should scale down with regime count. The
current implementation uses one `gamma` for every partition size, which the
table shows is wrong. POGO removes the parameter entirely and is the principled
answer; scaling `gamma` by the number of active regimes would be the cheap one.
Neither is implemented yet.
