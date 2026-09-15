# Requirements 1, 5 and 6: latency spikes, availability, and cost

Run 2026-09-02 on all three StarNet locations. `scripts/run_requirements.py`,
stride 6, horizon 5, 50 epochs, seed 1337.

The brief names six industry needs and permits addressing three. The throughput
pipeline addressed three (predict degradation, detect congestion, optimize
allocation). These are the other three, on the same traces through the same
calibration layer.

---

## Requirement 1: predict latency spikes

Casparsen's period-level Good/Degraded framing, l_t = 50 ms, period boundaries
aligned to the recovered scheduling phase. The spike decision is read off a
risk-controlled **upper** bound rather than a second trained classifier, so the
operating point is set by the risk budget instead of a tuned threshold.

### The bound holds its budget where the static one does not

Budget 0.10. UnderRate is the fraction of decisions where the bound promised a
delay the link did not meet.

| location | latency MAE | degraded periods | budget | **online** | static |
|---|---|---|---|---|---|
| Chicago | 3.37 ms | 14.7% | 0.10 | **0.101** | 0.143 (+43%) |
| Osnabruck | 7.59 ms | 39.3% | 0.10 | **0.100** | 0.313 (+213%) |
| Victoria | 27.22 ms | 43.5% | 0.10 | **0.088** | 0.033 (-67%) |

This is the two-directional failure from the throughput work, reproduced on a
second target and **more severely**. Static split conformal overshoots its
latency budget by a factor of three on Osnabruck and undershoots by two thirds
on Victoria. The online layer lands within 0.012 of the budget on all three.

That the same failure appears on a different target, with the bound direction
flipped, is better evidence that it is a property of the link's
non-stationarity than the throughput result was on its own.

### Spike detection, and a structural point about static calibration

AUPRC against the base rate, which is what a coin flip scores.

| location | chance | point forecast | static conformal | **online regime** |
|---|---|---|---|---|
| Chicago | 0.046 | 0.628 (13.6x) | 0.628 (13.6x) | 0.600 (13.0x) |
| Osnabruck | 0.065 | 0.348 (5.4x) | 0.348 (5.4x) | **0.527 (8.1x)** |
| Victoria | 0.123 | 0.360 (2.9x) | 0.360 (2.9x) | **0.516 (4.2x)** |

**The point forecast and the static bound have identical AUPRC, exactly, on all
three locations. That is not a coincidence and it is not a bug.** A static
conformal bound is the point forecast plus a constant, and adding a constant
cannot change the ordering of the predictions. AUPRC depends only on the
ordering. So static calibration is mathematically incapable of improving the
ranking; it can only move the operating point along an unchanged curve.

The online per-regime bound is not a constant offset. It applies a different
correction in different regimes and at different times, so it *can* reorder the
predictions, and on two of three locations it does, by 43% and 51% in AUPRC.

This is a clean argument for the online form that does not depend on any of the
calibration results: it is the only one of the three that can change what the
detector ranks highest.

F1 for the same detectors, which is less favourable and is reported anyway:

| location | point | static | online |
|---|---|---|---|
| Chicago | 0.580 | **0.601** | 0.508 |
| Osnabruck | 0.344 | 0.349 | **0.429** |
| Victoria | 0.253 | 0.229 | **0.348** |

Online wins two of three on F1 and loses on Chicago, where its higher recall
(0.818 against 0.587) costs more precision than it buys. The budget is the dial
that trades these, and it was fixed at 0.10 for all three rather than tuned per
location.

---

## Requirements 5 and 6: availability and operational cost

Availability as delivered-over-promised session-slots, reported in nines, with
MTBF and MTTR over outage runs. Cost as SLA credits for over-allocation against
foregone revenue for under-allocation.

### Availability

| location | policy | availability | nines | outages | MTTR |
|---|---|---|---|---|---|
| Chicago | point forecast | 93.50% | 1.19 | 8,967 | 9.0 s |
| Chicago | online, budget 0.05 | **99.16%** | **2.08** | 1,494 | 5.3 s |
| Osnabruck | point forecast | 93.37% | 1.18 | 4,514 | 9.2 s |
| Osnabruck | static, budget 0.05 | **99.18%** | **2.09** | 651 | 5.1 s |
| Victoria | point forecast | 93.44% | 1.18 | 1,001 | 8.8 s |
| Victoria | static, budget 0.05 | **99.25%** | **2.13** | 146 | 5.0 s |

Calibration moves the link from roughly **one nine to two**, cuts the number of
outages by five to six times, and shortens the mean outage from about nine
seconds to about five. The uncalibrated forecaster sits at 93.4% on all three
locations, which is a strikingly consistent figure and reflects that it is
optimising for accuracy rather than for a promise.

**No policy reaches three nines.** The best is 2.13. An operator wanting 99.9%
on this link cannot get it by forecasting better; they would need to commit to a
rate low enough that the bound almost never misses, which the cost model below
prices.

### Cost, and why the ranking is not the interesting part

Under the default prices the *aggressive* policy is cheapest everywhere:
Chicago 0.473 per hour at budget 0.35 against 0.629 at 0.05.

That is a statement about the prices, not about the link. Foregone revenue
accrues on every unsold session-hour, while credits are a fraction of a small
prorated fee that only starts biting below three nines. Under commodity pricing,
over-allocating and paying the credits is rational.

The invariant quantity is the **break-even ratio**: how much more a violated
session-hour must cost than a sold one before risk control pays.

| location | break-even ratio |
|---|---|
| Chicago | **3.87x** |
| Osnabruck | **3.92x** |
| Victoria | **4.05x** |

Within 5% of each other across three independent links on two continents, which
makes this the most stable number in the project. The operational reading:

- **Consumer broadband**, where a degraded minute costs little more than the
  revenue on it, does not clear 4x. Allocate aggressively.
- **Enterprise connectivity and URLLC**, where SLA credits run 10 to 25% of a
  monthly fee and the reputational cost of a violation is larger still, clears
  4x comfortably. Risk control pays.

That is a decision rule an operator can check against their own contract without
adopting any of our price assumptions.

---

## Caveats

- The break-even model prices a violation linearly in session-hours. Real SLA
  remedies are stepped and capped, and a step function has no single crossing.
  4x is the linear approximation to the crossing, not the crossing itself.
- Availability is measured over the test split only, a few days to a month per
  location, whereas carrier SLAs are written monthly or annually. The nines here
  are not directly comparable to a contractual figure.
- The default prices are illustrative. They are carried in the config snapshot
  beside every result, and every conclusion above that depends on them says so.
- Latency at 1 Hz means fifteen samples per scheduling period, so Casparsen's
  99th-quantile criterion is effectively a maximum. `PeriodLabels.summary()`
  reports this flag on every run rather than leaving it implicit.
