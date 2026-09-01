# When Does Risk-Controlled Throughput Forecasting Actually Control Risk? A Reproduction Study on LEO Satellite Links

Mayan Sharma, Kriti Saini, Devansh Behl

---

## Abstract

Risk-budgeted throughput forecasting has been proposed as a way to make LEO
satellite capacity estimates safe for admission control: a point forecast is
converted into a conservative bound whose overestimation rate is held at a
user-specified budget. We reproduce the state of the art, BG-CFQS, on the three
Starlink traces its authors used and on an independent fourth dataset, and
report three findings.

First, its risk guarantee is reproducible but conditional on the calibration
split. Under a random split of the same data we recover every published
quantity, including the risk-pass column at 3 of 3 datasets. Under a contiguous
temporal split, which is what a deployed terminal faces, it holds on 1 of 3.

Second, and independent of any split, the method cannot serve a risk budget
below 0.15. Its published candidate quantile set is `T = [0.15, 0.40]`, and
when the budget is tighter than the risk at the most conservative candidate the
selection interval collapses to a point and the achieved rate pins there. On all
four datasets the achieved rate is identical at budgets of 0.05, 0.10 and 0.15,
overshooting by 1.4x to 3.7x at the tightest. The paper evaluates only at 0.35,
where the floor never binds.

Third, we evaluate the natural remedy, group-conditional online recalibration,
and find that it splits into two mechanisms with very different standing. The
online part replicates: it tracks the budget within 1% at every point from 0.05
to 0.35, where static calibration is 16 to 21% off, and it dominates static
calibration on both tail risk and accuracy at matched achieved risk. The
group-conditional part does not: on three of four datasets, conditioning on any
covariate we have, including *measured* serving-satellite geometry, is worse
than not conditioning at all. We attempted to predict which datasets benefit,
from a statistic computable on the calibration split, and failed. We report
that failure in full.

**What is new here is empirical, not algorithmic.** The candidate-set floor,
the split-conditionality of the guarantee, the failure of measured satellite
geometry to carry residual structure, and the two negative results are, to our
knowledge, unreported. The *mechanisms* we evaluate are all published, and
Section 6 says which paper each comes from. We claim the findings, not the
methods.

---

## 1. Introduction

A LEO access link is a hard thing to allocate bandwidth over. Capacity moves on
a 15-second scheduling cadence, handovers are frequent, and the achievable rate
varies by more than a factor of three within a single hour. An allocator that
acts on a point forecast will over-promise whenever the forecast is optimistic,
and over-promising drops sessions.

The natural response is to allocate against a *conservative bound* rather than
a point forecast, and to control how often that bound is wrong. BG-CFQS [Xie et
al. 2026] does this by selecting a quantile level whose overestimation rate on a
calibration set meets a user-specified budget. It reports a global
overestimation rate of 0.349 against a budget of 0.35, and a risk-pass on all
three of its datasets.

This paper asks whether that guarantee survives contact with the conditions a
deployed terminal actually operates in, and what the remedies cost. We do not
propose a method. Both mechanisms we evaluate are published, and Section 6 is
explicit about which paper each comes from.

### 1.1 What is new in this paper

Novelty in a measurement paper is novelty of *knowledge*, not of algorithm. Each
of the following is, to our knowledge, unreported, and each is a fact about
published methods or about LEO links that was not previously on record. We
searched for prior work on all five before claiming them; Section 6 gives the
full prior-art accounting, including what we found *and* what closed off.

1. **BG-CFQS cannot serve a risk budget below the low end of its candidate
   quantile set.** The achieved overestimation rate is identical at budgets of
   0.05, 0.10 and 0.15 on all four datasets, overshooting by up to 3.7x, because
   the selection interval collapses and the method returns the same quantile.
   Derived from their Algorithm 1, confirmed against their Table II, and
   measured. Their own evaluation is conducted at a single budget where the
   floor never binds, so it cannot surface. **This is the strongest result in
   the paper and it is checkable by a reader in ten minutes.** (Section 4.2)

2. **The method's risk guarantee is an artifact of an exchangeable calibration
   split.** We recover every published quantity under a random split, including
   the risk-pass column at 3/3, and 1/3 under a contiguous temporal split. That
   split conformal degrades under non-exchangeability is textbook; that *this
   method, on these traces, moves from 3/3 to 1/3* is not on record, and it is
   the difference between the method working and not working in deployment.
   (Section 4.1)

3. **Measured serving-satellite geometry predicts throughput level but not
   residual structure.** StarNet establishes that elevation, distance and
   visible-satellite count correlate with throughput, and we reproduce that.
   We show they carry no information about the forecaster's *uncertainty*:
   conditioning a calibration layer on them is worse than not conditioning at
   all, on all three of their own traces. The level/residual distinction is not
   drawn for these features anywhere we could find, and it is what determines
   whether the covariates are useful for risk control. (Section 5.3)

4. **Static calibration misses a risk budget in both directions on LEO links,
   with the sign set by the drift.** It overshoots on one dataset and
   undershoots by a fifth on another. The undershoot half is under-reported in a
   literature that emphasises under-coverage, and it is not a safe failure: it
   silently withholds capacity the operator said it would risk. (Section 5.2)

5. **A negative result on predicting when group conditioning helps.** The
   intuition that measurable heterogeneity predicts benefit is appealing and
   cheap to implement. It is wrong on our data, in two independent ways, and we
   report the failure with the mechanism. (Section 5.4)

**What we do not claim.** The two mechanisms evaluated here are both published:
group-conditional online recalibration is GCACI, and deciding whether to
condition from data is Clustered Conformal Prediction. Our implementation of the
former was written independently and is its naive special case. We include both
as baselines rather than presenting either as ours.

---

## 2. Setup

**Datasets.** Three Starlink traces released with StarNet [Liu et al. 2025],
covering Chicago (1,123,832 samples, 2024-04-26 to 05-28), Osnabruck (613,295,
2024-07-13 to 07-31) and Victoria (145,053, 2024-07-11 to 07-28), each at 1 Hz
with serving-satellite identity, elevation, distance, visible-satellite count
and co-located weather. Plus the per-second iperf release of WetLinks
[Laniewski et al. 2024], 1,019,109 samples of measured capacity at Osnabruck
over 180 days, which is an independent measurement campaign with a different
instrument and no satellite telemetry.

**Forecaster.** The StarNet backbone: a two-layer GRU sequence-to-sequence model
with a periodical embedding over the 15-second cycle and attention. We reproduce
its published accuracy within 1.1% RMSE and 2.4% MAE on average across the three
locations (Appendix A), which is the gate everything downstream rests on.

**Metrics.** Following BG-CFQS exactly, so the numbers are comparable: OverRate,
the fraction of decisions where the bound exceeded the realised capacity; MPE,
the mean positive error; P95+Err. Reported globally and on the two risk slices
the paper defines, **P30** and **P10**, the lowest 30% and 10% of realised
throughput. These are where over-allocation actually drops sessions.

**Splits.** Contiguous temporal blocks, 60/20/20, with a purge band so no time
step crosses a boundary. Every run carries a leak check.

---

## 3. What a risk budget is supposed to buy

An allocator sets a budget epsilon and expects the bound to over-promise at most
that often. The whole value proposition is that epsilon is a dial the operator
sets, not a number the method reports afterwards.

Two things can go wrong, and both do:

- the achieved rate can **exceed** the budget, which drops sessions;
- the achieved rate can fall **below** the budget, which wastes capacity the
  operator explicitly said it was willing to risk.

The second failure is under-discussed. It is not safe, it is expensive: an
allocator that spends 0.29 of a 0.35 budget is leaving utilisation on the table
and getting no risk reduction the operator asked for.

---

## 4. BG-CFQS

### 4.1 The guarantee is conditional on the split

Reproducing BG-CFQS with its published configuration (L=75, H=15, epsilon=0.35,
T=[0.15,0.40], delta=0.05, M=5, XGBoost with pinball loss) under contiguous
temporal splits, the conditional failure it reports comes out clearly, but its
risk column does not: average OverRate 0.391 against their 0.349, risk pass 0/3
against 3/3, while accuracy came out *better* than published. Better accuracy
with worse risk is not a signature of a broken reimplementation, so we looked
for the confound.

Holding everything constant and varying only the split:

| split | OverRate | P30 | P10 | risk pass |
|---|---|---|---|---|
| temporal | 0.377 | 0.750 | 0.909 | 1/3 |
| **random (exchangeable)** | **0.340** | **0.671** | **0.848** | **3/3** |
| **published** | **0.349** | **0.65 to 0.71** | **0.83 to 0.86** | **3/3** |

Under the exchangeable split every published quantity lands in range. **The
reimplementation is correct**, and the guarantee holds under the assumption it
inherits from split conformal.

It does not survive a temporal split. This is not a criticism of the paper,
which does not report the temporal case, and it is not a discovery about
conformal prediction, where non-exchangeability is textbook [Barber et al.
2023]. It is a statement about deployment: the assumption is not available to a
terminal, and the guarantee is the reason to use the method.

**The detail that matters most is in the P10 column of the random split.** At
0.848, in the arm where the method passes its budget 3/3, the conditional
failure is nearly as bad as under drift. So the tail failure is intrinsic to
selecting one global quantile and is not a symptom of non-stationarity. The two
problems are independent, and a fix for one is not a fix for the other.

### 4.2 Budgets below 0.15 are unservable

This is a structural limitation and it holds under any split.

From Algorithm 1 of the paper, verbatim:

```
4:  if R(tau_max) <= e then
5:      Set boundary interval [t-, t+] = [tau_max, tau_max]
6:  else if R(tau_min) > e then
7:      Set boundary interval [t-, t+] = [tau_min, tau_min]
...
21: Build Q_fine = LinSpace(t-, t+, M)
23: if exists tau in Q_fine such that R(tau) <= e then
24:     Select tau* = argmin A(tau) over the feasible set
25: else
26:     Select tau* = argmin A(tau) + lambda max(R(tau) - e, 0)
```

When the requested budget is tighter than the risk achievable at the most
conservative candidate `tau_min`, line 6 fires. The interval collapses to a
single point, line 21 produces `M` copies of `tau_min`, the feasible set at line
23 is empty, and line 26 returns `tau_min` however small the budget gets.

With their published `T = [0.15, 0.40]` (Table II), the achieved rate therefore
pins at `R(0.15)`. The signature is an achieved rate that is *identical* at
every budget at or below 0.15, and it appears on all four datasets:

| budget | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | overshoot at 0.05 |
|---|---|---|---|---|---|---|---|---|
| Chicago | 0.165 | 0.165 | 0.165 | 0.209 | 0.267 | 0.322 | 0.378 | 3.3x |
| Osnabruck (StarNet) | 0.127 | 0.127 | 0.127 | 0.169 | 0.213 | 0.251 | 0.295 | 2.5x |
| Victoria | 0.070 | 0.070 | 0.070 | 0.092 | 0.120 | 0.159 | 0.194 | 1.4x |
| Osnabruck (WetLinks) | 0.185 | 0.185 | 0.185 | 0.244 | 0.297 | 0.349 | 0.405 | 3.7x |

The floor value differs by dataset because it is `R(tau_min)`, the risk the most
conservative available quantile happens to achieve on that link. The flatness
does not: on every dataset the three tightest budgets return the same number,
because the method is returning the same quantile.

For comparison, online recalibration on the same traces tracks every one of
those budgets within 1%.

Their evaluation is conducted at epsilon = 0.35 (Table II, "Default risk
budget"), where the floor never binds on any of the four, so their own results
cannot expose it.

A budget of 0.35 permits over-promising on more than a third of decisions. We
know of no allocator that would accept that, which makes the tighter budgets the
interesting ones and the floor a practical obstacle rather than a curiosity.

Widening `T` fixes it, and would no longer be their method. The general lesson
is about the method class: budget-guided selection over a bounded candidate set
inherits that set's bounds as hard limits on the achievable risk.

---

## 5. The remedy, and how much of it works

### 5.1 Two mechanisms

The natural fix has two parts, and they are separable:

- **Online recalibration.** Stop treating the operating point as a constant.
  Update it from outcomes as they are observed. This is ACI [Gibbs and Candes
  2021].
- **Group conditioning.** Maintain a separate operating point per regime, where
  a regime is a bucket over covariates the terminal can compute at prediction
  time. Combined with the above this is GCACI [Ramalingam, Kiyani and Roth
  2025].

### 5.2 The online mechanism replicates

Budget tracking across the sweep, averaged over three locations:

| budget | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |
|---|---|---|---|---|---|---|---|
| online achieved | 0.050 | 0.100 | 0.150 | 0.200 | 0.250 | 0.300 | 0.350 |
| static achieved | 0.040 | 0.080 | 0.120 | 0.160 | 0.204 | 0.248 | 0.293 |
| static error | -21% | -20% | -20% | -20% | -19% | -17% | -16% |

The online layer tracks within 0.8% everywhere. Static split conformal is 16 to
21% **low**, the second failure mode from Section 3.

Note the sign against the WetLinks result, where static calibration *overshot*
(0.423 against 0.35) because the test month was slower than the calibration
month. Here the test period is easier and it undershoots. **The direction of the
miss is a property of the drift, not of the method**, which is a stronger claim
than either dataset alone supports.

At matched achieved risk, the comparison a naive table gets wrong:

| achieved risk | online P10 | static+regime P10 | gain | online MAE | static MAE |
|---|---|---|---|---|---|
| 0.050 | 0.145 | 0.170 | -14.6% | 51.54 | 52.66 |
| 0.100 | 0.256 | 0.289 | -11.3% | 39.62 | 40.64 |
| 0.150 | 0.357 | 0.398 | -10.4% | 33.67 | 34.47 |
| 0.200 | 0.438 | 0.499 | -12.1% | 29.76 | 30.53 |
| 0.250 | 0.521 | 0.589 | -11.6% | 27.23 | 28.00 |
| 0.300 | 0.598 | 0.666 | -10.1% | 25.36 | 26.39 |

Better tail risk and better accuracy at every operating point. **Methods must be
compared at matched achieved risk**: ranked by raw P10 alone, static BG-CFQS
appears best, but it is spending 0.289 of a 0.35 budget and conservatism buys
P10 directly.

### 5.3 The group-conditional mechanism mostly does not

Adaptive, budget 0.35, mean P10 over three locations. n = 3,718 in the Chicago
P10 slice, so the standard error is 0.008 and these differences are real:

| regime axes | P10 | vs no conditioning |
|---|---|---|
| **none** | **0.666** | - |
| level (observed look-back mean) | 0.673 | +1.1% |
| visible satellite count | 0.676 | +1.4% |
| serving elevation | 0.685 | +2.9% |
| all measured geometry | 0.693 | +4.0% |
| 15 s scheduling phase | 0.694 | +4.2% |
| all four axes | 0.709 | +6.3% |

Every axis is worse than not conditioning, including the *measured*
serving-satellite geometry. StarNet reports that elevation, distance and visible
count correlate with throughput *level*, and we reproduce that. They do not
carry *residual* structure, which is what a calibration layer needs, and the
distinction matters.

The result is not an artifact of the learning rate. Across seven values spanning
two orders of magnitude, conditioning beat no conditioning in 8 of 42 cells, and
seven of the eight are Victoria. On Chicago it loses at every setting.

The degradation tracks regime count rather than axis choice: 1 regime 0.666, 4
regimes 0.673, 18 regimes 0.693, 51 regimes 0.709, correlation -0.571 between
log10(outcomes per regime) and P10. Online per-regime calibration must converge
one parameter per regime from a *stream*, and 51 regimes leave ~400 outcomes
each. Static per-regime calibration, which fits once, is less affected. This is
the estimation-noise effect Ding et al. [2023] describe for class-conditional
conformal prediction, appearing here in an online setting.

### 5.4 A negative result on predicting when conditioning helps

Because conditioning helps on one dataset and hurts on three, we tried to build
a test that decides which case a trace is in, using only the calibration split.

The statistic: fit one offset per regime on the calibration split, weight each
by the standard error of an empirical quantile, and run Cochran's Q against
chi-squared, requiring both significance and a substantial `I^2`. This is the
standard heterogeneity test from meta-analysis, and doing it for conformal
calibration is Clustered Conformal Prediction [Ding et al. 2023] in binary form.

**It does not work, for two independent reasons.**

It fires everywhere. On Chicago, where conditioning costs 6.1%, the test returns
Q = 18.3 on 3 dof, p = 0.0004, `I^2` = 0.84, for an offset spread of 1.99 Mbps.
With ~1,200 calibration points per regime a 2 Mbps difference is comfortably
significant, and `I^2` is scale free so it agrees that the difference is real.
Both hurdles ask whether a difference *exists*; neither asks whether it is worth
anything.

And the statistic does not order the outcome:

| location | pre-test spread | measured effect |
|---|---|---|
| Chicago | 1.99 Mbps | +6.1% (hurts) |
| Osnabruck | **5.83 Mbps** | +0.1% (neutral) |
| Victoria | 5.17 Mbps | **-1.7% (helps)** |

Osnabruck has the largest pre-test spread and gets no benefit; Victoria has a
smaller one and gets the only benefit.

**A correction to our own earlier analysis.** We initially reported this
relationship as monotonic across four datasets. Those figures were taken from
the offsets *applied during the test replay*, a post-hoc quantity not available
before a decision is made. Computed correctly on the calibration split, the
relationship does not hold. We report this because the failure is the result:
the intuition that heterogeneity predicts benefit is appealing, cheap to
implement, and wrong on our data.

What would be needed is a **stability** test rather than a spread test. Both
Osnabruck and Victoria have real per-regime differences on the calibration
split; only Victoria's persist into the test period. That is a different
question and we did not attempt it.

---

## 6. What is and is not new here

Stated plainly because the answer is mostly "not new".

| mechanism | status | prior work |
|---|---|---|
| Per-regime conformal calibration | prior work | Mondrian conformal prediction, Vovk et al. 2003 |
| Per-regime *online* conformal calibration | prior work | GCACI, Ramalingam, Kiyani and Roth 2025; concurrently Angelopoulos et al. 2025 |
| Parameter-free version | prior work | POGO, Bharti et al. 2026 |
| Online control of a user-specified risk | prior work | Rolling RC, Feldman et al. 2023 |
| Deciding whether to condition, from data | prior work | Clustered Conformal Prediction, Ding et al. 2023; AFCP, Zhou and Sesia 2024 |
| Conformal risk control for resource allocation | prior work | Cohen et al. 2023, URLLC scheduling |
| Admission control from a safe throughput bound | prior work | BG-CFQS eqs. (23)-(26) |

Our implementation of per-regime adaptive calibration was written independently
and is the naive special case of GCACI: for a partition, GCACI's group
membership vector is one-hot and its update reduces to one parameter per regime.
We include GCACI and Rolling RC as baselines.

**What this paper contributes is the five findings in Section 1.1.** They are
new knowledge, not new machinery: a structural limitation in a published method
that its own evaluation cannot surface, the condition its guarantee depends on,
a covariate class that predicts level but not uncertainty, a two-directional
failure of static calibration, and two negatives.

We were careful about this distinction because we got it wrong first. The
project was designed around a per-regime calibration layer believed to be novel,
and that belief survived until it was checked against the literature rather than
against our own reading of two papers. The check cost a day and removed the
methods claim entirely. We record that here because the temptation in a project
like this is to check late, or to check in a way that confirms.

---

## 7. Limitations

- **Four datasets, three of them from one measurement campaign.** Any
  cross-dataset claim rests on four observations.
- **Cross-location means two European sites 150 km apart** on the WetLinks side,
  and three sites from one collection on the StarNet side.
- **The 15-sample constraint on WetLinks** caps look-back plus horizon at 15, so
  neither StarNet's 30/5 nor BG-CFQS's 75/15 runs there.
- **The online layer's budget control is empirical**, not a finite-sample
  guarantee, and it requires outcome feedback after each decision.
- **No live terminal.** Every result is replay over recorded traces.
- **We did not search the shrinkage and empirical-Bayes literature**, where the
  pooling-versus-conditioning question has its own century of work. Our negative
  in Section 5.4 should be read against that gap.

---

## Appendix A. Reproduction gate

StarNet, look-back 30, output 5, their published per-location step sizes,
contiguous 8:2 split:

| location | RMSE | published | gap | MAE | published | gap |
|---|---|---|---|---|---|---|
| Chicago | 42.56 | 40.33 | +5.5% | 31.94 | 29.88 | +6.9% |
| Victoria | 38.38 | 41.08 | -6.6% | 29.55 | 30.84 | -4.2% |
| Osnabruck | 38.24 | 36.48 | +4.8% | 28.42 | 27.11 | +4.8% |
| **average** | **39.73** | **39.30** | **+1.1%** | **29.97** | **29.28** | **+2.4%** |

Gaps scatter in both directions. We do not reproduce their whole-trace scaler,
which fits before splitting; ours fits on training windows only.

## Appendix B. Artifacts

All code, run configurations and result files are in the project repository.
Every table in this paper is generated from a saved `result.json` with its
configuration snapshot beside it. The two reproduction gates, the split
sensitivity experiment, the learning-rate sweep and the heterogeneity gate are
each a single script.
