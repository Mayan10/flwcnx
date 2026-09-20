# Limitations

Written while building, not retrofitted at the end. Anything here that turns
out to be wrong should be corrected in place rather than quietly dropped.

## 1. Exchangeability breaks under a temporal split

**Status: measured, and mitigated. This section was rewritten on 2026-08-31
after the mitigation was built. The original text predicted the effect and
listed the fix as unimplemented; both halves turned out right.**

Split conformal guarantees `P(bound > y) <= epsilon` only if the calibration
and test residuals are exchangeable. Temporal splits deliberately violate this,
and on the Osnabruck per-second release the violation is large and measurable:

    train  2023-09-14 to 2024-01-05   mean 208.6 Mbps
    calib  2024-01-05 to 2024-02-06   mean 219.0 Mbps
    test   2024-02-06 to 2024-03-12   mean 197.2 Mbps

The test month is 10% slower than the month the operating point was fit on. The
cost, against a 0.35 budget on 13,530 test decisions:

| method | global OverRate | within budget |
|---|---|---|
| point forecast | 0.5048 | no |
| split conformal, global | 0.4227 | no |
| regime conformal, static (ours) | 0.4086 | no |
| adaptive conformal, global | 0.3498 | yes |
| adaptive regime conformal (ours) | 0.3508 | to within 0.001 |

Every static method missed, including ours. This is not an implementation
defect: `tests/test_calibrate.py` confirms the bound hits its budget to within
0.02 on exchangeable data at three different budgets. The data is the problem.

**The mitigation.** `calibrate/adaptive.py` keeps one alpha and one rolling
residual window per regime and updates both from outcomes as they become
observable. The update rule is Gibbs and Candes 2021 and is theirs; running it
per regime is ours. Across the full budget sweep it tracks the target within
0.002 at every point:

| budget | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |
|---|---|---|---|---|---|---|---|
| achieved | 0.0516 | 0.1007 | 0.1508 | 0.2010 | 0.2511 | 0.3010 | 0.3508 |

What remains true, and must stay in view:

- **"Within budget" is still empirical, never a guarantee.** Adaptive conformal
  has a regret bound on the long-run rate, not a finite-sample coverage
  guarantee. Every risk-pass column in this repo is a measurement on a test
  block, and none of them is a theorem.
- **It needs feedback.** The online layer consumes realised outcomes. A
  deployment that cannot measure what it actually got after each decision
  cannot run this layer, and falls back to the static version and its drift.
- **It costs accuracy and utilisation.** Against the static global conformal
  bound it is the closest like-for-like comparison to make: MAE 24.77 to 26.02
  Mbps, about 5%, and link utilisation 0.9294 to 0.9124, 1.7 points. Against the
  uncalibrated forecaster the utilisation cost is 3.0 points (0.9431 to 0.9131).
  Quote whichever baseline is meant and say which; the two differ by nearly a
  factor of two.
- **Conditional risk on the P30 and P10 slices is improved, not solved.** Those
  slices are defined by the *true* throughput, which is not observable at
  prediction time. No amount of conditioning on observable covariates can
  equalise risk across a partition defined by the answer. P10 OverRate falls
  from 0.841 to 0.604; it does not reach 0.35 and could not.
- Horizon's window-length asymmetry (latency best at two months, throughput
  improving out to eleven) is independent evidence that this drift is real and
  differs by signal.

## 2. The 1 Hz ceiling on the Casparsen work

The traces carry latency, so the period-level Good/Degraded framing is usable.
The boundary analysis is not. At 1 Hz a 15 second period holds fifteen samples,
so "at least 99% of packets meet the threshold" degenerates to "all fifteen
do", and the 140 ms opening and 75 ms closing regions cannot be resolved at
all. `state.phase.boundary_mask` raises rather than pretending otherwise, and
none of the 74 ms excess figures may be claimed on this data.

## 3. Regime estimation noise

Per-regime operating points are estimated from finite calibration samples, so
each carries its own sampling error. On the synthetic check the worst-regime
OverRate improves but does not reach the budget, and that residual gap is
estimation noise rather than a modelling failure. The `min_samples` guard
trades this against coverage: raising it shrinks the noise and pushes mass up
to coarser levels, which is the whole point of reporting mass-by-level next to
every result.

## 3b. The satellite geometry is reconstructed, and two axes of it are dead

Added 2026-08-31, after the geometry was actually computed.

WetLinks records no serving satellite. Candidate count is recovered exactly by
propagating 181 consecutive days of Space-Track elements against the known site
coordinates: mean 37.6, range 4 to 59, against StarNet's reported 15 to 45.

Serving elevation and distance are **not recoverable**, and the obvious proxy
fails for a structural reason. With roughly 38 satellites above the 25 degree
service floor, the highest one is near zenith almost always: its elevation has
standard deviation 3.9 degrees around 81, and its correlation with throughput
is -0.02. Conditioning on it is conditioning on noise.

The consequence for the ablation is that objective O2 is only half answered
here. Candidate count is a real, exact axis and it contributes very little
(P10 OverRate 0.6460 against 0.6393 for no conditioning at all). Whether
*serving* satellite geometry would have carried more is a question this data
cannot answer, and the answer must not be inferred from the failure of the
proxy.

## 4. What is not reproduced from the baselines

- BG-CFQS has no public code. Everything here is written from the paper, and
  the check on it is their published selected quantiles and their conditional
  OverRate table. If those do not come out, the reimplementation is wrong and
  the comparison is worthless. **Neither check has been run**, because both
  need the traces.
- **BG-CFQS cannot serve a budget below 0.15, and this is structural rather
  than a defect in the reimplementation.** Their candidate quantile set is
  T = [0.15, 0.40] (CLAUDE.md section 6, their published configuration), so the
  boundary search cannot select an operating point below the low end of its own
  candidate range. Measured on the Osnabruck sweep, the achieved global
  OverRate is pinned at 0.1854 for every budget at or below 0.15:

  | budget | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |
  |---|---|---|---|---|---|---|---|
  | BG-CFQS | 0.1854 | 0.1854 | 0.1854 | 0.2438 | 0.2969 | 0.3494 | 0.4054 |
  | ours, online | 0.0516 | 0.1007 | 0.1508 | 0.2010 | 0.2511 | 0.3010 | 0.3508 |

  At a budget of 0.05 it overshoots by a factor of 3.7. Their paper reports
  only epsilon = 0.35, where the floor never binds, so this does not contradict
  anything they published. It does mean the method as specified cannot be used
  at the budgets a real allocator would want, and it is the clearest
  demonstration in this project of why the sweep was worth running. Widening T
  would fix it and would no longer be their method.
- StarNet's obstruction-map-to-TLE resolution is implemented but untested
  against real obstruction maps, because the released traces already carry the
  resolved serving satellite. It is live-path code with unit tests on geometry,
  not a validated reproduction.
- Their released code fits its scaler on the whole trace before splitting. We
  do not reproduce that leak, so a small shortfall against their published
  numbers is expected rather than alarming.

## 4b. Results were computed on one machine, and one bug from that was caught

Added 2026-09-02, when CI was set up and immediately failed.

Every number in this project was produced on a single macOS machine with one
pinned set of library versions. Adding CI on Linux across Python 3.11, 3.12 and
3.13 surfaced a defect on the first run that 176 local tests had not:

`Series.astype("int64")` on a pandas datetime column returns the count in
whatever resolution that column carries, and pandas 2 supports seconds,
milliseconds, microseconds and nanoseconds. Four call sites divided that by 1e9
and assumed nanoseconds. Locally the columns were nanoseconds and everything
worked; on the runner they were not, the scheduling-phase aliasing guard
inverted its answer, and thirteen tests failed.

Fixed in `flwcnx/timeutil.py` with regression tests at all four resolutions.

**What this implies for the reported results, stated rather than assumed.** The
runs behind `results/summary/` were executed before the fix, on the machine
where the columns were nanoseconds, so the code path they took was the correct
one and the numbers are unaffected. That is an argument, not a re-run: they have
not been recomputed on a second machine, and no result in this project has
independent hardware confirmation.

A second defect surfaced in the same run: `scikit-learn` had been removed from
the dependencies by a static import audit, and it is genuinely required because
`xgb.XGBRegressor` is xgboost's scikit-learn API. It was present locally as a
transitive dependency. An import audit cannot see that, and CI could.

## 4c. The protection layer is evaluated on a modelled workload

This is the largest qualification in the document and it applies to every
number about `decide/flows.py` and `decide/protect.py`.

**The capacity is measured and the flows are a model.** The per-slot capacity
series is the calibrated bound and the realised throughput from a trained
pipeline over the real StarNet traces, which is as real as anything else in
this repository. The flows contending for it come from
`flwcnx/eval/workload.py` and are generated.

There was no alternative. The protection layer needs per-flow evidence and
per-flow ground truth, and neither StarNet nor WetLinks carries either: both
are link-level throughput series with no flow table, no process attribution and
no user attention signal. No public LEO dataset has them, because collecting
them means instrumenting the host rather than the link. Driving a simulated
workload with a real capacity trace is what every adaptive-streaming evaluation
does, and it is the honest form of the thing, but it is not a measurement of
real traffic.

What that means concretely, in descending order of how much it should worry a
reader:

1. **`truly_critical` is a stipulation, not an observation.** It is defined as
   "throttling this flow below its floor causes harm the operator would not
   accept", and which archetypes carry it is our choice. Every accuracy figure
   is accuracy against our own model of what matters.
2. **The confusion in the workload is ours too.** The archetypes are chosen so
   that a rate-based shaper and a declaration-trusting policy each fail on a
   case they cannot see, because a generator without those cases would make
   the trivial policy win and measure nothing. That choice is argued for in the
   module docstring, and it is still a choice that favours the layer being
   evaluated. A workload whose critical flows were all small and all correctly
   declared would show this layer buying nothing.
3. **The channel weights are hand set and were not learned.** Fitting them on
   this generator and evaluating on the same generator is circular, so they
   were not fitted at all. `scripts/run_protection.py --sensitivity` perturbs
   each weight by up to a factor of four either way and reports the range; that
   bounds how much the conclusions depend on the exact values, and bounds
   nothing about whether the channels are the right ones.
4. **Elasticity, packet-size profiles, arrival rates and the mix are modelled.**
   The elasticity response is generated from behaviour rather than read off the
   label, which is the minimum needed for the channel measuring it to be
   measuring something, but an exponential backoff toward the granted rate is
   not TCP.
5. **The shortfall model is a simplification.** When the bound overestimates,
   the excess is shared in proportion to the allocated rates. A real bottleneck
   distributes loss by queue occupancy and RTT, and a real shaper would enforce
   the rates rather than let them collide.

The one claim that does not depend on the workload is the allocator's
guarantee, which is algebraic: whenever the protected floors sum to no more
than the capacity, every protected flow receives at least its floor. That is
proved by the form of the water-filling solution and pinned on random instances
in `tests/test_protect.py`. It says nothing about whether the right flows were
protected.

## 4d. What the scorer still gets wrong

Reported here rather than left in a results file. At the reference load, the
scorer's recall is high and its precision is not, and the precision cost has a
single identifiable source: an entertainment video stream and a
teleconsultation look alike on every channel this layer has. Both are
inelastic, both sit in the foreground, both show the same packet-size profile,
and neither has a deadline. The only signal separating them is the declared
class, which is exactly the signal the layer was built not to depend on.

A protected flow that should not have been protected costs capacity that a
genuinely critical flow could have used, so this is not a free error. It is
visible in `docs/figures/protection-scorer.png` as the one blue archetype
sitting above the threshold, and it is the first thing to fix.

## 5. Things this system does not do

- No live terminal. `LiveSource.connect` raises.
- Congestion is a threshold rule on the bound, not a trained detector. That is
  a design choice with a stated rationale, but it means congestion performance
  is entirely inherited from the forecaster and the calibration.
- The admission layer assumes uniform 10 Mbps services and an oracle defined
  by realised throughput. Real admission control has heterogeneous demands,
  holding times, and a scheduler. `decide/protect.py` relaxes the uniform
  demand assumption but only against a modelled workload, see section 4c.
- The protection layer never enforces anything. It computes rates; wiring them
  to a queueing discipline on a real host is not done and is not stubbed.
- Criticality is scored per flow and the allocator treats flows as
  independent. An application whose usefulness depends on several flows at
  once, which is most of them, is not modelled.
- The sequence length is capped by the data, not chosen. Every iperf run in the
  WetLinks seconds release is exactly 15 samples, so look-back plus horizon
  cannot exceed 15 and every run here is 10/5. StarNet's 30/5 and BG-CFQS's
  75/15 are impossible on this data, which is one of several reasons no number
  in this repo is comparable to a published one.
- Cross location means two European sites 150 km apart, measured by the same
  instrument in the same months. It is not the three continents the StarNet
  traces would have given, and it should not be described as though it were.
