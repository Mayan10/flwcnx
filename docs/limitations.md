# Limitations

Written while building, not retrofitted at the end. Anything here that turns
out to be wrong should be corrected in place rather than quietly dropped.

## 1. Exchangeability breaks under a temporal split

Split conformal guarantees `P(bound > y) <= epsilon` only if the calibration
and test residuals are exchangeable. Temporal splits deliberately violate this:
the test block is later in time than the calibration block, and Starlink
throughput is non-stationary on exactly the timescales that matters over.

This is visible in every smoke run. Against a budget of 0.35 the global
conformal bound lands well above it on the test block, even though it lands
exactly on it when calibration and test are drawn from the same period. The
effect is not a bug in the implementation: `tests/test_calibrate.py` confirms
the bound hits its budget to within 0.02 on exchangeable data at three
different budgets.

Consequences to keep in view:

- **Do not report "risk pass" as though the guarantee were unconditional.**
  Every risk-pass column in this repo is empirical, measured on the test block.
- Our layer inherits this. Regime conditioning fixes *conditional* risk given
  the same marginal drift; it does not fix the drift. If the real traces show
  large drift, expect both the baseline and our layer to overshoot, and the
  claim narrows to "we overshoot less unevenly", which is a weaker claim and
  has to be written as one.
- The honest mitigations, in order of preference: a rolling or online
  calibration that refits the operating point on recent residuals; a shorter
  gap between calibration and test; or reporting against both a temporal and a
  same-period split so the drift cost is separated from the conditioning gain.
  None of these is implemented yet.
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

## 4. What is not reproduced from the baselines

- BG-CFQS has no public code. Everything here is written from the paper, and
  the check on it is their published selected quantiles and their conditional
  OverRate table. If those do not come out, the reimplementation is wrong and
  the comparison is worthless.
- StarNet's obstruction-map-to-TLE resolution is implemented but untested
  against real obstruction maps, because the released traces already carry the
  resolved serving satellite. It is live-path code with unit tests on geometry,
  not a validated reproduction.
- Their released code fits its scaler on the whole trace before splitting. We
  do not reproduce that leak, so a small shortfall against their published
  numbers is expected rather than alarming.

## 5. Things this system does not do

- No live terminal. `LiveSource.connect` raises.
- Congestion is a threshold rule on the bound, not a trained detector. That is
  a design choice with a stated rationale, but it means congestion performance
  is entirely inherited from the forecaster and the calibration.
- The allocation layer assumes uniform 10 Mbps services and an oracle defined
  by realised throughput. Real admission control has heterogeneous demands,
  holding times, and a scheduler.
