# The calibration layer on the StarNet traces, with measured satellite geometry

Run 2026-09-01 on all three locations, stride 6, look-back 30, horizon 5,
budget sweep 0.05 to 0.35, nine regime granularities, six calibration methods,
50 epochs, seed 1337. Test decisions per location: 20,714 (USA), 12,251
(Germany), 4,676 (Canada).

This is the run the WetLinks work could not do. There the satellite geometry
was reconstructed from propagated elements and the elevation proxy was
degenerate, so `level` won largely by default. Here elevation, distance and
candidate count are **measured**, so the brief's axes get a fair test.

**The headline is mixed, and the half that fails is the half the project was
named after.** Reported in that order rather than the flattering one.

## 1. Online recalibration replicates, and dominates

Budget tracking, asked against achieved, averaged over three locations:

| budget | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |
|---|---|---|---|---|---|---|---|
| adaptive achieved | 0.050 | 0.100 | 0.150 | 0.200 | 0.250 | 0.300 | 0.350 |
| static achieved | 0.040 | 0.080 | 0.120 | 0.160 | 0.204 | 0.248 | 0.293 |
| static error | -21% | -20% | -20% | -20% | -19% | -17% | -16% |

The online layer tracks its budget within 0.8% at every point. Static split
conformal is 16 to 21% **low** at every point.

Note the direction. On WetLinks static calibration *overshot* the budget
(0.42 against 0.35) because the test month was slower than the calibration
month. Here it *undershoots* by a fifth. Both are failures of risk control: an
overshoot drops sessions, an undershoot silently wastes capacity, and in
neither case can an operator set a budget and get it. The sign of the miss is a
property of the drift, not of the method, which is a stronger statement than
the WetLinks run alone could support.

At **matched achieved risk**, the comparison the naive table gets wrong:

| achieved risk | adaptive P10 | static+level P10 | adaptive gain | adaptive MAE | static MAE |
|---|---|---|---|---|---|
| 0.050 | 0.145 | 0.170 | -14.6% | 51.54 | 52.66 |
| 0.100 | 0.256 | 0.289 | -11.3% | 39.62 | 40.64 |
| 0.150 | 0.357 | 0.398 | -10.4% | 33.67 | 34.47 |
| 0.200 | 0.438 | 0.499 | -12.1% | 29.76 | 30.53 |
| 0.250 | 0.521 | 0.589 | -11.6% | 27.23 | 28.00 |
| 0.300 | 0.598 | 0.666 | -10.1% | 25.36 | 26.39 |

Better conditional risk **and** better accuracy at every operating point. The
0.35 row is omitted because static never achieves 0.35 and comparing there
would be extrapolation.

Ranking methods by raw P10 without matching achieved risk puts static BG-CFQS
on top at 0.659. That ranking is an artifact: it is spending only 0.289 of its
0.35 budget, and conservatism buys P10 directly. Any table of these methods has
to match on achieved risk or it is measuring budget usage rather than method
quality.

## 2. Regime conditioning largely does not replicate here

The brief's ablation, adaptive regime conformal at budget 0.35, mean P10 over
three locations. n = 3,718 in the USA P10 slice, so the standard error is
0.0077 and these differences are real rather than noise:

| axes | P10 | vs no conditioning |
|---|---|---|
| **global (none)** | **0.6663** | - |
| candidates | 0.6755 | +1.4% |
| level | 0.6733 | +1.1% |
| elevation | 0.6854 | +2.9% |
| level+geometry | 0.6856 | +2.9% |
| level+full | 0.6868 | +3.1% |
| geometry | 0.6931 | +4.0% |
| phase | 0.6944 | +4.2% |
| full | 0.7085 | +6.3% |

**Every axis is worse than not conditioning at all**, including the measured
satellite geometry the brief expected to carry the signal. Objective O2's
premise does not hold on this data: elevation, distance and candidate count,
measured rather than reconstructed, do not improve conditional risk control.

In the **static** setting conditioning does help, slightly and consistently:
`level` gives P10 0.6655 against 0.6869 for global, about 3%. So the axes carry
a little signal. It is an order of magnitude smaller than what adaptation buys.

## 3. Why per-regime adaptation hurts: the feedback stream gets divided

The degradation tracks the number of regimes, not the choice of axis:

| axes | regimes | test outcomes per regime | P10 |
|---|---|---|---|
| global | 1 | 20,714 | 0.6663 |
| level | 4 | 5,179 | 0.6733 |
| geometry | 18 | 1,151 | 0.6931 |
| full | 51 | 404 | 0.7085 |
| level+full | 174 | 120 | 0.6868 |

Correlation between log10(outcomes per regime) and P10 OverRate is -0.571.

The mechanism is specific to the online layer. Static per-regime calibration
fits one offset per regime **once**, from the calibration split. Per-regime
online calibration must converge one alpha per regime from a **stream**, and
each regime only receives feedback on the decisions routed to it. Splitting
20,714 outcomes across 51 regimes leaves roughly 400 each, which is not enough
for the update rule to settle before the test split ends. The per-regime alpha
noise then exceeds the conditional signal it was meant to capture.

This is why the two mechanisms composed on WetLinks and conflict here. On
WetLinks the winning partition was `level+candidates`: 12 regimes over 13,530
decisions, about 1,100 outcomes each. That is above the threshold. The StarNet
grid mostly is not.

**The practical rule this suggests**, and it should be tested rather than
assumed: the online layer needs a coarser partition than the static layer, and
regime count should be chosen from the expected feedback volume rather than
from how finely the covariates can be cut.

## 4. What this means for the contribution

Stated plainly because it changes the claim:

- **Online per-regime recalibration**: replicates on StarNet, dominates static
  calibration on both risk and accuracy at every matched operating point, and
  holds its budget within 1% where static misses by a fifth. This is the part
  that works.
- **Regime conditioning**: gives about 3% in the static setting and is
  net-negative in the online setting on this dataset. On WetLinks it gave 5.5%.
  The honest summary is that it is dataset dependent and small, not that it is
  a reliable gain.
- **Objective O2 as the brief framed it**: not supported. Measured serving
  satellite elevation, distance and candidate count did not improve conditional
  risk control on any of the three locations.

The WetLinks result is not withdrawn and this one is not preferred over it.
They are two datasets and they disagree about the regime layer; the writeup has
to carry both.

## Reproduce

```
python -m flwcnx.eval.runner --location usa --stride 6 --lookback 30 --horizon 5 \
    --epochs 50 --granularities global phase elevation candidates geometry full \
    level level+geometry level+full --output results/starnet_regime/usa
```

An earlier run at StarNet's published per-location strides (46/29/6) is kept at
`results/starnet_regime_stride_published/`. It is underpowered and should not be
quoted: only ~450 decisions land in its P10 slice, and `level+full` spread 4,700
calibration points over 173 regimes with a 37% fallback rate. It was rerun at
stride 6 for that reason, stride 6 being the densest windowing that still keeps
scored horizons disjoint at horizon 5.
