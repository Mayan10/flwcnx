# Phase 2 gate: BG-CFQS reproduction, and what it took to pass

Run 2026-09-01 on the released StarNet traces. **The gate passes**, but only
after the split was identified as the confound, and the identification is
itself the most useful result in this file.

L = 75, H = 15, epsilon = 0.35, candidate set T = [0.15, 0.40], coarse
tolerance 0.05, fine grid 5, XGBoost with pinball loss, calibration used only
for quantile selection. Stride 15, so scored decisions do not overlap
(`scripts/reproduce_bgcfqs.py` documents why).

## First attempt: conditional failure reproduced, risk column did not

Contiguous temporal splits, which is what `eval/splits.py` does everywhere else
in this project:

| location | selected tau | published tau | OverRate | P30 | published P30 | P10 | published P10 |
|---|---|---|---|---|---|---|---|
| CHI | 0.350 | 0.314 | 0.385 | 0.657 | 0.65 | 0.780 | 0.83 |
| OSN | 0.283 | 0.244 | 0.431 | 0.883 | 0.71 | 0.990 | 0.86 |
| VIC | 0.217 | 0.306 | 0.358 | 0.804 | 0.71 | 0.994 | 0.86 |

Average OverRate 0.391 against their 0.349, **risk pass 0/3 against their 3/3**.
Accuracy came out better than published (MAE 38.30 against 40.36, RMSE 48.74
against 52.48) while risk came out worse, which is a strange combination and
the reason this was not written up as a failed reproduction straight away.

## The confound: their split is not specified, ours is temporal

`scripts/split_sensitivity.py` holds everything constant except the split. Same
data, same backbone, same boundary search, same budget. One arm uses contiguous
temporal blocks; the other permutes the same windows first, which makes
calibration and test exchangeable. Exchangeability is precisely the assumption
the boundary search inherits from split conformal.

| split | OverRate | P30 | P10 | risk pass |
|---|---|---|---|---|
| temporal | 0.377 | 0.750 | 0.909 | 1/3 |
| **random (exchangeable)** | **0.340** | **0.671** | **0.848** | **3/3** |
| **published** | **0.349** | **0.65 to 0.71** | **0.83 to 0.86** | **3/3** |

Under the exchangeable split every published quantity lands in range: the
global rate within 0.009, P30 and P10 inside their stated intervals, and the
risk-pass column exactly. **The reimplementation is correct.** Nothing
downstream of it needs to be doubted.

## What that means, and it is not a criticism of their paper

BG-CFQS's risk guarantee holds under the assumption it is derived from. It does
not survive a temporal split, and a temporal split is what a deployed terminal
faces. Their paper does not claim otherwise; it simply does not report the
temporal case. This is a scope finding, not an error found in their work.

It also lands exactly where this project already was. `docs/limitations.md`
section 1 predicted this failure from first principles before the traces
existed, then measured it on WetLinks, then built the online layer that fixes
it. The same failure is now confirmed on the original authors' own data with
their own method.

## Two failures, two fixes, and they are independent

The important detail is in the P10 column of the random split: **0.848, under
the split where the method passes its budget 3/3.** The conditional failure is
not caused by drift. It is intrinsic to selecting one global quantile, and it
persists in full even when the exchangeability assumption is satisfied.

So the two problems this project addresses are genuinely separate:

| failure | visible when | our fix |
|---|---|---|
| risk lost in the low-capacity regime | always, including under a random split | regime conditioning |
| budget lost entirely | only under a temporal split | online recalibration |

That separation is worth stating plainly, because it means the regime layer is
motivated on its own terms and not as a side effect of the drift work.

## Reproduce

```
python scripts/reproduce_bgcfqs.py --all --stride 15
python scripts/split_sensitivity.py
```


---

## Verified from the primary source, 2026-09-01

The independent review flagged that the `T = [0.15, 0.40]` bound had not been
confirmed: its fetch of Table II was truncated. It is now confirmed by direct
text extraction from the arXiv PDF of arXiv:2605.09508.

**Table II, Experimental Configuration, verbatim:**

    History length      L   75
    Prediction horizon  H   15
    Default risk budget e   0.35
    Candidate quantile set T [0.15, 0.40]
    Coarse tolerance    d   0.05
    Fine-grid size      M   5

The same table gives used samples 1,123,832 / 613,295 / 145,053 for CHI / OSN /
VIC, which match this project's loader exactly and independently confirm the US
file identification recorded in `docs/data-access.md`.

**Algorithm 1, the floor, verbatim:**

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

When the budget is tighter than the risk at the most conservative candidate,
line 6 fires, the interval collapses to a point, line 21's `LinSpace` produces
M copies of `tau_min`, the feasible set at line 23 is empty, and line 26
returns `tau_min` however small the budget gets. The achieved rate therefore
pins at `R(tau_min)`, which is what we measure at 0.1854 for every budget at or
below 0.15.

**The finding is confirmed end to end**: the parameter from their Table II, the
mechanism from their Algorithm 1, and the pinned rate from our runs.

One further confirmation of the review's reading: their equations (23) to (26)
define admission control as `n_admit = floor(y_safe / b)` and
`n_drop = max(n_admit - n_oracle, 0)`, which is the same arithmetic as this
project's decision layer. The decision layer is therefore a reproduction of
theirs, not an addition, and is described that way.
