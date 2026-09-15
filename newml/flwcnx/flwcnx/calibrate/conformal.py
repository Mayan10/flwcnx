"""One-sided split conformal lower bound.

Reproduction of standard split conformal prediction (Vovk, Gammerman and
Shafer), specialised to a one-sided lower bound because the asymmetry is the
whole point: predicting below the actual costs utilisation, predicting above
it drops sessions.

The construction. On a calibration set held out from fitting, take residuals

    r_i = y_i - yhat_i

and let q be the k-th smallest residual with k = floor(epsilon * (n + 1)).
The bound is

    L = yhat + q

Under exchangeability of the calibration and test residuals,

    P(L > y) = P(r < q) <= k / (n + 1) <= epsilon

which is exactly the overestimation budget. Note the guarantee is *marginal*.
It says nothing about the rate inside any particular subpopulation, and the
BG-CFQS conditional results are the empirical demonstration of that gap. Fixing
it is what `regime_cal.py` is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flwcnx.eval.metrics import check_direction, over_rate


@dataclass(frozen=True)
class ConformalBound:
    """A fitted additive offset plus the evidence for it."""

    offset: float
    epsilon: float
    n_calibration: int
    rank: int                    # k, the order statistic used
    achieved_over_rate: float    # on the calibration set itself, in-sample
    degenerate: bool = False     # too few points for any finite bound at this epsilon
    direction: str = "lower"

    def apply(self, predictions: np.ndarray, floor: float = 0.0) -> np.ndarray:
        """Turn point predictions into safe bounds, clipped at `floor`.

        The floor is not cosmetic. On the throughput path a bound below zero is
        a promise to deliver negative capacity, which the decision layer reads
        as "admit no sessions" anyway, so clipping makes the two agree. On the
        latency path a negative delay is simply impossible.
        """
        return np.maximum(np.asarray(predictions, dtype=float) + self.offset, floor)


def conformal_rank(n: int, epsilon: float) -> int:
    """k = floor(epsilon * (n + 1)), the order statistic the guarantee needs."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    return int(np.floor(epsilon * (n + 1)))


def fit_conformal(predicted: np.ndarray, actual: np.ndarray, epsilon: float,
                  floor: float = 0.0, direction: str = "lower") -> ConformalBound:
    """Fit the additive offset on a calibration split.

    `predicted` and `actual` must come from data the forecaster never saw.
    Fitting this on training residuals produces an offset that is far too
    optimistic, because training residuals are smaller than test residuals and
    the whole guarantee rests on the two being exchangeable.

    The two directions are mirror images of the same order statistic:

      lower  q is the k-th *smallest* residual, so P(r < q) <= k/(n+1) <= eps
             and the bound sits below the truth all but eps of the time.

      upper  q is the k-th *largest* residual, so P(r > q) <= k/(n+1) <= eps
             and the bound sits above the truth all but eps of the time.

    The upper form is what the latency target needs, where the unsafe side is
    promising a delay the link will not meet.
    """
    check_direction(direction)
    predicted = np.asarray(predicted, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    finite = np.isfinite(predicted) & np.isfinite(actual)
    residuals = np.sort((actual - predicted)[finite])
    n = residuals.size

    if n == 0:
        raise ValueError("no finite calibration residuals")

    k = conformal_rank(n, epsilon)
    if k < 1:
        # No finite offset can be certified at this epsilon with this many
        # points. Degrade to the most conservative available bound rather than
        # quietly returning something with no guarantee behind it.
        offset = (float(residuals[0]) - 1e-9 if direction == "lower"
                  else float(residuals[-1]) + 1e-9)
        return ConformalBound(offset, epsilon, n, 0, 0.0, degenerate=True,
                              direction=direction)

    offset = float(residuals[k - 1] if direction == "lower" else residuals[n - k])
    achieved = over_rate(predicted + offset, actual, direction)
    return ConformalBound(offset, epsilon, n, k, achieved, direction=direction)
