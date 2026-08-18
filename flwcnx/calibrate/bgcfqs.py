"""BG-CFQS: budget guided coarse to fine quantile selection.

Reimplementation of Xie et al., *Risk-Aware Safe Throughput Forecasting for
Starlink Networks*, arXiv 2605.09508, Algorithm 1. Not ours. There is no
public code, so this is written from the paper.

The method in one line: train quantile regressors over a candidate quantile
set, and pick the *largest* quantile whose overestimation rate on a held out
calibration set still sits under the risk budget. Largest, because a lower
quantile is safer but wastes capacity, so the budget is spent rather than
hoarded.

Configuration from the paper (CLAUDE.md section 6):

    history L = 75, horizon H = 15, epsilon = 0.35,
    T = [0.15, 0.40], coarse delta = 0.05, fine grid M = 5,
    XGBoost backbone trained with pinball loss

and the protocol: the calibration set is used for quantile selection only and
never for fitting.

The published selected quantiles are CHI 0.314, OSN 0.244, VIC 0.306. Those are
the check on this implementation. What we are really after is downstream: their
global OverRate of 0.349 against a 0.35 budget, alongside 0.65 to 0.71 on P30
and 0.83 to 0.86 on P10. That conditional failure is the gap our layer exists
to close, so reproducing it faithfully matters more than beating it here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flwcnx.config import BGCFQSConfig
from flwcnx.eval.metrics import over_rate
from flwcnx.forecast.baselines import XGBoostForecaster


@dataclass
class SelectionTrace:
    """Every quantile tried and what it achieved. The audit trail for a result."""

    coarse: list[tuple[float, float]] = field(default_factory=list)   # (tau, over_rate)
    fine: list[tuple[float, float]] = field(default_factory=list)
    selected_tau: float = float("nan")
    selected_over_rate: float = float("nan")
    feasible_found: bool = False

    def to_dict(self) -> dict:
        return {
            "coarse": [(round(t, 4), round(r, 4)) for t, r in self.coarse],
            "fine": [(round(t, 4), round(r, 4)) for t, r in self.fine],
            "selected_tau": self.selected_tau,
            "selected_over_rate": self.selected_over_rate,
            "feasible_found": self.feasible_found,
        }


class BGCFQS:
    """Budget guided coarse to fine quantile selection over an XGBoost backbone.

    Quantile models are trained on the training split and cached, because the
    search visits each candidate quantile once and refitting would dominate the
    cost. Selection reads the calibration split only.
    """

    def __init__(self, config: BGCFQSConfig | None = None, *, seed: int = 1337) -> None:
        self.config = config or BGCFQSConfig()
        self.seed = seed
        self._models: dict[float, XGBoostForecaster] = {}
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self.trace = SelectionTrace()
        self.selected_tau: float | None = None

    # -- backbone ----------------------------------------------------------

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> BGCFQS:
        """Hold the training data. Quantile models are fit lazily as searched."""
        self._x_train = np.asarray(x_train, dtype=float)
        self._y_train = np.asarray(y_train, dtype=float).ravel()
        return self

    def _model_for(self, tau: float) -> XGBoostForecaster:
        key = round(float(tau), 6)
        if key not in self._models:
            if self._x_train is None:
                raise RuntimeError("BGCFQS.fit must be called before selection")
            self._models[key] = XGBoostForecaster(
                quantile_alpha=key,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                seed=self.seed,
            ).fit(self._x_train, self._y_train)
        return self._models[key]

    def predict_quantile(self, x: np.ndarray, tau: float) -> np.ndarray:
        return self._model_for(tau).predict(np.asarray(x, dtype=float))

    # -- Algorithm 1 -------------------------------------------------------

    def select(self, x_cal: np.ndarray, y_cal: np.ndarray,
               epsilon: float | None = None) -> float:
        """Coarse to fine search for the largest budget respecting quantile.

        Coarse pass walks T at spacing delta. Fine pass subdivides the interval
        between the best feasible coarse point and the next one up into M
        steps, which is where the published quantiles land: 0.314 and 0.306 are
        not on the coarse grid of 0.15, 0.20, ..., 0.40, so the fine pass is
        doing real work rather than decorating the result.
        """
        epsilon = self.config.epsilon if epsilon is None else epsilon
        x_cal = np.asarray(x_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        self.trace = SelectionTrace()

        n_coarse = int(round((self.config.tau_hi - self.config.tau_lo) /
                             self.config.coarse_delta)) + 1
        coarse_grid = np.linspace(self.config.tau_lo, self.config.tau_hi, n_coarse)

        best_tau: float | None = None
        best_rate = float("nan")
        next_tau: float | None = None

        for tau in coarse_grid:
            rate = over_rate(self.predict_quantile(x_cal, float(tau)), y_cal)
            self.trace.coarse.append((float(tau), float(rate)))
            if rate <= epsilon:
                best_tau, best_rate = float(tau), float(rate)
            else:
                next_tau = float(tau)
                break

        if best_tau is None:
            # Even the lowest candidate quantile overshoots the budget. Report
            # it rather than silently returning tau_lo as if it had passed: the
            # honest answer is that this budget is unreachable with this
            # candidate set on this data.
            self.selected_tau = float(self.config.tau_lo)
            self.trace.selected_tau = self.selected_tau
            self.trace.selected_over_rate = (
                self.trace.coarse[0][1] if self.trace.coarse else float("nan")
            )
            self.trace.feasible_found = False
            return self.selected_tau

        upper = next_tau if next_tau is not None else min(
            best_tau + self.config.coarse_delta, self.config.tau_hi
        )
        if upper > best_tau:
            for tau in np.linspace(best_tau, upper, self.config.fine_grid + 2)[1:-1]:
                rate = over_rate(self.predict_quantile(x_cal, float(tau)), y_cal)
                self.trace.fine.append((float(tau), float(rate)))
                if rate <= epsilon and tau > best_tau:
                    best_tau, best_rate = float(tau), float(rate)

        self.selected_tau = best_tau
        self.trace.selected_tau = best_tau
        self.trace.selected_over_rate = best_rate
        self.trace.feasible_found = True
        return best_tau

    def predict(self, x: np.ndarray, floor: float = 0.0) -> np.ndarray:
        """Safe lower bound at the selected quantile."""
        if self.selected_tau is None:
            raise RuntimeError("BGCFQS.select must run before predict")
        return np.maximum(self.predict_quantile(x, self.selected_tau), floor)


def bgcfqs_on_residuals(predicted_cal: np.ndarray, actual_cal: np.ndarray,
                        epsilon: float, config: BGCFQSConfig | None = None,
                        direction: str = "lower") -> float:
    """The BG-CFQS boundary search applied to residuals rather than to a model.

    Used two ways. As a per regime operating point selector inside our
    calibration layer, so the comparison against BG-CFQS is a comparison of
    *conditioning* and not of backbone. And as a cheap global baseline that
    isolates the search from the quantile regression.

    Selects the largest residual quantile whose empirical overestimation rate
    on the calibration residuals stays inside the budget. Unlike the conformal
    order statistic this has no finite sample correction, so it is slightly
    anti-conservative on small regimes. That difference is exactly what the two
    modes of `regime_cal` are there to measure.
    """
    config = config or BGCFQSConfig()
    predicted_cal = np.asarray(predicted_cal, dtype=float).ravel()
    actual_cal = np.asarray(actual_cal, dtype=float).ravel()
    finite = np.isfinite(predicted_cal) & np.isfinite(actual_cal)
    residuals = (actual_cal - predicted_cal)[finite]
    if residuals.size == 0:
        raise ValueError("no finite calibration residuals")

    # On an upper bound the candidate set is mirrored: tau indexes how much
    # risk is being spent, so the offset is the (1 - tau) residual quantile.
    # Both directions stay monotone in tau, so one search serves both.
    def offset_for(tau: float) -> float:
        level = tau if direction == "lower" else 1.0 - tau
        return float(np.quantile(residuals, level))

    def rate_for(tau: float) -> float:
        return over_rate(predicted_cal[finite] + offset_for(tau), actual_cal[finite],
                         direction)

    n_coarse = int(round((config.tau_hi - config.tau_lo) / config.coarse_delta)) + 1
    grid = np.linspace(config.tau_lo, config.tau_hi, n_coarse)

    best_tau = None
    upper = config.tau_hi
    for tau in grid:
        if rate_for(float(tau)) <= epsilon:
            best_tau = float(tau)
        else:
            upper = float(tau)
            break

    if best_tau is None:
        # Budget unreachable inside the candidate set. Fall back to the most
        # conservative candidate rather than pretending the search succeeded.
        return offset_for(config.tau_lo)

    for tau in np.linspace(best_tau, upper, config.fine_grid + 2)[1:-1]:
        if rate_for(float(tau)) <= epsilon and tau > best_tau:
            best_tau = float(tau)
    return offset_for(best_tau)
