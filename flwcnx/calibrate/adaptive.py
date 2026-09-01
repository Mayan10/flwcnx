"""Regime conditioned *online* calibration.

Why this module exists, measured rather than assumed. On the Osnabruck
per-second release, split into contiguous temporal thirds, the link is not
stationary across the splits:

    train  2023-09-14 to 2024-01-05   mean 208.6 Mbps
    calib  2024-01-05 to 2024-02-06   mean 219.0 Mbps
    test   2024-02-06 to 2024-03-12   mean 197.2 Mbps

The test month is 10% slower than the month the calibration offset was fit on.
Split conformal's guarantee is conditional on the calibration and test
residuals being exchangeable, and across a one month gap on this link they are
not. The consequence is measurable: a static bound fit for a 0.35 budget lands
at OverRate 0.42 on test, and misses the budget it was constructed to hold.

This is not a defect in the conformal construction. It is the standard
distribution shift failure, and Horizon (CLAUDE.md section 6) reports the same
non-stationarity from the other direction: throughput prediction keeps
improving out to an eleven month training window while latency peaks at two,
which only happens if the two signals drift on different timescales.

The fix is to stop treating the operating point as a constant.

**Attribution.** The update rule is Adaptive Conformal Inference, Gibbs and
Candes 2021, and is not ours:

    alpha_{t+1} = alpha_t + gamma * (epsilon - err_t)

where err_t is 1 when the bound realised the risk event at step t. It is a
standard online method with a known regret bound, and it is used here as a
component exactly as the GRU backbone is.

**This module is a reproduction, not a contribution.** It was written believing
only the marginal single-alpha version was in print. That was wrong, and the
correction is recorded rather than quietly absorbed:

- **GCACI**, Ramalingam et al. 2025 (arXiv:2502.10947), already generalises ACI
  to group-conditional guarantees, with an FTRL formulation that carries a
  finite-time group-coverage bound. What this module does is the naive special
  case of that, with no bound.
- **POGO** (arXiv:2606.00419, 2026) goes further and is parameter-free. The
  `gamma` below is exactly the learning rate POGO exists to eliminate.

So one alpha per regime is prior work. What this project contributes is the
measurement around it: which regimes actually differ on real LEO links, and the
finding that conditioning helps only when they do. See
`docs/novelty-assessment.md`.

**Causality.** Nothing here reads an outcome before it is observable. At step t
the bound is computed from residuals of steps strictly before t, then the step
t outcome is revealed and used to update alpha for step t+1. This is the order
a terminal actually experiences: it commits an allocation for the next H
seconds, those H seconds happen, and it measures what it got. The unit test
`test_adaptive_bounds_do_not_depend_on_future_outcomes` pins the property by
rewriting the tail of the outcome stream and checking the earlier bounds are
byte identical.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flwcnx.calibrate.regime_cal import RegimeCalibrator
from flwcnx.config import CalibrationConfig
from flwcnx.eval.metrics import check_direction
from flwcnx.state.regime import GLOBAL_LABEL, RegimeAssigner

#: Below this many residuals a regime's own window is not trusted to supply a
#: quantile, and the pooled global window is used instead. Same motivation as
#: `RegimeConfig.min_samples`: a quantile from twenty points is noise, and noise
#: in the offset is paid for in dropped sessions.
MIN_WINDOW = 50


def offset_for_alpha(residuals: np.ndarray, alpha: float, direction: str) -> float:
    """The additive offset whose risk rate is `alpha` on these residuals.

    Residuals are r = actual - predicted, so the bound is predicted + offset.

    lower: the risk is predicting above the truth, and
           P(pred + off > actual) = P(r < off), so off is the alpha quantile.
    upper: the risk is predicting below the truth, and
           P(pred + off < actual) = P(r > off), so off is the 1 - alpha one.
    """
    if residuals.size == 0:
        return 0.0
    q = alpha if direction == "lower" else 1.0 - alpha
    return float(np.quantile(residuals, np.clip(q, 0.0, 1.0)))


@dataclass
class AdaptiveTrace:
    """Per step record, kept because the adaptation is itself a result.

    A reviewer's first question about an online method is whether it converged
    or oscillated. Storing alpha and the offset at every step is what lets the
    answer be a figure instead of an assertion.
    """

    alpha: np.ndarray
    offset: np.ndarray
    regime: np.ndarray
    risk_event: np.ndarray
    fell_back: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "alpha": self.alpha, "offset": self.offset, "regime": self.regime,
            "risk_event": self.risk_event, "fell_back": self.fell_back,
        })

    def summary(self) -> dict:
        return {
            "n_steps": int(self.alpha.size),
            "alpha_start": float(self.alpha[0]) if self.alpha.size else float("nan"),
            "alpha_end": float(self.alpha[-1]) if self.alpha.size else float("nan"),
            "alpha_mean": float(np.mean(self.alpha)) if self.alpha.size else float("nan"),
            "alpha_min": float(np.min(self.alpha)) if self.alpha.size else float("nan"),
            "alpha_max": float(np.max(self.alpha)) if self.alpha.size else float("nan"),
            "offset_mean": float(np.mean(self.offset)) if self.offset.size else float("nan"),
            "realised_risk_rate": (float(np.mean(self.risk_event))
                                   if self.risk_event.size else float("nan")),
            "fallback_rate": (float(np.mean(self.fell_back))
                              if self.fell_back.size else float("nan")),
            "n_regimes_active": int(len(set(self.regime.tolist()))),
        }


@dataclass
class AdaptiveRegimeCalibrator:
    """One alpha and one residual window per regime, both updated online.

    `fit` seeds the windows from the calibration split, which is the same data
    the static calibrator uses, so the two start from the same place and any
    difference on test is attributable to the adaptation alone.
    """

    config: CalibrationConfig = field(default_factory=CalibrationConfig)
    assigner: RegimeAssigner | None = None
    direction: str = "lower"
    #: Step size. 0.02 moves alpha by two points per miss, so a regime that is
    #: systematically wrong corrects within tens of decisions rather than
    #: thousands, while a regime that is right jitters by well under a point.
    gamma: float = 0.02
    #: Rolling residual window per regime. Finite so that a shift is forgotten
    #: rather than averaged against forever, which is the whole point.
    window: int = 2000
    #: alpha is kept off the boundary: 0 or 1 would pin the offset to an
    #: extreme order statistic and the update could never climb back.
    alpha_floor: float = 0.005
    alpha_ceiling: float = 0.95

    _windows: dict[str, deque] = field(default_factory=dict, repr=False)
    _alphas: dict[str, float] = field(default_factory=dict, repr=False)
    _global_window: deque | None = field(default=None, repr=False)
    _global_alpha: float = field(default=0.0, repr=False)
    _fitted: bool = field(default=False, repr=False)
    trace: AdaptiveTrace | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        check_direction(self.direction)
        if self.assigner is None:
            self.assigner = RegimeAssigner(self.config.regime)

    # -- seeding -----------------------------------------------------------

    def _labels(self, covariates: pd.DataFrame) -> np.ndarray:
        assert self.assigner is not None
        if not self.config.regime.axes:
            return np.full(len(covariates), GLOBAL_LABEL, dtype=object)
        return np.asarray(self.assigner.assign(covariates), dtype=object)

    def fit(self, predicted_cal: np.ndarray, actual_cal: np.ndarray,
            covariates_cal: pd.DataFrame) -> AdaptiveRegimeCalibrator:
        """Seed each regime's residual window and alpha from the calibration split."""
        assert self.assigner is not None
        if self.config.regime.axes and not self.assigner._fitted:
            raise RuntimeError(
                "RegimeAssigner is not fitted. Fit it on the training split before "
                "seeding the adaptive calibrator, or the regime edges are chosen "
                "from the calibration data."
            )
        predicted = np.asarray(predicted_cal, dtype=float).ravel()
        actual = np.asarray(actual_cal, dtype=float).ravel()
        if predicted.size != len(covariates_cal):
            raise ValueError(
                f"{predicted.size} predictions against {len(covariates_cal)} covariate "
                "rows; these must be the same samples in the same order"
            )
        residuals = actual - predicted
        labels = self._labels(covariates_cal)

        self._global_window = deque(residuals.tolist(), maxlen=self.window)
        self._global_alpha = float(self.config.epsilon)
        self._windows, self._alphas = {}, {}
        for label in set(labels.tolist()):
            mask = labels == label
            self._windows[label] = deque(residuals[mask].tolist(), maxlen=self.window)
            # Every regime starts at the budget. Starting from the static
            # per-regime rate instead would bake in the calibration month's
            # shift, which is the thing being corrected.
            self._alphas[label] = float(self.config.epsilon)
        self._fitted = True
        return self

    # -- online application ------------------------------------------------

    def transform_online(self, predicted: np.ndarray, actual: np.ndarray,
                         covariates: pd.DataFrame, *, floor: float = 0.0) -> np.ndarray:
        """Stream the test split, emitting each bound before seeing its outcome.

        `actual` is consumed, but only ever at a step strictly earlier than the
        bound it influences. That is the honest version of what a deployed
        terminal has: yesterday's outcomes, not tomorrow's.
        """
        if not self._fitted:
            raise RuntimeError("AdaptiveRegimeCalibrator.fit must run before transform_online")
        predicted = np.asarray(predicted, dtype=float).ravel()
        actual = np.asarray(actual, dtype=float).ravel()
        if predicted.size != actual.size or predicted.size != len(covariates):
            raise ValueError("predicted, actual and covariates must have the same length")

        labels = self._labels(covariates)
        n = predicted.size
        bounds = np.empty(n, dtype=float)
        alpha_log = np.empty(n, dtype=float)
        offset_log = np.empty(n, dtype=float)
        risk_log = np.zeros(n, dtype=bool)
        fell_back = np.zeros(n, dtype=bool)

        for t in range(n):
            label = labels[t]
            window = self._windows.get(label)
            alpha = self._alphas.get(label)
            # A regime unseen or under-populated in calibration borrows the
            # pooled window but keeps its own alpha, so it adapts on its own
            # evidence from the first step it appears.
            if window is None or len(window) < MIN_WINDOW:
                source = self._global_window
                fell_back[t] = True
                if alpha is None:
                    alpha = self._global_alpha
                    self._alphas[label] = alpha
                if window is None:
                    self._windows[label] = deque(maxlen=self.window)
            else:
                source = window

            assert source is not None
            offset = offset_for_alpha(np.fromiter(source, dtype=float, count=len(source)),
                                      alpha, self.direction)
            bound = max(float(predicted[t]) + offset, floor)
            bounds[t] = bound
            alpha_log[t] = alpha
            offset_log[t] = offset

            # --- the outcome becomes observable only now ---
            truth = actual[t]
            risk = bound > truth if self.direction == "lower" else bound < truth
            risk_log[t] = bool(risk)

            updated = alpha + self.gamma * (self.config.epsilon - float(risk))
            self._alphas[label] = float(np.clip(updated, self.alpha_floor, self.alpha_ceiling))

            residual = float(truth - predicted[t])
            self._windows[label].append(residual)
            assert self._global_window is not None
            self._global_window.append(residual)

        self.trace = AdaptiveTrace(alpha=alpha_log, offset=offset_log, regime=labels,
                                   risk_event=risk_log, fell_back=fell_back)
        return bounds

    def summary(self) -> dict:
        out: dict = {
            "method": "adaptive_regime_conformal",
            "reference": "Gibbs and Candes 2021, adaptive conformal inference",
            "gamma": self.gamma,
            "window": self.window,
            "epsilon": self.config.epsilon,
            "axes": list(self.config.regime.axes),
            "n_regimes_seeded": len(self._windows),
        }
        if self.trace is not None:
            out.update(self.trace.summary())
            per_regime = self.trace.to_frame().groupby("regime", observed=True).agg(
                n=("alpha", "size"), alpha_end=("alpha", "last"),
                realised_risk=("risk_event", "mean"), offset_mean=("offset", "mean"),
            )
            out["per_regime"] = {
                str(k): {kk: (float(vv) if kk != "n" else int(vv)) for kk, vv in v.items()}
                for k, v in per_regime.to_dict(orient="index").items()
            }
        return out


def static_then_adaptive(
    config: CalibrationConfig,
    assigner: RegimeAssigner | None,
    predicted_cal: np.ndarray,
    actual_cal: np.ndarray,
    covariates_cal: pd.DataFrame,
    predicted_test: np.ndarray,
    actual_test: np.ndarray,
    covariates_test: pd.DataFrame,
    *,
    direction: str = "lower",
    gamma: float = 0.02,
    window: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Both bounds from one seeding, for a like-for-like static/online pair.

    Returns (static_bounds, adaptive_bounds, summary). The static calibrator is
    the existing `RegimeCalibrator`, unchanged, so the comparison isolates the
    adaptation and nothing else.
    """
    static = RegimeCalibrator(config=config, assigner=assigner, selector="conformal",
                              direction=direction)
    static.fit(predicted_cal, actual_cal, covariates_cal)
    static_bounds = static.transform(predicted_test, covariates_test)

    online = AdaptiveRegimeCalibrator(config=config, assigner=assigner,
                                      direction=direction, gamma=gamma, window=window)
    online.fit(predicted_cal, actual_cal, covariates_cal)
    adaptive_bounds = online.transform_online(predicted_test, actual_test, covariates_test)

    return static_bounds, adaptive_bounds, {"static": static.summary(),
                                            "adaptive": online.summary()}
