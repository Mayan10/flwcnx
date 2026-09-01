"""Published online calibration baselines. All reproduction, none of it ours.

The independent novelty review (`docs/novelty-review.md` section 3) named these
as the comparisons a reviewer asks about first, and their absence as a hole in
the related work independent of any novelty question.

**GCACI**, Ramalingam, Kiyani and Roth 2025, arXiv:2502.10947, Algorithm 2.
This is the paper that closed the project's methods claim. It maintains a
parameter vector with one coordinate per group and updates the coordinate the
current point belongs to. On a partition, which is what this project's regimes
are, it reduces exactly to one adaptive threshold per regime. Implemented here
so the comparison is a measured table rather than an argument.

**Rolling RC**, Feldman, Ringel, Bates and Romano, TMLR 2023, arXiv:2205.09095.
The closest published method to what this project's calibration layer actually
does: online control of a *user-specified risk* rather than coverage, under
arbitrary and even adversarial drift, via an additive calibration term with a
stretching function for faster adaptation.

One difference from GCACI worth stating because it is easy to miss. GCACI
updates the threshold directly in the units of the score. This project's
`AdaptiveRegimeCalibrator` updates `alpha`, the quantile *level*, and re-reads
the empirical quantile from a rolling residual window each step. Under a
stationary stream the two coincide; under drift they differ, because the
window-based version inherits the window's own adaptation. Neither is a
contribution over the other and the table below is how the difference gets
measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flwcnx.config import CalibrationConfig
from flwcnx.eval.metrics import check_direction
from flwcnx.state.regime import GLOBAL_LABEL, RegimeAssigner


def _labels(assigner: RegimeAssigner | None, config: CalibrationConfig,
            covariates: pd.DataFrame) -> np.ndarray:
    if not config.regime.axes or assigner is None:
        return np.full(len(covariates), GLOBAL_LABEL, dtype=object)
    return np.asarray(assigner.assign(covariates), dtype=object)


@dataclass
class GCACI:
    """Group Conditional ACI, Ramalingam, Kiyani and Roth 2025, Algorithm 2.

    Their update, for one-hot group membership `g_t` on a partition:

        tau_hat_t = <theta_t, g_t>                       (the threshold)
        theta_{t+1} = theta_t + eta * q * g_t            if undercovered
        theta_{t+1} = theta_t - eta * (1 - q) * g_t      otherwise

    which is online gradient descent on the pinball loss, one coordinate at a
    time. `eta = 1` follows the paper's own experiments, and POGO also uses
    `eta = 1` "following [41]".

    Here the threshold is an additive offset on the point forecast, and the risk
    event is direction-aware so the same class serves the throughput lower bound
    and the latency upper bound.
    """

    config: CalibrationConfig = field(default_factory=CalibrationConfig)
    assigner: RegimeAssigner | None = None
    direction: str = "lower"
    eta: float = 1.0

    _theta: dict[str, float] = field(default_factory=dict, repr=False)
    _fitted: bool = field(default=False, repr=False)
    trace: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        check_direction(self.direction)

    def fit(self, predicted_cal: np.ndarray, actual_cal: np.ndarray,
            covariates_cal: pd.DataFrame) -> GCACI:
        """Seed each coordinate at its regime's empirical offset.

        The paper starts from zero; starting from the calibration quantile is a
        kinder initialisation and matches how the other methods here are seeded,
        so the comparison isolates the update rule rather than the warm start.
        """
        predicted = np.asarray(predicted_cal, dtype=float).ravel()
        actual = np.asarray(actual_cal, dtype=float).ravel()
        residuals = actual - predicted
        labels = _labels(self.assigner, self.config, covariates_cal)
        epsilon = self.config.epsilon
        q = epsilon if self.direction == "lower" else 1.0 - epsilon

        self._theta = {}
        pooled = float(np.quantile(residuals, q)) if residuals.size else 0.0
        for label in set(labels.tolist()):
            part = residuals[labels == label]
            self._theta[str(label)] = (float(np.quantile(part, q))
                                       if part.size >= 30 else pooled)
        self._theta.setdefault(GLOBAL_LABEL, pooled)
        self._fitted = True
        return self

    def transform_online(self, predicted: np.ndarray, actual: np.ndarray,
                         covariates: pd.DataFrame, *, floor: float = 0.0) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("GCACI.fit must run before transform_online")
        predicted = np.asarray(predicted, dtype=float).ravel()
        actual = np.asarray(actual, dtype=float).ravel()
        labels = _labels(self.assigner, self.config, covariates)
        epsilon = self.config.epsilon
        n = predicted.size
        bounds = np.empty(n, dtype=float)
        risk = np.zeros(n, dtype=bool)

        for t in range(n):
            label = str(labels[t])
            theta = self._theta.get(label)
            if theta is None:
                theta = self._theta[GLOBAL_LABEL]
                self._theta[label] = theta
            bound = max(float(predicted[t]) + theta, floor)
            bounds[t] = bound

            # Outcome becomes observable only now.
            hit = bound > actual[t] if self.direction == "lower" else bound < actual[t]
            risk[t] = bool(hit)
            # Gradient of the pinball loss, one coordinate. Moving the threshold
            # down on a risk event is the "lower" convention; "upper" flips it,
            # because there the bound must rise to stop under-promising.
            step = (self.eta * (1.0 - epsilon)) if hit else -(self.eta * epsilon)
            self._theta[label] = theta - step if self.direction == "lower" else theta + step

        self.trace = {"realised_risk_rate": float(np.mean(risk)) if n else float("nan"),
                      "n_groups": len(self._theta)}
        return bounds

    def summary(self) -> dict:
        return {"method": "gcaci", "eta": self.eta,
                "reference": "Ramalingam, Kiyani and Roth 2025, arXiv:2502.10947, Alg. 2",
                "axes": list(self.config.regime.axes),
                "epsilon": self.config.epsilon,
                **(self.trace or {})}


@dataclass
class RollingRC:
    """Rolling Risk Control, Feldman, Ringel, Bates and Romano, TMLR 2023.

    Their scheme maintains a scalar calibration parameter updated by

        theta_{t+1} = theta_t + lr * (risk_t - alpha)

    where `risk_t` is the realised value of the user-specified risk at step t,
    and the parameter is passed through a *stretching* function before being
    applied, so the correction can move quickly when the parameter is far from
    where it needs to be. The stretching is what distinguishes this from plain
    ACI: `phi(theta) = theta` near the origin and grows superlinearly outside
    it.

    Implemented for the one-sided overestimation risk this project controls,
    which is a 0/1 risk, so `risk_t - alpha` is the same signal ACI uses. The
    difference that matters here is the stretching function and the fact that
    the parameter lives in score units rather than quantile-level units.
    """

    config: CalibrationConfig = field(default_factory=CalibrationConfig)
    assigner: RegimeAssigner | None = None
    direction: str = "lower"
    learning_rate: float = 1.0
    #: Superlinear beyond this magnitude, in units of the calibration scale.
    stretch_knee: float = 1.0

    _theta: dict[str, float] = field(default_factory=dict, repr=False)
    _scale: float = field(default=1.0, repr=False)
    _fitted: bool = field(default=False, repr=False)
    trace: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        check_direction(self.direction)

    def _stretch(self, theta: float) -> float:
        """phi(theta): identity near zero, superlinear outside the knee.

        Feldman et al. use a stretching function so a parameter that is badly
        wrong recovers in a few steps rather than a few hundred. The exact form
        is a design choice in their paper too; this is the simplest one with the
        stated shape.
        """
        knee = self.stretch_knee
        if abs(theta) <= knee:
            return theta
        sign = 1.0 if theta > 0 else -1.0
        return sign * (knee + (abs(theta) - knee) ** 2 / knee)

    def fit(self, predicted_cal: np.ndarray, actual_cal: np.ndarray,
            covariates_cal: pd.DataFrame) -> RollingRC:
        predicted = np.asarray(predicted_cal, dtype=float).ravel()
        actual = np.asarray(actual_cal, dtype=float).ravel()
        residuals = actual - predicted
        labels = _labels(self.assigner, self.config, covariates_cal)
        epsilon = self.config.epsilon
        q = epsilon if self.direction == "lower" else 1.0 - epsilon

        # The parameter is in score units, so the step size has to be too.
        self._scale = float(np.std(residuals)) if residuals.size else 1.0
        if not np.isfinite(self._scale) or self._scale <= 0:
            self._scale = 1.0

        pooled = float(np.quantile(residuals, q)) if residuals.size else 0.0
        self._theta = {}
        for label in set(labels.tolist()):
            part = residuals[labels == label]
            self._theta[str(label)] = (float(np.quantile(part, q))
                                       if part.size >= 30 else pooled)
        self._theta.setdefault(GLOBAL_LABEL, pooled)
        self._fitted = True
        return self

    def transform_online(self, predicted: np.ndarray, actual: np.ndarray,
                         covariates: pd.DataFrame, *, floor: float = 0.0) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RollingRC.fit must run before transform_online")
        predicted = np.asarray(predicted, dtype=float).ravel()
        actual = np.asarray(actual, dtype=float).ravel()
        labels = _labels(self.assigner, self.config, covariates)
        epsilon = self.config.epsilon
        n = predicted.size
        bounds = np.empty(n, dtype=float)
        risk = np.zeros(n, dtype=bool)
        step = self.learning_rate * self._scale * 0.05

        for t in range(n):
            label = str(labels[t])
            theta = self._theta.get(label)
            if theta is None:
                theta = self._theta[GLOBAL_LABEL]
                self._theta[label] = theta
            applied = self._stretch(theta / self._scale) * self._scale
            bound = max(float(predicted[t]) + applied, floor)
            bounds[t] = bound

            hit = bound > actual[t] if self.direction == "lower" else bound < actual[t]
            risk[t] = bool(hit)
            delta = step * (float(hit) - epsilon)
            self._theta[label] = theta - delta if self.direction == "lower" else theta + delta

        self.trace = {"realised_risk_rate": float(np.mean(risk)) if n else float("nan"),
                      "n_groups": len(self._theta), "scale": round(self._scale, 4)}
        return bounds

    def summary(self) -> dict:
        return {"method": "rolling_rc", "learning_rate": self.learning_rate,
                "stretch_knee": self.stretch_knee,
                "reference": "Feldman, Ringel, Bates and Romano, TMLR 2023, arXiv:2205.09095",
                "axes": list(self.config.regime.axes),
                "epsilon": self.config.epsilon,
                **(self.trace or {})}
