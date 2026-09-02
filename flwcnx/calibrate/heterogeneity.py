"""Should the calibration layer condition on regimes at all?

The measured motivation. Conditioning helps only when the regimes actually
differ, and across four datasets the spread of the fitted per-regime offset
predicts the sign of the effect on tail risk:

    StarNet USA          1.85 Mbps spread   conditioning costs 6.1%
    StarNet Germany      5.44 Mbps          neutral
    StarNet Canada      11.30 Mbps          gains 1.7%
    WetLinks Osnabruck   9.28 Mbps          gains 5.6%

On the US trace the four regimes want offsets of -10.69, -11.91, -10.06 and
-10.99 Mbps. They are the same number three times over, so conditioning buys
nothing and pays estimation noise for it.

**Attribution, and it matters here.** None of this is a new idea and the
implementation deliberately uses the standard tool rather than an invented
threshold:

- Deciding whether to pool by testing whether between-group variance exceeds
  sampling variance is **Cochran's Q** (1954), with **I-squared** (Higgins and
  Thompson 2002) as the effect-size companion. Both come from meta-analysis and
  are roughly as old as the field.
- Doing this for *conformal* calibration specifically is **Clustered Conformal
  Prediction** (Ding et al., NeurIPS 2023, arXiv:2306.09335), which measures
  per-group calibration quantiles and pools groups whose quantiles are similar.
  CCP produces a clustering with k between 1 and K; the gate here is the
  degenerate binary case of that, k = 1 or k = K.
- **AFCP** (Zhou and Sesia, NeurIPS 2024) also selects groups from data, but by
  a different statistic: it tests whether any group's *miscoverage* exceeds the
  budget, a question about level. This module tests whether the groups' offsets
  *differ from each other*, a question about dispersion. Regimes can all sit
  inside budget and still differ, or all sit outside it and not differ.

So the contribution here is applying a standard heterogeneity test to one-sided
conformal risk control under temporal drift, not the test and not the idea of
data-driven conditioning. See `docs/novelty-review.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flwcnx.calibrate.adaptive import offset_for_alpha
from flwcnx.config import CalibrationConfig
from flwcnx.eval.metrics import check_direction
from flwcnx.state.regime import RegimeAssigner

#: Regimes with fewer calibration points than this do not get a vote. Their
#: offset standard error is enormous, and Q weights by inverse variance, so
#: they contribute almost nothing anyway; excluding them keeps the degrees of
#: freedom honest rather than inflating k with noise.
MIN_REGIME_FOR_TEST = 30


@dataclass(frozen=True)
class HeterogeneityTest:
    """The result of asking whether per-regime offsets differ at all."""

    q: float                 # Cochran's Q
    dof: int                 # k - 1
    p_value: float
    i_squared: float         # share of spread not attributable to sampling error
    spread: float            # max - min of the fitted offsets, in target units
    n_regimes: int
    offsets: dict[str, float] = field(default_factory=dict)
    standard_errors: dict[str, float] = field(default_factory=dict)
    #: The decision, and the reason for it, so a log line explains itself.
    condition: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "q": round(self.q, 4), "dof": self.dof,
            "p_value": round(self.p_value, 6),
            "i_squared": round(self.i_squared, 4),
            "spread": round(self.spread, 4),
            "n_regimes": self.n_regimes,
            "condition": self.condition, "reason": self.reason,
            "offsets": {k: round(v, 4) for k, v in self.offsets.items()},
        }


def _chi2_sf(x: float, dof: int) -> float:
    """Upper tail of a chi-squared, without pulling in scipy.

    Uses the regularised upper incomplete gamma Q(dof/2, x/2) via a Lentz
    continued fraction, falling back to the series for small x. scipy is not a
    dependency of this project and adding one for a single tail probability
    would be a poor trade.
    """
    if dof <= 0:
        return 1.0
    if x <= 0:
        return 1.0
    a, z = dof / 2.0, x / 2.0

    if z < a + 1.0:
        # Series for the lower regularised gamma P(a, z), then complement.
        term = 1.0 / a
        total = term
        for n in range(1, 1000):
            term *= z / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        lower = total * math.exp(-z + a * math.log(z) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - lower))

    # Continued fraction for the upper regularised gamma Q(a, z).
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    upper = h * math.exp(-z + a * math.log(z) - math.lgamma(a))
    return max(0.0, min(1.0, upper))


def offset_standard_error(residuals: np.ndarray, alpha: float) -> float:
    """Standard error of an empirical quantile, by the Siddiqui formula.

        se = sqrt(alpha (1 - alpha) / n) / f(q_alpha)

    The density at the quantile is estimated from the spacing of the order
    statistics around it, which avoids choosing a kernel bandwidth. This is the
    weight Cochran's Q needs: without it, a regime with 40 points and one with
    4,000 would count equally and Q would find heterogeneity everywhere.
    """
    values = np.sort(np.asarray(residuals, dtype=float))
    n = values.size
    if n < MIN_REGIME_FOR_TEST:
        return float("inf")
    # Bandwidth in rank space, the usual n^(-1/5) rate, kept inside the array.
    h = max(int(round(n ** 0.8 / 2.0)), 2)
    index = int(np.clip(round(alpha * (n - 1)), 0, n - 1))
    lo, hi = max(index - h, 0), min(index + h, n - 1)
    if hi <= lo:
        return float("inf")
    # Density from the empirical spacing: (F(hi) - F(lo)) / (x_hi - x_lo).
    span = values[hi] - values[lo]
    if span <= 0:
        return float("inf")
    density = ((hi - lo) / (n - 1)) / span
    if density <= 0:
        return float("inf")
    return float(math.sqrt(alpha * (1.0 - alpha) / n) / density)


def cochran_q(offsets: np.ndarray, standard_errors: np.ndarray) -> tuple[float, int, float, float]:
    """Cochran's Q, its degrees of freedom, the p-value, and I-squared.

        w_i = 1 / se_i^2
        Q   = sum_i w_i (theta_i - theta_bar_w)^2  ~ chi^2_{k-1} under H0
        I^2 = max(0, (Q - (k - 1)) / Q)

    H0 is that every regime shares one offset and the observed differences are
    sampling error. This is exactly "measure the spread against its own
    estimation noise", with a null distribution attached.
    """
    offsets = np.asarray(offsets, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    finite = np.isfinite(offsets) & np.isfinite(standard_errors) & (standard_errors > 0)
    offsets, standard_errors = offsets[finite], standard_errors[finite]
    k = offsets.size
    if k < 2:
        return 0.0, 0, 1.0, 0.0

    weights = 1.0 / standard_errors ** 2
    weighted_mean = float(np.sum(weights * offsets) / np.sum(weights))
    q = float(np.sum(weights * (offsets - weighted_mean) ** 2))
    dof = k - 1
    p = _chi2_sf(q, dof)
    i_squared = max(0.0, (q - dof) / q) if q > 0 else 0.0
    return q, dof, p, i_squared


# Named `assess_` rather than `test_`: a public function whose name starts with
# `test_` is collected as a test case by pytest in any module that imports it,
# which is exactly what happened the first time this was written.
def assess_regime_heterogeneity(
    predicted: np.ndarray,
    actual: np.ndarray,
    covariates: pd.DataFrame,
    config: CalibrationConfig,
    assigner: RegimeAssigner | None = None,
    *,
    direction: str = "lower",
    alpha_level: float = 0.05,
    min_i_squared: float = 0.25,
) -> HeterogeneityTest:
    """Do the regimes differ enough to be worth conditioning on?

    Runs on the **calibration split only**, so the answer is available before
    any test decision is made. That is the property that makes this a usable
    gate rather than a post-hoc explanation.

    Two conditions have to hold, and requiring both is deliberate. The p-value
    answers "is the difference real", which with tens of thousands of
    calibration points will eventually be yes for a difference of no practical
    size. `I^2` answers "is it big enough to matter", and is the guard against
    a significant but useless split.
    """
    check_direction(direction)
    predicted = np.asarray(predicted, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    residuals = actual - predicted
    epsilon = config.epsilon

    if not config.regime.axes:
        return HeterogeneityTest(0.0, 0, 1.0, 0.0, 0.0, 1, condition=False,
                                 reason="no regime axes configured")

    assigner = assigner or RegimeAssigner(config.regime)
    labels = np.asarray(assigner.assign(covariates), dtype=object)

    offsets: dict[str, float] = {}
    errors: dict[str, float] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        part = residuals[mask]
        if part.size < MIN_REGIME_FOR_TEST:
            continue
        offsets[str(label)] = offset_for_alpha(part, epsilon, direction)
        errors[str(label)] = offset_standard_error(part, epsilon)

    if len(offsets) < 2:
        return HeterogeneityTest(0.0, 0, 1.0, 0.0, 0.0, len(offsets), offsets, errors,
                                 condition=False,
                                 reason=f"only {len(offsets)} regimes cleared "
                                        f"{MIN_REGIME_FOR_TEST} calibration points")

    values = np.array(list(offsets.values()))
    ses = np.array(list(errors.values()))
    q, dof, p, i_squared = cochran_q(values, ses)
    spread = float(values.max() - values.min())

    significant = p < alpha_level
    substantial = i_squared >= min_i_squared
    condition = bool(significant and substantial)
    if condition:
        reason = (f"Q={q:.1f} on {dof} dof, p={p:.2g}, I2={i_squared:.2f}: regimes "
                  f"differ by more than sampling error, spread {spread:.2f}")
    elif not significant:
        reason = (f"Q={q:.1f} on {dof} dof, p={p:.2g}: offsets are within sampling "
                  f"error of each other, spread only {spread:.2f}")
    else:
        reason = (f"I2={i_squared:.2f} below {min_i_squared}: difference is real but "
                  f"too small to pay estimation noise for, spread {spread:.2f}")
    return HeterogeneityTest(q, dof, p, i_squared, spread, len(offsets),
                             offsets, errors, condition, reason)


@dataclass
class GatedCalibrator:
    """Conditions on regimes only when the calibration split says it is worth it.

    Wraps any calibrator exposing `fit(predicted, actual, covariates)` and
    either `transform(predicted, covariates)` or
    `transform_online(predicted, actual, covariates)`. On a negative test the
    regime axes are dropped and the same calibrator runs globally, so the
    fallback costs one refit and never leaves the layer unconfigured.
    """

    config: CalibrationConfig = field(default_factory=CalibrationConfig)
    assigner: RegimeAssigner | None = None
    direction: str = "lower"
    alpha_level: float = 0.05
    min_i_squared: float = 0.25
    #: Built lazily so the gate can hand it either the regime axes or none.
    factory: object = None

    test: HeterogeneityTest | None = field(default=None, repr=False)
    inner: object = field(default=None, repr=False)
    used_axes: tuple[str, ...] = field(default=(), repr=False)

    def fit(self, predicted, actual, covariates) -> GatedCalibrator:
        from dataclasses import replace as _replace

        self.test = assess_regime_heterogeneity(
            predicted, actual, covariates, self.config, self.assigner,
            direction=self.direction, alpha_level=self.alpha_level,
            min_i_squared=self.min_i_squared,
        )
        if self.test.condition:
            config, assigner = self.config, self.assigner
            self.used_axes = tuple(self.config.regime.axes)
        else:
            config = _replace(self.config,
                              regime=_replace(self.config.regime, axes=()))
            assigner = None
            self.used_axes = ()

        assert self.factory is not None, "GatedCalibrator needs a factory"
        self.inner = self.factory(config, assigner)          # type: ignore[operator]
        self.inner.fit(predicted, actual, covariates)        # type: ignore[attr-defined]
        return self

    def transform(self, predicted, covariates):
        return self.inner.transform(predicted, covariates)   # type: ignore[attr-defined]

    def transform_online(self, predicted, actual, covariates):
        return self.inner.transform_online(                  # type: ignore[attr-defined]
            predicted, actual, covariates)

    def summary(self) -> dict:
        out = {"method": "gated", "gate": self.test.to_dict() if self.test else {},
               "used_axes": list(self.used_axes)}
        if self.inner is not None and hasattr(self.inner, "summary"):
            out["inner"] = self.inner.summary()              # type: ignore[attr-defined]
        return out
