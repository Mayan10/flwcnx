"""Metric definitions.

Follows the BG-CFQS definitions exactly (Xie et al., section 5.2) so our
numbers sit in the same table as theirs without a footnote:

  MAE       mean absolute error
  RMSE      root mean squared error
  OverRate  fraction of predictions strictly greater than the actual
  MPE       mean positive error, mean of max(predicted - actual, 0)
  P95+Err   95th percentile of the positive error component

OverRate is the one that matters. A prediction above the actual is an
over allocation, and an over allocation is what drops sessions. Predicting low
costs utilisation; predicting high costs service.

Every metric is computed globally, per regime, and on two risk slices:

  High-risk P30    the lowest 30% of true throughput
  Severe-risk P10  the lowest 10%

The slices are defined on the *true* values, so they are an evaluation device
and not something the predictor can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: The two conditional slices BG-CFQS report and where our contribution lives.
RISK_SLICES: dict[str, float] = {"P30": 0.30, "P10": 0.10}


@dataclass
class MetricSet:
    """Metrics for one set of predictions."""

    n: int
    mae: float
    rmse: float
    over_rate: float
    mpe: float
    p95_pos_err: float
    mean_actual: float = float("nan")
    mean_predicted: float = float("nan")
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        out = {
            "n": self.n, "MAE": self.mae, "RMSE": self.rmse,
            "OverRate": self.over_rate, "MPE": self.mpe, "P95+Err": self.p95_pos_err,
            "mean_actual": self.mean_actual, "mean_predicted": self.mean_predicted,
        }
        out.update(self.extras)
        return out

    def risk_pass(self, epsilon: float) -> bool:
        """Whether the overestimation budget was held."""
        return bool(self.over_rate <= epsilon)


def _clean(predicted: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if predicted.shape != actual.shape:
        raise ValueError(f"shape mismatch: predicted {predicted.shape}, actual {actual.shape}")
    finite = np.isfinite(predicted) & np.isfinite(actual)
    return predicted[finite], actual[finite]


def over_rate(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of predictions strictly above the actual.

    Strict, so a prediction exactly equal to the actual is not counted as an
    overestimate. With continuous throughput this is a measure zero
    distinction, but it matters on the calibration path where the bound is set
    to an observed residual and ties are therefore common.
    """
    predicted, actual = _clean(predicted, actual)
    if predicted.size == 0:
        return float("nan")
    return float(np.mean(predicted > actual))


def mean_positive_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted, actual = _clean(predicted, actual)
    if predicted.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(predicted - actual, 0.0)))


def p95_positive_error(predicted: np.ndarray, actual: np.ndarray,
                       positive_only: bool = False) -> float:
    """95th percentile of max(predicted - actual, 0).

    Taken over all samples by default, which is how a budget of 0.35 makes the
    statistic meaningful: with a third of predictions overestimating, the 95th
    percentile of the clipped error is well inside the positive region.
    `positive_only` restricts to the overestimating samples for anyone who
    reads the definition the other way.
    """
    predicted, actual = _clean(predicted, actual)
    if predicted.size == 0:
        return float("nan")
    positive = np.maximum(predicted - actual, 0.0)
    if positive_only:
        positive = positive[positive > 0]
        if positive.size == 0:
            return 0.0
    return float(np.percentile(positive, 95))


def compute_metrics(predicted: np.ndarray, actual: np.ndarray) -> MetricSet:
    """All five metrics at once."""
    predicted, actual = _clean(predicted, actual)
    if predicted.size == 0:
        nan = float("nan")
        return MetricSet(0, nan, nan, nan, nan, nan)
    error = predicted - actual
    return MetricSet(
        n=int(predicted.size),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error ** 2))),
        over_rate=float(np.mean(predicted > actual)),
        mpe=float(np.mean(np.maximum(error, 0.0))),
        p95_pos_err=float(np.percentile(np.maximum(error, 0.0), 95)),
        mean_actual=float(np.mean(actual)),
        mean_predicted=float(np.mean(predicted)),
    )


def risk_slice_mask(actual: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask for the lowest `fraction` of true throughput.

    Uses a quantile on the actual values rather than a fixed Mbps cut, so the
    slice means the same thing across locations with different capacity.
    """
    actual = np.asarray(actual, dtype=float).ravel()
    finite = np.isfinite(actual)
    if not finite.any():
        return np.zeros_like(actual, dtype=bool)
    threshold = np.quantile(actual[finite], fraction)
    return finite & (actual <= threshold)


def conditional_metrics(predicted: np.ndarray, actual: np.ndarray,
                        slices: dict[str, float] | None = None) -> dict[str, MetricSet]:
    """Metrics globally and on each risk slice.

    This is the table that motivates the project. BG-CFQS hold a global
    OverRate of 0.349 against a 0.35 budget and then report 0.65 to 0.71 on
    P30 and 0.83 to 0.86 on P10. The global number is not wrong, it is just
    not the number an allocator cares about.
    """
    slices = slices or RISK_SLICES
    out = {"global": compute_metrics(predicted, actual)}
    for name, fraction in slices.items():
        mask = risk_slice_mask(actual, fraction)
        out[name] = compute_metrics(np.asarray(predicted).ravel()[mask],
                                    np.asarray(actual).ravel()[mask])
    return out


def per_regime_metrics(predicted: np.ndarray, actual: np.ndarray,
                       regimes: np.ndarray, min_samples: int = 1) -> pd.DataFrame:
    """Metrics broken out by regime label, largest regime first."""
    predicted = np.asarray(predicted, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    regimes = np.asarray(regimes).ravel()

    rows = []
    for label in pd.unique(regimes):
        mask = regimes == label
        if mask.sum() < min_samples:
            continue
        row = {"regime": label}
        row.update(compute_metrics(predicted[mask], actual[mask]).to_dict())
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["regime", "n", "MAE", "RMSE", "OverRate", "MPE", "P95+Err"])
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def worst_regime_over_rate(predicted: np.ndarray, actual: np.ndarray,
                           regimes: np.ndarray, min_samples: int = 30) -> float:
    """The worst OverRate any adequately sized regime suffers.

    A single headline number for the thing our layer exists to fix. A global
    OverRate at budget with a worst regime far above it is exactly the failure
    mode BG-CFQS exhibit, and driving this number down without wrecking MAE is
    the result we are after.
    """
    table = per_regime_metrics(predicted, actual, regimes, min_samples=min_samples)
    if table.empty:
        return float("nan")
    return float(table["OverRate"].max())


def summarise(results: dict[str, MetricSet], epsilon: float | None = None) -> pd.DataFrame:
    """Turn a slice name to MetricSet mapping into a reportable table."""
    rows = []
    for name, metrics in results.items():
        row = {"slice": name}
        row.update(metrics.to_dict())
        if epsilon is not None:
            row["risk_pass"] = metrics.risk_pass(epsilon)
        rows.append(row)
    return pd.DataFrame(rows)
