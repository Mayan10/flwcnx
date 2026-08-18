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

#: Which way a bound is unsafe.
#:
#:   "lower"  the bound sits below the actual, as for throughput. The risk is
#:            predicting *above* the truth, which over-allocates and drops
#:            sessions. This is the BG-CFQS setting and OverRate is its metric.
#:
#:   "upper"  the bound sits above the actual, as for latency. The risk is
#:            predicting *below* the truth, which promises a delay the link will
#:            not meet. UnderRate is its metric.
#:
#: The two are mirror images and every function below takes the direction
#: rather than assuming throughput, because the supplied dataset's usable
#: target is latency (docs/supplied-dataset.md).
DIRECTIONS = ("lower", "upper")


def check_direction(direction: str) -> str:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return direction


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
    direction: str = "lower"
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        out = {
            "n": self.n, "MAE": self.mae, "RMSE": self.rmse,
            "OverRate": self.over_rate, "MPE": self.mpe, "P95+Err": self.p95_pos_err,
            "mean_actual": self.mean_actual, "mean_predicted": self.mean_predicted,
            "direction": self.direction,
        }
        out.update(self.extras)
        return out

    @property
    def risk_rate(self) -> float:
        """The rate of unsafe predictions, whichever way unsafe runs.

        Kept named OverRate in `to_dict` even on the upper-bound path, so our
        tables stay column-compatible with the BG-CFQS ones. The `direction`
        field is what says which quantity it is, and it travels with the row.
        """
        return self.over_rate

    def risk_pass(self, epsilon: float) -> bool:
        """Whether the risk budget was held."""
        return bool(self.over_rate <= epsilon)


def _clean(predicted: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if predicted.shape != actual.shape:
        raise ValueError(f"shape mismatch: predicted {predicted.shape}, actual {actual.shape}")
    finite = np.isfinite(predicted) & np.isfinite(actual)
    return predicted[finite], actual[finite]


def over_rate(predicted: np.ndarray, actual: np.ndarray,
              direction: str = "lower") -> float:
    """Fraction of predictions on the unsafe side of the actual.

    On a lower bound that is `predicted > actual`; on an upper bound it is
    `predicted < actual`.

    Strict either way, so a prediction exactly equal to the actual is never
    counted as unsafe. With continuous data this is a measure zero distinction,
    but it matters on the calibration path where the bound is set to an
    observed residual and ties are therefore common.
    """
    check_direction(direction)
    predicted, actual = _clean(predicted, actual)
    if predicted.size == 0:
        return float("nan")
    return float(np.mean(predicted > actual) if direction == "lower"
                 else np.mean(predicted < actual))


def under_rate(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of predictions strictly below the actual. The upper-bound risk."""
    return over_rate(predicted, actual, direction="upper")


def _unsafe_error(predicted: np.ndarray, actual: np.ndarray, direction: str) -> np.ndarray:
    """The unsafe component of the error, clipped at zero."""
    return (np.maximum(predicted - actual, 0.0) if direction == "lower"
            else np.maximum(actual - predicted, 0.0))


def mean_positive_error(predicted: np.ndarray, actual: np.ndarray,
                        direction: str = "lower") -> float:
    check_direction(direction)
    predicted, actual = _clean(predicted, actual)
    if predicted.size == 0:
        return float("nan")
    return float(np.mean(_unsafe_error(predicted, actual, direction)))


def p95_positive_error(predicted: np.ndarray, actual: np.ndarray,
                       positive_only: bool = False,
                       direction: str = "lower") -> float:
    """95th percentile of max(predicted - actual, 0).

    Taken over all samples by default, which is how a budget of 0.35 makes the
    statistic meaningful: with a third of predictions overestimating, the 95th
    percentile of the clipped error is well inside the positive region.
    `positive_only` restricts to the overestimating samples for anyone who
    reads the definition the other way.
    """
    check_direction(direction)
    predicted, actual = _clean(predicted, actual)
    if predicted.size == 0:
        return float("nan")
    positive = _unsafe_error(predicted, actual, direction)
    if positive_only:
        positive = positive[positive > 0]
        if positive.size == 0:
            return 0.0
    return float(np.percentile(positive, 95))


def compute_metrics(predicted: np.ndarray, actual: np.ndarray,
                    direction: str = "lower") -> MetricSet:
    """All five metrics at once, on whichever side is the unsafe one."""
    check_direction(direction)
    predicted, actual = _clean(predicted, actual)
    if predicted.size == 0:
        nan = float("nan")
        return MetricSet(0, nan, nan, nan, nan, nan, direction=direction)
    error = predicted - actual
    unsafe = _unsafe_error(predicted, actual, direction)
    return MetricSet(
        n=int(predicted.size),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error ** 2))),
        over_rate=over_rate(predicted, actual, direction),
        mpe=float(np.mean(unsafe)),
        p95_pos_err=float(np.percentile(unsafe, 95)),
        mean_actual=float(np.mean(actual)),
        mean_predicted=float(np.mean(predicted)),
        direction=direction,
    )


def risk_slice_mask(actual: np.ndarray, fraction: float,
                    direction: str = "lower") -> np.ndarray:
    """Boolean mask for the riskiest `fraction` of true values.

    Which tail is risky follows the direction. For throughput it is the
    *lowest* capacity, where over-allocation drops sessions. For latency it is
    the *highest* delay, where under-promising breaks the service. Taking the
    low tail on a latency target would slice out the easiest samples and
    report them as the hard case, which is the sort of error that produces an
    excellent looking table.

    Uses a quantile rather than a fixed cut, so the slice means the same thing
    across sites with different baselines.
    """
    check_direction(direction)
    actual = np.asarray(actual, dtype=float).ravel()
    finite = np.isfinite(actual)
    if not finite.any():
        return np.zeros_like(actual, dtype=bool)
    if direction == "lower":
        return finite & (actual <= np.quantile(actual[finite], fraction))
    return finite & (actual >= np.quantile(actual[finite], 1.0 - fraction))


def conditional_metrics(predicted: np.ndarray, actual: np.ndarray,
                        slices: dict[str, float] | None = None,
                        direction: str = "lower") -> dict[str, MetricSet]:
    """Metrics globally and on each risk slice.

    This is the table that motivates the project. BG-CFQS hold a global
    OverRate of 0.349 against a 0.35 budget and then report 0.65 to 0.71 on
    P30 and 0.83 to 0.86 on P10. The global number is not wrong, it is just
    not the number an allocator cares about.
    """
    slices = slices or RISK_SLICES
    out = {"global": compute_metrics(predicted, actual, direction)}
    for name, fraction in slices.items():
        mask = risk_slice_mask(actual, fraction, direction)
        out[name] = compute_metrics(np.asarray(predicted).ravel()[mask],
                                    np.asarray(actual).ravel()[mask], direction)
    return out


def per_regime_metrics(predicted: np.ndarray, actual: np.ndarray,
                       regimes: np.ndarray, min_samples: int = 1,
                       direction: str = "lower") -> pd.DataFrame:
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
        row.update(compute_metrics(predicted[mask], actual[mask], direction).to_dict())
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["regime", "n", "MAE", "RMSE", "OverRate", "MPE", "P95+Err"])
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def worst_regime_over_rate(predicted: np.ndarray, actual: np.ndarray,
                           regimes: np.ndarray, min_samples: int = 30,
                           direction: str = "lower") -> float:
    """The worst OverRate any adequately sized regime suffers.

    A single headline number for the thing our layer exists to fix. A global
    OverRate at budget with a worst regime far above it is exactly the failure
    mode BG-CFQS exhibit, and driving this number down without wrecking MAE is
    the result we are after.
    """
    table = per_regime_metrics(predicted, actual, regimes, min_samples=min_samples,
                               direction=direction)
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
