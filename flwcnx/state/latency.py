"""Period-level latency classification, following Casparsen et al.

Requirement 1 of the brief, "predict latency spikes", and the first one it
names. The StarNet traces carry latency alongside throughput, which the project
brief did not expect, so this is runnable on the same data as everything else.

**The framing is Casparsen et al.** (arXiv:2601.08439, npj Wireless Technology
2026), reproduced not invented. They segment the link into 15-second scheduling
periods and label each one:

    Good       at least `good_fraction` of packets meet the latency threshold
    Degraded   otherwise

with a threshold of l_t = 50 ms applied to the 99th latency quantile within the
period, which they justify as typical for cyber-physical systems and 5G NR
satellite access. Both numbers are configurable here and both default to theirs.

**What does not transfer, and is not attempted.** Their sampling rate is 500 Hz,
7,500 probes per period. The StarNet traces are 1 Hz, so a period holds fifteen
samples. Consequences, all of them stated rather than worked around:

- A 99th quantile over fifteen samples is the maximum. The classifier is
  therefore closer to "did any sample in this period exceed the threshold" than
  to a real tail quantile, and the module reports the effective sample count so
  this is visible in the output rather than buried.
- Their boundary regions, the first 140 ms and last 75 ms of each period
  carrying a handover spike averaging 74 ms above the period mean, cannot be
  resolved at 1 Hz at all. `boundary_mask` in `state/phase.py` refuses rates
  that cannot see them, and nothing here claims those numbers.

What the 1 Hz data does support is the period-level Good/Degraded label itself,
which is what a scheduler or an admission controller would act on, and a
risk-controlled *upper* bound on latency, which is the same calibration layer
used for throughput with the direction flipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flwcnx.config import LATENCY_COL, PERIOD_SECONDS, TIME_COL
from flwcnx.timeutil import to_epoch_seconds

#: Casparsen et al.'s threshold. 50 ms on the 99th latency quantile, described
#: there as typical for cyber-physical systems and 5G NR satellite access.
DEFAULT_LATENCY_THRESHOLD_MS: float = 50.0

#: Their Good criterion: the period is Good when at least this fraction of
#: packets meet the threshold.
DEFAULT_GOOD_FRACTION: float = 0.99


@dataclass(frozen=True)
class PeriodLabels:
    """One row per 15 s scheduling period, with the evidence for its label."""

    frame: pd.DataFrame
    threshold_ms: float
    good_fraction: float
    samples_per_period: float

    @property
    def degraded_rate(self) -> float:
        return float(self.frame["degraded"].mean()) if len(self.frame) else float("nan")

    def summary(self) -> dict:
        return {
            "n_periods": int(len(self.frame)),
            "degraded_rate": round(self.degraded_rate, 6),
            "threshold_ms": self.threshold_ms,
            "good_fraction": self.good_fraction,
            "median_samples_per_period": round(self.samples_per_period, 2),
            # Below about 100 samples a 99th quantile is really the maximum.
            # Reported rather than silently accepted.
            "quantile_is_effectively_max": bool(self.samples_per_period < 100),
            "reference": "Casparsen et al., arXiv:2601.08439",
        }


def label_periods(
    frame: pd.DataFrame,
    *,
    column: str = LATENCY_COL,
    threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
    good_fraction: float = DEFAULT_GOOD_FRACTION,
    period_seconds: int = PERIOD_SECONDS,
    phase_offset: float = 0.0,
) -> PeriodLabels:
    """Segment into scheduling periods and label each Good or Degraded.

    `phase_offset` aligns the period boundaries to the recovered scheduling
    phase. It matters: periods cut on arbitrary boundaries straddle two
    scheduling intervals and mix a handover spike into two labels instead of
    one. Pass the offset from `state.phase.recover_phase`.
    """
    if column not in frame.columns:
        raise KeyError(f"{column!r} not in frame; this trace carries no latency")
    work = frame[[TIME_COL, column]].dropna().copy()
    if work.empty:
        raise ValueError("no latency samples after dropping nulls")

    # Via the helper, not astype("int64"): pandas 2 carries datetime64 at any
    # of four resolutions and the naive conversion is only correct at ns. See
    # flwcnx/timeutil.py, and the thirteen CI failures that found it.
    seconds = to_epoch_seconds(work[TIME_COL])
    work["period"] = np.floor((seconds - phase_offset) / period_seconds).astype("int64")

    grouped = work.groupby("period")[column]
    # The Good criterion is stated on the fraction meeting the threshold, so it
    # is computed that way rather than via a quantile. At 1 Hz the two coincide
    # for good_fraction = 0.99, and the fraction form degrades gracefully.
    meeting = grouped.apply(lambda s: float((s <= threshold_ms).mean()))
    labels = pd.DataFrame({
        "period": meeting.index,
        "fraction_meeting": meeting.to_numpy(),
        "n_samples": grouped.size().to_numpy(),
        "latency_max_ms": grouped.max().to_numpy(),
        "latency_median_ms": grouped.median().to_numpy(),
        "latency_p99_ms": grouped.quantile(0.99).to_numpy(),
    })
    labels["degraded"] = labels["fraction_meeting"] < good_fraction
    labels["period_start_s"] = labels["period"] * period_seconds + phase_offset

    return PeriodLabels(
        frame=labels.reset_index(drop=True),
        threshold_ms=threshold_ms,
        good_fraction=good_fraction,
        samples_per_period=float(labels["n_samples"].median()),
    )


@dataclass(frozen=True)
class SpikeMetrics:
    """Classification quality for the Degraded class, which is the positive."""

    precision: float
    recall: float
    f1: float
    auprc: float
    baseline_auprc: float      # the positive rate, what a coin flip scores
    lift: float                # auprc over baseline
    tp: int
    fp: int
    fn: int
    tn: int
    n: int

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4), "recall": round(self.recall, 4),
            "f1": round(self.f1, 4), "auprc": round(self.auprc, 4),
            "baseline_auprc": round(self.baseline_auprc, 4),
            "lift": round(self.lift, 3),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn, "n": self.n,
        }


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, by the step-wise definition.

    AUPRC rather than AUROC because the positive class is the rare and
    expensive one, and AUROC is optimistic under class imbalance. Casparsen et
    al. report AUPRC for the same reason.

    Implemented directly rather than via sklearn so this module has no
    dependency beyond numpy, matching the rest of the project.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    if labels.size == 0 or not labels.any():
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, hits.size + 1)
    # Sum of precision at each true positive, over the number of positives.
    return float(precision[hits].sum() / labels.sum())


def spike_metrics(predicted_degraded: np.ndarray, actual_degraded: np.ndarray,
                  scores: np.ndarray | None = None) -> SpikeMetrics:
    """Precision, recall, F1 and AUPRC for the Degraded class.

    `scores` is a continuous ranking (a predicted latency, or a bound) used for
    AUPRC. Without it AUPRC is computed from the binary prediction, which is a
    degenerate curve and is reported as such rather than omitted.
    """
    predicted = np.asarray(predicted_degraded).astype(bool)
    actual = np.asarray(actual_degraded).astype(bool)
    if predicted.size != actual.size:
        raise ValueError("predicted and actual must have the same length")

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    ranking = np.asarray(scores, dtype=float) if scores is not None else predicted.astype(float)
    auprc = average_precision(ranking, actual)
    baseline = float(actual.mean()) if actual.size else float("nan")
    return SpikeMetrics(
        precision=precision, recall=recall, f1=f1, auprc=auprc,
        baseline_auprc=baseline,
        lift=(auprc / baseline) if baseline and np.isfinite(auprc) else float("nan"),
        tp=tp, fp=fp, fn=fn, tn=tn, n=int(actual.size),
    )


def degraded_from_bound(bound_ms: np.ndarray,
                        threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS) -> np.ndarray:
    """Predict Degraded when the calibrated upper bound breaches the threshold.

    This is the latency analogue of the congestion rule on the throughput path:
    the decision is read off the calibrated bound rather than from a second
    trained classifier. It inherits the bound's risk guarantee, so the operating
    point is set by the risk budget rather than by tuning a decision threshold,
    which is the property that makes it worth doing this way.
    """
    return np.asarray(bound_ms, dtype=float) > threshold_ms
