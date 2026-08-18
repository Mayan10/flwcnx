"""15 second scheduling phase recovery.

Reproduction of the procedure in Casparsen et al. (arXiv 2601.08439), section
III. Not ours.

Starlink reschedules on a 15 second cadence, and the published reference is
that the rescheduling points land on the 12th, 27th, 42nd and 57th second of
each minute. That is a measured constant, not a protocol guarantee, so it is
not hardcoded: the offset is recovered from the signal and the fixed value is
only a fallback when recovery is not confident.

Recovery, following the paper:

  1. first difference the signal;
  2. threshold it to get candidate rescheduling edges;
  3. enforce a minimum spacing of T = 15 s between accepted edges;
  4. histogram the accepted edge times over phase bins;
  5. take the dominant bin;
  6. weighted circular mean over the top-k bins in its neighbourhood.

Note on sampling rate. Casparsen probe at 500 Hz (2 ms, S = 7500 samples per
period) and analyse a 140 ms opening and 75 ms closing boundary region. The
StarNet traces are 1 Hz. The phase recovery and the period level framing carry
over; the boundary numbers do not, and `boundary_mask` refuses to pretend
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flwcnx.config import PERIOD_SECONDS, TARGET_COL, TIME_COL

# 12, 27, 42 and 57 are all congruent to 12 modulo 15, so the published
# rescheduling points reduce to a single phase offset.
FIXED_PHASE_OFFSET: float = 12.0

# Casparsen section III-B. Reported for 500 Hz latency probes. Kept here so the
# constant is documented, and guarded so it is never applied to 1 Hz data.
BOUNDARY_OPEN_SECONDS: float = 0.140
BOUNDARY_CLOSE_SECONDS: float = 0.075
BOUNDARY_EXCESS_MS: float = 74.0    # mean latency above the period mean

# Their period level threshold: Good if at least 99% of packets meet l_t, and
# l_t = 50 ms on the 99th latency quantile, which is typical for cyber physical
# systems and 5G NR satellite access.
GOOD_PERIOD_QUANTILE: float = 0.99
GOOD_LATENCY_MS: float = 50.0


@dataclass(frozen=True)
class PhaseReference:
    """The recovered scheduling phase and the evidence behind it."""

    offset_seconds: float
    confidence: float          # dominant bin mass divided by the uniform expectation
    n_edges: int
    method: str                # "recovered" or "fallback"
    histogram: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    bin_edges: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def is_recovered(self) -> bool:
        return self.method == "recovered"


def _epoch_seconds(times: pd.Series) -> np.ndarray:
    """Seconds since the epoch. 60 is a multiple of 15, so phase measured this
    way is the same phase as second-of-minute modulo 15."""
    return times.astype("int64").to_numpy() / 1e9


def detect_edges(
    signal: np.ndarray,
    times: np.ndarray,
    *,
    polarity: str = "drop",
    mad_multiplier: float = 3.0,
    min_spacing: float = float(PERIOD_SECONDS),
) -> np.ndarray:
    """Candidate rescheduling times from thresholded first differences.

    `polarity` selects what counts as an edge. On throughput the rescheduling
    event shows up as a drop, on latency as a spike, so the caller picks. The
    threshold is median plus k times the median absolute deviation rather than
    a fixed value, because throughput scale varies by location and by hour.

    Edges are selected greedily by descending magnitude subject to the minimum
    spacing constraint, which is the step that stops one large transient from
    contributing a cluster of neighbouring candidates.
    """
    if len(signal) < 3:
        return np.empty(0)

    diff = np.diff(signal.astype(float))
    edge_times = times[1:]

    if polarity == "drop":
        strength = -diff
    elif polarity == "spike":
        strength = diff
    elif polarity == "abs":
        strength = np.abs(diff)
    else:
        raise ValueError("polarity must be one of 'drop', 'spike', 'abs'")

    finite = np.isfinite(strength)
    if not finite.any():
        return np.empty(0)

    median = float(np.nanmedian(strength[finite]))
    mad = float(np.nanmedian(np.abs(strength[finite] - median)))
    # 1.4826 makes the MAD a consistent estimator of sigma under normality.
    scale = 1.4826 * mad if mad > 0 else float(np.nanstd(strength[finite]))
    if scale <= 0:
        return np.empty(0)
    threshold = median + mad_multiplier * scale

    candidate = np.flatnonzero(finite & (strength > threshold))
    if candidate.size == 0:
        return np.empty(0)

    order = candidate[np.argsort(-strength[candidate])]
    accepted: list[float] = []
    for index in order:
        t = float(edge_times[index])
        if all(abs(t - other) >= min_spacing for other in accepted):
            accepted.append(t)
    return np.sort(np.asarray(accepted))


def recover_phase(
    frame: pd.DataFrame,
    *,
    column: str = TARGET_COL,
    polarity: str = "drop",
    n_bins: int = PERIOD_SECONDS,
    mad_multiplier: float = 3.0,
    top_k: int = 3,
    neighbourhood: int = 1,
    min_confidence: float = 1.8,
    min_edges: int = 20,
) -> PhaseReference:
    """Recover the scheduling phase offset from a trace.

    Falls back to the fixed 12/27/42/57 offset when the histogram peak is not
    convincing, and says so in the returned `method` rather than silently
    substituting it.
    """
    if column not in frame.columns:
        raise KeyError(f"{column!r} not in frame")

    signal = frame[column].to_numpy(dtype=float)
    times = _epoch_seconds(frame[TIME_COL])
    edges = detect_edges(signal, times, polarity=polarity,
                         mad_multiplier=mad_multiplier,
                         min_spacing=float(PERIOD_SECONDS))

    # Bins are centred on the candidate phases rather than spanning between
    # them. At 1 Hz an edge lands on an integer phase, and bins running [p, p+1)
    # would report every recovered offset half a bin late.
    width = float(PERIOD_SECONDS) / n_bins
    bin_edges = np.linspace(-width / 2, PERIOD_SECONDS - width / 2, n_bins + 1)
    if edges.size < min_edges:
        return PhaseReference(FIXED_PHASE_OFFSET, 0.0, int(edges.size), "fallback",
                              np.zeros(n_bins), bin_edges)

    phases = np.mod(edges + width / 2, PERIOD_SECONDS) - width / 2
    histogram, _ = np.histogram(phases, bins=bin_edges)

    # Confidence is the dominant bin mass over what a uniform scatter of the
    # same number of edges would put in one bin. A trace with no scheduling
    # structure sits near 1.
    uniform = edges.size / n_bins
    confidence = float(histogram.max() / uniform) if uniform > 0 else 0.0
    if confidence < min_confidence:
        return PhaseReference(FIXED_PHASE_OFFSET, confidence, int(edges.size), "fallback",
                              histogram, bin_edges)

    offset = _circular_peak(histogram, bin_edges, top_k=top_k, neighbourhood=neighbourhood)
    return PhaseReference(float(offset), confidence, int(edges.size), "recovered",
                          histogram, bin_edges)


def _circular_peak(histogram: np.ndarray, bin_edges: np.ndarray, *,
                   top_k: int, neighbourhood: int) -> float:
    """Weighted circular mean over the top-k bins near the dominant bin.

    Circular because phase 14.9 and phase 0.1 are 0.2 s apart, not 14.8. A
    plain weighted mean would put an offset straddling the wrap point in the
    middle of the period, which is the worst possible answer.
    """
    n_bins = len(histogram)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dominant = int(np.argmax(histogram))

    window = [(dominant + d) % n_bins for d in range(-neighbourhood, neighbourhood + 1)]
    window = sorted(set(window), key=lambda b: -histogram[b])[:top_k]

    weights = histogram[window].astype(float)
    if weights.sum() <= 0:
        return float(centres[dominant])

    angles = 2 * np.pi * centres[window] / PERIOD_SECONDS
    x = float(np.sum(weights * np.cos(angles)))
    y = float(np.sum(weights * np.sin(angles)))
    if x == 0.0 and y == 0.0:
        return float(centres[dominant])
    return float(np.mod(np.arctan2(y, x) * PERIOD_SECONDS / (2 * np.pi), PERIOD_SECONDS))


def assign_phase(times: pd.Series, reference: PhaseReference | float) -> np.ndarray:
    """Seconds since the start of the current 15 s period, in [0, 15)."""
    offset = reference.offset_seconds if isinstance(reference, PhaseReference) else float(reference)
    return np.mod(_epoch_seconds(times) - offset, PERIOD_SECONDS)


def assign_period(times: pd.Series, reference: PhaseReference | float) -> np.ndarray:
    """Integer period index, so intra-period statistics can be grouped."""
    offset = reference.offset_seconds if isinstance(reference, PhaseReference) else float(reference)
    seconds = _epoch_seconds(times) - offset
    return np.floor(seconds / PERIOD_SECONDS).astype(np.int64)


def boundary_mask(
    times: pd.Series,
    reference: PhaseReference | float,
    *,
    open_seconds: float = BOUNDARY_OPEN_SECONDS,
    close_seconds: float = BOUNDARY_CLOSE_SECONDS,
    sampling_hz: float | None = None,
) -> np.ndarray:
    """True where a sample falls in a period boundary region.

    Casparsen isolate the opening 140 ms and closing 75 ms because the handover
    spike lives there, averaging 74 ms above the period mean. That analysis
    needs a sampling rate that can resolve those windows. At 1 Hz it cannot,
    and applying the mask anyway would either select nothing or select a whole
    second and call it 140 ms, so this raises instead.
    """
    if sampling_hz is None:
        deltas = np.diff(_epoch_seconds(times))
        positive = deltas[deltas > 0]
        sampling_hz = float(1.0 / np.median(positive)) if positive.size else 1.0

    narrowest = min(open_seconds, close_seconds)
    if sampling_hz * narrowest < 2.0:
        raise ValueError(
            f"boundary regions of {open_seconds * 1000:.0f} ms / "
            f"{close_seconds * 1000:.0f} ms cannot be resolved at {sampling_hz:g} Hz. "
            "The Casparsen boundary analysis needs their 500 Hz probes; on 1 Hz "
            "traces use the phase recovery and the period level framing only."
        )

    phase = assign_phase(times, reference)
    return (phase < open_seconds) | (phase >= PERIOD_SECONDS - close_seconds)


def period_statistics(
    frame: pd.DataFrame,
    reference: PhaseReference | float,
    *,
    column: str = TARGET_COL,
    exclude_boundary: bool = False,
) -> pd.DataFrame:
    """Per period summary of a column, optionally dropping boundary samples."""
    work = frame.copy()
    work["period"] = assign_period(work[TIME_COL], reference)
    work["phase"] = assign_phase(work[TIME_COL], reference)
    if exclude_boundary:
        work = work[~boundary_mask(work[TIME_COL], reference)]
    grouped = work.groupby("period")[column]
    return pd.DataFrame({
        "n": grouped.size(),
        "mean": grouped.mean(),
        "std": grouped.std(),
        "min": grouped.min(),
        "q99": grouped.quantile(0.99),
        "max": grouped.max(),
    })


def classify_periods(
    frame: pd.DataFrame,
    reference: PhaseReference | float,
    *,
    latency_column: str = "latency_ms",
    threshold_ms: float = GOOD_LATENCY_MS,
    good_fraction: float = GOOD_PERIOD_QUANTILE,
) -> pd.DataFrame:
    """Casparsen's period level Good/Degraded label.

    Good if at least `good_fraction` of packets in the period meet the latency
    threshold, Degraded otherwise. Needs a latency column, which the StarNet
    throughput traces do not carry; this is here for the stretch contribution
    in CLAUDE.md section 2 and for any trace that does have one.
    """
    if latency_column not in frame.columns:
        raise KeyError(
            f"{latency_column!r} not in frame. The StarNet traces are throughput "
            "only, so period classification needs a latency carrying source."
        )
    work = frame[[TIME_COL, latency_column]].copy()
    work["period"] = assign_period(work[TIME_COL], reference)
    grouped = work.groupby("period")[latency_column]
    met = grouped.apply(lambda s: float((s <= threshold_ms).mean()))
    return pd.DataFrame({
        "n": grouped.size(),
        "fraction_meeting_threshold": met,
        "q99_latency_ms": grouped.quantile(0.99),
        "label": np.where(met >= good_fraction, "Good", "Degraded"),
    })
