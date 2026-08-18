"""Congestion detection, derived from the calibrated bound.

No separate classifier, by design (CLAUDE.md section 4). Congestion is:

    the calibrated lower bound falls below the currently committed allocation
    for a sustained window of W slots

That definition gets issue 2 of the brief nearly free and keeps the system
coherent: there is one model, one calibration, and every downstream signal is
a decision rule on top of the same bound. A second trained classifier would
have its own errors, its own calibration, and no guarantee of agreeing with
the allocator about when the link is in trouble.

Requiring W sustained slots rather than a single dip is what separates an alert
from a twitch. A one second drop below commitment on a link that reschedules
every 15 seconds is normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CongestionResult:
    """Predicted and actual congestion, plus how early the alert fired."""

    predicted: np.ndarray        # bool per slot
    actual: np.ndarray           # bool per slot, from the realised throughput
    lead_times: np.ndarray       # slots of warning per actual episode, may be empty

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"predicted": self.predicted, "actual": self.actual})

    def confusion(self) -> dict[str, int]:
        tp = int(np.sum(self.predicted & self.actual))
        fp = int(np.sum(self.predicted & ~self.actual))
        fn = int(np.sum(~self.predicted & self.actual))
        tn = int(np.sum(~self.predicted & ~self.actual))
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def scores(self) -> dict[str, float]:
        c = self.confusion()
        precision = c["tp"] / max(c["tp"] + c["fp"], 1)
        recall = c["tp"] / max(c["tp"] + c["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mean_lead_slots": float(np.mean(self.lead_times)) if self.lead_times.size else 0.0,
            "median_lead_slots": (float(np.median(self.lead_times))
                                  if self.lead_times.size else 0.0),
        }


def sustained(flags: np.ndarray, window: int) -> np.ndarray:
    """True where `flags` has been true for `window` consecutive slots.

    Causal: a slot is flagged on the strength of itself and the slots before
    it, never the ones after. An alert that needs the future is not an alert.
    """
    flags = np.asarray(flags, dtype=bool)
    if window <= 1:
        return flags.copy()
    if len(flags) < window:
        return np.zeros(len(flags), dtype=bool)
    # Running count of consecutive trues, reset by any false.
    out = np.zeros(len(flags), dtype=bool)
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        out[i] = run >= window
    return out


def detect_congestion(safe_bound: np.ndarray, actual: np.ndarray,
                      commitment_mbps: float, window: int = 5) -> CongestionResult:
    """Flag congestion from the bound and score it against what happened.

    The predicted flag uses the bound only, so it is available before the
    slot is served. The actual flag uses realised throughput and exists purely
    to score the predictor.
    """
    safe_bound = np.asarray(safe_bound, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if safe_bound.shape != actual.shape:
        raise ValueError("bound and actual must be the same length")

    predicted = sustained(safe_bound < commitment_mbps, window)
    truth = sustained(actual < commitment_mbps, window)
    return CongestionResult(predicted, truth, _lead_times(predicted, truth))


def _lead_times(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Slots of warning before each actual episode starts.

    Zero means the alert fired on the same slot the episode began. An episode
    with no preceding alert contributes nothing rather than a zero, so the mean
    is a mean over detected episodes and recall carries the misses.
    """
    starts = np.flatnonzero(actual & ~np.roll(actual, 1))
    if actual.size and actual[0]:
        starts = np.unique(np.append(starts, 0))

    leads = []
    for start in starts:
        window = predicted[: start + 1]
        if not window.any():
            continue
        fired = int(np.flatnonzero(window)[-1])
        # Walk back to the beginning of that contiguous alert run.
        while fired > 0 and predicted[fired - 1]:
            fired -= 1
        leads.append(start - fired)
    return np.asarray(leads, dtype=float)


def congestion_episodes(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of a boolean flag as (start, end_exclusive) pairs."""
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return []
    padded = np.concatenate([[False], flags, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]
