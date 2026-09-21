"""Admission control from the safe bound.

The downstream evaluation from CLAUDE.md section 8. Each service needs
b = 10 Mbps. Given a safe forecast the allocator admits floor(safe / b)
sessions. An oracle with perfect knowledge would admit floor(actual / b). Any
session admitted beyond what the link can carry is dropped.

This is the layer that makes the risk metrics mean something. OverRate is a
statistic; dropped sessions is what a user experiences. A calibration layer
that improves OverRate but not dropped sessions has not improved anything.

BG-CFQS report 6.8%, 11.0% and 12.6% relative reductions in dropped sessions on
all decisions, P30 and P10 against budget scale baselines. That is the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thalweg.eval.metrics import RISK_SLICES, risk_slice_mask


@dataclass
class AdmissionResult:
    """One policy's admission outcomes."""

    admitted: np.ndarray
    oracle: np.ndarray
    dropped: np.ndarray
    violations: np.ndarray       # bool, admitted more than the link could carry
    unused: np.ndarray           # sessions the link could have carried but did not

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "admitted": self.admitted, "oracle": self.oracle,
            "dropped": self.dropped, "violation": self.violations, "unused": self.unused,
        })


def admit_sessions(safe_forecast: np.ndarray, bandwidth_per_session_mbps: float = 10.0
                   ) -> np.ndarray:
    """floor(safe / b), never negative."""
    if bandwidth_per_session_mbps <= 0:
        raise ValueError("bandwidth per session must be positive")
    safe = np.asarray(safe_forecast, dtype=float).ravel()
    return np.maximum(np.floor(safe / bandwidth_per_session_mbps), 0.0).astype(int)


def run_admission(safe_forecast: np.ndarray, actual: np.ndarray,
                  bandwidth_per_session_mbps: float = 10.0) -> AdmissionResult:
    """Admit against the bound, then score against what the link actually did."""
    admitted = admit_sessions(safe_forecast, bandwidth_per_session_mbps)
    oracle = admit_sessions(actual, bandwidth_per_session_mbps)
    dropped = np.maximum(admitted - oracle, 0)
    unused = np.maximum(oracle - admitted, 0)
    return AdmissionResult(admitted, oracle, dropped, dropped > 0, unused)


def summarise_admission(result: AdmissionResult, actual: np.ndarray,
                        slices: dict[str, float] | None = None) -> pd.DataFrame:
    """Dropped sessions globally and on the risk slices.

    Reports P95 dropped alongside the mean, because a policy that drops nothing
    most of the time and eleven sessions occasionally is not the same as one
    that steadily drops one, and the mean cannot tell them apart.
    """
    slices = slices or RISK_SLICES
    actual = np.asarray(actual, dtype=float).ravel()

    rows = []
    for name, mask in [("all", np.ones(len(actual), dtype=bool)),
                       *[(k, risk_slice_mask(actual, v)) for k, v in slices.items()]]:
        if mask.sum() == 0:
            continue
        rows.append({
            "slice": name,
            "n": int(mask.sum()),
            "mean_dropped": float(np.mean(result.dropped[mask])),
            "violation_rate": float(np.mean(result.violations[mask])),
            "p95_dropped": float(np.percentile(result.dropped[mask], 95)),
            "mean_unused": float(np.mean(result.unused[mask])),
            "mean_admitted": float(np.mean(result.admitted[mask])),
            "mean_oracle": float(np.mean(result.oracle[mask])),
            # Utilisation: what fraction of the admissible capacity was used.
            # Without this a policy can win on drops by admitting nobody.
            "utilisation": float(
                np.sum(np.minimum(result.admitted[mask], result.oracle[mask]))
                / max(np.sum(result.oracle[mask]), 1)
            ),
        })
    return pd.DataFrame(rows)


def evaluate_admission(safe_forecast: np.ndarray, actual: np.ndarray,
                       bandwidth_per_session_mbps: float = 10.0,
                       slices: dict[str, float] | None = None) -> pd.DataFrame:
    """Run and summarise in one call."""
    return summarise_admission(
        run_admission(safe_forecast, actual, bandwidth_per_session_mbps), actual, slices
    )


def relative_reduction(baseline: pd.DataFrame, ours: pd.DataFrame,
                       column: str = "mean_dropped") -> pd.DataFrame:
    """Relative improvement per slice, in the form BG-CFQS report it.

    Positive means fewer dropped sessions than the baseline. Utilisation is
    carried through so a reduction bought entirely by admitting fewer sessions
    is visible rather than hidden.
    """
    merged = baseline.merge(ours, on="slice", suffixes=("_baseline", "_ours"))
    merged["relative_reduction"] = np.where(
        merged[f"{column}_baseline"] > 0,
        1.0 - merged[f"{column}_ours"] / merged[f"{column}_baseline"],
        np.nan,
    )
    merged["utilisation_delta"] = merged["utilisation_ours"] - merged["utilisation_baseline"]
    return merged[["slice", f"{column}_baseline", f"{column}_ours",
                   "relative_reduction", "utilisation_baseline", "utilisation_ours",
                   "utilisation_delta"]]
