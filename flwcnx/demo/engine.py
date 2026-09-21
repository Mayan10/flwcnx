"""The system running as a system, one decision at a time.

Everything else in this repository evaluates the pipeline in batch: fit, predict
the whole test split, score. That is the right way to measure it and the wrong
way to *show* it, because it hides the thing that makes the design defensible,
which is that every step is causal. At decision `t` the terminal has the
look-back window, the regime covariates, and the outcomes of decisions before
`t`. It does not have the answer.

This module replays a trace through the full stack in that order:

    look-back window
      -> point forecast              (forecast/)
      -> regime assignment           (state/regime.py)
      -> safe lower bound            (calibrate/)
      -> admitted sessions           (decide/admission.py)
      -> congestion flag             (decide/congestion.py)
      -> the horizon elapses, the outcome is revealed
      -> the calibrator learns from it

and emits one record per step. A step is only ever handed data that existed
before it, which is the property `tests/test_demo.py` pins by feeding the engine
a trace whose future has been overwritten and checking the earlier decisions are
unchanged.

There is no dish. `ingest/live.py` still raises. This replays a recorded trace
at whatever speed the caller asks for, which is the honest form of a live demo
and is exactly what `ReplaySource` was built for.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from flwcnx.calibrate.adaptive import AdaptiveRegimeCalibrator
from flwcnx.config import CalibrationConfig, DecisionConfig
from flwcnx.state.regime import GLOBAL_LABEL, RegimeAssigner


@dataclass
class DecisionRecord:
    """One decision, and everything needed to audit it afterwards."""

    step: int
    timestamp: str
    regime: str
    predicted_mbps: float
    bound_mbps: float
    actual_mbps: float
    offset_mbps: float
    alpha: float
    admitted: int
    oracle: int
    dropped: int
    unused: int
    risk_event: bool
    congested: bool
    realised_risk_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DemoEngine:
    """Streams decisions through the calibration and decision layers.

    `calibrator` is anything exposing `fit` and `transform_online`; the default
    is this project's online layer. It is seeded once from the calibration split
    and then never sees a batch again: the loop hands it one outcome at a time,
    in order.
    """

    calibrator: AdaptiveRegimeCalibrator
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    assigner: RegimeAssigner | None = None
    config: CalibrationConfig = field(default_factory=CalibrationConfig)
    #: Sustained slots below commitment before congestion is declared. The
    #: congestion signal is derived from the bound, never a separate model.
    congestion_window: int = 5

    _risk_history: list[bool] = field(default_factory=list, repr=False)
    _below: int = field(default=0, repr=False)

    def _regime_of(self, row: pd.DataFrame) -> str:
        if not self.config.regime.axes or self.assigner is None:
            return GLOBAL_LABEL
        return str(self.assigner.assign(row)[0])

    def step(self, index: int, timestamp, predicted: float, actual: float,
             covariate_row: pd.DataFrame) -> DecisionRecord:
        """One decision. The outcome is used only after the bound is emitted."""
        regime = self._regime_of(covariate_row)

        # --- everything up to here is knowable before the horizon elapses ---
        bound = float(self.calibrator.transform_online(
            np.array([predicted]), np.array([actual]), covariate_row)[0])

        trace = self.calibrator.trace
        offset = float(trace.offset[-1]) if trace is not None and trace.offset.size else 0.0
        alpha = float(trace.alpha[-1]) if trace is not None and trace.alpha.size else float("nan")

        per_session = self.decision.bandwidth_per_session_mbps
        admitted = int(max(bound, 0.0) // per_session)
        oracle = int(max(actual, 0.0) // per_session)
        dropped = max(admitted - oracle, 0)
        unused = max(oracle - admitted, 0)

        risk = bound > actual
        self._risk_history.append(risk)

        # Congestion: the bound has sat below the commitment for a sustained
        # window. Derived, not modelled (CLAUDE.md section 4).
        self._below = self._below + 1 if bound < self.decision.commitment_mbps else 0
        congested = self._below >= self.congestion_window

        return DecisionRecord(
            step=index,
            timestamp=str(timestamp),
            regime=regime,
            predicted_mbps=round(float(predicted), 3),
            bound_mbps=round(bound, 3),
            actual_mbps=round(float(actual), 3),
            offset_mbps=round(offset, 3),
            alpha=round(alpha, 4),
            admitted=admitted, oracle=oracle, dropped=dropped, unused=unused,
            risk_event=bool(risk), congested=bool(congested),
            realised_risk_rate=round(float(np.mean(self._risk_history)), 4),
        )

    def run(self, predicted: np.ndarray, actual: np.ndarray,
            covariates: pd.DataFrame, timestamps=None,
            limit: int | None = None) -> Iterator[DecisionRecord]:
        """Replay a whole split, yielding one record per decision."""
        n = len(predicted) if limit is None else min(limit, len(predicted))
        for t in range(n):
            stamp = timestamps[t] if timestamps is not None else t
            yield self.step(t, stamp, float(predicted[t]), float(actual[t]),
                            covariates.iloc[[t]].reset_index(drop=True))

    def summary(self) -> dict:
        risk = np.asarray(self._risk_history, dtype=float)
        return {
            "n_decisions": int(risk.size),
            "realised_risk_rate": float(risk.mean()) if risk.size else float("nan"),
            "budget": self.config.epsilon,
            "within_budget": bool(risk.mean() <= self.config.epsilon) if risk.size else False,
        }


def records_to_frame(records: list[DecisionRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in records])
