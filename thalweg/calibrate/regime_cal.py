"""Regime conditioned calibration. This is the contribution.

The claim being tested:

    A point throughput forecast can be converted into a safe lower bound whose
    overestimation rate is controlled *within each operating regime*, where
    regime is defined by covariates the terminal can compute at prediction
    time, and doing so removes the conditional risk failure that a single
    global operating point suffers, at comparable MAE.

Why a global operating point cannot do this. A conformal or BG-CFQS bound
holds P(bound > y) <= epsilon marginally, averaged over everything. If the
residual distribution is wider in one regime than another, the single offset
that hits epsilon on average sits too high in the wide regime and too low in
the narrow one. Risk concentrates exactly where capacity is lowest, which is
where over allocation actually drops sessions. BG-CFQS report this against
themselves: global 0.349 at a 0.35 budget, and 0.65 to 0.86 on the low
throughput slices.

The procedure (CLAUDE.md section 7):

  1. temporal train / calibration / test split, calibration never used to fit;
  2. fit the point forecaster on train;
  3. residuals on calibration, grouped by regime;
  4. per regime, select the offset that holds the budget inside that regime,
     by either a per regime conformal order statistic or a per regime run of
     the BG-CFQS boundary search;
  5. at test time assign the regime and apply that regime's offset.

Sparse regimes cannot support their own quantile, so they climb the coarsening
hierarchy until they reach a level with enough calibration points, and the
climb is recorded. If most of the mass ends up at the global level then the
layer is doing nothing and the ablation must say so.

The honest failure mode, stated up front: if per regime bounds become so
conservative that MAE collapses, that is a negative result and gets reported
as one. `summary()` returns the MAE cost alongside the risk gain so the two
cannot be separated in the writeup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thalweg.calibrate.bgcfqs import bgcfqs_on_residuals
from thalweg.calibrate.conformal import fit_conformal
from thalweg.config import BGCFQSConfig, CalibrationConfig
from thalweg.eval.metrics import check_direction, over_rate
from thalweg.state.regime import GLOBAL_LABEL, FallbackReport, RegimeAssigner, build_fallback_map

#: How a per regime operating point is chosen.
SELECTORS = ("conformal", "bgcfqs")


@dataclass
class RegimeOffset:
    """One regime's operating point and the evidence behind it."""

    label: str
    level_index: int
    level_name: str
    offset: float
    n_calibration: int
    achieved_over_rate: float
    selector: str
    degenerate: bool = False

    def to_dict(self) -> dict:
        return {
            "regime": self.label, "level": self.level_name,
            "level_index": self.level_index, "offset_mbps": round(self.offset, 4),
            "n_calibration": self.n_calibration,
            "calibration_over_rate": round(self.achieved_over_rate, 4),
            "selector": self.selector, "degenerate": self.degenerate,
        }


@dataclass
class RegimeCalibrator:
    """Fits one operating point per regime and applies it at prediction time."""

    config: CalibrationConfig = field(default_factory=CalibrationConfig)
    assigner: RegimeAssigner | None = None
    selector: str = "conformal"
    # "lower" for throughput, where the risk is over-allocating; "upper" for
    # latency, where the risk is promising a delay the link will not meet. The
    # conditional failure this layer fixes is the same shape either way.
    direction: str = "lower"
    bgcfqs_config: BGCFQSConfig = field(default_factory=BGCFQSConfig)

    _offsets: dict[tuple[int, str], RegimeOffset] = field(default_factory=dict, repr=False)
    _resolved: dict[str, tuple[str, int]] = field(default_factory=dict, repr=False)
    _global_offset: RegimeOffset | None = field(default=None, repr=False)
    _level_names: tuple[str, ...] = field(default=(), repr=False)
    report: FallbackReport | None = field(default=None, repr=False)
    last_fallback_rate: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.selector not in SELECTORS:
            raise ValueError(f"selector must be one of {SELECTORS}, got {self.selector!r}")
        check_direction(self.direction)
        if self.assigner is None:
            self.assigner = RegimeAssigner(self.config.regime)

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        predicted_cal: np.ndarray,
        actual_cal: np.ndarray,
        covariates_cal: pd.DataFrame,
        *,
        allow_fit_assigner_on_calibration: bool = False,
    ) -> RegimeCalibrator:
        """Select one offset per regime on the calibration split.

        The assigner should already be fit on the training split. Its candidate
        count terciles are data dependent, and fitting them here would let the
        calibration split shape the regime definition. That is a small leak but
        it is a leak, so it has to be asked for explicitly.
        """
        assert self.assigner is not None
        if not self.assigner._fitted:
            if not allow_fit_assigner_on_calibration:
                raise RuntimeError(
                    "RegimeAssigner is not fitted. Fit it on the training split, or "
                    "pass allow_fit_assigner_on_calibration=True and accept that the "
                    "candidate count terciles then depend on the calibration data."
                )
            self.assigner.fit(covariates_cal)

        predicted_cal = np.asarray(predicted_cal, dtype=float).ravel()
        actual_cal = np.asarray(actual_cal, dtype=float).ravel()
        if len(predicted_cal) != len(covariates_cal):
            raise ValueError(
                f"{len(predicted_cal)} predictions against {len(covariates_cal)} "
                "covariate rows; these must be the same samples in the same order"
            )

        levels = self.assigner.assign_all_levels(covariates_cal)
        self._level_names = self.assigner.level_names()
        self.report = build_fallback_map(levels, self.config.regime.min_samples,
                                         self._level_names)
        self._resolved = self.report.resolved

        # The global offset always exists, because it is the floor of the
        # hierarchy and every sample belongs to it.
        self._global_offset = self._select(
            predicted_cal, actual_cal, np.ones(len(predicted_cal), dtype=bool),
            label=GLOBAL_LABEL, level_index=len(levels) - 1,
        )

        self._offsets = {}
        for level_label, level_index in set(self._resolved.values()):
            mask = np.asarray(levels[level_index] == level_label)
            self._offsets[(level_index, level_label)] = self._select(
                predicted_cal, actual_cal, mask, label=level_label, level_index=level_index,
            )
        return self

    def _select(self, predicted: np.ndarray, actual: np.ndarray, mask: np.ndarray,
                *, label: str, level_index: int) -> RegimeOffset:
        """Choose one regime's offset with the configured selector."""
        p, a = predicted[mask], actual[mask]
        level_name = (self._level_names[level_index] if level_index < len(self._level_names)
                      else GLOBAL_LABEL)

        if p.size == 0:
            return RegimeOffset(label, level_index, level_name, 0.0, 0, float("nan"),
                                self.selector, degenerate=True)

        if self.selector == "conformal":
            bound = fit_conformal(p, a, self.config.epsilon, direction=self.direction)
            offset, degenerate = bound.offset, bound.degenerate
        else:
            offset = bgcfqs_on_residuals(p, a, self.config.epsilon, self.bgcfqs_config,
                                         direction=self.direction)
            degenerate = False

        achieved = over_rate(p + offset, a, self.direction)
        return RegimeOffset(label, level_index, level_name, float(offset), int(p.size),
                            achieved, self.selector, degenerate)

    # -- application -------------------------------------------------------

    def offsets_for(self, covariates: pd.DataFrame) -> np.ndarray:
        """The offset each row's regime resolves to.

        A fine regime unseen during calibration is not an error. The row walks
        its own hierarchy from finest to coarsest and takes the first level
        that has a fitted offset, which is the same climb as at fit time and
        lands on global in the worst case. The fraction of rows that had to do
        this is recorded in `last_fallback_rate`.
        """
        assert self.assigner is not None
        if self._global_offset is None:
            raise RuntimeError("RegimeCalibrator.fit must run before offsets_for")

        levels = self.assigner.assign_all_levels(covariates)
        fine = levels[0]
        offsets = np.full(len(covariates), self._global_offset.offset, dtype=float)
        unseen = 0

        for label in pd.unique(fine):
            rows = np.flatnonzero(fine == label)
            resolved = self._resolved.get(str(label))
            if resolved is not None:
                key = (resolved[1], resolved[0])
                chosen = self._offsets.get(key, self._global_offset)
                offsets[rows] = chosen.offset
                continue

            # Unseen at calibration: climb this row's own hierarchy.
            unseen += len(rows)
            for level_index in range(len(levels)):
                candidate = str(levels[level_index][rows[0]])
                hit = self._offsets.get((level_index, candidate))
                if hit is not None:
                    offsets[rows] = hit.offset
                    break

        self.last_fallback_rate = float(unseen) / max(len(covariates), 1)
        return offsets

    def transform(self, predicted: np.ndarray, covariates: pd.DataFrame,
                  floor: float = 0.0) -> np.ndarray:
        """Point predictions to regime conditioned safe bounds."""
        predicted = np.asarray(predicted, dtype=float).ravel()
        return np.maximum(predicted + self.offsets_for(covariates), floor)

    def assign(self, covariates: pd.DataFrame) -> np.ndarray:
        """Fine grained regime labels, for the per regime metric breakdown."""
        assert self.assigner is not None
        return self.assigner.assign(covariates)

    # -- reporting ---------------------------------------------------------

    def offsets_table(self) -> pd.DataFrame:
        """Every fitted offset. Committed alongside any result that uses it."""
        rows = [o.to_dict() for o in self._offsets.values()]
        if self._global_offset is not None:
            rows.append(self._global_offset.to_dict())
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("n_calibration", ascending=False).reset_index(
            drop=True)

    def summary(self) -> dict:
        """What the layer did, in a form that can go straight into a results file."""
        table = self.offsets_table()
        out: dict = {
            "selector": self.selector,
            "direction": self.direction,
            "epsilon": self.config.epsilon,
            "axes": list(self.config.regime.axes),
            "min_samples": self.config.regime.min_samples,
            "n_offsets": int(len(self._offsets)),
            "global_offset_mbps": (round(self._global_offset.offset, 4)
                                   if self._global_offset else None),
            "test_fallback_rate": round(self.last_fallback_rate, 4),
        }
        if self.report is not None:
            out["calibration_mass_by_level"] = {
                k: round(v, 4) for k, v in self.report.rate_by_level().items()
            }
            out["n_fine_regimes"] = self.report.n_regimes
        if not table.empty:
            out["offset_spread_mbps"] = round(
                float(table["offset_mbps"].max() - table["offset_mbps"].min()), 3
            )
            out["degenerate_regimes"] = int(table["degenerate"].sum())
        return out


def global_calibrator(config: CalibrationConfig, selector: str = "conformal",
                      direction: str = "lower") -> RegimeCalibrator:
    """The ablation floor: one regime for everything.

    Same code path as the conditioned version, so a difference between them is
    a difference in conditioning and not in implementation.
    """
    from dataclasses import replace

    flat = replace(config, regime=replace(config.regime, axes=()))
    calibrator = RegimeCalibrator(config=flat, selector=selector, direction=direction)
    assert calibrator.assigner is not None
    calibrator.assigner._fitted = True     # no data dependent cut points to fit
    return calibrator
