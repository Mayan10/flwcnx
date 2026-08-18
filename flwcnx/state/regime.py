"""Regime assignment. This is ours.

A regime is a bucket over covariates the terminal can compute at prediction
time. The point of the whole project is that a single global operating point
cannot hold a risk budget across regimes that behave differently, so the
regimes have to be defined before anything can be calibrated inside them.

The four axes and their cut points come from measured structure, not from
convenience (CLAUDE.md section 7):

  - phase: StarNet's attention concentrates on the opening of the 15 s period
    and that is where the throughput drops are largest;
  - elevation: throughput rises with elevation and plateaus above 60 degrees,
    and the FCC filing floors serving satellites at 25 degrees;
  - distance: throughput holds around 230 Mbps until roughly 645 km and then
    declines;
  - candidates: average throughput rises about 26% going from 15 to 45
    visible candidates.

The full cross product is too sparse to calibrate in directly, so the assigner
also exposes a coarsening hierarchy. Axes are dropped from the end of the
`axes` tuple, so the axis you least want to lose goes first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flwcnx.config import PERIOD_SECONDS, RegimeConfig

GLOBAL_LABEL = "global"

#: Regime axis -> the column it reads.
#:
#: The first four are the StarNet axes from the brief. The last three exist
#: because the supplied WetLinks data has none of the satellite geometry and
#: its 30 s grid aliases the phase; obstruction, dish pointing and hour of day
#: are what is actually observable there (docs/supplied-dataset.md).
REQUIRED_COLUMNS: dict[str, str] = {
    "phase": "phase_seconds",
    "elevation": "elevation_deg",
    "distance": "distance_km",
    "candidates": "candidate_count",
    "obstruction": "fraction_obstructed",
    "azimuth": "dish_azimuth_deg",
    "hour": "hour",
}


@dataclass(frozen=True)
class FallbackReport:
    """Which fine regimes could not be calibrated directly, and where they went.

    Kept as a returned object rather than a log line because the fallback rate
    is a result: if most of the mass falls back to global, the regime layer is
    not doing anything and the ablation should say so.
    """

    resolved: dict[str, tuple[str, int]]     # fine label -> (level label, level index)
    level_names: tuple[str, ...]
    counts: dict[str, int]                   # fine label -> calibration sample count

    @property
    def n_regimes(self) -> int:
        return len(self.resolved)

    def rate_by_level(self) -> dict[str, float]:
        """Fraction of calibration samples resolved at each level of the hierarchy."""
        total = sum(self.counts.values())
        if total == 0:
            return {}
        mass: dict[str, float] = {}
        for label, (_, level_index) in self.resolved.items():
            name = self.level_names[level_index]
            mass[name] = mass.get(name, 0.0) + self.counts.get(label, 0)
        return {k: v / total for k, v in sorted(mass.items())}

    def summary(self) -> str:
        parts = [f"{k}={v:.1%}" for k, v in self.rate_by_level().items()]
        return f"{self.n_regimes} regimes, mass by level: " + ", ".join(parts)


@dataclass
class RegimeAssigner:
    """Assigns regime labels, at every level of the coarsening hierarchy.

    Cut points that depend on the data (the candidate count terciles) are fit
    on the training split only. Fitting them on everything would leak the test
    distribution into the regime definition, which is a subtle enough leak to
    be worth being loud about.
    """

    config: RegimeConfig = field(default_factory=RegimeConfig)
    _candidate_edges: tuple[float, ...] = field(default=(), repr=False)
    _obstruction_edges: tuple[float, ...] = field(default=(), repr=False)
    _fitted: bool = field(default=False, repr=False)

    def _fit_edges(self, frame: pd.DataFrame, axis: str,
                   quantiles: tuple[float, ...]) -> tuple[float, ...]:
        values = pd.to_numeric(frame[REQUIRED_COLUMNS[axis]], errors="coerce").dropna()
        if values.empty:
            return ()
        # Collapse duplicate edges rather than creating an empty bucket. This
        # matters for obstruction, which is exactly zero in most rows.
        return tuple(float(e) for e in np.unique(np.quantile(values, quantiles)))

    def fit(self, frame: pd.DataFrame) -> RegimeAssigner:
        if "candidates" in self.config.axes:
            self._candidate_edges = self._fit_edges(frame, "candidates",
                                                    self.config.candidate_quantiles)
        if "obstruction" in self.config.axes:
            self._obstruction_edges = self._fit_edges(frame, "obstruction",
                                                      self.config.obstruction_quantiles)
        self._fitted = True
        return self

    @property
    def candidate_edges(self) -> tuple[float, ...]:
        return self._candidate_edges

    # -- individual axes ---------------------------------------------------

    def _phase_bucket(self, values: np.ndarray) -> np.ndarray:
        """Opening, middle, closing of the 15 s period."""
        opening = self.config.phase_open_seconds
        closing = PERIOD_SECONDS - self.config.phase_close_seconds
        out = np.full(len(values), "mid", dtype=object)
        out[values < opening] = "open"
        out[values >= closing] = "close"
        out[~np.isfinite(values)] = "na"
        return out

    def _binned(self, values: np.ndarray, edges: tuple[float, ...], prefix: str) -> np.ndarray:
        if not edges:
            return np.where(np.isfinite(values), f"{prefix}0", "na").astype(object)
        index = np.digitize(values, np.asarray(edges, dtype=float), right=False)
        out = np.array([f"{prefix}{int(i)}" for i in index], dtype=object)
        out[~np.isfinite(values)] = "na"
        return out

    def _axis_labels(self, axis: str, frame: pd.DataFrame) -> np.ndarray:
        column = REQUIRED_COLUMNS[axis]
        if column not in frame.columns:
            raise KeyError(
                f"regime axis {axis!r} needs column {column!r}. "
                "Build the frame through state.features first."
            )
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)

        if axis == "phase":
            return self._phase_bucket(values)
        if axis == "elevation":
            return self._binned(values, self.config.elevation_edges, "el")
        if axis == "distance":
            return self._binned(values, self.config.distance_edges, "d")
        if axis == "candidates":
            if not self._fitted:
                raise RuntimeError("RegimeAssigner.assign before fit (candidate terciles)")
            return self._binned(values, self._candidate_edges, "c")
        if axis == "obstruction":
            if not self._fitted:
                raise RuntimeError("RegimeAssigner.assign before fit (obstruction cuts)")
            return self._binned(values, self._obstruction_edges, "o")
        if axis == "azimuth":
            sectors = max(int(self.config.azimuth_sectors), 1)
            # Dish azimuth arrives in [-180, 180); wrap to [0, 360) before
            # sectoring or north splits across two buckets.
            wrapped = np.mod(values, 360.0)
            index = np.floor(wrapped / (360.0 / sectors))
            out = np.array([f"s{int(i) % sectors}" for i in np.nan_to_num(index)],
                           dtype=object)
            out[~np.isfinite(values)] = "na"
            return out
        if axis == "hour":
            return self._binned(values, self.config.hour_edges, "h")
        raise ValueError(f"unknown regime axis {axis!r}")

    # -- hierarchy ---------------------------------------------------------

    def levels(self) -> list[tuple[str, ...]]:
        """Coarsening hierarchy, finest first, ending at the global level."""
        axes = tuple(self.config.axes)
        return [axes[:k] for k in range(len(axes), -1, -1)]

    def level_names(self) -> tuple[str, ...]:
        return tuple("+".join(level) if level else GLOBAL_LABEL for level in self.levels())

    def assign(self, frame: pd.DataFrame) -> np.ndarray:
        """Labels at the finest level."""
        return self.assign_all_levels(frame)[0]

    def assign_all_levels(self, frame: pd.DataFrame) -> list[np.ndarray]:
        """Labels at every level, finest first. Coarse labels are prefixes of fine
        ones, which is what makes the fallback a lookup rather than a re-bucket."""
        axes = tuple(self.config.axes)
        per_axis = {axis: self._axis_labels(axis, frame) for axis in axes}
        n = len(frame)

        out: list[np.ndarray] = []
        for level in self.levels():
            if not level:
                out.append(np.full(n, GLOBAL_LABEL, dtype=object))
                continue
            parts = [np.char.add(f"{axis}=", per_axis[axis].astype(str)) for axis in level]
            joined = parts[0]
            for part in parts[1:]:
                joined = np.char.add(np.char.add(joined, "|"), part)
            out.append(joined.astype(object))
        return out


def build_fallback_map(
    levels: list[np.ndarray], min_samples: int, level_names: tuple[str, ...]
) -> FallbackReport:
    """Resolve each fine regime to the finest level with enough samples.

    A fine regime with fewer than `min_samples` calibration points cannot have
    its own quantile estimated: the quantile of 12 residuals at epsilon = 0.05
    is not an estimate, it is the minimum. So it climbs the hierarchy until it
    finds a level that does have enough, and ultimately lands on global, which
    always does.

    The climb is recorded rather than applied silently. The fraction of mass
    that ends up at each level is reported alongside every result that uses it.
    """
    if not levels:
        raise ValueError("no levels to resolve")
    n_levels = len(levels)
    counts_per_level = [pd.Series(level).value_counts().to_dict() for level in levels]

    fine = levels[0]
    fine_counts = counts_per_level[0]

    resolved: dict[str, tuple[str, int]] = {}
    for label in dict.fromkeys(fine):
        chosen_level = n_levels - 1                       # global, the guaranteed floor
        chosen_label = str(levels[-1][0]) if len(levels[-1]) else GLOBAL_LABEL
        rows = np.flatnonzero(fine == label)
        for level_index in range(n_levels):
            candidate = str(levels[level_index][rows[0]])
            if counts_per_level[level_index].get(candidate, 0) >= min_samples:
                chosen_level, chosen_label = level_index, candidate
                break
        resolved[str(label)] = (chosen_label, chosen_level)

    return FallbackReport(
        resolved=resolved,
        level_names=level_names,
        counts={str(k): int(v) for k, v in fine_counts.items()},
    )


def regime_granularity_presets() -> dict[str, tuple[str, ...]]:
    """The ablation grid from CLAUDE.md section 8."""
    return {
        "global": (),
        "phase": ("phase",),
        "phase+elevation": ("phase", "elevation"),
        "full": ("phase", "elevation", "distance", "candidates"),
    }


def wetlinks_granularity_presets() -> dict[str, tuple[str, ...]]:
    """The ablation grid for the supplied dataset.

    Phase and satellite geometry are unavailable there, so the question the
    ablation answers changes: does conditioning on obstruction, dish pointing
    and time of day recover the conditional risk control that the StarNet
    axes would have given.
    """
    return {
        "global": (),
        "obstruction": ("obstruction",),
        "obstruction+hour": ("obstruction", "hour"),
        "full": ("obstruction", "hour", "azimuth"),
    }
