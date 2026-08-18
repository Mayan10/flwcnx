"""Feature assembly and sequence windowing.

Produces the 11 input features StarNet uses, plus the per-sample phase channel
their periodical embedding needs, plus the regime covariates our calibration
layer conditions on.

One rule governs this module: everything a sample carries must be computable at
the moment the forecast is made. The regime covariates are read from the last
look-back step, never from the horizon, because a regime label that peeked at
the horizon would make the calibration layer look good for the wrong reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flwcnx.config import (
    FEATURE_CLASSES,
    FEATURE_COLUMNS,
    PERIOD_SECONDS,
    TARGET_COL,
    TIME_COL,
    WETLINKS_FEATURE_CLASSES,
    WETLINKS_FEATURE_COLUMNS,
    WETLINKS_SECONDS_BASE_CLASSES,
    WETLINKS_SECONDS_BASE_COLUMNS,
    WETLINKS_SECONDS_FEATURE_CLASSES,
    WETLINKS_SECONDS_FEATURE_COLUMNS,
    FeatureConfig,
)
from flwcnx.state.phase import FIXED_PHASE_OFFSET, PhaseReference, assign_phase, recover_phase
from flwcnx.state.satellite import SatelliteEncoder, resolve_from_frame

# Covariates the regime assigner reads. Carried alongside every sample so the
# calibration layer never has to go back to the frame.
REGIME_COVARIATES: tuple[str, ...] = (
    # StarNet axes.
    "phase_seconds", "elevation_deg", "distance_km", "candidate_count",
    # WetLinks axes. Carried for every dataset because the cost is four float
    # columns per window and the alternative is a second SequenceSet shape.
    "fraction_obstructed", "dish_azimuth_deg", "hour",
)


@dataclass
class Standardizer:
    """Per column mean and standard deviation, fit on the training split only.

    Fitting on the whole trace leaks the test distribution's scale into
    training. It is a small leak and it is still a leak, and CLAUDE.md section
    11 says to stop and check rather than run more experiments on top of one.
    """

    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.ones(0))
    columns: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> Standardizer:
        values = frame[list(columns)].to_numpy(dtype=float)
        self.mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        # A constant column standardises to zero rather than to infinity.
        self.std = np.where(std > 1e-8, std, 1.0)
        self.columns = tuple(columns)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def fit_windows(self, x: np.ndarray, columns: tuple[str, ...]) -> Standardizer:
        """Fit from already windowed inputs, (n, lookback, n_features).

        Needed because the split is computed on windows, not on frame rows, and
        the standardiser must see the training windows only. Overlapping
        windows reweight the sample slightly, which is harmless next to the
        alternative of fitting on data the model is about to be tested on.
        """
        flat = np.asarray(x, dtype=float).reshape(-1, x.shape[-1])
        self.mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        self.std = np.where(std > 1e-8, std, 1.0)
        self.columns = tuple(columns)
        return self

    def inverse_transform_target(self, values: np.ndarray, target_index: int = 0) -> np.ndarray:
        """Undo standardisation for the target channel only.

        Every reported metric is in Mbps, so predictions come back through
        here before anything is measured.
        """
        return values * self.std[target_index] + self.mean[target_index]


@dataclass
class SequenceSet:
    """Windowed sequences plus everything needed to interpret them.

    `regime` holds the covariate values at the forecast origin, which is the
    last look-back step. `origin_time` is that step's timestamp.
    """

    x: np.ndarray                 # (n, lookback, n_features)
    y: np.ndarray                 # (n, horizon), in Mbps
    phase: np.ndarray             # (n, lookback), seconds into the period, [0, 15)
    regime: pd.DataFrame          # (n, len(REGIME_COVARIATES))
    origin_time: np.ndarray       # (n,) datetime64
    segment: np.ndarray           # (n,)
    feature_names: tuple[str, ...]
    target_index: int = 0

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, index: np.ndarray) -> SequenceSet:
        return SequenceSet(
            x=self.x[index], y=self.y[index], phase=self.phase[index],
            regime=self.regime.iloc[index].reset_index(drop=True),
            origin_time=self.origin_time[index], segment=self.segment[index],
            feature_names=self.feature_names, target_index=self.target_index,
        )

    def class_slices(self) -> dict[str, list[int]]:
        """Feature index groups for StarNet's per class periodical embedding.

        Picks the grouping that matches the feature set the sequences actually
        carry, so a WetLinks run gets the WetLinks classes rather than a set of
        empty groups named after satellite columns it does not have.
        """
        lookup = {name: i for i, name in enumerate(self.feature_names)}
        names = set(self.feature_names)
        if names == set(WETLINKS_SECONDS_FEATURE_COLUMNS):
            classes = WETLINKS_SECONDS_FEATURE_CLASSES
        elif names == set(WETLINKS_SECONDS_BASE_COLUMNS):
            classes = WETLINKS_SECONDS_BASE_CLASSES
        elif names == set(WETLINKS_FEATURE_COLUMNS):
            classes = WETLINKS_FEATURE_CLASSES
        else:
            classes = FEATURE_CLASSES
        groups = {
            klass: [lookup[c] for c in cols if c in lookup]
            for klass, cols in classes.items()
        }
        return {k: v for k, v in groups.items() if v}


def build_features(
    frame: pd.DataFrame,
    *,
    config: FeatureConfig | None = None,
    phase_reference: PhaseReference | float | None = None,
    encoder: SatelliteEncoder | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, PhaseReference, SatelliteEncoder]:
    """Turn a normalized frame into a feature frame.

    Returns the frame together with the phase reference and satellite encoder
    actually used, so a caller fitting on train can hand the same two objects
    to the calibration and test splits instead of refitting and leaking.
    """
    config = config or FeatureConfig()
    work = frame.copy()

    if phase_reference is None:
        phase_reference = (
            recover_phase(work) if config.recover_phase
            else PhaseReference(FIXED_PHASE_OFFSET, 0.0, 0, "fallback")
        )
    reference = (
        phase_reference if isinstance(phase_reference, PhaseReference)
        else PhaseReference(float(phase_reference), 0.0, 0, "fallback")
    )

    work["phase_seconds"] = assign_phase(work[TIME_COL], reference)

    # StarNet feed minute and hour as separate channels (t_minute, t_hour in
    # their loader) rather than a single second-of-day. Derived here so the
    # normalized frame does not have to carry redundant columns.
    work["minute"] = work[TIME_COL].dt.minute
    work["hour"] = work[TIME_COL].dt.hour

    # WetLinks carries no serving satellite at all, so the encoder and the
    # handover derivation have nothing to work on. Producing a column of zeros
    # and calling it an encoded satellite ID would be a fabricated feature, so
    # the satellite work is skipped and the columns stay null.
    has_satellites = "sat_id" in work.columns and work["sat_id"].notna().any()
    if has_satellites:
        if encoder is None:
            encoder = SatelliteEncoder().fit(work["sat_id"])
        work["sat_id_encoded"] = encoder.transform(work["sat_id"])
        work = resolve_from_frame(work)
    else:
        encoder = encoder or SatelliteEncoder()
        work["sat_id_encoded"] = np.nan
        work["handover"] = False
        work["dwell_seconds"] = np.nan

    # Time of day is cyclic, and a raw second-of-day tells a model that 23:59
    # and 00:00 are as far apart as it is possible to be. StarNet feed time of
    # day directly, so the raw column stays in the 11 for reproduction and the
    # cyclic pair is carried alongside for the baselines that want it.
    angle = 2 * np.pi * work["second_of_day"] / 86400.0
    work["tod_sin"] = np.sin(angle)
    work["tod_cos"] = np.cos(angle)
    phase_angle = 2 * np.pi * work["phase_seconds"] / PERIOD_SECONDS
    work["phase_sin"] = np.sin(phase_angle)
    work["phase_cos"] = np.cos(phase_angle)

    for column in feature_columns:
        if column not in work.columns:
            raise KeyError(f"feature column {column!r} missing after assembly")
        # Weather is hourly and satellite geometry can drop out for a sample or
        # two. Interpolating within a segment is honest; across a gap it is not.
        if work[column].isna().any():
            work[column] = (
                work.groupby("segment")[column]
                .transform(lambda s: s.interpolate(limit_direction="both"))
            )
    empty = [c for c in feature_columns if work[c].isna().all()]
    if empty:
        raise ValueError(
            f"feature columns {empty} are entirely missing. Zero-filling them "
            "would train the model on a fabricated input that looks like data. "
            "Either attach the source they come from, or select a feature set "
            "that does not include them."
        )
    work[list(feature_columns)] = work[list(feature_columns)].fillna(0.0)
    return work, reference, encoder


def make_sequences(
    frame: pd.DataFrame,
    *,
    config: FeatureConfig | None = None,
    standardizer: Standardizer | None = None,
    feature_names: tuple[str, ...] = FEATURE_COLUMNS,
) -> SequenceSet:
    """Window a feature frame into look-back and horizon pairs.

    Windows never cross a segment boundary. The target stays in Mbps whatever
    the inputs are scaled to, because every metric in the evaluation protocol
    is defined in Mbps and converting back and forth invites an error that is
    invisible in the numbers.
    """
    config = config or FeatureConfig()
    lookback, horizon, stride = config.lookback, config.horizon, config.stride
    total = lookback + horizon

    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise KeyError(f"feature frame is missing {missing}; call build_features first")

    xs, ys, phases, origins, segments = [], [], [], [], []
    regime_rows: list[np.ndarray] = []

    for segment, part in frame.groupby("segment", sort=True):
        if len(part) < total:
            continue
        values = part[list(feature_names)].to_numpy(dtype=float)
        if standardizer is not None:
            values = standardizer.transform(values)
        target = part[TARGET_COL].to_numpy(dtype=float)
        phase = part["phase_seconds"].to_numpy(dtype=float)
        times = part[TIME_COL].to_numpy()
        # reindex rather than select: a dataset that lacks an axis gets NaN
        # for it, and the regime assigner buckets NaN as "na" rather than
        # inventing a value.
        covariates = part.reindex(columns=list(REGIME_COVARIATES)).to_numpy(dtype=float)

        for start in range(0, len(part) - total + 1, stride):
            origin = start + lookback - 1          # last observed step
            xs.append(values[start : start + lookback])
            ys.append(target[start + lookback : start + total])
            phases.append(phase[start : start + lookback])
            origins.append(times[origin])
            segments.append(segment)
            # Read at the origin, never from the horizon.
            regime_rows.append(covariates[origin])

    if not xs:
        raise ValueError(
            f"no window of length {total} fits in any segment; "
            "the trace is shorter than lookback + horizon or too fragmented"
        )

    return SequenceSet(
        x=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.float32),
        phase=np.asarray(phases, dtype=np.float32),
        regime=pd.DataFrame(np.asarray(regime_rows), columns=list(REGIME_COVARIATES)),
        origin_time=np.asarray(origins),
        segment=np.asarray(segments, dtype=np.int64),
        feature_names=tuple(feature_names),
        target_index=list(feature_names).index(TARGET_COL),
    )


def flatten_for_tabular(sequences: SequenceSet, *, horizon_step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Flatten sequences for the tree based models.

    XGBoost has no notion of a sequence, so the look-back window is flattened
    into one row. `horizon_step` selects a single step of the horizon; None
    means the horizon mean, which is what a per-window allocation decision
    actually acts on.
    """
    n = len(sequences)
    x = sequences.x.reshape(n, -1)
    y = sequences.y.mean(axis=1) if horizon_step is None else sequences.y[:, horizon_step]
    return x, y


def standardize_sequences(sequences: SequenceSet, standardizer: Standardizer) -> SequenceSet:
    """Apply a fitted standardiser to a SequenceSet's inputs.

    The target stays in Mbps. Only `x` is scaled.
    """
    scaled = standardizer.transform(sequences.x.astype(float)).astype(np.float32)
    return SequenceSet(
        x=scaled, y=sequences.y, phase=sequences.phase, regime=sequences.regime,
        origin_time=sequences.origin_time, segment=sequences.segment,
        feature_names=sequences.feature_names, target_index=sequences.target_index,
    )
