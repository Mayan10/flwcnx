"""Configuration objects. Dataclasses only, no module level mutable state.

Every experiment writes one of these next to its results so any figure can be
traced back to the settings that produced it (CLAUDE.md section 11).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# The normalized frame
# ---------------------------------------------------------------------------

# Every Source produces exactly these columns, in this order, whatever the
# upstream format was. Downstream layers are allowed to assume this and nothing
# else. Throughput is downlink in Mbps at 1 Hz.
TIME_COL = "timestamp"
TARGET_COL = "throughput_mbps"

LATENCY_COL = "latency_ms"

SAT_COLS = ("sat_id", "elevation_deg", "azimuth_deg", "distance_km", "candidate_count")
TIME_FEATURE_COLS = ("second_of_day", "day_of_week")
# `humidity_pct` is the StarNet weather variable. `precipitation_mm` is kept
# because Open-Meteo supplies it on the live path, but it is not a StarNet
# feature and must not be treated as one. See docs/data.md.
WEATHER_COLS = ("cloud_cover_pct", "pressure_hpa", "humidity_pct", "precipitation_mm")

NORMALIZED_COLUMNS: tuple[str, ...] = (
    TIME_COL,
    TARGET_COL,
    LATENCY_COL,
    *SAT_COLS,
    *TIME_FEATURE_COLS,
    *WEATHER_COLS,
)

# The model inputs, pinned against their loader rather than against the paper
# text. `get_data_loader` in model/NN_TP_ours/sat_dataset.py excludes
# ['timestamp', 'latency', 'throughput'] from the attribute channels, leaving
# twelve, and carries throughput as its own channel. Thirteen in total.
#
# That twelve is exactly the auxiliary variable list BG-CFQS report, which is
# the strongest available evidence that both papers read the same columns.
# CLAUDE.md section 6 says eleven; the discrepancy is recorded in docs/data.md
# rather than resolved by picking whichever number is convenient.
#
# The raw timestamp is excluded deliberately, by both papers: a model given
# absolute time can memorise the trace order instead of learning the link.
FEATURE_COLUMNS: tuple[str, ...] = (
    TARGET_COL,
    "sat_id_encoded",
    "elevation_deg",
    "azimuth_deg",
    "distance_km",
    "candidate_count",
    "phase_seconds",
    "minute",
    "hour",
    "day_of_week",
    "cloud_cover_pct",
    "pressure_hpa",
    "humidity_pct",
)

# StarNet's periodical embedding runs a separate 1D conv over each of four
# feature classes (paper section 5.2), so the grouping is part of the model
# definition rather than a presentation detail.
FEATURE_CLASSES: dict[str, tuple[str, ...]] = {
    "throughput": (TARGET_COL,),
    "satellite": ("sat_id_encoded", "elevation_deg", "azimuth_deg", "distance_km",
                  "candidate_count"),
    "time": ("phase_seconds", "minute", "hour", "day_of_week"),
    "weather": ("cloud_cover_pct", "pressure_hpa", "humidity_pct"),
}

# The feature set for the latency target. StarNet excludes latency from their
# model inputs (their loader drops ['timestamp', 'latency', 'throughput'] from
# the attribute channels and carries throughput as its own), so forecasting
# latency needs its own set rather than a flag.
#
# Latency is the target and therefore its own history channel. Throughput joins
# as an attribute: it is observable at prediction time and a loaded link and a
# slow one are not independent, so excluding it would be throwing away a
# legitimate predictor. Everything else is the StarNet attribute list unchanged,
# which keeps the two targets comparable.
LATENCY_FEATURE_COLUMNS: tuple[str, ...] = (
    LATENCY_COL,
    TARGET_COL,
    "sat_id_encoded",
    "elevation_deg",
    "azimuth_deg",
    "distance_km",
    "candidate_count",
    "phase_seconds",
    "minute",
    "hour",
    "day_of_week",
    "cloud_cover_pct",
    "pressure_hpa",
    "humidity_pct",
)

LATENCY_FEATURE_CLASSES: dict[str, tuple[str, ...]] = {
    "latency": (LATENCY_COL,),
    "throughput": (TARGET_COL,),
    "satellite": ("sat_id_encoded", "elevation_deg", "azimuth_deg", "distance_km",
                  "candidate_count"),
    "time": ("phase_seconds", "minute", "hour", "day_of_week"),
    "weather": ("cloud_cover_pct", "pressure_hpa", "humidity_pct"),
}

# Starlink reschedules on a 15 second cadence. This shows up everywhere from
# the periodical embedding to the regime definition, so it lives here once.
PERIOD_SECONDS: int = 15

# The feature set for the supplied WetLinks data. Different from the StarNet
# one because the datasets are different, not because we changed our minds:
# WetLinks has no satellite geometry at all and its 30 s grid aliases the
# scheduling phase to a constant (docs/supplied-dataset.md). What it does have
# is obstruction, dish pointing, offered load and ping statistics.
#
# `offered_downlink_mbps` is a feature here and never a target. How much
# traffic the link was carrying is a legitimate predictor of a latency spike;
# it is useless as a capacity label because 91% of samples are idle.
WETLINKS_FEATURE_COLUMNS: tuple[str, ...] = (
    TARGET_COL,
    "offered_downlink_mbps",
    "offered_uplink_mbps",
    "ping_drop_pct",
    "ping_stdvar_ms",
    "fraction_obstructed",
    "obstruction_duration",
    "obstruction_interval",
    "dish_azimuth_deg",
    "dish_elevation_deg",
    "minute",
    "hour",
    "day_of_week",
)

# The feature set for the per-second iperf release, which is a different frame
# again: it has real capacity at 1 Hz and co-located weather, but none of the
# dish status fields (obstruction, pointing, ping counters) that the 30 s
# stream carries. `phase_seconds` is a genuine feature here, unlike in the
# status stream where 30 s sampling aliases it to a constant.
WETLINKS_SECONDS_FEATURE_COLUMNS: tuple[str, ...] = (
    TARGET_COL,
    "offered_uplink_mbps",
    "candidate_count",
    "best_elevation_deg",
    "best_distance_km",
    "phase_seconds",
    "minute",
    "hour",
    "day_of_week",
    "temp",
    "humidity_pct",
    "pressure_hpa",
    "precipitation_mm",
)

# The same set without the reconstructed satellite columns, for runs where no
# orbital elements have been fetched. Kept as an explicit alternative rather
# than letting the geometry columns arrive as NaN: an all-null feature that is
# then zero-filled is a fabricated input that costs nothing to train on and
# quietly changes what the model saw.
WETLINKS_SECONDS_BASE_COLUMNS: tuple[str, ...] = tuple(
    c for c in (
        TARGET_COL, "offered_uplink_mbps", "phase_seconds", "minute", "hour",
        "day_of_week", "temp", "humidity_pct", "pressure_hpa", "precipitation_mm",
    )
)

WETLINKS_SECONDS_BASE_CLASSES: dict[str, tuple[str, ...]] = {
    "throughput": (TARGET_COL, "offered_uplink_mbps"),
    "time": ("phase_seconds", "minute", "hour", "day_of_week"),
    "weather": ("temp", "humidity_pct", "pressure_hpa", "precipitation_mm"),
}

WETLINKS_SECONDS_FEATURE_CLASSES: dict[str, tuple[str, ...]] = {
    "throughput": (TARGET_COL, "offered_uplink_mbps"),
    # Reconstructed from propagated elements, not measured. See
    # ingest/spacetrack.py: best_* is the highest satellite in view, a proxy
    # for the serving one.
    "satellite": ("candidate_count", "best_elevation_deg", "best_distance_km"),
    "time": ("phase_seconds", "minute", "hour", "day_of_week"),
    "weather": ("temp", "humidity_pct", "pressure_hpa", "precipitation_mm"),
}

WETLINKS_FEATURE_CLASSES: dict[str, tuple[str, ...]] = {
    "throughput": (TARGET_COL,),
    "satellite": ("fraction_obstructed", "obstruction_duration", "obstruction_interval",
                  "dish_azimuth_deg", "dish_elevation_deg"),
    "time": ("minute", "hour", "day_of_week"),
    "weather": ("offered_downlink_mbps", "offered_uplink_mbps", "ping_drop_pct",
                "ping_stdvar_ms"),
}

LOCATIONS: tuple[str, ...] = ("usa", "canada", "germany")

# BG-CFQS renames the three StarNet locations. Kept so the baseline comparison
# tables line up without anyone having to remember the mapping.
BGCFQS_ALIASES: dict[str, str] = {"usa": "CHI", "germany": "OSN", "canada": "VIC"}


@dataclass(frozen=True)
class DataConfig:
    """Where the traces are and how much of them to use."""

    root: Path = Path("data/starnet")
    location: str = "usa"
    resample_hz: float = 1.0
    # BG-CFQS section 5.1 restricts each location to a contiguous date range.
    # Left as None for StarNet reproduction, set for the BG-CFQS comparison.
    date_start: str | None = None
    date_end: str | None = None
    max_gap_seconds: int = 2  # a larger jump splits the trace into two segments

    def __post_init__(self) -> None:
        if self.location not in LOCATIONS:
            raise ValueError(f"location must be one of {LOCATIONS}, got {self.location!r}")


@dataclass(frozen=True)
class FeatureConfig:
    """Look-back and horizon, plus which optional feature work is switched on."""

    lookback: int = 30   # StarNet default, seconds
    horizon: int = 5     # StarNet reports both 5 and 15; the headline table is 5
    stride: int = 1
    recover_phase: bool = True     # else fall back to the fixed 12/27/42/57 offset
    standardize: bool = True
    #: Which column the model forecasts. Throughput is the default and drives
    #: the allocation story; `latency_ms` is the other target the StarNet traces
    #: support, and it flips the bound direction (see `ExperimentConfig.direction`).
    #: The column must also be present in the feature set, because the model
    #: reads its own history.
    target_column: str = TARGET_COL


@dataclass(frozen=True)
class StarNetConfig:
    """StarNet backbone hyperparameters, exactly as published (section 6)."""

    hidden_size: int = 128
    num_layers: int = 2
    n_features: int = 13
    embed_dim: int = 48          # each feature class is convolved to L x 48
    head_hidden: int = 128
    dropout: float = 0.0
    lr: float = 1e-3
    lr_decay: float = 0.99       # per step, not per epoch
    batch_size: int = 512
    epochs: int = 50
    weight_decay: float = 1e-2   # AdamW default
    grad_clip: float = 1.0


@dataclass(frozen=True)
class BGCFQSConfig:
    """BG-CFQS configuration from Xie et al. section 5.1 and Algorithm 1."""

    lookback: int = 75
    horizon: int = 15
    epsilon: float = 0.35                       # risk budget
    tau_lo: float = 0.15                        # candidate quantile set T
    tau_hi: float = 0.40
    coarse_delta: float = 0.05                  # coarse tolerance
    fine_grid: int = 5                          # M
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05


@dataclass(frozen=True)
class RegimeConfig:
    """Our regime definition. See CLAUDE.md section 7.

    `axes` is ordered coarsest-last: the fallback hierarchy drops axes from the
    end of the tuple, so put the axis you least want to lose first.
    """

    axes: tuple[str, ...] = ("phase", "elevation", "distance", "candidates")
    min_samples: int = 200        # below this a regime falls back up the hierarchy
    phase_open_seconds: float = 2.0   # StarNet section 4: attention concentrates here
    phase_close_seconds: float = 2.0
    elevation_edges: tuple[float, ...] = (45.0, 60.0)   # FCC floor is 25 degrees
    distance_edges: tuple[float, ...] = (645.0,)        # StarNet knee
    candidate_quantiles: tuple[float, ...] = (1 / 3, 2 / 3)
    # WetLinks axes. Obstruction cuts are fit from the data because the scale is
    # tiny and site specific (max 0.023 at uos-rz), so fixed edges would put
    # every sample in one bucket. Azimuth is quartered into compass sectors.
    # The hour buckets split the diurnal cycle the latency actually follows:
    # 29.8 ms at 08:00 against 34.9 ms at 02:00.
    obstruction_quantiles: tuple[float, ...] = (0.5, 0.9)
    azimuth_sectors: int = 4
    hour_edges: tuple[float, ...] = (6.0, 12.0, 18.0)
    # Look-back window axes. Quartiles rather than terciles because the risk
    # failure these are meant to catch is concentrated in the bottom slice, and
    # a quarter is close to the P30 subset the conditional metrics report on.
    level_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    volatility_quantiles: tuple[float, ...] = (0.5, 0.9)


@dataclass(frozen=True)
class CalibrationConfig:
    """How the point forecast becomes a lower bound."""

    method: str = "regime_conformal"   # regime_conformal | regime_bgcfqs | conformal | bgcfqs
    epsilon: float = 0.35
    regime: RegimeConfig = field(default_factory=RegimeConfig)


@dataclass(frozen=True)
class SplitConfig:
    """Temporal splits only. Random splits flatter these models (Horizon)."""

    scheme: str = "temporal"          # temporal | contiguous_82 | leave_one_location_out
    train_frac: float = 0.6
    calib_frac: float = 0.2           # never used for fitting, only for the operating point
    test_frac: float = 0.2


@dataclass(frozen=True)
class DecisionConfig:
    """Admission control and congestion detection."""

    bandwidth_per_session_mbps: float = 10.0
    congestion_window: int = 5        # W consecutive slots below commitment
    commitment_mbps: float = 50.0


#: Which dataset a run is against. This selects the feature set, the regime
#: ablation grid and the risk direction together, because on these two datasets
#: those three choices are not independent: WetLinks has no satellite geometry,
#: aliases the scheduling phase, and its usable target is latency.
DATASETS = ("starnet", "wetlinks", "wetlinks_seconds")


@dataclass(frozen=True)
class ExperimentConfig:
    """One full run. Serialised next to its results."""

    name: str = "default"
    dataset: str = "starnet"
    # Whether reconstructed satellite geometry is available. Recorded in the
    # config snapshot because it changes the input width, so two runs that
    # differ only in this are not comparable without saying so.
    geometry: bool = True
    seed: int = 1337
    device: str = "auto"              # auto | cpu | cuda | mps
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: StarNetConfig = field(default_factory=StarNetConfig)
    bgcfqs: BGCFQSConfig = field(default_factory=BGCFQSConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        # The target decides the set before the dataset does: forecasting
        # latency needs latency as an input channel, and StarNet's own set
        # deliberately excludes it.
        if self.features.target_column == LATENCY_COL:
            return LATENCY_FEATURE_COLUMNS
        if self.dataset == "wetlinks":
            return WETLINKS_FEATURE_COLUMNS
        if self.dataset == "wetlinks_seconds":
            return (WETLINKS_SECONDS_FEATURE_COLUMNS if self.geometry
                    else WETLINKS_SECONDS_BASE_COLUMNS)
        return FEATURE_COLUMNS

    @property
    def direction(self) -> str:
        """Capacity is bounded from below; latency from above.

        `wetlinks` is the 30 s status stream, whose only usable target is
        latency. `wetlinks_seconds` is the per-second iperf release, which is
        real capacity, so it goes back to the lower bound the whole
        allocation story is built on.

        The target column wins over the dataset: a latency target is an upper
        bound wherever it is run, because the risk is always promising a delay
        the link will not meet.
        """
        if self.features.target_column == LATENCY_COL:
            return "upper"
        return "upper" if self.dataset == "wetlinks" else "lower"

    def to_dict(self) -> dict[str, Any]:
        return _as_jsonable(dataclasses.asdict(self))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


def _as_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_as_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def seed_everything(seed: int) -> int:
    """Seed every RNG we touch and return the seed so callers can record it."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device. Delegates to `thalweg.device`.

    Kept here because every call site imports it from this module. The real
    logic moved once the original version turned out to be Mac-shaped: it
    called `torch.backends.mps.is_available()` unguarded, which raises on a
    torch build without that backend.
    """
    from thalweg.device import resolve_device as _resolve

    return _resolve(preference)
