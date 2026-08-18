"""WetLinks: the dataset supplied with the problem statement.

Laniewski, Lanfer, Meijerink, van Rijswijk-Deij, Aschenbruck, TMA 2024
(references.bib entry 9). Two European vantage points, `uos-rz` (University of
Osnabrück) and `utwente` (University of Twente), sampled every 30 seconds from
the terminal's gRPC status interface.

Read `docs/supplied-dataset.md` before using this. The three things that matter:

  1. Sampling is 30 s and 30 is a multiple of 15, so the scheduling phase is
     aliased to a constant. There is no phase information in this data.
  2. `direction_azimuth` / `direction_elevation` are the dish's own boresight,
     not satellite geometry. There is no serving satellite here at all.
  3. `downlink` is offered load, not capacity, for 91% of samples. Only the
     periodic speedtest bursts measure what the link could carry.

Because of (3) this module never hands `downlink` over as a throughput target
without being asked to. `target="latency"` is the default, and the capacity
path is opt-in through `SpeedtestFilter` so that nobody trains a throughput
model on idle telemetry by accident.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flwcnx.config import LATENCY_COL, NORMALIZED_COLUMNS, TARGET_COL, TIME_COL
from flwcnx.ingest.base import (
    RawWindow,
    Source,
    iter_windows_from_frame,
    segment_frame,
    validate_frame,
)

#: Site coordinates. uos-rz carries lat/lon in 29% of its rows and the values
#: below are its measured medians; utwente carries none, so its coordinates are
#: the university campus. Needed for the TLE candidate count and the weather
#: join, both of which are location keyed.
SITE_COORDINATES: dict[str, tuple[float, float]] = {
    "uos-rz": (52.2853, 8.0220),      # University of Osnabrück, measured
    "utwente": (52.2396, 6.8567),     # University of Twente campus, Enschede
}

SITE_TO_LOCATION: dict[str, str] = {"uos-rz": "germany", "utwente": "germany"}

#: Below this the sample is idle telemetry rather than a capacity measurement.
#: 90.7% of rows sit under 0.1 Mbps and the speedtest bursts have a median of
#: 230.8 Mbps, so the boundary is wide and the exact value is not delicate.
SPEEDTEST_FLOOR_MBPS: float = 50.0

#: Casparsen's period threshold, reused here as the spike definition. 3.15% of
#: pop_ping_latency samples at uos-rz exceed it.
SPIKE_THRESHOLD_MS: float = 50.0

TARGETS = ("latency", "latency_pop", "capacity")


@dataclass(frozen=True)
class WetLinksConfig:
    root: Path = Path("data/supplied")
    site: str | None = None            # None means every site in the files
    target: str = "latency"
    # None resolves per target: the latency series is on a 30 s grid, while the
    # capacity series is speedtests roughly 6 minutes apart, so one gap
    # threshold cannot serve both. With the latency threshold every speedtest
    # becomes its own segment and no window can ever be formed.
    max_gap_seconds: float | None = None
    connected_only: bool = True
    speedtest_floor_mbps: float = SPEEDTEST_FLOOR_MBPS

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {self.target!r}")

    @property
    def gap_seconds(self) -> float:
        if self.max_gap_seconds is not None:
            return self.max_gap_seconds
        # 3 missed samples on the 30 s grid; 2 missed speedtests on the 6 min one.
        return 900.0 if self.target == "capacity" else 90.0


def _mbps(series: pd.Series) -> pd.Series:
    """The dish reports bits per second."""
    return pd.to_numeric(series, errors="coerce") / 1e6


def normalize_wetlinks(raw: pd.DataFrame, config: WetLinksConfig) -> pd.DataFrame:
    """Map WetLinks columns onto the normalized contract.

    The awkward part is `throughput_mbps`. On the latency targets it is still
    populated, because offered load is a legitimate *feature* even where it is
    a useless target: how much traffic the link was carrying is exactly the
    kind of thing that predicts a latency spike. It is the target selection,
    not the column, that has to be careful.
    """
    frame = pd.DataFrame(index=raw.index)
    frame[TIME_COL] = pd.to_datetime(raw["timestamp"], errors="coerce")

    if config.connected_only and "state" in raw.columns:
        frame = frame[raw["state"] == "CONNECTED"]
        raw = raw.loc[frame.index]

    frame["site_name"] = raw["site_name"].astype(str)
    frame["offered_downlink_mbps"] = _mbps(raw["downlink"])
    frame["offered_uplink_mbps"] = _mbps(raw["uplink"])
    frame["pop_ping_latency_ms"] = pd.to_numeric(raw["pop_ping_latency"], errors="coerce")
    frame["mean_ping_latency_ms"] = pd.to_numeric(raw["mean_ping_latency"], errors="coerce")
    frame["ping_drop_pct"] = pd.to_numeric(raw["ping_drop"], errors="coerce")
    frame["ping_stdvar_ms"] = pd.to_numeric(raw["ping_stdvar"], errors="coerce")
    frame["fraction_obstructed"] = pd.to_numeric(raw["fraction_obstructed"], errors="coerce")
    frame["obstruction_duration"] = pd.to_numeric(raw["obstruction_duration"], errors="coerce")
    frame["obstruction_interval"] = pd.to_numeric(raw["obstruction_interval"], errors="coerce")
    # The dish's own pointing. Named to make it impossible to mistake for the
    # serving satellite's look angles, which this dataset does not contain.
    frame["dish_azimuth_deg"] = pd.to_numeric(raw["direction_azimuth"], errors="coerce")
    frame["dish_elevation_deg"] = pd.to_numeric(raw["direction_elevation"], errors="coerce")
    frame["is_speedtest"] = frame["offered_downlink_mbps"] >= config.speedtest_floor_mbps

    coordinates = frame["site_name"].map(SITE_COORDINATES)
    frame["lat"] = pd.to_numeric(raw.get("lat"), errors="coerce")
    frame["lon"] = pd.to_numeric(raw.get("lon"), errors="coerce")
    frame["lat"] = frame["lat"].fillna(coordinates.map(lambda c: c[0] if c else np.nan))
    frame["lon"] = frame["lon"].fillna(coordinates.map(lambda c: c[1] if c else np.nan))

    # Target selection.
    if config.target == "latency":
        frame[TARGET_COL] = frame["mean_ping_latency_ms"]
    elif config.target == "latency_pop":
        frame[TARGET_COL] = frame["pop_ping_latency_ms"]
    else:
        # Capacity: keep only the speedtest bursts. These are isolated samples
        # roughly six minutes apart, so the resulting series is 6 minute
        # granularity, not 30 s. A look-back is a look-back over speedtests.
        frame = frame[frame["is_speedtest"]]
        frame[TARGET_COL] = frame["offered_downlink_mbps"]

    frame[LATENCY_COL] = frame["pop_ping_latency_ms"]

    # Contract columns this dataset genuinely does not have. Left as NaN rather
    # than zero, so that a regime axis built on them is empty rather than
    # silently uniform.
    for column in ("sat_id", "elevation_deg", "azimuth_deg", "distance_km",
                   "candidate_count"):
        frame[column] = np.nan
    for column in ("cloud_cover_pct", "pressure_hpa", "humidity_pct", "precipitation_mm"):
        frame[column] = np.nan

    times = frame[TIME_COL].dt
    frame["second_of_day"] = times.hour * 3600 + times.minute * 60 + times.second
    frame["day_of_week"] = times.dayofweek

    frame = frame[frame[TIME_COL].notna() & frame[TARGET_COL].notna()]
    frame = frame.sort_values([ "site_name", TIME_COL], kind="stable")

    # Segment per site, then offset so two sites never share a segment id.
    parts, offset = [], 0
    for _, part in frame.groupby("site_name", sort=True):
        part = segment_frame(part.reset_index(drop=True),
                             max_gap_seconds=config.gap_seconds)
        part["segment"] = part["segment"] + offset
        offset = int(part["segment"].max()) + 1
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)

    extra = [c for c in frame.columns if c not in NORMALIZED_COLUMNS and c != "segment"]
    return validate_frame(frame)[list(NORMALIZED_COLUMNS) + ["segment"] + extra]


class WetLinksSource(Source):
    """Reads the supplied WetLinks CSVs."""

    def __init__(self, config: WetLinksConfig | None = None) -> None:
        self.config = config or WetLinksConfig()
        self.location = self.config.site or "wetlinks"
        self._frame: pd.DataFrame | None = None

    def _files(self) -> list[Path]:
        root = self.config.root
        if root.is_file():
            return [root]
        files = sorted(p for p in root.glob("*.csv") if p.is_file())
        if not files:
            raise FileNotFoundError(f"no CSV files under {root}")
        return files

    def load_frame(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame
        raw = pd.concat([pd.read_csv(f) for f in self._files()], ignore_index=True)
        if self.config.site is not None:
            raw = raw[raw["site_name"] == self.config.site]
            if raw.empty:
                raise ValueError(f"site {self.config.site!r} not present in {self._files()}")
        self._frame = normalize_wetlinks(raw, self.config).reset_index(drop=True)
        return self._frame

    def iter_windows(self, length: int, stride: int = 1) -> Iterator[RawWindow]:
        yield from iter_windows_from_frame(self.load_frame(), length, stride, self.location)

    def describe(self) -> dict:
        frame = self.load_frame()
        described = {
            "location": self.location,
            "target": self.config.target,
            "samples": int(len(frame)),
            "sites": sorted(frame["site_name"].unique().tolist()),
            "segments": int(frame["segment"].nunique()),
            "span_days": round(
                (frame[TIME_COL].max() - frame[TIME_COL].min()).total_seconds() / 86400.0, 1
            ),
            "median_sampling_seconds": float(
                frame.groupby("segment")[TIME_COL].diff().dt.total_seconds().median()
            ),
            "target_median": round(float(frame[TARGET_COL].median()), 3),
            "target_p99": round(float(frame[TARGET_COL].quantile(0.99)), 3),
        }
        if self.config.target != "capacity":
            described["spike_rate"] = round(
                float((frame["pop_ping_latency_ms"] > SPIKE_THRESHOLD_MS).mean()), 5
            )
        return described


def phase_is_aliased(frame: pd.DataFrame, period_seconds: float = 15.0,
                     tolerance: float = 1.0) -> bool:
    """True when the sampling grid destroys the scheduling phase.

    A guard, not a diagnostic. WetLinks samples every 30 s against a 15 s
    period, so every sample sits at the same phase and the standard deviation
    collapses to 0.156 s. Any code about to build a phase regime axis should
    call this first and refuse rather than produce a constant feature that
    looks like a working one.
    """
    seconds = frame[TIME_COL].astype("int64").to_numpy() / 1e9
    phase = np.mod(seconds, period_seconds)
    # Circular standard deviation, so the wrap point does not fake a spread.
    angle = 2 * np.pi * phase / period_seconds
    resultant = np.hypot(np.mean(np.cos(angle)), np.mean(np.sin(angle)))
    circular_std = np.sqrt(-2 * np.log(max(resultant, 1e-12))) * period_seconds / (2 * np.pi)
    return bool(circular_std < tolerance)
