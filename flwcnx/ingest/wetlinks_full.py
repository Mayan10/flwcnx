"""The full WetLinks release, which the supplied CSVs were only a subset of.

Source: https://github.com/sys-uos/WetLinks (Laniewski et al., TMA 2024).

Two files matter here and neither is in the supplied subset:

  `iperf_cleaned_seconds_*.csv`  per-second iperf throughput. This is measured
      capacity, not offered load, and it is sampled at 1 Hz, so both of the
      limitations the supplied subset imposed are lifted: the 15 s scheduling
      phase is no longer aliased, and the target is a real capacity figure.

  `analysis_data_*.csv`  one row per measurement run, roughly every 3 minutes,
      carrying co-located weather-station readings alongside the run's mean
      throughput and ping statistics.

Sentinel values. The weather columns carry -9999 style missing markers that the
upstream preprocessing then averaged with real readings, producing values like
-6666 and -3333. They affect only 61 of 68,597 rows at Osnabrück, but those
rows are not missing at random: their median throughput is 190 Mbps against 212
for the rest. So they are masked to NaN rather than dropped, and the masking is
by physical plausibility rather than by matching a magic number, which also
catches the partially-averaged cases.
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
from flwcnx.ingest.wetlinks import SITE_COORDINATES

#: Physically possible ranges for the weather columns. Anything outside is a
#: sentinel or a sentinel averaged with real readings, and becomes NaN.
WEATHER_BOUNDS: dict[str, tuple[float, float]] = {
    "temp": (-60.0, 60.0),
    "dewpt": (-60.0, 60.0),
    "windchill": (-80.0, 60.0),
    "humidity": (0.0, 100.0),
    "winddir": (0.0, 360.0),
    "windspeed": (0.0, 120.0),
    "windgust": (0.0, 150.0),
    "rain": (0.0, 500.0),
    "solarradiation": (0.0, 1500.0),
    "uv": (0.0, 20.0),
    "barom": (800.0, 1100.0),
}

#: Upstream weather name -> our normalized column, where one exists.
WEATHER_RENAME: dict[str, str] = {
    "humidity": "humidity_pct",
    "barom": "pressure_hpa",
    "rain": "precipitation_mm",
}


def clean_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Mask implausible weather readings to NaN, in place on a copy."""
    work = frame.copy()
    for column, (low, high) in WEATHER_BOUNDS.items():
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        work[column] = values.where((values >= low) & (values <= high))
    return work


def weather_sentinel_report(frame: pd.DataFrame) -> pd.DataFrame:
    """How much of each weather column is implausible. Worth printing once."""
    rows = []
    for column, (low, high) in WEATHER_BOUNDS.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = ~values.between(low, high) & values.notna()
        rows.append({"column": column, "n_bad": int(bad.sum()),
                     "fraction": float(bad.mean()),
                     "min_observed": float(values.min()),
                     "max_observed": float(values.max())})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class WetLinksFullConfig:
    root: Path = Path("data/wetlinks_full")
    site: str = "Osnabruck"                 # Osnabruck or Enschede
    #: A gap this long ends an iperf burst. Runs are ~3 minutes apart and each
    #: burst is contiguous at 1 Hz, so anything above a few seconds is a break.
    max_gap_seconds: float = 3.0
    join_weather: bool = True
    #: Weather is per measurement run, so a per-second sample takes the reading
    #: from its own run. Beyond this the join is left null rather than carried.
    weather_tolerance_minutes: int = 10

    @property
    def seconds_path(self) -> Path:
        return self.root / f"iperf_cleaned_seconds_{self.site}.csv"

    @property
    def analysis_path(self) -> Path:
        return self.root / f"analysis_data_{self.site}.csv"


class WetLinksSecondsSource(Source):
    """Per-second measured capacity, optionally joined to weather.

    This is the source that restores the original framing: 1 Hz throughput
    where the throughput is what the link carried under load, so the scheduling
    phase is recoverable and the target means what StarNet's target means.
    """

    def __init__(self, config: WetLinksFullConfig | None = None) -> None:
        self.config = config or WetLinksFullConfig()
        self.location = f"wetlinks-{self.config.site.lower()}"
        self._frame: pd.DataFrame | None = None

    def _load_weather(self) -> pd.DataFrame | None:
        path = self.config.analysis_path
        if not (self.config.join_weather and path.exists()):
            return None
        analysis = pd.read_csv(path, low_memory=False)
        analysis = clean_weather(analysis)
        analysis[TIME_COL] = pd.to_datetime(analysis["timestamp_start"], errors="coerce")
        keep = [TIME_COL, *[c for c in WEATHER_BOUNDS if c in analysis.columns]]
        for extra in ("ping_avg", "ping_worst", "ping_best", "ping_stddev",
                      "ping_packet_loss"):
            if extra in analysis.columns:
                keep.append(extra)
        weather = analysis[keep].dropna(subset=[TIME_COL]).sort_values(TIME_COL)
        return weather.rename(columns=WEATHER_RENAME)

    def load_frame(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame
        path = self.config.seconds_path
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Fetch the full release with "
                "scripts/download_data.py --dataset wetlinks."
            )
        raw = pd.read_csv(path, low_memory=False)

        frame = pd.DataFrame(index=raw.index)
        frame[TIME_COL] = pd.to_datetime(raw["timestamp_start"], errors="coerce")
        frame["site_name"] = raw["site_name"].astype(str)
        # iperf reports bits per second.
        frame[TARGET_COL] = pd.to_numeric(raw["download"], errors="coerce") / 1e6
        frame["offered_uplink_mbps"] = pd.to_numeric(raw["upload"], errors="coerce") / 1e6
        frame["transport_protocol"] = raw.get("transport_protocol", "unknown")
        frame = frame[frame[TIME_COL].notna() & frame[TARGET_COL].notna()]

        times = frame[TIME_COL].dt
        frame["second_of_day"] = times.hour * 3600 + times.minute * 60 + times.second
        frame["day_of_week"] = times.dayofweek

        coordinates = frame["site_name"].map(SITE_COORDINATES)
        frame["lat"] = coordinates.map(lambda c: c[0] if c else np.nan)
        frame["lon"] = coordinates.map(lambda c: c[1] if c else np.nan)

        weather = self._load_weather()
        if weather is not None:
            frame = pd.merge_asof(
                frame.sort_values(TIME_COL), weather, on=TIME_COL, direction="nearest",
                tolerance=pd.Timedelta(minutes=self.config.weather_tolerance_minutes),
            )

        # Contract columns this release still does not have. Satellite geometry
        # is absent here too; it has to come from propagated elements
        # (ingest/spacetrack.py) and is a reconstruction, not a measurement.
        for column in ("sat_id", "elevation_deg", "azimuth_deg", "distance_km",
                       "candidate_count"):
            if column not in frame.columns:
                frame[column] = np.nan
        for column in ("cloud_cover_pct", "pressure_hpa", "humidity_pct",
                       "precipitation_mm"):
            if column not in frame.columns:
                frame[column] = np.nan
        frame[LATENCY_COL] = frame.get("ping_avg", np.nan)

        frame = segment_frame(frame.reset_index(drop=True),
                              max_gap_seconds=self.config.max_gap_seconds)
        extra = [c for c in frame.columns if c not in NORMALIZED_COLUMNS and c != "segment"]
        self._frame = validate_frame(frame)[list(NORMALIZED_COLUMNS) + ["segment"] + extra]
        return self._frame

    def iter_windows(self, length: int, stride: int = 1) -> Iterator[RawWindow]:
        yield from iter_windows_from_frame(self.load_frame(), length, stride, self.location)

    def describe(self) -> dict:
        frame = self.load_frame()
        burst = frame.groupby("segment").size()
        gaps = frame.groupby("segment")[TIME_COL].diff().dt.total_seconds()
        return {
            "location": self.location,
            "samples": int(len(frame)),
            "bursts": int(frame["segment"].nunique()),
            "median_burst_samples": int(burst.median()),
            "max_burst_samples": int(burst.max()),
            "within_burst_median_gap_s": float(gaps.median()),
            "span_days": round(
                (frame[TIME_COL].max() - frame[TIME_COL].min()).total_seconds() / 86400.0, 1
            ),
            "throughput_median_mbps": round(float(frame[TARGET_COL].median()), 2),
            "throughput_p05_mbps": round(float(frame[TARGET_COL].quantile(0.05)), 2),
            "throughput_p95_mbps": round(float(frame[TARGET_COL].quantile(0.95)), 2),
            "weather_joined": bool(frame.get("temp", pd.Series(dtype=float)).notna().any()),
        }


# ---------------------------------------------------------------------------
# Reconstructed satellite geometry
# ---------------------------------------------------------------------------


def load_cached_elements(cache_dir: Path = Path("data/tle_cache")) -> pd.DataFrame:
    """Read every cached Space-Track day into one frame."""
    import gzip
    import json

    rows = []
    for path in sorted(Path(cache_dir).glob("gp_*.json.gz")):
        with gzip.open(path, "rt") as handle:
            rows.extend(json.load(handle))
    if not rows:
        raise FileNotFoundError(
            f"no cached elements under {cache_dir}. Run "
            "scripts/fetch_elements.py after putting credentials in .env."
        )
    frame = pd.DataFrame(rows)
    frame["EPOCH"] = pd.to_datetime(frame["EPOCH"], errors="coerce", utc=True)
    return frame.dropna(subset=["EPOCH"])


def attach_geometry(
    frame: pd.DataFrame,
    elements: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    altitude_m: float = 0.0,
    refresh_hours: float = 12.0,
) -> pd.DataFrame:
    """Add reconstructed candidate count and best-in-view geometry.

    **This is a reconstruction, not a measurement.** WetLinks does not record
    which satellite served the terminal. What is computed here is the number of
    satellites above the service elevation floor, and the elevation and
    distance of the highest one, from orbital elements propagated to the
    measurement time. `geometry_source` is set to "reconstructed" on every row
    so the distinction survives into any table built from this frame.

    Computed once per burst rather than once per sample. An iperf run lasts 15
    seconds, over which a satellite at 550 km moves about half a degree in
    elevation, which is far below the width of the regime buckets these values
    feed. Doing it per sample would cost 15 times as much for no resolution
    that the downstream use can see.
    """
    from flwcnx.ingest.spacetrack import elements_to_tles, latest_per_satellite
    from flwcnx.ingest.tle import Observer, visible_counts

    if frame.empty:
        return frame
    observer = Observer(latitude, longitude, altitude_m)
    work = frame.copy()

    anchors = work.groupby("segment")[TIME_COL].min().sort_values()
    # Element sets are re-selected every `refresh_hours`, so SGP4 never
    # propagates further than that from an epoch.
    window = (anchors.astype("int64") // int(refresh_hours * 3600 * 1e9))

    pieces = []
    for _, group in anchors.groupby(window):
        midpoint = group.iloc[len(group) // 2].to_pydatetime()
        current = latest_per_satellite(elements, midpoint)
        if current.empty:
            continue
        tles = elements_to_tles(current)
        counts = visible_counts(group.to_numpy(dtype="datetime64[ns]"), observer, tles)
        counts["segment"] = group.index.to_numpy()
        pieces.append(counts)

    if not pieces:
        raise ValueError("no element sets covered the measurement window")
    geometry = pd.concat(pieces, ignore_index=True).drop(columns=[TIME_COL])

    # The loader lays down empty placeholders for every normalized column, so
    # `candidate_count` already exists and is all null. Left in place it would
    # win the merge and silently suffix the real values to `candidate_count_geo`,
    # which is how this column came back 100% null the first time it was run.
    work = work.drop(columns=[c for c in ("candidate_count", "elevation_deg",
                                          "distance_km") if c in work.columns])
    work = work.merge(geometry, on="segment", how="left", suffixes=("", "_geo"))
    # Canonical names so the existing regime axes work unchanged. The alias and
    # the marker column are what keep the provenance visible.
    work["elevation_deg"] = work["best_elevation_deg"]
    work["distance_km"] = work["best_distance_km"]
    work["geometry_source"] = "reconstructed"
    return work
