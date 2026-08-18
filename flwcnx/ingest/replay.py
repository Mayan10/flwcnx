"""Replay of the published StarNet CSV traces. This is the primary path.

Source of the data: https://github.com/ConnectedSystemsLab/StarNet

The upstream column names are not fixed by any spec, so mapping is done by
alias table rather than by position. `inspect_schema` reports exactly which
upstream column fed each normalized column and which upstream columns were
ignored. Run it once on a new drop of the data before trusting anything
downstream.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from flwcnx.config import (
    NORMALIZED_COLUMNS,
    TARGET_COL,
    TIME_COL,
    DataConfig,
)
from flwcnx.ingest.base import (
    RawWindow,
    Source,
    iter_windows_from_frame,
    segment_frame,
    validate_frame,
)

# Normalized name -> upstream candidates, most specific first. Matching is done
# on a normalized key (lowercase, alphanumerics only) so `Elevation (deg)`,
# `elevation_deg` and `elevationDeg` all collapse to the same thing.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    TIME_COL: ("timestamp", "time", "datetime", "ts", "unixtime", "epoch"),
    TARGET_COL: (
        "throughputmbps", "throughput", "downlinkthroughput", "downlinkbps",
        "downlinkthroughputmbps", "dlthroughput", "tputmbps", "tput", "bandwidth",
        "downlink",
    ),
    "sat_id": ("satid", "satelliteid", "servingsatellite", "servingsat", "noradid",
               "sat", "satellite"),
    "elevation_deg": ("elevationdeg", "elevation", "elev", "el"),
    "azimuth_deg": ("azimuthdeg", "azimuth", "azim", "az"),
    "distance_km": ("distancekm", "distance", "range", "rangekm", "slantrange", "dist"),
    "candidate_count": ("candidatecount", "numcandidates", "candidates", "ncandidates",
                        "visiblesatellites", "candidatesatellites", "numsats"),
    "second_of_day": ("secondofday", "timeofday", "tod"),
    "day_of_week": ("dayofweek", "weekday", "dow"),
    "precipitation_mm": ("precipitationmm", "precipitation", "precip", "rain",
                         "rainfall", "precipitationrate"),
    "cloud_cover_pct": ("cloudcoverpct", "cloudcover", "cloudiness", "clouds",
                        "cloudcoverage", "totalcloudcover"),
    "pressure_hpa": ("pressurehpa", "pressure", "surfacepressure", "airpressure",
                     "msl", "pressuremsl"),
}

# From the StarNet paper, reproduced in CLAUDE.md section 5. The loader checks
# itself against these. A loader that reads the wrong column makes every number
# downstream meaningless, and this is the cheapest place to catch it.
PUBLISHED_STATS: dict[str, dict[str, float]] = {
    "usa": {
        "trace_minutes": 41252, "samples": 2475163,
        "unique_satellites": 6052, "handovers": 86808,
    },
    "canada": {
        "trace_minutes": 2417, "samples": 145053,
        "unique_satellites": 3166, "handovers": 7257,
    },
    "germany": {
        "trace_minutes": 10221, "samples": 613295,
        "unique_satellites": 3956, "handovers": 26782,
    },
}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def map_columns(columns: list[str]) -> dict[str, str]:
    """Resolve upstream columns to normalized names.

    Returns normalized name -> upstream name. Ambiguity is resolved by alias
    order, and an upstream column is never mapped to two normalized names.
    """
    by_key: dict[str, list[str]] = {}
    for col in columns:
        by_key.setdefault(_key(col), []).append(col)

    resolved: dict[str, str] = {}
    claimed: set[str] = set()
    for normalized, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            hits = [c for c in by_key.get(alias, []) if c not in claimed]
            if hits:
                resolved[normalized] = hits[0]
                claimed.add(hits[0])
                break
    return resolved


def inspect_schema(path: str | Path, nrows: int = 2000) -> dict[str, object]:
    """Report how a CSV would be mapped, without building anything on it."""
    head = pd.read_csv(path, nrows=nrows)
    resolved = map_columns(list(head.columns))
    return {
        "path": str(path),
        "upstream_columns": list(head.columns),
        "resolved": resolved,
        "unmapped_normalized": [c for c in NORMALIZED_COLUMNS if c not in resolved],
        "ignored_upstream": [c for c in head.columns if c not in set(resolved.values())],
        "dtypes": {c: str(t) for c, t in head.dtypes.items()},
        "rows_sampled": int(len(head)),
    }


def _parse_time(series: pd.Series) -> pd.Series:
    """Accept epoch seconds, epoch milliseconds, or a parseable string."""
    if pd.api.types.is_numeric_dtype(series):
        finite = series.dropna()
        scale = "ms" if not finite.empty and finite.abs().median() > 1e11 else "s"
        return pd.to_datetime(series, unit=scale, utc=True).dt.tz_localize(None)
    parsed = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    return parsed.dt.tz_localize(None)


def normalize_frame(raw: pd.DataFrame, *, max_gap_seconds: float = 2.0) -> pd.DataFrame:
    """Rename, coerce and derive until the frame satisfies the contract."""
    resolved = map_columns(list(raw.columns))
    if TIME_COL not in resolved:
        raise ValueError(
            f"no timestamp column found among {list(raw.columns)}. "
            "Add the upstream name to COLUMN_ALIASES rather than guessing here."
        )
    if TARGET_COL not in resolved:
        raise ValueError(
            f"no throughput column found among {list(raw.columns)}. "
            "Add the upstream name to COLUMN_ALIASES."
        )

    frame = pd.DataFrame(index=raw.index)
    for normalized, upstream in resolved.items():
        frame[normalized] = raw[upstream]

    frame[TIME_COL] = _parse_time(frame[TIME_COL])
    frame = frame[frame[TIME_COL].notna()]

    # Throughput is reported in bps by some of the tooling. Convert on the
    # magnitude rather than on the column name, and record nothing silently:
    # the describe() check against the published mean is the backstop.
    tput = pd.to_numeric(frame[TARGET_COL], errors="coerce")
    if tput.notna().any() and tput.abs().median() > 1e5:
        tput = tput / 1e6
    frame[TARGET_COL] = tput

    # Time of day and day of week are derivable, so derive them if absent
    # rather than requiring the upstream to have precomputed them.
    if "second_of_day" not in frame.columns:
        t = frame[TIME_COL].dt
        frame["second_of_day"] = t.hour * 3600 + t.minute * 60 + t.second
    if "day_of_week" not in frame.columns:
        frame["day_of_week"] = frame[TIME_COL].dt.dayofweek

    for col in NORMALIZED_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
        elif col not in (TIME_COL, "sat_id"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.drop_duplicates(subset=[TIME_COL], keep="first")
    frame = frame[frame[TARGET_COL].notna()]
    frame = segment_frame(frame, max_gap_seconds=max_gap_seconds)
    return validate_frame(frame)


class ReplaySource(Source):
    """Reads one location's StarNet traces off disk.

    Accepts either a single CSV or a directory of them. Directories are read in
    sorted filename order and concatenated before segmentation, so a location
    split across daily files still yields correct segments.
    """

    def __init__(self, config: DataConfig | None = None, *, path: str | Path | None = None,
                 location: str = "usa") -> None:
        self.config = config or DataConfig(location=location)
        self.location = self.config.location
        self._path = Path(path) if path is not None else self.config.root / self.location
        self._frame: pd.DataFrame | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _files(self) -> list[Path]:
        if self._path.is_file():
            return [self._path]
        if not self._path.exists():
            raise FileNotFoundError(
                f"no traces at {self._path}. Fetch them with "
                "`python scripts/download_data.py --dataset starnet`."
            )
        files = sorted(p for p in self._path.rglob("*.csv") if p.is_file())
        files += sorted(p for p in self._path.rglob("*.csv.gz") if p.is_file())
        if not files:
            raise FileNotFoundError(f"no CSV files under {self._path}")
        return files

    def load_frame(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame
        parts = [pd.read_csv(f) for f in self._files()]
        raw = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        frame = normalize_frame(raw, max_gap_seconds=self.config.max_gap_seconds)

        # BG-CFQS section 5.1 restricts each location to a contiguous date
        # range. Applying it here keeps the baseline comparison apples to
        # apples without a second loader.
        if self.config.date_start is not None:
            frame = frame[frame[TIME_COL] >= pd.Timestamp(self.config.date_start)]
        if self.config.date_end is not None:
            frame = frame[frame[TIME_COL] <= pd.Timestamp(self.config.date_end)]
        if self.config.date_start or self.config.date_end:
            frame = segment_frame(frame.reset_index(drop=True),
                                  max_gap_seconds=self.config.max_gap_seconds)

        self._frame = frame.reset_index(drop=True)
        return self._frame

    def iter_windows(self, length: int, stride: int = 1) -> Iterator[RawWindow]:
        yield from iter_windows_from_frame(self.load_frame(), length, stride, self.location)

    def verify_against_published(self, tolerance: float = 0.05) -> dict[str, object]:
        """Compare loader output to the StarNet paper's dataset table.

        This is a check on the loader, not on the model. A mismatch means the
        traces on disk are a different subset or a column is being misread, and
        either way nothing downstream should be believed until it is explained.
        Returns the comparison instead of raising, because a legitimate partial
        download will fail it and that is worth seeing rather than crashing on.
        """
        observed = self.describe()
        published = PUBLISHED_STATS.get(self.location, {})
        report: dict[str, object] = {"location": self.location, "checks": {}, "passed": True}
        for key, expected in published.items():
            got = float(observed.get(key, float("nan")))
            rel = abs(got - expected) / expected if expected else float("inf")
            ok = bool(rel <= tolerance)
            report["checks"][key] = {
                "published": expected, "observed": got,
                "relative_error": round(rel, 4), "within_tolerance": ok,
            }
            report["passed"] = bool(report["passed"] and ok)
        report["observed"] = observed
        return report
