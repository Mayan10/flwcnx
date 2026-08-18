"""Serving satellite resolution and satellite ID encoding.

The resolution procedure is reproduction, not ours: StarNet (Liu et al.,
CoNEXT 2025, section 4) projects the terminal's 2D obstruction map into 3D
look angles and DTW matches the resulting track against SGP4 propagated TLE
tracks to work out which satellite is actually serving the dish. The terminal
never reports this directly.

On the replay path the published traces already carry the resolved serving
satellite, so `resolve_from_frame` only has to derive handovers and dwell. The
projection and matching below are for the live path and for validating the
published resolution against a TLE set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flwcnx.config import TIME_COL
from flwcnx.ingest.tle import MIN_SERVICE_ELEVATION_DEG

# The terminal reports obstruction over a field of view centred on boresight.
# 70 degrees off boresight is the usable cone for a Gen 3 dish; anything beyond
# it is structure, not sky.
DEFAULT_FIELD_OF_VIEW_DEG = 70.0


@dataclass(frozen=True)
class SatelliteTrack:
    """A candidate satellite's look angles over a window."""

    sat_id: str
    times: np.ndarray          # epoch seconds
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray

    def as_matrix(self) -> np.ndarray:
        return np.column_stack([self.azimuth_deg, self.elevation_deg])


@dataclass(frozen=True)
class MatchResult:
    sat_id: str
    distance: float
    runner_up: str | None
    margin: float              # runner-up distance minus best distance


def project_obstruction_map(
    obstruction: np.ndarray,
    *,
    boresight_azimuth_deg: float = 0.0,
    boresight_elevation_deg: float = 90.0,
    field_of_view_deg: float = DEFAULT_FIELD_OF_VIEW_DEG,
) -> pd.DataFrame:
    """2D obstruction raster to 3D look angles (StarNet section 4.1).

    The terminal exposes obstruction as a square raster in a dish centred
    frame. Pixel distance from the centre is angular distance from boresight
    and pixel bearing is angular bearing around it, so the projection is a
    polar unwrap followed by a rotation onto the boresight direction.

    Returns one row per non-null pixel with its azimuth, elevation and the
    obstruction value, which is what the DTW matcher consumes.
    """
    if obstruction.ndim != 2:
        raise ValueError(f"obstruction map must be 2D, got shape {obstruction.shape}")

    rows, cols = obstruction.shape
    centre_row, centre_col = (rows - 1) / 2.0, (cols - 1) / 2.0
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

    dy = (rr - centre_row) / max(centre_row, 1e-9)
    dx = (cc - centre_col) / max(centre_col, 1e-9)
    radius = np.hypot(dx, dy)

    inside = (radius <= 1.0) & np.isfinite(obstruction)
    # Angular distance from boresight scales linearly with pixel radius.
    off_boresight = radius * field_of_view_deg
    bearing = np.degrees(np.arctan2(dx, -dy)) % 360.0

    elevation = boresight_elevation_deg - off_boresight * np.cos(np.radians(bearing))
    azimuth = (boresight_azimuth_deg + bearing) % 360.0

    return pd.DataFrame({
        "azimuth_deg": azimuth[inside],
        "elevation_deg": np.clip(elevation[inside], -90.0, 90.0),
        "obstruction": obstruction[inside],
    })


def angular_difference(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Signed difference between two bearings, wrapped into [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def _point_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Cost between two (azimuth, elevation) points, wrap aware in azimuth."""
    d_az = float(angular_difference(a[0], b[0]))
    d_el = float(a[1] - b[1])
    return float(np.hypot(d_az, d_el))


def dtw_distance(a: np.ndarray, b: np.ndarray, *, band: int | None = None) -> float:
    """Dynamic time warping distance between two look angle tracks.

    A Sakoe-Chiba band keeps the alignment from drifting arbitrarily far in
    time, which matters because two satellites on adjacent orbital planes trace
    similar arcs offset by minutes and an unconstrained warp will happily match
    them. `band` is in samples; None means unconstrained.

    Normalised by path length so tracks of different lengths compare.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    if band is None:
        band = max(n, m)
    band = max(band, abs(n - m))

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            step = _point_cost(a[i - 1], b[j - 1])
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))


def match_serving_satellite(
    observed: np.ndarray,
    candidates: list[SatelliteTrack],
    *,
    band: int | None = 5,
    min_elevation_deg: float = MIN_SERVICE_ELEVATION_DEG,
) -> MatchResult | None:
    """Pick the candidate track that best explains an observed track.

    Candidates that spend the window below the service elevation floor are
    dropped before matching: the FCC filing puts all serving satellites above
    25 degrees, so a lower one is not a plausible answer however well it warps.

    `margin` is reported so the caller can reject an ambiguous match rather
    than accept a coin flip between two adjacent-plane satellites.
    """
    scored: list[tuple[float, str]] = []
    for track in candidates:
        if np.nanmax(track.elevation_deg) < min_elevation_deg:
            continue
        scored.append((dtw_distance(observed, track.as_matrix(), band=band), track.sat_id))
    if not scored:
        return None

    scored.sort()
    best_distance, best_id = scored[0]
    runner_up = scored[1][1] if len(scored) > 1 else None
    margin = (scored[1][0] - best_distance) if len(scored) > 1 else float("inf")
    return MatchResult(best_id, best_distance, runner_up, margin)


def resolve_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive handover and dwell columns from an already resolved trace.

    `handover` marks the first sample under a new serving satellite.
    `dwell_seconds` is time since that handover, which is the covariate that
    makes the start-of-service throughput dip visible to a model that has no
    explicit phase feature.

    Both are computed within segments, so a trace gap is never mistaken for a
    handover.
    """
    work = frame.copy()
    if "segment" not in work.columns:
        work["segment"] = 0

    handover = np.zeros(len(work), dtype=bool)
    dwell = np.zeros(len(work), dtype=float)

    for _, part in work.groupby("segment", sort=False):
        idx = part.index.to_numpy()
        ids = part["sat_id"].to_numpy()
        times = part[TIME_COL].astype("int64").to_numpy() / 1e9

        changed = np.zeros(len(ids), dtype=bool)
        changed[0] = True
        if len(ids) > 1:
            prev, cur = ids[:-1], ids[1:]
            known = pd.notna(prev) & pd.notna(cur)
            changed[1:] = known & (prev != cur)

        handover[np.searchsorted(work.index.to_numpy(), idx)] = changed
        anchor = np.maximum.accumulate(np.where(changed, times, -np.inf))
        dwell[np.searchsorted(work.index.to_numpy(), idx)] = times - anchor

    work["handover"] = handover
    work["dwell_seconds"] = dwell
    return work


class SatelliteEncoder:
    """Stable integer encoding of serving satellite IDs.

    BG-CFQS feed an encoded satellite ID as an auxiliary variable. It must be
    fit on the training split only: fitting on everything leaks the test
    split's satellite population into the encoding, and with thousands of
    unique satellites per location that is not a small leak.

    Unseen IDs at test time map to 0, which is reserved and never assigned to a
    training satellite.
    """

    UNKNOWN = 0

    def __init__(self) -> None:
        self._mapping: dict[str, int] = {}
        self._fitted = False

    def fit(self, sat_ids: pd.Series | np.ndarray) -> SatelliteEncoder:
        values = pd.Series(sat_ids).dropna().astype(str)
        # Sorted by descending frequency so the low codes are the satellites
        # the model actually sees often. Ties broken by name for determinism.
        counts = values.value_counts()
        ordered = sorted(counts.index, key=lambda k: (-counts[k], k))
        self._mapping = {name: i + 1 for i, name in enumerate(ordered)}
        self._fitted = True
        return self

    def transform(self, sat_ids: pd.Series | np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SatelliteEncoder.transform before fit")
        values = pd.Series(sat_ids).astype("object")
        return np.array([
            self.UNKNOWN if pd.isna(v) else self._mapping.get(str(v), self.UNKNOWN)
            for v in values
        ], dtype=np.int64)

    def fit_transform(self, sat_ids: pd.Series | np.ndarray) -> np.ndarray:
        return self.fit(sat_ids).transform(sat_ids)

    @property
    def n_known(self) -> int:
        return len(self._mapping)

    def unseen_rate(self, sat_ids: pd.Series | np.ndarray) -> float:
        """Fraction of IDs the encoder has never seen. Worth logging per split."""
        codes = self.transform(sat_ids)
        return float(np.mean(codes == self.UNKNOWN))
