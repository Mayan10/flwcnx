"""Orbital elements and candidate satellite counting.

Reproduction, not ours. StarNet resolves the serving satellite by projecting
the terminal's 2D obstruction map into 3D and DTW matching the result against
propagated TLE tracks. This module supplies the propagation and geometry half
of that; the matching half is in `state/satellite.py`.

TLEs come from CelesTrak (daily). Propagation is SGP4 via the `sgp4` package,
which is an optional dependency: the replay path does not need it because the
published traces already carry the resolved serving satellite.

The coordinate work is done here rather than delegated to skyfield so that the
geometry is inspectable and testable without a network fetch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

CELESTRAK_STARLINK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
)

# WGS-84, the datum SGP4 output is referenced against.
EARTH_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1.0 / 298.257223563

# The FCC filing puts the Starlink service floor at 25 degrees elevation, so a
# satellite below that is not a candidate however visible it is (StarNet
# section 3).
MIN_SERVICE_ELEVATION_DEG = 25.0


@dataclass(frozen=True)
class Observer:
    """Terminal position. Altitude in metres above the ellipsoid."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0


@dataclass(frozen=True)
class Topocentric:
    """Look angles from an observer to a satellite."""

    azimuth_deg: float
    elevation_deg: float
    distance_km: float


def load_tles(path: str | Path) -> list[tuple[str, str, str]]:
    """Read a three line element file into (name, line1, line2) triples."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    triples: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            triples.append(("UNNAMED", lines[i], lines[i + 1]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            triples.append((lines[i], lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return triples


def fetch_tles(dest: str | Path, url: str = CELESTRAK_STARLINK_URL, timeout: int = 60) -> Path:
    """Download the current Starlink TLE set. Network call, kept explicit."""
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        dest.write_bytes(response.read())
    return dest


def gmst_radians(when: datetime) -> float:
    """Greenwich mean sidereal time. IAU 1982 series, adequate at this scale."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    jd = _julian_date(when)
    t = (jd - 2451545.0) / 36525.0
    seconds = (67310.54841
               + (876600.0 * 3600.0 + 8640184.812866) * t
               + 0.093104 * t * t
               - 6.2e-6 * t * t * t)
    return math.radians((seconds % 86400.0) / 240.0)


def _julian_date(when: datetime) -> float:
    y, m = when.year, when.month
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    day_fraction = (when.hour + when.minute / 60.0
                    + (when.second + when.microsecond / 1e6) / 3600.0) / 24.0
    return (math.floor(365.25 * (y + 4716))
            + math.floor(30.6001 * (m + 1))
            + when.day + day_fraction + b - 1524.5)


def observer_ecef_km(observer: Observer) -> np.ndarray:
    """Geodetic position to ECEF, accounting for the ellipsoid flattening."""
    lat = math.radians(observer.latitude_deg)
    lon = math.radians(observer.longitude_deg)
    alt_km = observer.altitude_m / 1000.0
    e2 = EARTH_FLATTENING * (2 - EARTH_FLATTENING)
    n = EARTH_RADIUS_KM / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return np.array([
        (n + alt_km) * math.cos(lat) * math.cos(lon),
        (n + alt_km) * math.cos(lat) * math.sin(lon),
        (n * (1 - e2) + alt_km) * math.sin(lat),
    ])


def eci_to_ecef(position_km: np.ndarray, when: datetime) -> np.ndarray:
    """Rotate TEME/ECI into ECEF by the sidereal angle."""
    theta = gmst_radians(when)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        c * position_km[0] + s * position_km[1],
        -s * position_km[0] + c * position_km[1],
        position_km[2],
    ])


def topocentric(observer: Observer, satellite_ecef_km: np.ndarray) -> Topocentric:
    """ECEF satellite position to azimuth, elevation and slant range."""
    lat = math.radians(observer.latitude_deg)
    lon = math.radians(observer.longitude_deg)
    delta = satellite_ecef_km - observer_ecef_km(observer)

    # ECEF to local east-north-up.
    east = -math.sin(lon) * delta[0] + math.cos(lon) * delta[1]
    north = (-math.sin(lat) * math.cos(lon) * delta[0]
             - math.sin(lat) * math.sin(lon) * delta[1]
             + math.cos(lat) * delta[2])
    up = (math.cos(lat) * math.cos(lon) * delta[0]
          + math.cos(lat) * math.sin(lon) * delta[1]
          + math.sin(lat) * delta[2])

    distance = float(np.linalg.norm(delta))
    elevation = math.degrees(math.asin(up / distance)) if distance > 0 else 0.0
    azimuth = math.degrees(math.atan2(east, north)) % 360.0
    return Topocentric(azimuth_deg=azimuth, elevation_deg=elevation, distance_km=distance)


def propagate(tle: tuple[str, str, str], when: datetime) -> np.ndarray:
    """SGP4 propagate one TLE to a time, returning the ECI position in km."""
    try:
        from sgp4.api import Satrec, jday
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "SGP4 propagation needs the `sgp4` package: "
            "pip install -e '.[orbital]'"
        ) from exc

    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    satellite = Satrec.twoline2rv(tle[1], tle[2])
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond / 1e6)
    error, position, _ = satellite.sgp4(jd, fr)
    if error != 0:
        raise ValueError(f"SGP4 propagation failed for {tle[0]} with code {error}")
    return np.asarray(position)


def look_angles(observer: Observer, tles: list[tuple[str, str, str]],
                when: datetime) -> dict[str, Topocentric]:
    """Look angles from the observer to every satellite in the set."""
    out: dict[str, Topocentric] = {}
    for tle in tles:
        try:
            eci = propagate(tle, when)
        except ValueError:
            continue  # a decayed or malformed element set is not a candidate
        out[tle[0]] = topocentric(observer, eci_to_ecef(eci, when))
    return out


def candidate_count(observer: Observer, tles: list[tuple[str, str, str]], when: datetime,
                    min_elevation_deg: float = MIN_SERVICE_ELEVATION_DEG) -> int:
    """Number of satellites above the service elevation floor at a time."""
    angles = look_angles(observer, tles, when)
    return sum(1 for t in angles.values() if t.elevation_deg >= min_elevation_deg)


# ---------------------------------------------------------------------------
# Vectorised propagation
# ---------------------------------------------------------------------------
#
# The scalar path above is for one satellite at one time and is what the unit
# tests exercise. Counting candidates across a six month trace is a different
# problem: roughly 4,700 satellites against 110,000 burst timestamps is 5x10^8
# propagations, which pure Python will not do in reasonable time. sgp4's
# SatrecArray does the whole matrix in C.


def gmst_radians_array(times: np.ndarray) -> np.ndarray:
    """GMST for an array of datetime64 values. Same series as the scalar form."""
    unix = times.astype("datetime64[ns]").astype("int64") / 1e9
    julian = unix / 86400.0 + 2440587.5
    t = (julian - 2451545.0) / 36525.0
    seconds = (67310.54841
               + (876600.0 * 3600.0 + 8640184.812866) * t
               + 0.093104 * t * t
               - 6.2e-6 * t * t * t)
    return np.radians(np.mod(seconds, 86400.0) / 240.0)


def _julian_arrays(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split into whole Julian day and fraction, which is what SGP4 wants."""
    unix = times.astype("datetime64[ns]").astype("int64") / 1e9
    julian = unix / 86400.0 + 2440587.5
    jd = np.floor(julian - 0.5) + 0.5
    return jd, julian - jd


def visible_counts(
    times: np.ndarray,
    observer: Observer,
    tles: list[tuple[str, str, str]],
    *,
    min_elevation_deg: float = MIN_SERVICE_ELEVATION_DEG,
    chunk: int = 512,
) -> pd.DataFrame:
    """Candidate count and best-in-view geometry for many timestamps at once.

    Returns one row per input time with the number of satellites above the
    service elevation floor, plus the elevation and distance of the highest
    one. That highest satellite is a **proxy** for the serving satellite, not a
    measurement of it: nothing in these datasets records which satellite
    actually served, and anything built on it must be labelled a
    reconstruction.

    Propagation failures (decayed objects, malformed elements) are dropped per
    satellite-time rather than per satellite, because an element set can be
    valid early in the window and fail later.
    """
    try:
        from sgp4.api import Satrec, SatrecArray
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("vectorised propagation needs `sgp4`: "
                          "pip install -e '.[orbital]'") from exc

    times = np.asarray(times, dtype="datetime64[ns]")
    if times.size == 0 or not tles:
        return pd.DataFrame(columns=["timestamp", "candidate_count",
                                     "best_elevation_deg", "best_distance_km"])

    satellites = []
    for tle in tles:
        try:
            satellites.append(Satrec.twoline2rv(tle[1], tle[2]))
        except Exception:
            continue          # a malformed element set is not a candidate
    if not satellites:
        raise ValueError("no valid element sets")
    array = SatrecArray(satellites)

    latitude = math.radians(observer.latitude_deg)
    longitude = math.radians(observer.longitude_deg)
    observer_ecef = observer_ecef_km(observer)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)

    counts = np.zeros(times.size, dtype=np.int32)
    best_elevation = np.full(times.size, np.nan)
    best_distance = np.full(times.size, np.nan)

    for start in range(0, times.size, chunk):
        block = times[start : start + chunk]
        jd, fr = _julian_arrays(block)
        error, position, _ = array.sgp4(jd, fr)      # (n_sat, n_time, 3), km TEME

        theta = gmst_radians_array(block)             # (n_time,)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x, y, z = position[..., 0], position[..., 1], position[..., 2]
        # TEME to ECEF: rotate about the pole by the sidereal angle.
        ex = cos_t * x + sin_t * y
        ey = -sin_t * x + cos_t * y
        ez = z

        dx = ex - observer_ecef[0]
        dy = ey - observer_ecef[1]
        dz = ez - observer_ecef[2]
        up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        distance = np.sqrt(dx * dx + dy * dy + dz * dz)

        with np.errstate(invalid="ignore", divide="ignore"):
            elevation = np.degrees(np.arcsin(np.clip(up / distance, -1.0, 1.0)))
        valid = (error == 0) & np.isfinite(elevation)
        visible = valid & (elevation >= min_elevation_deg)

        counts[start : start + block.size] = visible.sum(axis=0)
        masked = np.where(visible, elevation, -np.inf)
        top = masked.argmax(axis=0)
        any_visible = visible.any(axis=0)
        rows = np.arange(block.size)
        best_elevation[start : start + block.size] = np.where(
            any_visible, elevation[top, rows], np.nan)
        best_distance[start : start + block.size] = np.where(
            any_visible, distance[top, rows], np.nan)

    return pd.DataFrame({
        "timestamp": times,
        "candidate_count": counts,
        "best_elevation_deg": best_elevation,
        "best_distance_km": best_distance,
    })
