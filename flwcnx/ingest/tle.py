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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
        when = when.replace(tzinfo=timezone.utc)
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
        when = when.replace(tzinfo=timezone.utc)
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
