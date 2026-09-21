"""Weather join. Open-Meteo, free hourly, roughly 9 km resolution.

The StarNet traces already carry precipitation, cloudiness and pressure, so
this is only needed on the live path and for locations the traces do not
cover. The join is nearest hour with an explicit tolerance rather than a merge
on equality, because the telemetry is 1 Hz and the weather is hourly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from thalweg.config import TIME_COL

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo variable name -> our normalized column.
VARIABLE_MAP: dict[str, str] = {
    "precipitation": "precipitation_mm",
    "cloud_cover": "cloud_cover_pct",
    "surface_pressure": "pressure_hpa",
}


@dataclass(frozen=True)
class WeatherQuery:
    latitude: float
    longitude: float
    start_date: str    # YYYY-MM-DD
    end_date: str


def fetch_hourly(query: WeatherQuery, timeout: int = 60) -> pd.DataFrame:
    """Pull hourly weather for a location and date range. Network call."""
    import urllib.parse
    import urllib.request

    params = {
        "latitude": query.latitude,
        "longitude": query.longitude,
        "start_date": query.start_date,
        "end_date": query.end_date,
        "hourly": ",".join(VARIABLE_MAP),
        "timezone": "UTC",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read())
    return parse_hourly(payload)


def parse_hourly(payload: dict) -> pd.DataFrame:
    """Turn an Open-Meteo response into a normalized weather frame."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Open-Meteo response carries no hourly block")
    frame = pd.DataFrame({TIME_COL: pd.to_datetime(hourly["time"])})
    for variable, column in VARIABLE_MAP.items():
        if variable in hourly:
            frame[column] = pd.to_numeric(pd.Series(hourly[variable]), errors="coerce")
    return frame.sort_values(TIME_COL).reset_index(drop=True)


def load_cached(path: str | Path) -> pd.DataFrame:
    """Read a previously saved Open-Meteo JSON response."""
    return parse_hourly(json.loads(Path(path).read_text()))


def join_weather(frame: pd.DataFrame, weather: pd.DataFrame,
                 tolerance_minutes: int = 90) -> pd.DataFrame:
    """Attach weather to telemetry by nearest timestamp within a tolerance.

    Beyond the tolerance the weather columns are left null rather than carried
    forward indefinitely, so a gap in the weather feed stays visible.
    """
    left = frame.sort_values(TIME_COL, kind="stable")
    right = weather.sort_values(TIME_COL, kind="stable")
    columns = [c for c in VARIABLE_MAP.values() if c in right.columns]
    merged = pd.merge_asof(
        left, right[[TIME_COL, *columns]], on=TIME_COL, direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes), suffixes=("", "_wx"),
    )
    for column in columns:
        joined = f"{column}_wx"
        if joined in merged.columns:
            merged[column] = merged[column].fillna(merged[joined])
            merged = merged.drop(columns=[joined])
    return merged.reset_index(drop=True)
