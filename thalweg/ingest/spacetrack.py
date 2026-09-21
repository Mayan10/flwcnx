"""Space-Track historical orbital elements.

CelesTrak serves only the current element set, and is in any case unreachable
from this machine (a plain HTTPS GET to celestrak.org returns no response while
github.com, open-meteo.com and space-track.org all answer). The WetLinks
measurements run September 2023 to March 2024, so historical elements are
required, and Space-Track is the archive that has them.

Credentials come from the environment, and `.env` is gitignored (line 19).
They are never arguments, never logged, and never written into a config
snapshot. `ExperimentConfig.to_dict` has no field for them by construction.

Space-Track's API is rate limited to roughly 30 requests per minute and 300 per
hour, and they ask that clients not hammer it. This module therefore fetches
one wide query per satellite-set per day range rather than per satellite, and
caches every response to disk so a re-run costs nothing.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from thalweg.timeutil import to_epoch_nanoseconds

BASE_URL = "https://www.space-track.org"
LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
QUERY_URL = f"{BASE_URL}/basicspacedata/query"

#: Space-Track asks for at most ~30 requests/minute. One request per day of
#: data is well inside that, but the pause is kept explicit and honest.
REQUEST_PAUSE_SECONDS = 3.0

ENV_USER = "SPACETRACK_USER"
ENV_PASS = "SPACETRACK_PASS"


class SpaceTrackError(RuntimeError):
    pass


def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=VALUE lines into the environment if not already set.

    Deliberately minimal and deliberately non-overriding: a value already in
    the environment wins, so a CI secret is never silently replaced by a stale
    file. Values are not echoed anywhere.
    """
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def credentials() -> tuple[str, str]:
    """Fetch credentials from the environment, with a useful error if absent."""
    load_dotenv()
    user, password = os.environ.get(ENV_USER), os.environ.get(ENV_PASS)
    if not user or not password:
        raise SpaceTrackError(
            f"{ENV_USER} and {ENV_PASS} must be set. Put them in .env, which is "
            "already gitignored:\n"
            f"  {ENV_USER}=<the email you registered with>\n"
            f"  {ENV_PASS}=<your password>\n"
            "Register free at https://www.space-track.org/auth/createAccount"
        )
    return user, password


@dataclass
class SpaceTrackClient:
    """Minimal authenticated client. Caches every response to disk.

    Uses urllib with a cookie jar rather than requests, to avoid adding a
    dependency for one session login.
    """

    cache_dir: Path = Path("data/tle_cache")
    pause_seconds: float = REQUEST_PAUSE_SECONDS
    _opener: object | None = None

    def _login(self):
        import urllib.parse
        import urllib.request
        from http.cookiejar import CookieJar

        if self._opener is not None:
            return self._opener
        user, password = credentials()
        jar = CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        payload = urllib.parse.urlencode(
            {"identity": user, "password": password}
        ).encode()
        try:
            with opener.open(LOGIN_URL, payload, timeout=60) as response:
                body = response.read().decode("utf-8", "replace")
        except Exception as exc:                      # network or auth failure
            raise SpaceTrackError(f"Space-Track login failed: {exc}") from exc
        if "Failed" in body or "login" in body.lower() and "success" not in body.lower():
            # Never include the response body in the message: it can echo the
            # submitted identity back.
            raise SpaceTrackError(
                "Space-Track rejected the credentials. Check SPACETRACK_USER and "
                "SPACETRACK_PASS in .env."
            )
        self._opener = opener
        return opener

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json.gz"

    def query(self, path: str, cache_key: str) -> list[dict]:
        """Run one query, caching the parsed result under `cache_key`."""
        cached = self._cache_path(cache_key)
        if cached.exists():
            with gzip.open(cached, "rt") as handle:
                return json.load(handle)

        opener = self._login()
        url = f"{QUERY_URL}/{path}"
        try:
            with opener.open(url, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except Exception as exc:
            raise SpaceTrackError(f"Space-Track query failed for {cache_key}: {exc}") from exc

        payload = self._thin(payload)
        cached.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cached, "wt") as handle:
            json.dump(payload, handle)
        time.sleep(self.pause_seconds)
        return payload

    @staticmethod
    def _thin(rows: list[dict]) -> list[dict]:
        """Keep the latest element set per satellite within the fetched window.

        Space-Track returns every update, several per satellite per day. For
        propagation we only ever use the newest set at or before a given time,
        so the intermediate ones are dead weight in both cache and memory.
        """
        if not rows or "NORAD_CAT_ID" not in rows[0]:
            return rows
        latest: dict[str, dict] = {}
        for row in rows:
            key = str(row.get("NORAD_CAT_ID"))
            epoch = str(row.get("EPOCH", ""))
            if key not in latest or epoch > str(latest[key].get("EPOCH", "")):
                latest[key] = row
        return list(latest.values())

    def gp_history(self, start: date, end: date, *, object_name: str = "STARLINK",
                   chunk_days: int = 1) -> pd.DataFrame:
        """Element sets for a date range, one query per `chunk_days`.

        Space-Track's `gp_history` class carries the general perturbations
        archive. Filtering by OBJECT_NAME rather than by NORAD id keeps the
        query to one request per chunk instead of thousands.
        """
        frames = []
        cursor = start
        while cursor <= end:
            stop = min(cursor + timedelta(days=chunk_days - 1), end)
            key = f"gp_{object_name}_{cursor:%Y%m%d}_{stop:%Y%m%d}"
            # Only the five predicates we actually use. The full record is
            # ~60 fields and a single day of Starlink is ~14,800 element sets,
            # so trimming is the difference between a few MB and a few hundred.
            path = (
                f"class/gp_history/EPOCH/{cursor:%Y-%m-%d}--{stop + timedelta(days=1):%Y-%m-%d}"
                f"/OBJECT_NAME/~~{object_name}/orderby/NORAD_CAT_ID/format/json"
                "/predicates/NORAD_CAT_ID,OBJECT_NAME,EPOCH,TLE_LINE1,TLE_LINE2"
            )
            rows = self.query(path, key)
            if rows:
                frames.append(pd.DataFrame(rows))
            cursor = stop + timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def elements_to_tles(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Turn a gp_history response into (name, line1, line2) triples."""
    if frame.empty:
        return []
    required = {"TLE_LINE1", "TLE_LINE2"}
    if not required.issubset(frame.columns):
        raise SpaceTrackError(
            f"response lacks {sorted(required - set(frame.columns))}; "
            "request format/json on the gp_history class"
        )
    names = frame.get("OBJECT_NAME", pd.Series(["UNNAMED"] * len(frame)))
    return list(zip(names.astype(str), frame["TLE_LINE1"].astype(str),
                    frame["TLE_LINE2"].astype(str), strict=True))


def latest_per_satellite(frame: pd.DataFrame, when: datetime) -> pd.DataFrame:
    """The most recent element set at or before `when`, per satellite.

    SGP4 accuracy degrades with propagation age, so using the newest element
    set that does not postdate the measurement is both the accurate choice and
    the honest one: propagating backwards from a later epoch would use
    information that did not exist at the time.
    """
    if frame.empty:
        return frame
    work = frame.copy()
    work["EPOCH"] = pd.to_datetime(work["EPOCH"], errors="coerce", utc=True)
    reference = pd.Timestamp(when, tz="UTC") if when.tzinfo is None else pd.Timestamp(when)
    work = work[work["EPOCH"] <= reference]
    if work.empty:
        return work
    work = work.sort_values("EPOCH")
    return work.groupby("NORAD_CAT_ID", as_index=False).last()


def candidate_counts_for_times(
    times: pd.Series,
    latitude: float,
    longitude: float,
    elements: pd.DataFrame,
    *,
    altitude_m: float = 0.0,
    min_elevation_deg: float | None = None,
    refresh_hours: float = 6.0,
) -> pd.DataFrame:
    """Visible satellite count at each timestamp, from propagated elements.

    This is the one regime axis the WetLinks measurements can recover, and it
    is recovered rather than measured, so it is only as good as SGP4 over the
    propagation gap. `refresh_hours` bounds that: the element set is re-selected
    that often rather than once for the whole six months.

    Returns a frame of timestamp, candidate_count, best_elevation_deg,
    best_distance_km. The "best" columns describe the highest satellite in
    view, which is a proxy for the serving one and must be labelled as a proxy
    wherever it is used: this dataset does not say which satellite served.
    """
    from thalweg.ingest.tle import MIN_SERVICE_ELEVATION_DEG, Observer, look_angles

    floor = MIN_SERVICE_ELEVATION_DEG if min_elevation_deg is None else min_elevation_deg
    observer = Observer(latitude, longitude, altitude_m)
    stamps = pd.to_datetime(pd.Series(times)).sort_values().reset_index(drop=True)
    if stamps.empty:
        return pd.DataFrame(columns=["timestamp", "candidate_count",
                                     "best_elevation_deg", "best_distance_km"])

    # Group the timeline into refresh windows so the element selection and the
    # propagation both stay bounded.
    bucket = (to_epoch_nanoseconds(stamps) // int(refresh_hours * 3600 * 1e9))
    rows = []
    for _, index in stamps.groupby(bucket).groups.items():
        window = stamps.loc[index]
        anchor = window.iloc[len(window) // 2].to_pydatetime()
        current = latest_per_satellite(elements, anchor)
        tles = elements_to_tles(current)
        for stamp in window:
            angles = look_angles(observer, tles, stamp.to_pydatetime())
            visible = [t for t in angles.values() if t.elevation_deg >= floor]
            if visible:
                best = max(visible, key=lambda t: t.elevation_deg)
                rows.append((stamp, len(visible), best.elevation_deg, best.distance_km))
            else:
                rows.append((stamp, 0, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["timestamp", "candidate_count",
                                       "best_elevation_deg", "best_distance_km"])
