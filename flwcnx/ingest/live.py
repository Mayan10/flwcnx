"""Live ingestion from a Starlink terminal.

Deliberately a stub. Mayan does not have a dish, so `ReplaySource` is the real
path. What matters is that the interface exists and is honest: the method
bodies say exactly what they would have to do and what they need, so the
architecture is not a diagram drawn over a notebook.

Three feeds are joined here:

  - terminal telemetry over gRPC, https://github.com/sparky8512/starlink-grpc-tools
  - orbital elements from CelesTrak, propagated with SGP4 (`ingest/tle.py`)
  - weather from Open-Meteo (`ingest/weather.py`)

The terminal reports obstruction and throughput but not which satellite is
serving it, which is why the TLE feed is needed at all: the serving satellite
is recovered, not read off (StarNet section 4).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from flwcnx.config import NORMALIZED_COLUMNS
from flwcnx.ingest.base import RawWindow, Source, segment_frame, validate_frame
from flwcnx.ingest.tle import Observer

DEFAULT_DISH_ADDRESS = "192.168.100.1:9200"


@dataclass
class LiveConfig:
    observer: Observer
    dish_address: str = DEFAULT_DISH_ADDRESS
    tle_path: Path | None = None
    poll_hz: float = 1.0
    buffer_seconds: int = 3600
    location: str = "usa"
    weather_cache: Path | None = None
    _buffer: list[dict] = field(default_factory=list, repr=False)


class LiveSource(Source):
    """Streams from a real terminal. Raises until a dish is available.

    Everything except `connect` and `_poll_once` is real, so the day a dish
    turns up only those two need writing.
    """

    def __init__(self, config: LiveConfig) -> None:
        self.config = config
        self.location = config.location
        self._connected = False
        self._rows: list[dict] = []

    def connect(self) -> None:
        """Open the gRPC channel to the terminal.

        Would call `starlink_grpc.status_data()` against `dish_address` and
        confirm the terminal reports a software version and a valid GPS fix
        before accepting any sample.
        """
        raise NotImplementedError(
            "LiveSource needs a Starlink terminal reachable at "
            f"{self.config.dish_address}. Use ReplaySource over the published "
            "StarNet traces instead."
        )

    def _poll_once(self, when: datetime | None = None) -> dict:
        """One telemetry sample, already in normalized column names.

        Would read `status_data()` for downlink throughput, obstruction
        fraction and GPS, then resolve the serving satellite from the
        obstruction map against the propagated TLE set, and finally attach the
        most recent weather row.
        """
        raise NotImplementedError("no terminal connected")

    def poll(self, seconds: int) -> pd.DataFrame:
        """Collect `seconds` of telemetry into a normalized frame."""
        if not self._connected:
            self.connect()
        rows = []
        for _ in range(int(seconds * self.config.poll_hz)):
            rows.append(self._poll_once(datetime.now(UTC)))
        self._rows.extend(rows)
        return self.load_frame()

    def load_frame(self) -> pd.DataFrame:
        """Normalize whatever has been buffered so far."""
        if not self._rows:
            raise RuntimeError(
                "LiveSource buffer is empty. Call poll() first, which requires "
                "a connected terminal."
            )
        frame = pd.DataFrame(self._rows)
        for column in NORMALIZED_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        return validate_frame(segment_frame(frame))

    def iter_windows(self, length: int, stride: int = 1) -> Iterator[RawWindow]:
        from flwcnx.ingest.base import iter_windows_from_frame

        yield from iter_windows_from_frame(self.load_frame(), length, stride, self.location)
