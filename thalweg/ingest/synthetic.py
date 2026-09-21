"""Synthetic trace generator.

For tests and smoke runs only. Nothing produced here may appear in any
reported result. It exists so the whole pipeline can be exercised end to end
without the real traces, and so the phase recovery has a signal whose true
offset is known.

The generative model is deliberately crude but carries the structure the code
under test is supposed to find:

  - a 15 second scheduling period with a known phase offset;
  - a throughput drop at the start of each period (the handover cost);
  - throughput rising with elevation and falling with distance past a knee;
  - serving satellite changes aligned to period boundaries.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from thalweg.config import PERIOD_SECONDS, TARGET_COL, TIME_COL
from thalweg.ingest.base import (
    RawWindow,
    Source,
    iter_windows_from_frame,
    segment_frame,
    validate_frame,
)


@dataclass(frozen=True)
class SyntheticSpec:
    n_seconds: int = 6000
    phase_offset: int = 12          # the ground truth the recovery must find
    base_mbps: float = 180.0
    drop_mbps: float = 55.0         # size of the start of period dip
    drop_seconds: int = 2
    noise_mbps: float = 8.0
    handover_period_multiple: int = 4   # a new serving satellite every N periods
    gap_every: int = 0              # if > 0, punch a trace gap every N seconds
    gap_length: int = 30
    seed: int = 0


def generate_frame(spec: SyntheticSpec | None = None) -> pd.DataFrame:
    """Build a normalized frame with known ground truth."""
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n = spec.n_seconds

    t0 = pd.Timestamp("2024-05-01 00:00:00")
    seconds = np.arange(n)
    times = t0 + pd.to_timedelta(seconds, unit="s")

    # Phase within the 15 s period, measured from the true offset.
    phase = (seconds - spec.phase_offset) % PERIOD_SECONDS
    period_index = (seconds - spec.phase_offset) // PERIOD_SECONDS

    # Satellite geometry, held constant within a serving assignment.
    n_periods = int(period_index.max() - period_index.min() + 1)
    n_assign = max(1, n_periods // spec.handover_period_multiple + 1)
    assign = (period_index - period_index.min()) // spec.handover_period_multiple
    sat_ids = np.array([f"SAT-{1000 + int(i)}" for i in rng.integers(0, 400, n_assign)])
    elevations = rng.uniform(25.0, 85.0, n_assign)
    azimuths = rng.uniform(0.0, 360.0, n_assign)
    distances = 550.0 + (90.0 - elevations) * 4.5 + rng.normal(0, 20.0, n_assign)
    candidates = rng.integers(12, 48, n_assign)

    idx = np.clip(assign.astype(int), 0, n_assign - 1)
    elevation = elevations[idx]
    azimuth = azimuths[idx]
    distance = distances[idx]
    candidate = candidates[idx].astype(float)

    # Throughput: elevation helps and plateaus, distance hurts past the knee,
    # candidate count helps, and every period opens with a dip.
    elev_gain = 60.0 * np.clip(elevation, 25.0, 60.0) / 60.0
    dist_loss = 0.12 * np.clip(distance - 645.0, 0.0, None)
    cand_gain = 0.9 * (candidate - 15.0)
    opening = np.where(phase < spec.drop_seconds, spec.drop_mbps, 0.0)
    slow = 12.0 * np.sin(2 * np.pi * seconds / 900.0)

    tput = (spec.base_mbps + elev_gain - dist_loss + cand_gain + slow
            - opening + rng.normal(0, spec.noise_mbps, n))
    tput = np.clip(tput, 0.0, None)

    frame = pd.DataFrame({
        TIME_COL: times,
        TARGET_COL: tput,
        "sat_id": sat_ids[idx],
        "elevation_deg": elevation,
        "azimuth_deg": azimuth,
        "distance_km": distance,
        "candidate_count": candidate,
        "second_of_day": (times.hour * 3600 + times.minute * 60 + times.second),
        "day_of_week": times.dayofweek,
        "cloud_cover_pct": np.clip(rng.normal(45.0, 25.0, n), 0.0, 100.0),
        "pressure_hpa": 1013.0 + rng.normal(0, 4.0, n),
        "humidity_pct": np.clip(rng.normal(60.0, 15.0, n), 0.0, 100.0),
        "precipitation_mm": np.clip(rng.normal(0.05, 0.2, n), 0.0, None),
        # The traces carry latency. Given a period opening penalty so the
        # Casparsen period classifier has something to find.
        "latency_ms": 35.0 + 30.0 * (phase < 1) + rng.gamma(2.0, 3.0, n),
    })

    if spec.gap_every > 0:
        keep = ((np.arange(len(frame)) % spec.gap_every) >= spec.gap_length)
        keep[: spec.gap_every] = True
        frame = frame[keep]

    frame = segment_frame(frame.reset_index(drop=True))
    return validate_frame(frame)


class SyntheticSource(Source):
    """A Source over generated data. Test fixture, never a result."""

    def __init__(self, spec: SyntheticSpec | None = None, location: str = "usa") -> None:
        self.spec = spec or SyntheticSpec()
        self.location = location
        self._frame: pd.DataFrame | None = None

    def load_frame(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = generate_frame(self.spec)
        return self._frame

    def iter_windows(self, length: int, stride: int = 1) -> Iterator[RawWindow]:
        yield from iter_windows_from_frame(self.load_frame(), length, stride, self.location)
