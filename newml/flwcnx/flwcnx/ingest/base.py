"""The Source interface.

Both ingestion modes sit behind this from day one. ReplaySource over the
published StarNet traces is the real path; LiveSource is a stub until a dish
exists. Keeping the interface honest is the point: the architecture should not
be a notebook wearing a diagram.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flwcnx.config import NORMALIZED_COLUMNS, TARGET_COL, TIME_COL


@dataclass(frozen=True)
class RawWindow:
    """A contiguous slice of the normalized frame.

    `segment` identifies the uninterrupted run of samples this window came
    from. Windows never straddle a segment boundary, because a gap in the
    trace is a gap in the signal and a look-back that spans one is fiction.
    """

    frame: pd.DataFrame
    segment: int
    location: str

    @property
    def start_time(self) -> pd.Timestamp:
        return self.frame[TIME_COL].iloc[0]

    @property
    def end_time(self) -> pd.Timestamp:
        return self.frame[TIME_COL].iloc[-1]

    def __len__(self) -> int:
        return len(self.frame)


class Source(ABC):
    """Anything that can produce normalized windows of terminal telemetry."""

    location: str

    @abstractmethod
    def load_frame(self) -> pd.DataFrame:
        """Return the whole trace as one normalized, segmented frame."""

    @abstractmethod
    def iter_windows(self, length: int, stride: int = 1) -> Iterator[RawWindow]:
        """Yield fixed length windows that never cross a segment boundary."""

    def describe(self) -> dict[str, float | int | str]:
        """Summary statistics used to check the loader against published numbers."""
        frame = self.load_frame()
        span = frame[TIME_COL].max() - frame[TIME_COL].min()
        return {
            "location": self.location,
            "samples": int(len(frame)),
            "trace_minutes": round(float(len(frame)) / 60.0, 1),
            "unique_satellites": int(frame["sat_id"].nunique(dropna=True)),
            "handovers": int(count_handovers(frame)),
            "segments": int(frame["segment"].nunique()),
            "span_days": round(span.total_seconds() / 86400.0, 2),
            "mean_throughput_mbps": round(float(frame[TARGET_COL].mean()), 2),
        }


def count_handovers(frame: pd.DataFrame) -> int:
    """Serving satellite changes, counted within segments only.

    A change across a trace gap is not an observed handover, it is a gap. The
    published handover counts (CLAUDE.md section 5) are the check on this.
    """
    total = 0
    for _, part in frame.groupby("segment", sort=False):
        ids = part["sat_id"].to_numpy()
        if len(ids) < 2:
            continue
        prev, cur = ids[:-1], ids[1:]
        both_known = pd.notna(prev) & pd.notna(cur)
        total += int(np.sum(both_known & (prev != cur)))
    return total


def segment_frame(frame: pd.DataFrame, max_gap_seconds: float = 2.0) -> pd.DataFrame:
    """Add a `segment` column that increments whenever the trace jumps."""
    frame = frame.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    deltas = frame[TIME_COL].diff().dt.total_seconds()
    breaks = (deltas > max_gap_seconds) | deltas.isna()
    breaks.iloc[0] = True
    frame["segment"] = breaks.cumsum().astype(int) - 1
    return frame


def validate_frame(frame: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Check the normalized contract and return the frame in canonical order.

    Raises rather than coercing when `strict`. A loader that silently invents a
    column is a loader that silently invents a result.
    """
    missing = [c for c in NORMALIZED_COLUMNS if c not in frame.columns]
    if missing:
        if strict:
            raise ValueError(
                f"normalized frame is missing {missing}. "
                "Map the upstream columns in ingest/replay.py COLUMN_ALIASES "
                "rather than filling them in downstream."
            )
        for col in missing:
            frame[col] = np.nan

    if not pd.api.types.is_datetime64_any_dtype(frame[TIME_COL]):
        raise TypeError(f"{TIME_COL} must be datetime64, got {frame[TIME_COL].dtype}")
    if frame[TARGET_COL].isna().all():
        raise ValueError(f"{TARGET_COL} is entirely missing")
    if (frame[TARGET_COL].dropna() < 0).any():
        raise ValueError("negative throughput in the normalized frame")

    ordered = [*NORMALIZED_COLUMNS]
    if "segment" in frame.columns:
        ordered.append("segment")
    extra = [c for c in frame.columns if c not in ordered]
    return frame[ordered + extra]


def iter_windows_from_frame(
    frame: pd.DataFrame, length: int, stride: int, location: str
) -> Iterator[RawWindow]:
    """Shared window walker. Segment aware, so no window spans a trace gap."""
    if length <= 0:
        raise ValueError("window length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    for segment, part in frame.groupby("segment", sort=True):
        part = part.reset_index(drop=True)
        if len(part) < length:
            continue
        for start in range(0, len(part) - length + 1, stride):
            yield RawWindow(
                frame=part.iloc[start : start + length],
                segment=int(segment),
                location=location,
            )
