"""Raw sources to a normalized frame. No ML and no feature engineering here."""

from thalweg.ingest.base import RawWindow, Source, count_handovers, segment_frame, validate_frame
from thalweg.ingest.live import LiveConfig, LiveSource
from thalweg.ingest.replay import ReplaySource, inspect_schema, normalize_frame
from thalweg.ingest.synthetic import SyntheticSource, SyntheticSpec

__all__ = [
    "LiveConfig",
    "LiveSource",
    "RawWindow",
    "ReplaySource",
    "Source",
    "SyntheticSource",
    "SyntheticSpec",
    "count_handovers",
    "inspect_schema",
    "normalize_frame",
    "segment_frame",
    "validate_frame",
]
