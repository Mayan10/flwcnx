"""The normalized frame contract and the column mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thalweg.config import NORMALIZED_COLUMNS, TARGET_COL, TIME_COL
from thalweg.ingest.base import count_handovers, segment_frame, validate_frame
from thalweg.ingest.replay import map_columns, normalize_frame
from thalweg.ingest.synthetic import SyntheticSpec


def test_normalized_frame_has_every_contract_column(raw_frame):
    for column in NORMALIZED_COLUMNS:
        assert column in raw_frame.columns


def test_real_starnet_columns_all_map():
    """The actual pickle column names, read off their get_data_loader."""
    upstream = ["timestamp", "throughput", "latency", "alt", "az", "distance",
                "sat_name", "n_candidates", "clouds", "pressure", "humidity"]
    mapped = map_columns(upstream)
    assert mapped[TIME_COL] == "timestamp"
    assert mapped[TARGET_COL] == "throughput"
    assert mapped["elevation_deg"] == "alt"
    assert mapped["azimuth_deg"] == "az"
    assert mapped["sat_id"] == "sat_name"
    assert mapped["candidate_count"] == "n_candidates"
    assert mapped["cloud_cover_pct"] == "clouds"
    assert mapped["humidity_pct"] == "humidity"
    assert mapped["latency_ms"] == "latency"


def test_one_upstream_column_is_never_claimed_twice():
    mapped = map_columns(["timestamp", "throughput", "distance", "pressure"])
    assert len(set(mapped.values())) == len(mapped)


def test_segments_split_on_a_gap():
    times = pd.to_datetime(["2024-05-01 00:00:00", "2024-05-01 00:00:01",
                            "2024-05-01 00:05:00", "2024-05-01 00:05:01"])
    frame = pd.DataFrame({TIME_COL: times, TARGET_COL: [1.0, 2.0, 3.0, 4.0]})
    assert segment_frame(frame)["segment"].tolist() == [0, 0, 1, 1]


def test_handovers_are_not_counted_across_a_gap():
    """A satellite change spanning a trace gap is a gap, not an observed handover."""
    frame = pd.DataFrame({
        "segment": [0, 0, 1, 1],
        "sat_id": ["A", "A", "B", "B"],
        TIME_COL: pd.to_datetime(["2024-05-01 00:00:00", "2024-05-01 00:00:01",
                                  "2024-05-01 00:05:00", "2024-05-01 00:05:01"]),
    })
    assert count_handovers(frame) == 0


def test_validate_rejects_a_missing_column():
    frame = pd.DataFrame({TIME_COL: pd.to_datetime(["2024-05-01"]), TARGET_COL: [1.0]})
    with pytest.raises(ValueError, match="missing"):
        validate_frame(frame)


def test_validate_rejects_negative_throughput(raw_frame):
    frame = raw_frame.copy()
    frame.loc[0, TARGET_COL] = -1.0
    with pytest.raises(ValueError, match="negative"):
        validate_frame(frame)


def test_bps_input_is_converted_to_mbps():
    n = 200
    raw = pd.DataFrame({
        "timestamp": pd.date_range("2024-05-01", periods=n, freq="s"),
        "throughput": np.full(n, 2.0e8),      # 200 Mbps expressed in bps
    })
    frame = normalize_frame(raw)
    assert 190.0 < frame[TARGET_COL].median() < 210.0


def test_epoch_seconds_are_parsed():
    n = 100
    raw = pd.DataFrame({
        "timestamp": np.arange(1714521600, 1714521600 + n),
        "throughput": np.full(n, 200.0),
    })
    frame = normalize_frame(raw)
    assert frame[TIME_COL].dt.year.iloc[0] == 2024


def test_windows_never_cross_a_segment_boundary():
    from thalweg.ingest.synthetic import SyntheticSource

    source = SyntheticSource(SyntheticSpec(n_seconds=3000, gap_every=500, gap_length=40))
    for window in source.iter_windows(30, stride=7):
        assert window.frame["segment"].nunique() == 1
        deltas = window.frame[TIME_COL].diff().dt.total_seconds().dropna()
        assert (deltas <= 2).all()


def test_synthetic_source_reports_sane_statistics():
    from thalweg.ingest.synthetic import SyntheticSource

    described = SyntheticSource(SyntheticSpec(n_seconds=3000)).describe()
    assert described["samples"] == 3000
    assert described["handovers"] > 0
    assert described["mean_throughput_mbps"] > 0
