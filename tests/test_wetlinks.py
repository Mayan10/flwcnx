"""The supplied dataset loader.

These run against the real supplied CSVs when present and skip otherwise, so
the suite stays green on a machine without the data. The assertions encode the
findings in docs/supplied-dataset.md, so if a future drop of the data breaks
one of them the assumption that broke is named.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thalweg.config import LATENCY_COL, NORMALIZED_COLUMNS, TARGET_COL
from thalweg.ingest.wetlinks import (
    SITE_COORDINATES,
    WetLinksConfig,
    WetLinksSource,
    normalize_wetlinks,
    phase_is_aliased,
)

SUPPLIED = Path("data/supplied")
has_data = pytest.mark.skipif(
    not (SUPPLIED.exists() and any(SUPPLIED.glob("*.csv"))),
    reason="supplied WetLinks CSVs not present",
)


def _synthetic_raw(n=400, site="uos-rz", step="30s"):
    """A WetLinks shaped frame, for the tests that must run without the data."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2023-09-14 15:12:00", periods=n, freq=step)
    idle = rng.normal(0.008e6, 2000, n)
    speedtest = np.zeros(n, dtype=bool)
    speedtest[::12] = True                       # one burst every 12 samples
    downlink = np.where(speedtest, rng.normal(230e6, 40e6, n), idle)
    return pd.DataFrame({
        "site_name": site, "timestamp": times, "state": "CONNECTED",
        "uplink": np.clip(rng.normal(5e3, 1e3, n), 0, None),
        "downlink": np.clip(downlink, 0, None),
        "pop_ping_latency": 30 + rng.gamma(2, 3, n),
        "ping_drop": rng.uniform(0, 5, n),
        "mean_ping_latency": 31 + rng.normal(0, 3, n),
        "ping_stdvar": rng.uniform(1, 15, n),
        "fraction_obstructed": rng.uniform(0, 0.002, n),
        "obstruction_duration": rng.uniform(0, 1, n),
        "obstruction_interval": rng.uniform(0, 43200, n),
        "direction_azimuth": rng.uniform(-180, 180, n),
        "direction_elevation": 70 + rng.normal(0, 0.4, n),
        "lat": np.nan, "lon": np.nan,
    })


def test_normalized_output_satisfies_the_contract():
    frame = normalize_wetlinks(_synthetic_raw(), WetLinksConfig())
    for column in NORMALIZED_COLUMNS:
        assert column in frame.columns
    assert frame[TARGET_COL].notna().all()


def test_satellite_columns_are_null_not_zero():
    """This dataset has no satellite geometry. A zero would look like data."""
    frame = normalize_wetlinks(_synthetic_raw(), WetLinksConfig())
    for column in ("sat_id", "elevation_deg", "distance_km", "candidate_count"):
        assert frame[column].isna().all()


def test_dish_pointing_is_named_so_it_cannot_be_mistaken_for_a_satellite():
    frame = normalize_wetlinks(_synthetic_raw(), WetLinksConfig())
    assert "dish_elevation_deg" in frame.columns
    assert "dish_azimuth_deg" in frame.columns
    assert frame["elevation_deg"].isna().all()


def test_missing_coordinates_are_filled_from_the_site_name():
    frame = normalize_wetlinks(_synthetic_raw(site="utwente"), WetLinksConfig())
    latitude, longitude = SITE_COORDINATES["utwente"]
    assert frame["lat"].iloc[0] == pytest.approx(latitude)
    assert frame["lon"].iloc[0] == pytest.approx(longitude)


def test_capacity_target_keeps_only_speedtests():
    raw = _synthetic_raw()
    capacity = normalize_wetlinks(raw, WetLinksConfig(target="capacity"))
    latency = normalize_wetlinks(raw, WetLinksConfig(target="latency"))
    assert len(capacity) < len(latency) / 5
    assert capacity[TARGET_COL].min() >= 50.0
    assert capacity["is_speedtest"].all()


def test_latency_target_keeps_offered_load_as_a_feature():
    """Offered load is a useless target here and a legitimate feature."""
    frame = normalize_wetlinks(_synthetic_raw(), WetLinksConfig(target="latency"))
    assert frame["offered_downlink_mbps"].notna().all()
    assert frame[TARGET_COL].median() < 100          # milliseconds, not Mbps


def test_latency_column_is_populated_for_period_classification():
    frame = normalize_wetlinks(_synthetic_raw(), WetLinksConfig())
    assert frame[LATENCY_COL].notna().all()


def test_gap_threshold_differs_by_target():
    assert WetLinksConfig(target="latency").gap_seconds < 200
    assert WetLinksConfig(target="capacity").gap_seconds > 600


def test_bad_target_is_rejected():
    with pytest.raises(ValueError, match="target must be one of"):
        WetLinksConfig(target="throughput")


def test_phase_aliasing_guard_fires_on_a_30s_grid():
    """30 s against a 15 s period puts every sample at the same phase."""
    aliased = normalize_wetlinks(_synthetic_raw(step="30s"), WetLinksConfig())
    assert phase_is_aliased(aliased)


def test_phase_aliasing_guard_passes_on_a_1hz_grid():
    dense = normalize_wetlinks(_synthetic_raw(n=2000, step="1s"), WetLinksConfig())
    assert not phase_is_aliased(dense)


# --- against the real supplied data -----------------------------------------


@has_data
def test_real_data_matches_the_documented_profile():
    described = WetLinksSource(WetLinksConfig(target="latency")).describe()
    assert set(described["sites"]) == {"uos-rz", "utwente"}
    assert described["samples"] > 800_000
    assert described["median_sampling_seconds"] == pytest.approx(30.0, abs=0.1)
    assert 25.0 < described["target_median"] < 40.0
    assert 0.02 < described["spike_rate"] < 0.06


@has_data
def test_real_capacity_series_is_six_minute_granularity():
    described = WetLinksSource(WetLinksConfig(target="capacity")).describe()
    assert described["median_sampling_seconds"] == pytest.approx(360.0, abs=15.0)
    # Median matches StarNet's reported ~230 Mbps plateau, which is the
    # evidence that these bursts really are capacity measurements.
    assert 200.0 < described["target_median"] < 270.0


@has_data
def test_real_capacity_segments_are_long_enough_to_window():
    source = WetLinksSource(WetLinksConfig(target="capacity"))
    assert len(list(source.iter_windows(20, stride=25))) > 500


@has_data
def test_real_data_phase_is_aliased():
    frame = WetLinksSource(WetLinksConfig(target="latency")).load_frame()
    assert phase_is_aliased(frame)


@has_data
def test_real_windows_never_cross_a_segment_boundary():
    source = WetLinksSource(WetLinksConfig(target="latency"))
    for index, window in enumerate(source.iter_windows(30, stride=997)):
        assert window.frame["segment"].nunique() == 1
        if index > 200:
            break
