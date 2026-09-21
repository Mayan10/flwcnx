"""Phase recovery, satellite geometry, regime assignment, feature assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thalweg.config import FEATURE_CLASSES, FEATURE_COLUMNS, PERIOD_SECONDS, RegimeConfig
from thalweg.ingest.synthetic import SyntheticSpec, generate_frame
from thalweg.state.features import REGIME_COVARIATES, WINDOW_COVARIATES, make_sequences
from thalweg.state.phase import (
    FIXED_PHASE_OFFSET,
    assign_phase,
    boundary_mask,
    classify_periods,
    recover_phase,
)
from thalweg.state.regime import RegimeAssigner, build_fallback_map
from thalweg.state.satellite import (
    SatelliteEncoder,
    angular_difference,
    dtw_distance,
    project_obstruction_map,
    resolve_from_frame,
)

# ---------------------------------------------------------------------------
# phase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0, 3, 7, 12, 14])
def test_phase_recovery_finds_the_true_offset(offset):
    """Including 0 and 14, which straddle the wrap point."""
    frame = generate_frame(SyntheticSpec(n_seconds=9000, phase_offset=offset, seed=offset))
    reference = recover_phase(frame)
    assert reference.is_recovered
    # Circular distance, because 14.9 and 0.1 are 0.2 apart.
    error = abs((reference.offset_seconds - offset + 7.5) % PERIOD_SECONDS - 7.5)
    assert error < 1.0


def test_phase_recovery_falls_back_on_structureless_signal():
    n = 4000
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-05-01", periods=n, freq="s"),
        "throughput_mbps": np.random.default_rng(0).normal(200, 15, n),
    })
    reference = recover_phase(frame)
    assert not reference.is_recovered
    assert reference.offset_seconds == FIXED_PHASE_OFFSET


def test_assigned_phase_stays_in_range(raw_frame):
    phase = assign_phase(raw_frame["timestamp"], 12.0)
    assert phase.min() >= 0.0
    assert phase.max() < PERIOD_SECONDS


def test_boundary_mask_refuses_1hz_data(raw_frame):
    """The 140 ms / 75 ms regions cannot be resolved at 1 Hz and must not pretend to be."""
    with pytest.raises(ValueError, match="cannot be resolved"):
        boundary_mask(raw_frame["timestamp"], 12.0)


def test_boundary_mask_works_at_a_rate_that_can_resolve_it():
    n = 5000
    times = pd.Series(pd.date_range("2024-05-01", periods=n, freq="2ms"))
    mask = boundary_mask(times, 0.0, sampling_hz=500.0)
    assert mask.any()
    assert not mask.all()


def test_period_classification_uses_the_latency_column(feature_frame):
    frame, reference, _ = feature_frame
    table = classify_periods(frame, reference)
    assert set(table["label"]) <= {"Good", "Degraded"}
    assert (table["n"] > 0).all()


def test_period_classification_needs_latency(raw_frame):
    with pytest.raises(KeyError):
        classify_periods(raw_frame.drop(columns=["latency_ms"]), 12.0)


# ---------------------------------------------------------------------------
# satellite
# ---------------------------------------------------------------------------


def test_dtw_is_zero_against_itself_and_larger_against_a_shifted_track():
    t = np.linspace(0, 60, 30)
    a = np.column_stack([(100 + 0.5 * t) % 360, 30 + 0.4 * t])
    near = np.column_stack([(100 + 0.5 * t + 0.3) % 360, 30 + 0.4 * t + 0.3])
    far = np.column_stack([(100 + 0.5 * t + 120) % 360, 30 + 0.4 * t])
    assert dtw_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert dtw_distance(a, near) < dtw_distance(a, far)


def test_azimuth_difference_wraps():
    assert angular_difference(359.0, 1.0) == pytest.approx(-2.0)
    assert angular_difference(1.0, 359.0) == pytest.approx(2.0)


def test_obstruction_projection_stays_inside_the_field_of_view():
    projected = project_obstruction_map(np.ones((21, 21)), field_of_view_deg=70.0)
    assert len(projected) > 0
    assert projected["elevation_deg"].min() >= 20.0 - 1e-6
    assert projected["elevation_deg"].max() <= 90.0 + 1e-6
    assert projected["azimuth_deg"].between(0, 360).all()


def test_dwell_resets_at_every_handover(raw_frame):
    resolved = resolve_from_frame(raw_frame)
    assert resolved["handover"].iloc[0]
    assert (resolved.loc[resolved["handover"], "dwell_seconds"] == 0).all()
    assert resolved["dwell_seconds"].min() >= 0


def test_encoder_reserves_zero_for_unseen_satellites():
    encoder = SatelliteEncoder().fit(pd.Series(["A", "A", "B"]))
    assert encoder.transform(pd.Series(["A", "B", "Z"])).tolist()[-1] == 0
    assert 0 not in [encoder.transform(pd.Series([s]))[0] for s in ("A", "B")]
    assert encoder.unseen_rate(pd.Series(["A", "Z"])) == pytest.approx(0.5)


def test_encoder_refuses_to_transform_before_fit():
    with pytest.raises(RuntimeError):
        SatelliteEncoder().transform(pd.Series(["A"]))


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------


def _covariates(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "phase_seconds": rng.uniform(0, 15, n),
        "elevation_deg": rng.uniform(25, 90, n),
        "distance_km": rng.uniform(500, 800, n),
        "candidate_count": rng.integers(10, 50, n).astype(float),
    })


def test_hierarchy_runs_finest_to_global():
    assigner = RegimeAssigner(RegimeConfig()).fit(_covariates())
    names = assigner.level_names()
    assert names[0] == "phase+elevation+distance+candidates"
    assert names[-1] == "global"
    assert len(names) == 5


def test_coarse_labels_are_prefixes_of_fine_labels():
    """This is what makes the fallback a lookup rather than a re-bucket."""
    covariates = _covariates()
    levels = RegimeAssigner(RegimeConfig()).fit(covariates).assign_all_levels(covariates)
    for row in range(0, len(covariates), 137):
        assert str(levels[0][row]).startswith(str(levels[1][row]))
        assert str(levels[1][row]).startswith(str(levels[2][row]))


def test_sparse_regimes_climb_and_the_climb_is_recorded():
    covariates = _covariates()
    assigner = RegimeAssigner(RegimeConfig()).fit(covariates)
    levels = assigner.assign_all_levels(covariates)

    strict = build_fallback_map(levels, 5000, assigner.level_names())
    assert strict.rate_by_level() == {"global": pytest.approx(1.0)}

    loose = build_fallback_map(levels, 5, assigner.level_names())
    assert loose.rate_by_level().get("global", 0.0) < 0.5
    assert sum(loose.rate_by_level().values()) == pytest.approx(1.0)


def test_assign_before_fit_raises_for_data_dependent_axes():
    with pytest.raises(RuntimeError, match="before fit"):
        RegimeAssigner(RegimeConfig()).assign(_covariates())


def test_missing_covariates_get_their_own_bucket_rather_than_a_wrong_one():
    covariates = _covariates(100)
    covariates.loc[0, "elevation_deg"] = np.nan
    assigner = RegimeAssigner(RegimeConfig()).fit(covariates)
    assert "elevation=na" in str(assigner.assign(covariates)[0])


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def test_feature_classes_partition_the_feature_columns():
    covered = sum(FEATURE_CLASSES.values(), ())
    assert set(covered) == set(FEATURE_COLUMNS)
    assert len(covered) == len(FEATURE_COLUMNS) == 13


def test_sequence_shapes_and_units(sequences):
    assert sequences.x.shape[1:] == (30, 13)
    assert sequences.y.shape[1] == 5
    assert sequences.phase.shape == (len(sequences), 30)
    assert list(sequences.regime.columns) == list(REGIME_COVARIATES) + list(WINDOW_COVARIATES)
    # StarNet axes must be real, not the NaN placeholders.
    assert sequences.regime["elevation_deg"].notna().all()
    # Standardising must not touch the target: every metric is in Mbps.
    assert sequences.y.mean() > 20.0


def test_window_covariates_summarise_the_lookback_not_the_horizon(sequences):
    """`level` and `volatility` must be functions of the observed history only.

    They are the axes most likely to carry the calibration gain, which makes
    them the ones most worth proving cannot see the answer. Recomputing them
    from the horizon would give a different number; recomputing from the
    look-back reproduces them exactly.
    """
    import numpy as np

    index = sequences.target_index
    # x is standardised, so compare on the standardised scale by rebuilding the
    # same statistic from the same channel the windower read.
    history = sequences.x[:, :, index]
    level = sequences.regime["level"].to_numpy()
    volatility = sequences.regime["volatility"].to_numpy()

    # Standardisation is affine, so an exact correlation of 1 with the
    # look-back mean is the check, and it must not hold against the horizon.
    assert np.corrcoef(level, history.mean(axis=1))[0, 1] > 0.999
    assert np.corrcoef(volatility, history.std(axis=1))[0, 1] > 0.999
    horizon_mean = sequences.y.mean(axis=1)
    assert np.corrcoef(level, horizon_mean)[0, 1] < 0.999


def test_regime_covariates_are_read_at_the_forecast_origin(feature_frame):
    """Not from the horizon. A regime that peeked would flatter the calibration."""
    frame, _, _ = feature_frame
    from thalweg.config import FeatureConfig

    built = make_sequences(frame, config=FeatureConfig(lookback=30, horizon=5, stride=1))
    single_segment = frame[frame["segment"] == frame["segment"].iloc[0]].reset_index(drop=True)
    for index in (0, 5, 50):
        origin_row = single_segment.iloc[index + 30 - 1]
        # Axes the dataset does not carry come through as NaN, which is what
        # the "na" regime bucket exists for. Only the present ones are checked.
        present = [c for c in REGIME_COVARIATES if c in single_segment.columns]
        assert present, "no regime covariates present to check"
        for column in present:
            assert built.regime.iloc[index][column] == pytest.approx(
                float(origin_row[column]), rel=1e-5
            )


def test_windows_shorter_than_lookback_plus_horizon_raise(feature_frame):
    from thalweg.config import FeatureConfig

    frame, _, _ = feature_frame
    with pytest.raises(ValueError, match="no window"):
        make_sequences(frame.head(10), config=FeatureConfig(lookback=30, horizon=5))
