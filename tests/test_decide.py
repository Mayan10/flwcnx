"""Admission control and the derived congestion signal."""

from __future__ import annotations

import numpy as np
import pytest

from flwcnx.decide.admission import (
    admit_sessions,
    evaluate_admission,
    relative_reduction,
    run_admission,
)
from flwcnx.decide.congestion import (
    congestion_episodes,
    detect_congestion,
    sustained,
)


def test_admission_floors_and_never_goes_negative():
    assert admit_sessions(np.array([0.0, 9.9, 10.0, 25.0]), 10.0).tolist() == [0, 0, 1, 2]
    assert admit_sessions(np.array([-50.0]), 10.0).tolist() == [0]


def test_zero_bandwidth_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        admit_sessions(np.array([100.0]), 0.0)


def test_a_conservative_bound_drops_nothing():
    actual = np.full(100, 100.0)
    result = run_admission(actual - 50.0, actual, 10.0)
    assert result.dropped.sum() == 0
    assert result.unused.sum() > 0


def test_an_optimistic_bound_drops_sessions():
    actual = np.full(100, 100.0)
    result = run_admission(actual + 50.0, actual, 10.0)
    assert (result.dropped == 5).all()
    assert result.violations.all()


def test_utilisation_exposes_a_policy_that_admits_nobody():
    """A policy can win on dropped sessions by refusing everything."""
    actual = np.full(200, 100.0)
    table = evaluate_admission(np.zeros(200), actual, 10.0)
    row = table.set_index("slice").loc["all"]
    assert row["mean_dropped"] == 0.0
    assert row["utilisation"] == 0.0


def test_relative_reduction_reports_the_utilisation_it_cost():
    actual = np.full(200, 100.0)
    baseline = evaluate_admission(actual + 20.0, actual, 10.0)
    ours = evaluate_admission(np.zeros(200), actual, 10.0)
    comparison = relative_reduction(baseline, ours).set_index("slice").loc["all"]
    assert comparison["relative_reduction"] == pytest.approx(1.0)
    assert comparison["utilisation_delta"] < 0


def test_sustained_requires_consecutive_slots():
    flags = np.array([1, 1, 0, 1, 1, 1, 1], dtype=bool)
    assert sustained(flags, 3).tolist() == [False] * 5 + [True, True]
    assert sustained(flags, 1).tolist() == flags.tolist()


def test_sustained_is_causal():
    """A slot must never be flagged by what comes after it."""
    flags = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    assert not sustained(flags, 3)[:3].any()


def test_congestion_detected_when_the_bound_falls_below_commitment():
    signal = np.concatenate([np.full(50, 80.0), np.full(40, 20.0), np.full(50, 80.0)])
    result = detect_congestion(signal, signal, commitment_mbps=50.0, window=5)
    assert result.scores()["recall"] == pytest.approx(1.0)
    assert congestion_episodes(result.actual) == [(54, 90)]


def test_lead_time_is_measured_from_the_alert_run_start():
    signal = np.concatenate([np.full(50, 80.0), np.full(40, 20.0), np.full(50, 80.0)])
    bound = signal.copy()
    bound[42:50] = 40.0                    # bound anticipates the drop
    result = detect_congestion(bound, signal, commitment_mbps=50.0, window=5)
    assert result.lead_times.tolist() == [8.0]
    assert result.scores()["mean_lead_slots"] == pytest.approx(8.0)


def test_a_healthy_link_raises_no_alert():
    signal = np.full(200, 90.0)
    result = detect_congestion(signal, signal, commitment_mbps=50.0, window=5)
    assert not result.predicted.any()
    assert result.confusion()["fp"] == 0


def test_length_mismatch_is_caught():
    with pytest.raises(ValueError, match="same length"):
        detect_congestion(np.zeros(5), np.zeros(6), 50.0)
