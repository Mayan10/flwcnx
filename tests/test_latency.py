"""Tests for period-level latency classification.

The AUPRC implementation is hand rolled to avoid a sklearn dependency in this
module, so it gets pinned against cases whose value can be computed by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flwcnx.state.latency import (
    DEFAULT_LATENCY_THRESHOLD_MS,
    average_precision,
    degraded_from_bound,
    label_periods,
    spike_metrics,
)

UNITS = ["s", "ms", "us", "ns"]


def _frame(latency: list[float], start="2024-05-01 00:00:00") -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=len(latency), freq="1s"),
        "latency_ms": latency,
    })


# -- period labelling --------------------------------------------------------


def test_a_clean_period_is_good_and_a_spiking_one_is_degraded():
    # 15 s at 1 Hz is one period. First all clean, then one with a spike.
    latency = [30.0] * 15 + [30.0] * 14 + [400.0]
    labels = label_periods(_frame(latency))
    assert len(labels.frame) == 2
    assert not labels.frame.loc[0, "degraded"]
    assert labels.frame.loc[1, "degraded"]


def test_the_threshold_is_casparsens_and_is_configurable():
    latency = [30.0] * 14 + [60.0]
    assert label_periods(_frame(latency)).frame["degraded"].all()
    # Raise the threshold above the spike and the same period is Good.
    loose = label_periods(_frame(latency), threshold_ms=100.0)
    assert not loose.frame["degraded"].any()
    assert DEFAULT_LATENCY_THRESHOLD_MS == 50.0


def test_good_fraction_is_the_criterion_not_a_quantile():
    """Casparsen state the rule on the fraction meeting the threshold."""
    latency = [30.0] * 14 + [400.0]          # 14/15 = 0.933 meeting
    strict = label_periods(_frame(latency), good_fraction=0.99)
    loose = label_periods(_frame(latency), good_fraction=0.90)
    assert strict.frame["degraded"].all()
    assert not loose.frame["degraded"].any()


def test_phase_offset_moves_the_period_boundaries():
    latency = [30.0] * 30
    a = label_periods(_frame(latency), phase_offset=0.0)
    b = label_periods(_frame(latency), phase_offset=7.0)
    assert list(a.frame["period_start_s"]) != list(b.frame["period_start_s"])


@pytest.mark.parametrize("unit", UNITS)
def test_labelling_is_independent_of_datetime_resolution(unit):
    """The bug that failed 13 tests on CI. It must not come back through here."""
    latency = [30.0] * 15 + [30.0] * 14 + [400.0]
    frame = _frame(latency)
    frame["timestamp"] = frame["timestamp"].dt.as_unit(unit)
    labels = label_periods(frame)
    assert len(labels.frame) == 2
    assert list(labels.frame["degraded"]) == [False, True]


def test_summary_flags_that_a_p99_over_fifteen_samples_is_really_a_maximum():
    labels = label_periods(_frame([30.0] * 45))
    s = labels.summary()
    assert s["quantile_is_effectively_max"] is True
    assert s["median_samples_per_period"] == 15.0
    assert "Casparsen" in s["reference"]


def test_missing_latency_column_is_refused_with_a_useful_message():
    with pytest.raises(KeyError, match="carries no latency"):
        label_periods(pd.DataFrame({"timestamp": pd.date_range("2024-05-01", periods=3,
                                                               freq="1s")}))


# -- AUPRC -------------------------------------------------------------------


def test_average_precision_is_one_for_a_perfect_ranking():
    scores = np.array([9.0, 8.0, 2.0, 1.0])
    labels = np.array([True, True, False, False])
    assert average_precision(scores, labels) == pytest.approx(1.0)


def test_average_precision_matches_a_hand_computed_case():
    # Ranking: T, F, T, F. Precision at the hits is 1/1 and 2/3.
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    labels = np.array([True, False, True, False])
    assert average_precision(scores, labels) == pytest.approx((1.0 + 2 / 3) / 2)


def test_average_precision_of_a_random_ranking_approaches_the_positive_rate():
    rng = np.random.default_rng(0)
    labels = rng.random(20000) < 0.3
    assert average_precision(rng.random(20000), labels) == pytest.approx(0.3, abs=0.02)


def test_average_precision_is_nan_without_positives():
    assert np.isnan(average_precision(np.array([1.0, 2.0]), np.array([False, False])))


# -- spike metrics -----------------------------------------------------------


def test_spike_metrics_confusion_counts_and_derived_rates():
    predicted = np.array([True, True, False, False])
    actual = np.array([True, False, True, False])
    m = spike_metrics(predicted, actual)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(0.5)


def test_lift_reports_how_much_better_than_chance_the_ranking_is():
    rng = np.random.default_rng(3)
    n = 8000
    actual = rng.random(n) < 0.2
    # A score that genuinely separates the classes should lift well above 1.
    scores = actual * 3.0 + rng.normal(0, 1.0, n)
    m = spike_metrics(scores > 1.5, actual, scores=scores)
    assert m.baseline_auprc == pytest.approx(0.2, abs=0.02)
    assert m.lift > 2.0


def test_spike_metrics_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        spike_metrics(np.zeros(3, bool), np.zeros(4, bool))


# -- the bound-driven decision ----------------------------------------------


def test_degraded_is_read_off_the_calibrated_bound():
    """No second classifier: the decision inherits the bound's risk guarantee."""
    bound = np.array([20.0, 49.9, 50.1, 300.0])
    np.testing.assert_array_equal(degraded_from_bound(bound),
                                  np.array([False, False, True, True]))


def test_a_more_conservative_bound_predicts_more_spikes():
    """Tightening the risk budget raises the bound, so recall rises."""
    base = np.linspace(20.0, 80.0, 200)
    loose = degraded_from_bound(base)
    conservative = degraded_from_bound(base + 15.0)
    assert conservative.sum() > loose.sum()
