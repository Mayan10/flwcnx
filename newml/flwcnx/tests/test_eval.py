"""Metrics, splits, and the leak checks that guard every experiment."""

from __future__ import annotations

import numpy as np
import pytest

from flwcnx.config import SplitConfig
from flwcnx.eval.metrics import (
    compute_metrics,
    conditional_metrics,
    mean_positive_error,
    over_rate,
    p95_positive_error,
    per_regime_metrics,
    risk_slice_mask,
    worst_regime_over_rate,
)
from flwcnx.eval.splits import (
    check_no_leak,
    contiguous_82,
    leave_one_location_out,
    purge_width,
    temporal_split,
)


def test_over_rate_is_strict():
    """A prediction exactly equal to the actual is not an overestimate.

    Ties are common on the calibration path, where the bound is set to an
    observed residual, so this is not a hypothetical distinction.
    """
    assert over_rate(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 4.0])) == pytest.approx(1 / 3)


def test_positive_error_metrics_ignore_underestimates():
    predicted = np.array([10.0, 0.0, 20.0])
    actual = np.array([5.0, 100.0, 10.0])
    assert mean_positive_error(predicted, actual) == pytest.approx(5.0)
    assert p95_positive_error(predicted, actual) >= 0.0


def test_metrics_match_hand_computed_values():
    predicted = np.array([10.0, 20.0, 30.0, 40.0])
    actual = np.array([12.0, 18.0, 33.0, 36.0])
    metrics = compute_metrics(predicted, actual)
    assert metrics.mae == pytest.approx(np.mean([2, 2, 3, 4]))
    assert metrics.rmse == pytest.approx(np.sqrt(np.mean([4, 4, 9, 16])))
    assert metrics.over_rate == pytest.approx(0.5)
    assert metrics.mpe == pytest.approx(np.mean([0, 2, 0, 4]))


def test_risk_slices_select_the_low_throughput_tail():
    actual = np.arange(1000, dtype=float)
    assert risk_slice_mask(actual, 0.30).sum() == pytest.approx(300, abs=2)
    assert risk_slice_mask(actual, 0.10).sum() == pytest.approx(100, abs=2)
    assert actual[risk_slice_mask(actual, 0.10)].max() < actual.mean()


def test_conditional_metrics_expose_a_risk_gap_the_global_number_hides():
    """The BG-CFQS failure mode, constructed: risk concentrated in the low tail."""
    rng = np.random.default_rng(0)
    n = 20000
    actual = rng.uniform(10, 400, n)
    # Overestimate almost always when capacity is low, rarely when it is high.
    low = actual < np.quantile(actual, 0.30)
    predicted = actual + np.where(low, 40.0, -40.0)

    results = conditional_metrics(predicted, actual)
    assert results["global"].over_rate < 0.4
    assert results["P30"].over_rate > 0.9
    assert results["P10"].over_rate > 0.9


def test_empty_input_yields_nan_not_a_crash():
    metrics = compute_metrics(np.array([]), np.array([]))
    assert metrics.n == 0
    assert np.isnan(metrics.mae)


def test_shape_mismatch_is_caught():
    with pytest.raises(ValueError, match="shape mismatch"):
        over_rate(np.zeros(3), np.zeros(4))


def test_non_finite_values_are_dropped():
    predicted = np.array([1.0, np.nan, 3.0])
    actual = np.array([2.0, 2.0, np.inf])
    assert compute_metrics(predicted, actual).n == 1


def test_per_regime_metrics_and_worst_case():
    predicted = np.concatenate([np.full(500, 10.0), np.full(500, 0.0)])
    actual = np.concatenate([np.full(500, 5.0), np.full(500, 5.0)])
    labels = np.array(["hot"] * 500 + ["cold"] * 500)

    table = per_regime_metrics(predicted, actual, labels)
    assert set(table["regime"]) == {"hot", "cold"}
    assert worst_regime_over_rate(predicted, actual, labels) == pytest.approx(1.0)


def test_regimes_below_the_minimum_are_excluded():
    predicted, actual = np.zeros(100), np.zeros(100)
    labels = np.array(["big"] * 95 + ["tiny"] * 5)
    assert set(per_regime_metrics(predicted, actual, labels, min_samples=10)["regime"]) == {"big"}


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def test_purge_width_covers_a_whole_window():
    assert purge_width(30, 5, 1) == 35
    assert purge_width(30, 5, 5) == 7


def test_temporal_split_is_ordered_and_clean(raw_sequences):
    split = temporal_split(raw_sequences, SplitConfig(), lookback=30, horizon=5)
    assert split.purged > 0
    check = check_no_leak(raw_sequences, split, lookback=30, horizon=5)
    assert check["clean"]

    times = raw_sequences.origin_time
    assert times[split.train].max() < times[split.calibration].min()
    assert times[split.calibration].max() < times[split.test].min()


def test_contiguous_82_matches_the_starnet_protocol(raw_sequences):
    split = contiguous_82(raw_sequences, lookback=30, horizon=5)
    total = len(split.train) + len(split.calibration) + len(split.test) + split.purged
    assert total == len(raw_sequences)
    # Test block is the last 20%, up to the purge band.
    assert len(split.test) <= round(0.2 * len(raw_sequences))
    assert check_no_leak(raw_sequences, split, lookback=30, horizon=5)["clean"]


def test_split_fractions_must_sum_to_one(raw_sequences):
    with pytest.raises(ValueError, match="sum to 1"):
        temporal_split(raw_sequences,
                       SplitConfig(train_frac=0.5, calib_frac=0.2, test_frac=0.2),
                       lookback=30, horizon=5)


def test_leave_one_location_out_holds_out_the_target(raw_sequences):
    half = len(raw_sequences) // 2
    sets = {"a": raw_sequences.subset(np.arange(0, half)),
            "b": raw_sequences.subset(np.arange(half, len(raw_sequences)))}
    train, calibration, test = leave_one_location_out(sets, "b", lookback=30, horizon=5)
    assert len(train) == len(sets["a"])
    assert len(calibration) + len(test) <= len(sets["b"])


def test_leave_one_location_out_rejects_an_unknown_location(raw_sequences):
    with pytest.raises(KeyError):
        leave_one_location_out({"a": raw_sequences}, "z", lookback=30, horizon=5)


def test_concatenated_locations_do_not_share_segment_ids(raw_sequences):
    from flwcnx.eval.splits import _concatenate

    half = len(raw_sequences) // 2
    a = raw_sequences.subset(np.arange(0, half))
    b = raw_sequences.subset(np.arange(half, len(raw_sequences)))
    joined = _concatenate([a, b])
    assert len(joined) == len(a) + len(b)
    assert joined.segment[: len(a)].max() < joined.segment[len(a) :].min() or \
        a.segment.max() == 0
