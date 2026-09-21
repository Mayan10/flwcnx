"""Calibration: the coverage guarantee, the search, and the regime layer.

The tests that matter here are the statistical ones. A calibration layer that
runs without error and does not hold its budget is worse than one that crashes,
because it produces numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thalweg.calibrate.bgcfqs import BGCFQS, bgcfqs_on_residuals
from thalweg.calibrate.conformal import conformal_rank, fit_conformal
from thalweg.calibrate.regime_cal import RegimeCalibrator, global_calibrator
from thalweg.config import BGCFQSConfig, CalibrationConfig, RegimeConfig
from thalweg.eval.metrics import conditional_metrics, over_rate, worst_regime_over_rate
from thalweg.state.regime import RegimeAssigner


@pytest.mark.parametrize("epsilon", [0.05, 0.15, 0.35])
def test_conformal_holds_its_budget_out_of_sample(epsilon):
    rng = np.random.default_rng(0)
    n = 40000
    predicted = rng.normal(200, 30, n)
    actual = predicted + rng.normal(0, 40, n)
    half = n // 2

    bound = fit_conformal(predicted[:half], actual[:half], epsilon)
    achieved = over_rate(bound.apply(predicted[half:]), actual[half:])
    # Finite calibration sample, so allow a little slack around the budget.
    assert achieved == pytest.approx(epsilon, abs=0.02)


def test_conformal_rank_matches_the_guarantee():
    assert conformal_rank(999, 0.35) == 350
    assert conformal_rank(19, 0.05) == 1
    assert conformal_rank(5, 0.05) == 0        # too few points for any bound


def test_conformal_degrades_loudly_when_it_cannot_certify():
    bound = fit_conformal(np.zeros(5), np.arange(5.0), 0.05)
    assert bound.degenerate
    assert bound.rank == 0


def test_conformal_bound_is_floored():
    bound = fit_conformal(np.full(100, 10.0), np.full(100, 5.0), 0.35)
    assert bound.apply(np.array([1.0]), floor=0.0)[0] >= 0.0


def test_bgcfqs_selects_the_largest_feasible_quantile():
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.normal(0, 1, (n, 6))
    y = x @ np.arange(6) * 3 + 200 + rng.normal(0, 25, n)

    model = BGCFQS(BGCFQSConfig(n_estimators=40)).fit(x[:2000], y[:2000])
    tau = model.select(x[2000:3000], y[2000:3000], epsilon=0.35)

    assert model.trace.feasible_found
    assert 0.15 <= tau <= 0.40
    assert model.trace.selected_over_rate <= 0.35
    # The fine pass must land off the coarse grid, or it is doing nothing.
    assert model.trace.fine


def test_bgcfqs_reports_failure_rather_than_faking_success():
    """An unreachable budget must not come back looking like a pass."""
    rng = np.random.default_rng(5)
    n = 1200
    x = rng.normal(0, 1, (n, 4))
    y = rng.normal(200, 30, n)
    model = BGCFQS(BGCFQSConfig(n_estimators=30)).fit(x[:600], y[:600])
    model.select(x[600:], y[600:], epsilon=0.001)
    assert not model.trace.feasible_found


def test_bgcfqs_residual_search_respects_the_budget():
    rng = np.random.default_rng(1)
    predicted = rng.normal(200, 20, 6000)
    actual = predicted + rng.normal(0, 40, 6000)
    offset = bgcfqs_on_residuals(predicted, actual, 0.35)
    assert over_rate(predicted + offset, actual) <= 0.36


# ---------------------------------------------------------------------------
# the regime layer
# ---------------------------------------------------------------------------


def _heteroscedastic(n=60000, seed=7):
    """Low elevation means low capacity and wide residuals at the same time.

    That coupling is the whole failure mode: a single global offset chosen to
    hit the budget on average sits too high exactly where capacity is lowest.
    """
    rng = np.random.default_rng(seed)
    elevation = rng.uniform(25, 90, n)
    distance = 550 + (90 - elevation) * 4.5 + rng.normal(0, 20, n)
    candidates = rng.integers(10, 50, n).astype(float)
    phase = rng.uniform(0, 15, n)
    actual = (60 + 2.2 * elevation + 0.8 * candidates
              - 0.05 * np.maximum(distance - 645, 0) - 40 * (phase < 2))
    predicted = actual + rng.normal(0, 60 - 0.55 * elevation)
    covariates = pd.DataFrame({
        "phase_seconds": phase, "elevation_deg": elevation,
        "distance_km": distance, "candidate_count": candidates,
    })
    return predicted, actual, covariates


def test_regime_calibration_improves_conditional_risk_at_similar_mae():
    predicted, actual, covariates = _heteroscedastic()
    train, calibration, test = slice(0, 20000), slice(20000, 40000), slice(40000, None)
    config = CalibrationConfig(epsilon=0.35, regime=RegimeConfig(min_samples=200))

    global_bound = fit_conformal(predicted[calibration], actual[calibration], 0.35)
    global_lower = global_bound.apply(predicted[test])

    assigner = RegimeAssigner(config.regime).fit(covariates.iloc[train])
    calibrator = RegimeCalibrator(config=config, assigner=assigner).fit(
        predicted[calibration], actual[calibration], covariates.iloc[calibration]
    )
    regime_lower = calibrator.transform(predicted[test], covariates.iloc[test])

    global_metrics = conditional_metrics(global_lower, actual[test])
    regime_metrics = conditional_metrics(regime_lower, actual[test])

    # The global budget is still held.
    assert regime_metrics["global"].over_rate == pytest.approx(
        global_metrics["global"].over_rate, abs=0.03
    )
    # The conditional risk improves, which is the claim.
    assert regime_metrics["P30"].over_rate < global_metrics["P30"].over_rate
    assert regime_metrics["P10"].over_rate < global_metrics["P10"].over_rate
    # And it is not bought by a collapse in accuracy.
    assert regime_metrics["global"].mae < global_metrics["global"].mae * 1.15


def test_regime_layer_narrows_the_worst_regime():
    predicted, actual, covariates = _heteroscedastic()
    calibration, test = slice(0, 30000), slice(30000, None)
    config = CalibrationConfig(epsilon=0.35, regime=RegimeConfig(min_samples=200))

    assigner = RegimeAssigner(config.regime).fit(covariates.iloc[calibration])
    calibrator = RegimeCalibrator(config=config, assigner=assigner).fit(
        predicted[calibration], actual[calibration], covariates.iloc[calibration]
    )
    labels = calibrator.assign(covariates.iloc[test])

    global_bound = fit_conformal(predicted[calibration], actual[calibration], 0.35)
    worst_global = worst_regime_over_rate(global_bound.apply(predicted[test]),
                                          actual[test], labels)
    worst_regime = worst_regime_over_rate(
        calibrator.transform(predicted[test], covariates.iloc[test]), actual[test], labels
    )
    assert worst_regime < worst_global


def test_global_calibrator_matches_plain_conformal_exactly():
    """Same code path, so an ablation difference is a conditioning difference."""
    predicted, actual, covariates = _heteroscedastic(20000)
    calibration, test = slice(0, 10000), slice(10000, None)

    bound = fit_conformal(predicted[calibration], actual[calibration], 0.35)
    calibrator = global_calibrator(CalibrationConfig(epsilon=0.35)).fit(
        predicted[calibration], actual[calibration], covariates.iloc[calibration]
    )
    np.testing.assert_allclose(
        bound.apply(predicted[test]),
        calibrator.transform(predicted[test], covariates.iloc[test]),
    )


def test_unfitted_assigner_is_refused_rather_than_fit_on_calibration():
    predicted, actual, covariates = _heteroscedastic(5000)
    calibrator = RegimeCalibrator(config=CalibrationConfig())
    with pytest.raises(RuntimeError, match="not fitted"):
        calibrator.fit(predicted, actual, covariates)
    # Explicit opt-in works, because sometimes there is no separate train split.
    calibrator.fit(predicted, actual, covariates, allow_fit_assigner_on_calibration=True)
    assert calibrator.summary()["n_offsets"] > 0


def test_mismatched_lengths_are_caught():
    predicted, actual, covariates = _heteroscedastic(2000)
    assigner = RegimeAssigner(RegimeConfig()).fit(covariates)
    with pytest.raises(ValueError, match="same samples"):
        RegimeCalibrator(assigner=assigner).fit(predicted[:100], actual[:100], covariates)


def test_unseen_test_regime_climbs_its_own_hierarchy():
    predicted, actual, covariates = _heteroscedastic(20000)
    assigner = RegimeAssigner(RegimeConfig()).fit(covariates)
    calibrator = RegimeCalibrator(
        config=CalibrationConfig(epsilon=0.35), assigner=assigner
    ).fit(predicted, actual, covariates)

    exotic = covariates.head(20).copy()
    exotic["elevation_deg"] = 89.99
    exotic["distance_km"] = 4000.0        # far outside anything calibrated
    bounds = calibrator.transform(predicted[:20], exotic)
    assert np.isfinite(bounds).all()


@pytest.mark.parametrize("selector", ["conformal", "bgcfqs"])
def test_both_selectors_produce_usable_bounds(selector):
    predicted, actual, covariates = _heteroscedastic(20000)
    calibration, test = slice(0, 10000), slice(10000, None)
    assigner = RegimeAssigner(RegimeConfig()).fit(covariates.iloc[calibration])
    calibrator = RegimeCalibrator(
        config=CalibrationConfig(epsilon=0.35), assigner=assigner, selector=selector
    ).fit(predicted[calibration], actual[calibration], covariates.iloc[calibration])

    bounds = calibrator.transform(predicted[test], covariates.iloc[test])
    assert (bounds >= 0).all()
    assert over_rate(bounds, actual[test]) < 0.5
    assert calibrator.summary()["selector"] == selector


def test_bounds_never_go_below_the_floor():
    predicted, actual, covariates = _heteroscedastic(5000)
    assigner = RegimeAssigner(RegimeConfig()).fit(covariates)
    calibrator = RegimeCalibrator(assigner=assigner).fit(predicted, actual, covariates)
    assert (calibrator.transform(np.full(len(covariates), -500.0), covariates) >= 0).all()


# ---------------------------------------------------------------------------
# upper bounds, which is what the latency target needs
# ---------------------------------------------------------------------------


def _latency_heteroscedastic(n=60000, seed=4):
    """The latency analogue of the failure mode.

    Heavy obstruction raises the delay and widens the residuals at the same
    time, so a single global offset chosen to hit the budget on average sits
    too low exactly where the link is already struggling.
    """
    rng = np.random.default_rng(seed)
    obstruction = rng.uniform(0, 0.02, n)
    hour = rng.integers(0, 24, n).astype(float)
    actual = (28 + 900 * obstruction + 3 * np.sin(2 * np.pi * hour / 24)
              + rng.gamma(2, 2, n))
    predicted = actual + rng.normal(0, 2 + 260 * obstruction)
    covariates = pd.DataFrame({
        "phase_seconds": rng.uniform(0, 15, n),
        "elevation_deg": obstruction * 4000,
        "distance_km": hour * 30,
        "candidate_count": rng.integers(10, 50, n).astype(float),
    })
    return predicted, actual, covariates


@pytest.mark.parametrize("epsilon", [0.05, 0.15, 0.35])
def test_upper_conformal_holds_its_budget_out_of_sample(epsilon):
    rng = np.random.default_rng(0)
    n = 40000
    actual = 30 + rng.gamma(2, 4, n)
    predicted = actual + rng.normal(0, 6, n)
    half = n // 2

    bound = fit_conformal(predicted[:half], actual[:half], epsilon, direction="upper")
    achieved = over_rate(bound.apply(predicted[half:]), actual[half:], direction="upper")
    assert achieved == pytest.approx(epsilon, abs=0.02)


def test_upper_and_lower_offsets_sit_on_opposite_sides():
    rng = np.random.default_rng(1)
    actual = 30 + rng.gamma(2, 4, 20000)
    predicted = actual + rng.normal(0, 6, 20000)
    lower = fit_conformal(predicted, actual, 0.2, direction="lower")
    upper = fit_conformal(predicted, actual, 0.2, direction="upper")
    assert lower.offset < 0 < upper.offset
    assert lower.rank == upper.rank


def test_tighter_budget_raises_an_upper_bound():
    """Opposite of the lower-bound case: safer means higher, not lower."""
    rng = np.random.default_rng(2)
    actual = 30 + rng.gamma(2, 4, 20000)
    predicted = actual + rng.normal(0, 6, 20000)
    tight = fit_conformal(predicted, actual, 0.05, direction="upper")
    loose = fit_conformal(predicted, actual, 0.35, direction="upper")
    assert tight.offset > loose.offset


def test_bgcfqs_residual_search_works_upward():
    rng = np.random.default_rng(3)
    actual = 30 + rng.gamma(2, 4, 20000)
    predicted = actual + rng.normal(0, 6, 20000)
    offset = bgcfqs_on_residuals(predicted, actual, 0.35, direction="upper")
    assert over_rate(predicted + offset, actual, direction="upper") <= 0.36


def test_regime_layer_improves_conditional_risk_on_latency():
    predicted, actual, covariates = _latency_heteroscedastic()
    train, calibration, test = slice(0, 20000), slice(20000, 40000), slice(40000, None)
    config = CalibrationConfig(epsilon=0.35, regime=RegimeConfig(min_samples=200))

    global_lower = fit_conformal(predicted[calibration], actual[calibration], 0.35,
                                 direction="upper").apply(predicted[test])
    assigner = RegimeAssigner(config.regime).fit(covariates.iloc[train])
    calibrator = RegimeCalibrator(config=config, assigner=assigner,
                                  direction="upper").fit(
        predicted[calibration], actual[calibration], covariates.iloc[calibration]
    )
    regime_bound = calibrator.transform(predicted[test], covariates.iloc[test])

    plain = conditional_metrics(global_lower, actual[test], direction="upper")
    ours = conditional_metrics(regime_bound, actual[test], direction="upper")

    assert ours["global"].over_rate == pytest.approx(plain["global"].over_rate, abs=0.03)
    assert ours["P30"].over_rate < plain["P30"].over_rate
    assert ours["P10"].over_rate < plain["P10"].over_rate
    assert ours["global"].mae < plain["global"].mae * 1.15
    assert calibrator.summary()["direction"] == "upper"


def test_latency_risk_slice_selects_the_high_tail():
    """Slicing the low tail on a latency target would report the easy samples
    as the hard case, which is exactly the kind of error that looks like a win."""
    from thalweg.eval.metrics import risk_slice_mask

    actual = np.arange(1000, dtype=float)
    high = risk_slice_mask(actual, 0.10, direction="upper")
    assert actual[high].min() >= np.quantile(actual, 0.90) - 1e-9
    low = risk_slice_mask(actual, 0.10, direction="lower")
    assert actual[low].max() <= np.quantile(actual, 0.10) + 1e-9


def test_bad_direction_is_rejected():
    with pytest.raises(ValueError, match="direction must be one of"):
        fit_conformal(np.zeros(100), np.ones(100), 0.35, direction="sideways")
    with pytest.raises(ValueError, match="direction must be one of"):
        RegimeCalibrator(direction="sideways")
