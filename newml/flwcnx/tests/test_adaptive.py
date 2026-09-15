"""Tests for the online calibration layer.

The two that matter are the causality test and the shift test. The first pins
the property that makes the method honest; the second pins the property that
makes it worth having.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flwcnx.calibrate.adaptive import (
    AdaptiveRegimeCalibrator,
    offset_for_alpha,
    static_then_adaptive,
)
from flwcnx.calibrate.conformal import fit_conformal
from flwcnx.config import CalibrationConfig, RegimeConfig
from flwcnx.eval.metrics import over_rate
from flwcnx.state.regime import RegimeAssigner

EPSILON = 0.2


def _covariates(level: np.ndarray) -> pd.DataFrame:
    """Regime frame carrying only the look-back level axis."""
    return pd.DataFrame({"level": level})


def _config(axes: tuple[str, ...] = ("level",), epsilon: float = EPSILON) -> CalibrationConfig:
    return CalibrationConfig(epsilon=epsilon,
                             regime=RegimeConfig(axes=axes, min_samples=50))


def _assigner(config: CalibrationConfig, frame: pd.DataFrame) -> RegimeAssigner:
    return RegimeAssigner(config.regime).fit(frame)


# -- the offset mapping ------------------------------------------------------


def test_offset_for_alpha_matches_the_requested_rate_both_directions():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0.0, 10.0, 20_000)

    lower = offset_for_alpha(residuals, 0.2, "lower")
    # bound = pred + offset overestimates when residual < offset
    assert np.mean(residuals < lower) == pytest.approx(0.2, abs=0.02)

    upper = offset_for_alpha(residuals, 0.2, "upper")
    assert np.mean(residuals > upper) == pytest.approx(0.2, abs=0.02)


def test_offset_for_alpha_handles_an_empty_window():
    assert offset_for_alpha(np.array([]), 0.3, "lower") == 0.0


# -- causality ---------------------------------------------------------------


def test_adaptive_bounds_do_not_depend_on_future_outcomes():
    """Rewriting the tail of the outcome stream must not move earlier bounds.

    This is the property that separates an online method from a leak. If the
    bound at step t changed when step t+k's actual changed, the layer would be
    reading the future and every risk number it produced would be fiction.
    """
    rng = np.random.default_rng(7)
    n_cal, n_test = 600, 400
    predicted_cal = rng.normal(200.0, 30.0, n_cal)
    actual_cal = predicted_cal + rng.normal(0.0, 15.0, n_cal)
    cov_cal = _covariates(predicted_cal)

    predicted_test = rng.normal(200.0, 30.0, n_test)
    actual_test = predicted_test + rng.normal(0.0, 15.0, n_test)
    cov_test = _covariates(predicted_test)

    config = _config()
    assigner = _assigner(config, cov_cal)

    def run(actual: np.ndarray) -> np.ndarray:
        cal = AdaptiveRegimeCalibrator(config=config, assigner=assigner)
        cal.fit(predicted_cal, actual_cal, cov_cal)
        return cal.transform_online(predicted_test, actual, cov_test)

    baseline = run(actual_test)

    cut = n_test // 2
    tampered = actual_test.copy()
    tampered[cut:] = 0.0            # destroy every outcome after the cut
    perturbed = run(tampered)

    np.testing.assert_array_equal(baseline[:cut], perturbed[:cut])
    # And the tail must actually respond, otherwise the test proves nothing.
    assert not np.array_equal(baseline[cut:], perturbed[cut:])


# -- the reason the module exists -------------------------------------------


def test_adaptation_recovers_the_budget_under_a_shift_that_breaks_static():
    """A level shift between calibration and test, exactly as measured on WetLinks.

    The static conformal offset is fit on a calibration block whose residuals
    are centred, then applied to a test block that has moved down. Static
    overshoots its budget; the online version pulls back to it.
    """
    rng = np.random.default_rng(11)
    n_cal, n_test = 3000, 3000

    predicted_cal = rng.normal(210.0, 40.0, n_cal)
    actual_cal = predicted_cal + rng.normal(0.0, 20.0, n_cal)

    # The test month is slower than the calibration month, so the forecaster
    # over-predicts and every residual moves down. WetLinks: 219 -> 197 Mbps.
    predicted_test = rng.normal(210.0, 40.0, n_test)
    actual_test = predicted_test + rng.normal(-22.0, 20.0, n_test)

    static = fit_conformal(predicted_cal, actual_cal, EPSILON, direction="lower")
    static_rate = over_rate(static.apply(predicted_test), actual_test, "lower")

    config = _config(axes=())
    online = AdaptiveRegimeCalibrator(config=config, assigner=RegimeAssigner(config.regime))
    online.fit(predicted_cal, actual_cal, _covariates(predicted_cal))
    bounds = online.transform_online(predicted_test, actual_test, _covariates(predicted_test))
    online_rate = over_rate(bounds, actual_test, "lower")

    assert static_rate > EPSILON + 0.10          # static loses the budget
    assert online_rate < static_rate             # adaptation recovers most of it
    # Measured over the second half, after alpha has had time to converge.
    settled = over_rate(bounds[n_test // 2:], actual_test[n_test // 2:], "lower")
    assert settled == pytest.approx(EPSILON, abs=0.06)


def test_alpha_follows_the_published_update_rule_step_by_step():
    """alpha_{t+1} = alpha_t + gamma * (epsilon - err_t), clipped.

    Pinned exactly rather than in spirit, because this is the one piece of the
    module that is a reproduction of Gibbs and Candes and it should stay
    recognisable as theirs.
    """
    rng = np.random.default_rng(3)
    predicted_cal = rng.normal(100.0, 10.0, 500)
    actual_cal = predicted_cal + rng.normal(0.0, 5.0, 500)
    config = _config(axes=())
    online = AdaptiveRegimeCalibrator(config=config, assigner=RegimeAssigner(config.regime))
    online.fit(predicted_cal, actual_cal, _covariates(predicted_cal))

    predicted_test = rng.normal(100.0, 10.0, 300)
    actual_test = predicted_test + rng.normal(-20.0, 5.0, 300)
    online.transform_online(predicted_test, actual_test, _covariates(predicted_test))

    trace = online.trace
    assert trace is not None
    expected = trace.alpha[:-1] + online.gamma * (EPSILON - trace.risk_event[:-1])
    expected = np.clip(expected, online.alpha_floor, online.alpha_ceiling)
    np.testing.assert_allclose(trace.alpha[1:], expected, rtol=0, atol=1e-12)
    assert trace.risk_event.any()      # the rule was exercised in both branches
    assert not trace.risk_event.all()


def test_the_rolling_window_absorbs_a_shift_without_alpha_having_to():
    """Two mechanisms adapt, and the window is the faster one.

    Under a hard level shift the window refills with shifted residuals within a
    few hundred steps, the risk event stops firing, and alpha is then free to
    drift back up rather than being pinned conservative forever. Worth a test
    because the first version of this file assumed the opposite.
    """
    rng = np.random.default_rng(3)
    predicted_cal = rng.normal(100.0, 10.0, 500)
    actual_cal = predicted_cal + rng.normal(0.0, 5.0, 500)
    config = _config(axes=())
    online = AdaptiveRegimeCalibrator(config=config, assigner=RegimeAssigner(config.regime))
    online.fit(predicted_cal, actual_cal, _covariates(predicted_cal))

    predicted_test = rng.normal(100.0, 10.0, 400)
    actual_test = predicted_test - 60.0
    online.transform_online(predicted_test, actual_test, _covariates(predicted_test))

    trace = online.trace
    assert trace is not None
    # Risk fires early while the window still holds the old residuals, then stops.
    assert trace.risk_event[:50].mean() > trace.risk_event[-50:].mean()
    assert trace.offset[-1] < trace.offset[0] - 20.0
    assert online.alpha_floor <= trace.alpha[-1] <= online.alpha_ceiling


# -- conditional behaviour ---------------------------------------------------


def test_per_regime_alphas_diverge_when_regimes_shift_differently():
    """One alpha per regime is the part that is ours, so it needs its own test.

    Two regimes, only one of which drifts. A single global alpha would split
    the difference and hold neither; per regime alphas should separate.
    """
    rng = np.random.default_rng(19)
    n = 4000
    level = np.where(rng.random(n) < 0.5, 50.0, 250.0)

    predicted_cal = level + rng.normal(0.0, 10.0, n)
    actual_cal = predicted_cal + rng.normal(0.0, 12.0, n)

    level_test = np.where(rng.random(n) < 0.5, 50.0, 250.0)
    predicted_test = level_test + rng.normal(0.0, 10.0, n)
    # Only the low regime degrades in the test period.
    drift = np.where(level_test < 150.0, -30.0, 0.0)
    actual_test = predicted_test + drift + rng.normal(0.0, 12.0, n)

    config = _config(axes=("level",))
    assigner = _assigner(config, _covariates(predicted_cal))
    online = AdaptiveRegimeCalibrator(config=config, assigner=assigner)
    online.fit(predicted_cal, actual_cal, _covariates(predicted_cal))
    online.transform_online(predicted_test, actual_test, _covariates(predicted_test))

    per_regime = online.summary()["per_regime"]
    assert len(per_regime) >= 2
    ends = sorted(v["alpha_end"] for v in per_regime.values())
    # The drifting regime must have been pushed materially more conservative
    # than the stable one, which is the whole claim.
    assert ends[0] < ends[-1] - 0.05


def test_static_then_adaptive_returns_a_like_for_like_pair():
    rng = np.random.default_rng(23)
    n = 1500
    predicted_cal = rng.normal(200.0, 30.0, n)
    actual_cal = predicted_cal + rng.normal(0.0, 18.0, n)
    predicted_test = rng.normal(200.0, 30.0, n)
    actual_test = predicted_test + rng.normal(-15.0, 18.0, n)

    config = _config(axes=("level",))
    assigner = _assigner(config, _covariates(predicted_cal))
    static, adaptive, summary = static_then_adaptive(
        config, assigner, predicted_cal, actual_cal, _covariates(predicted_cal),
        predicted_test, actual_test, _covariates(predicted_test),
    )
    assert static.shape == adaptive.shape == (n,)
    assert summary["adaptive"]["realised_risk_rate"] <= summary["adaptive"]["epsilon"] + 0.1
    assert "reference" in summary["adaptive"]      # attribution survives into results


def test_fit_refuses_an_unfitted_assigner():
    config = _config(axes=("level",))
    online = AdaptiveRegimeCalibrator(config=config, assigner=RegimeAssigner(config.regime))
    with pytest.raises(RuntimeError, match="not fitted"):
        online.fit(np.zeros(10), np.zeros(10), _covariates(np.zeros(10)))


def test_transform_online_refuses_mismatched_lengths():
    config = _config(axes=())
    online = AdaptiveRegimeCalibrator(config=config, assigner=RegimeAssigner(config.regime))
    online.fit(np.zeros(10), np.zeros(10), _covariates(np.zeros(10)))
    with pytest.raises(ValueError, match="same length"):
        online.transform_online(np.zeros(5), np.zeros(4), _covariates(np.zeros(5)))
