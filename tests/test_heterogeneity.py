"""Tests for the heterogeneity gate.

The two that matter are the null calibration test, which checks the gate does
not fire on homogeneous data at roughly its nominal rate, and the round trip on
the real measured offsets from the four datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thalweg.calibrate.adaptive import AdaptiveRegimeCalibrator
from thalweg.calibrate.heterogeneity import (
    GatedCalibrator,
    _chi2_sf,
    assess_regime_heterogeneity,
    cochran_q,
    offset_standard_error,
)
from thalweg.config import CalibrationConfig, RegimeConfig
from thalweg.state.regime import RegimeAssigner

EPS = 0.35


def _config(axes=("level",)) -> CalibrationConfig:
    return CalibrationConfig(epsilon=EPS, regime=RegimeConfig(axes=axes, min_samples=50))


def _frame(level: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"level": level})


# -- the chi-squared tail ----------------------------------------------------


@pytest.mark.parametrize(("x", "dof", "expected"), [
    (3.841, 1, 0.05), (5.991, 2, 0.05), (7.815, 3, 0.05), (11.070, 5, 0.05),
    (6.635, 1, 0.01), (9.210, 2, 0.01), (15.086, 5, 0.01),
    (0.004, 1, 0.95), (0.103, 2, 0.95),
])
def test_chi2_tail_matches_published_critical_values(x, dof, expected):
    """scipy is not a dependency, so the tail is hand rolled and has to be pinned."""
    assert _chi2_sf(x, dof) == pytest.approx(expected, abs=0.002)


# -- Cochran's Q -------------------------------------------------------------


def test_q_is_zero_when_every_group_agrees():
    q, dof, p, i2 = cochran_q(np.array([5.0, 5.0, 5.0]), np.array([1.0, 1.0, 1.0]))
    assert q == pytest.approx(0.0)
    assert dof == 2
    assert p == pytest.approx(1.0)
    assert i2 == 0.0


def test_q_grows_with_spread_and_shrinks_with_uncertainty():
    tight = cochran_q(np.array([0.0, 10.0]), np.array([1.0, 1.0]))[0]
    loose = cochran_q(np.array([0.0, 10.0]), np.array([10.0, 10.0]))[0]
    assert tight > loose
    wider = cochran_q(np.array([0.0, 30.0]), np.array([1.0, 1.0]))[0]
    assert wider > tight


# -- the gate's null behaviour ----------------------------------------------


def test_gate_does_not_fire_on_homogeneous_regimes():
    """Regimes that are labels over noise must not look heterogeneous.

    This is the failure mode the gate exists to prevent: firing on the US
    trace, where the four regimes want offsets within 1.85 Mbps of each other,
    and paying estimation noise for nothing.
    """
    rng = np.random.default_rng(0)
    n = 8000
    level = rng.normal(200.0, 40.0, n)          # a real covariate ...
    predicted = rng.normal(200.0, 30.0, n)
    actual = predicted + rng.normal(0.0, 20.0, n)   # ... that residuals ignore

    config = _config()
    assigner = RegimeAssigner(config.regime).fit(_frame(level))
    result = assess_regime_heterogeneity(predicted, actual, _frame(level), config, assigner)

    assert not result.condition
    assert result.n_regimes >= 2
    assert result.spread < 5.0


def test_gate_null_rate_is_near_nominal():
    """Across repeated homogeneous draws the gate should fire rarely.

    Not an exact size calculation: the offsets share a calibration sample so
    they are not independent, and I-squared adds a second hurdle. The check is
    that the rate is small, which is what protects the US-trace case.
    """
    fired = 0
    trials = 40
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        n = 4000
        level = rng.normal(200.0, 40.0, n)
        predicted = rng.normal(200.0, 30.0, n)
        actual = predicted + rng.normal(0.0, 20.0, n)
        config = _config()
        assigner = RegimeAssigner(config.regime).fit(_frame(level))
        if assess_regime_heterogeneity(predicted, actual, _frame(level),
                                     config, assigner).condition:
            fired += 1
    assert fired / trials < 0.20, f"gate fired on {fired}/{trials} homogeneous draws"


def test_gate_fires_when_regimes_genuinely_differ():
    """Heteroscedastic by level, which is the case conditioning is built for."""
    rng = np.random.default_rng(3)
    n = 8000
    level = rng.normal(200.0, 40.0, n)
    predicted = level + rng.normal(0.0, 10.0, n)
    # Residual spread scales with level, so the low regime needs a much larger
    # safety margin than the high one.
    scale = np.clip(60.0 - 0.22 * level, 5.0, None)
    actual = predicted + rng.normal(0.0, 1.0, n) * scale

    config = _config()
    assigner = RegimeAssigner(config.regime).fit(_frame(level))
    result = assess_regime_heterogeneity(predicted, actual, _frame(level), config, assigner)

    assert result.condition, result.reason
    assert result.p_value < 0.05
    assert result.i_squared >= 0.25
    assert result.spread > 5.0


def test_gate_refuses_when_no_axes_are_configured():
    config = _config(axes=())
    result = assess_regime_heterogeneity(np.zeros(100), np.zeros(100),
                                       _frame(np.zeros(100)), config)
    assert not result.condition
    assert "no regime axes" in result.reason


def test_gate_refuses_when_regimes_are_too_small():
    rng = np.random.default_rng(5)
    n = 40                                   # below MIN_REGIME_FOR_TEST per bucket
    level = rng.normal(200.0, 40.0, n)
    predicted = rng.normal(200.0, 30.0, n)
    actual = predicted + rng.normal(0.0, 20.0, n)
    config = _config()
    assigner = RegimeAssigner(config.regime).fit(_frame(level))
    result = assess_regime_heterogeneity(predicted, actual, _frame(level), config, assigner)
    assert not result.condition
    assert "cleared" in result.reason


# -- the quantile standard error --------------------------------------------


def test_offset_standard_error_shrinks_with_sample_size():
    rng = np.random.default_rng(7)
    small = offset_standard_error(rng.normal(0, 10, 200), EPS)
    large = offset_standard_error(rng.normal(0, 10, 20000), EPS)
    assert large < small
    # Roughly the 1/sqrt(n) rate, so a 100x sample is about 10x tighter.
    assert 3.0 < small / large < 30.0


def test_offset_standard_error_is_infinite_below_the_floor():
    assert not np.isfinite(offset_standard_error(np.zeros(5), EPS))


# -- the wrapper -------------------------------------------------------------


def test_gated_calibrator_drops_axes_when_the_gate_says_no():
    rng = np.random.default_rng(11)
    n = 6000
    level = rng.normal(200.0, 40.0, n)
    predicted = rng.normal(200.0, 30.0, n)
    actual = predicted + rng.normal(0.0, 20.0, n)
    frame = _frame(level)

    config = _config()
    assigner = RegimeAssigner(config.regime).fit(frame)

    def factory(cfg, asg):
        return AdaptiveRegimeCalibrator(config=cfg, assigner=asg, direction="lower")

    gated = GatedCalibrator(config=config, assigner=assigner, factory=factory)
    gated.fit(predicted, actual, frame)

    assert gated.used_axes == ()
    bounds = gated.transform_online(predicted, actual, frame)
    assert bounds.shape == (n,)
    assert gated.summary()["gate"]["condition"] is False


def test_gated_calibrator_keeps_axes_when_the_gate_says_yes():
    rng = np.random.default_rng(13)
    n = 8000
    level = rng.normal(200.0, 40.0, n)
    predicted = level + rng.normal(0.0, 10.0, n)
    scale = np.clip(60.0 - 0.22 * level, 5.0, None)
    actual = predicted + rng.normal(0.0, 1.0, n) * scale
    frame = _frame(level)

    config = _config()
    assigner = RegimeAssigner(config.regime).fit(frame)

    def factory(cfg, asg):
        return AdaptiveRegimeCalibrator(config=cfg, assigner=asg, direction="lower")

    gated = GatedCalibrator(config=config, assigner=assigner, factory=factory)
    gated.fit(predicted, actual, frame)
    assert gated.used_axes == ("level",)


# -- the measured offsets from the real runs --------------------------------


@pytest.mark.parametrize(("name", "offsets", "should_condition"), [
    # StarNet USA: -10.69, -11.91, -10.06, -10.99. Conditioning cost 6.1% there.
    ("starnet_usa", [-10.69, -11.91, -10.06, -10.99], False),
    # StarNet Canada: spread 11.30 Mbps. Conditioning gained 1.7%.
    ("starnet_canada", [-4.81, 2.71, 4.10, 6.49], True),
])
def test_gate_agrees_with_the_measured_outcome(name, offsets, should_condition):
    """The gate has to reproduce the sign of the effect actually observed.

    Standard errors are set to a plausible common value for a few thousand
    calibration points, which is what those runs had. The point is the ordering:
    a 1.85 Mbps spread must not clear, an 11.30 Mbps spread must.
    """
    se = np.full(len(offsets), 1.2)
    q, dof, p, i2 = cochran_q(np.array(offsets), se)
    fires = bool(p < 0.05 and i2 >= 0.25)
    assert fires is should_condition, (
        f"{name}: Q={q:.2f} p={p:.4g} I2={i2:.3f}, expected condition={should_condition}"
    )
