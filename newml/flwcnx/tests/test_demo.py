"""Tests for the replay engine.

The engine exists to *show* that the system is causal, so the test that matters
is the one proving it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flwcnx.calibrate.adaptive import AdaptiveRegimeCalibrator
from flwcnx.config import CalibrationConfig, DecisionConfig, RegimeConfig
from flwcnx.demo import DemoEngine, records_to_frame
from flwcnx.state.regime import RegimeAssigner

EPS = 0.35


def _setup(n_cal=2000, n_test=600, seed=0, axes=("level",)):
    rng = np.random.default_rng(seed)
    predicted_cal = rng.normal(200.0, 30.0, n_cal)
    actual_cal = predicted_cal + rng.normal(0.0, 20.0, n_cal)
    cov_cal = pd.DataFrame({"level": predicted_cal})

    predicted = rng.normal(200.0, 30.0, n_test)
    actual = predicted + rng.normal(0.0, 20.0, n_test)
    cov = pd.DataFrame({"level": predicted})

    config = CalibrationConfig(epsilon=EPS,
                               regime=RegimeConfig(axes=axes, min_samples=50))
    assigner = RegimeAssigner(config.regime).fit(cov_cal) if axes else None
    calibrator = AdaptiveRegimeCalibrator(config=config, assigner=assigner,
                                          direction="lower")
    calibrator.fit(predicted_cal, actual_cal, cov_cal)
    engine = DemoEngine(calibrator=calibrator, decision=DecisionConfig(),
                        assigner=assigner, config=config)
    return engine, predicted, actual, cov


def test_engine_emits_one_record_per_decision():
    engine, predicted, actual, cov = _setup()
    records = list(engine.run(predicted, actual, cov))
    assert len(records) == len(predicted)
    frame = records_to_frame(records)
    assert set(frame["step"]) == set(range(len(predicted)))


def test_decisions_do_not_depend_on_future_outcomes():
    """The property the whole demo exists to demonstrate.

    Overwrite every outcome after a cut point and the decisions before the cut
    must be byte identical. If they are not, the engine is reading the future
    and every number it displays is fiction.
    """
    cut = 300

    def run(mutate: bool):
        engine, predicted, actual, cov = _setup()
        stream = actual.copy()
        if mutate:
            stream[cut:] = 0.0
        return records_to_frame(list(engine.run(predicted, stream, cov)))

    baseline, perturbed = run(False), run(True)
    columns = ["predicted_mbps", "bound_mbps", "offset_mbps", "alpha", "admitted"]
    pd.testing.assert_frame_equal(baseline.loc[:cut - 1, columns],
                                  perturbed.loc[:cut - 1, columns])
    # And the tail must respond, or the test proves nothing.
    assert not baseline.loc[cut:, "bound_mbps"].equals(perturbed.loc[cut:, "bound_mbps"])


def test_realised_risk_lands_near_the_budget():
    engine, predicted, actual, cov = _setup(n_test=4000)
    list(engine.run(predicted, actual, cov))
    summary = engine.summary()
    assert summary["realised_risk_rate"] == pytest.approx(EPS, abs=0.06)


def test_admission_arithmetic_is_consistent():
    engine, predicted, actual, cov = _setup()
    frame = records_to_frame(list(engine.run(predicted, actual, cov)))
    per = engine.decision.bandwidth_per_session_mbps
    assert (frame["admitted"] == (frame["bound_mbps"].clip(lower=0) // per)).all()
    assert (frame["dropped"] == (frame["admitted"] - frame["oracle"]).clip(lower=0)).all()
    assert (frame["unused"] == (frame["oracle"] - frame["admitted"]).clip(lower=0)).all()
    # A dropped session means the bound promised capacity that was not there.
    assert (frame.loc[frame["dropped"] > 0, "risk_event"]).all()


def test_congestion_needs_a_sustained_window():
    """One dip below the commitment is not congestion; a run of them is."""
    engine, predicted, actual, cov = _setup()
    engine.decision = DecisionConfig(commitment_mbps=1e9)   # always below
    engine.congestion_window = 5
    frame = records_to_frame(list(engine.run(predicted, actual, cov)))
    assert not frame.loc[:3, "congested"].any()
    assert frame.loc[4:, "congested"].all()


def test_congestion_never_fires_when_the_bound_clears_the_commitment():
    engine, predicted, actual, cov = _setup()
    engine.decision = DecisionConfig(commitment_mbps=0.0)
    frame = records_to_frame(list(engine.run(predicted, actual, cov)))
    assert not frame["congested"].any()


def test_engine_runs_without_regime_axes():
    engine, predicted, actual, cov = _setup(axes=())
    frame = records_to_frame(list(engine.run(predicted, actual, cov)))
    assert (frame["regime"] == "global").all()


def test_limit_truncates_the_replay():
    engine, predicted, actual, cov = _setup()
    records = list(engine.run(predicted, actual, cov, limit=50))
    assert len(records) == 50
