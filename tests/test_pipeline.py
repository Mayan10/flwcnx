"""End to end: ingest through to allocation, on synthetic data.

Guards the seams between layers, which is where the errors that survive unit
tests live: units silently changing, a standardiser applied twice, a regime
label built from covariates the forecaster never saw.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from thalweg.config import (
    CalibrationConfig,
    DataConfig,
    DecisionConfig,
    ExperimentConfig,
    FeatureConfig,
    RegimeConfig,
    SplitConfig,
    StarNetConfig,
    resolve_device,
    seed_everything,
)
from thalweg.eval.runner import comparison_table, prepare, risk_table, run_experiment
from thalweg.ingest.synthetic import SyntheticSource, SyntheticSpec


@pytest.fixture(scope="module")
def small_config():
    return ExperimentConfig(
        name="pipeline-test",
        seed=7,
        device="cpu",
        data=DataConfig(location="usa"),
        features=FeatureConfig(lookback=30, horizon=5, stride=3),
        model=StarNetConfig(epochs=2, batch_size=256),
        split=SplitConfig(scheme="temporal"),
        calibration=CalibrationConfig(epsilon=0.35, regime=RegimeConfig(min_samples=50)),
        decision=DecisionConfig(),
    )


@pytest.fixture(scope="module")
def source():
    return SyntheticSource(SyntheticSpec(n_seconds=12000, phase_offset=12, seed=3))


def test_prepare_fits_the_standardizer_on_training_windows_only(source, small_config):
    prepared = prepare(source, small_config)
    train = prepared["train"]
    # Training windows are centred; the standardiser saw them and nothing else.
    assert abs(float(train.x[:, :, 0].mean())) < 0.2
    assert prepared["leak"]["clean"]
    # The target is never scaled, whatever happens to the inputs.
    assert train.y.mean() > 20.0


def test_phase_is_recovered_end_to_end(source, small_config):
    reference = prepare(source, small_config)["phase_reference"]
    assert reference.is_recovered
    assert abs(reference.offset_seconds - 12.0) < 1.0


def test_full_experiment_produces_every_downstream_artifact(source, small_config, tmp_path):
    result = run_experiment(
        source, small_config, backbone="starnet",
        calibration_methods=("point", "global_conformal", "regime_conformal"),
        epsilons=(0.35,), granularities=("global", "full"), verbose=False,
    )

    assert result.leak_check["clean"]
    assert result.point_metrics["backbone"] == "starnet"
    assert len(result.calibration) == 6      # 3 methods x 2 granularities

    for key, entry in result.calibration.items():
        for slice_name in ("global", "P30", "P10"):
            assert entry[slice_name]["n"] > 0
            assert 0.0 <= entry[slice_name]["OverRate"] <= 1.0
        assert key in result.admission
        assert key in result.congestion
        assert np.isfinite(entry["worst_regime_over_rate"])

    written = result.save(tmp_path / "run")
    payload = json.loads((written / "result.json").read_text())
    assert payload["config"]["seed"] == 7
    # A config snapshot must sit beside every result, or no figure is traceable.
    assert json.loads((written / "config.json").read_text())["features"]["lookback"] == 30
    assert (written / "config.json").exists()

    assert not comparison_table(result).empty
    assert not risk_table(result).empty


def test_calibrated_bound_is_never_worse_than_the_point_forecast_on_risk(
    source, small_config
):
    """Calibration exists to lower the overestimation rate. If it raises it,
    something is wired backwards."""
    result = run_experiment(
        source, small_config, backbone="starnet",
        calibration_methods=("point", "global_conformal"),
        epsilons=(0.35,), granularities=("global",), verbose=False,
    )
    point = result.calibration["point|eps=0.35|regime=global"]["global"]["OverRate"]
    calibrated = result.calibration["global_conformal|eps=0.35|regime=global"]["global"]
    assert calibrated["OverRate"] <= point + 1e-9


def test_global_granularity_reproduces_the_global_calibrator(source, small_config):
    """The ablation floor must equal plain global conformal exactly."""
    result = run_experiment(
        source, small_config, backbone="starnet",
        calibration_methods=("global_conformal", "regime_conformal"),
        epsilons=(0.35,), granularities=("global",), verbose=False,
    )
    a = result.calibration["global_conformal|eps=0.35|regime=global"]["global"]
    b = result.calibration["regime_conformal|eps=0.35|regime=global"]["global"]
    assert a["OverRate"] == pytest.approx(b["OverRate"])
    assert a["MAE"] == pytest.approx(b["MAE"])


def test_tighter_budgets_produce_more_conservative_bounds(source, small_config):
    result = run_experiment(
        source, small_config, backbone="starnet",
        calibration_methods=("global_conformal",),
        epsilons=(0.05, 0.35), granularities=("global",), verbose=False,
    )
    tight = result.calibration["global_conformal|eps=0.05|regime=global"]
    loose = result.calibration["global_conformal|eps=0.35|regime=global"]
    assert tight["detail"]["offset_mbps"] < loose["detail"]["offset_mbps"]
    assert tight["global"]["OverRate"] <= loose["global"]["OverRate"]


def test_seeding_is_reproducible():
    seed_everything(123)
    first = np.random.rand(5)
    seed_everything(123)
    np.testing.assert_allclose(first, np.random.rand(5))


def test_device_resolution_never_returns_auto():
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    assert resolve_device("cpu") == "cpu"


def test_config_round_trips_to_json(tmp_path, small_config):
    path = small_config.save(tmp_path / "config.json")
    loaded = json.loads(path.read_text())
    assert loaded["seed"] == 7
    assert loaded["features"]["horizon"] == 5
    assert loaded["calibration"]["regime"]["min_samples"] == 50


def test_invalid_location_is_rejected():
    with pytest.raises(ValueError, match="location must be"):
        DataConfig(location="mars")
