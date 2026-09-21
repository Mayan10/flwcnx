"""Tests for device selection and the memory guard.

Both exist because the project was developed on one machine. The device check
was Mac-shaped and would raise on a torch build without MPS; the windowing had
no size guard and would OOM without explanation on a laptop.
"""

from __future__ import annotations

import numpy as np
import pytest

from thalweg.device import (
    available_devices,
    check_memory_budget,
    describe_device,
    estimate_window_bytes,
    resolve_device,
)

# -- device selection --------------------------------------------------------


def test_cpu_is_always_available_and_last():
    devices = available_devices()
    assert devices[-1] == "cpu"
    assert len(set(devices)) == len(devices)


def test_auto_picks_the_best_available():
    assert resolve_device("auto") == available_devices()[0]


def test_an_explicit_available_device_is_honoured():
    assert resolve_device("cpu") == "cpu"


def test_an_unavailable_device_warns_and_falls_back_rather_than_failing():
    """A borrowed machine without a GPU should still run, just slower."""
    with pytest.warns(RuntimeWarning, match="was requested"):
        got = resolve_device("definitely-not-a-device")
    assert got in available_devices()


def test_environment_variable_overrides_everything(monkeypatch):
    """What makes a constrained CI runner reproducible without editing config."""
    monkeypatch.setenv("THALWEG_DEVICE", "cpu")
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda") == "cpu"


def test_mps_probe_survives_a_torch_without_the_backend(monkeypatch):
    """The original bug: `torch.backends.mps` does not always exist."""
    import torch

    class _Backends:
        pass                      # no `mps` attribute at all

    monkeypatch.setattr(torch, "backends", _Backends())
    assert "cpu" in available_devices()      # no AttributeError


def test_describe_device_is_json_safe():
    import json

    json.dumps(describe_device(resolve_device("auto")))


# -- memory guard ------------------------------------------------------------


def test_estimate_counts_inputs_targets_and_the_split_copy():
    # 1000 windows x 30 x 13 float32 = 1.56 MB, doubled for the split copy.
    got = estimate_window_bytes(1000, 30, 13, horizon=5)
    assert got == 1000 * (30 * 13 + 5) * 4 * 2


def test_a_run_that_fits_passes_silently():
    check_memory_budget(10_000, 30, 13, horizon=5, limit_gb=8.0)


def test_a_run_that_does_not_fit_raises_before_allocating():
    with pytest.raises(MemoryError) as excinfo:
        check_memory_budget(50_000_000, 30, 13, horizon=5, limit_gb=1.0)
    message = str(excinfo.value)
    # The error has to name both escape hatches, or it is just a crash with
    # extra words.
    assert "THALWEG_MEMORY_LIMIT_GB" in message
    assert "--stride" in message


def test_the_suggested_stride_actually_brings_the_run_under_the_limit():
    n, lookback, features, horizon, limit = 4_000_000, 30, 13, 5, 2.0
    with pytest.raises(MemoryError) as excinfo:
        check_memory_budget(n, lookback, features, horizon, limit_gb=limit, stride=1)
    suggested = int(str(excinfo.value).split("--stride ")[1].split()[0])
    assert suggested > 1
    # Windowing at the suggested stride yields proportionally fewer windows.
    check_memory_budget(n // suggested, lookback, features, horizon, limit_gb=limit)


def test_the_limit_is_configurable_by_environment(monkeypatch):
    monkeypatch.setenv("THALWEG_MEMORY_LIMIT_GB", "0.001")
    with pytest.raises(MemoryError):
        check_memory_budget(100_000, 30, 13, horizon=5)


def test_windowing_refuses_an_oversized_request_rather_than_dying(monkeypatch):
    """End to end through make_sequences, which is where it bites."""
    import pandas as pd

    from thalweg.config import FEATURE_COLUMNS, FeatureConfig
    from thalweg.state.features import make_sequences

    n = 5000
    frame = pd.DataFrame({c: np.linspace(1.0, 2.0, n) for c in FEATURE_COLUMNS})
    frame["segment"] = 0
    frame["timestamp"] = pd.date_range("2024-05-01", periods=n, freq="1s")
    frame["phase_seconds"] = np.arange(n) % 15

    monkeypatch.setenv("THALWEG_MEMORY_LIMIT_GB", "0.0001")
    with pytest.raises(MemoryError, match="--stride"):
        make_sequences(frame, config=FeatureConfig(lookback=30, horizon=5, stride=1),
                       feature_names=FEATURE_COLUMNS)
