"""Regression tests for the datetime-resolution bug.

This bug passed 176 local tests and failed 13 on CI. The cause was that pandas 2
carries datetime64 at second, millisecond, microsecond or nanosecond resolution,
and `Series.astype("int64")` returns the count in whatever unit the column
happens to have. Local runs produced nanoseconds and the runner produced
something else, so the same code computed the scheduling phase correctly on one
machine and not the other.

These tests pin the behaviour at every resolution so it cannot come back
quietly. See `thalweg/timeutil.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thalweg.ingest.synthetic import SyntheticSpec, generate_frame
from thalweg.state.phase import recover_phase, sampling_aliases_period
from thalweg.timeutil import to_epoch_nanoseconds, to_epoch_seconds

UNITS = ["s", "ms", "us", "ns"]


@pytest.mark.parametrize("unit", UNITS)
def test_epoch_seconds_is_identical_at_every_resolution(unit):
    index = pd.date_range("2024-05-01", periods=500, freq="1s")
    reference = to_epoch_seconds(pd.Series(index.as_unit("ns")))
    got = to_epoch_seconds(pd.Series(index.as_unit(unit)))
    np.testing.assert_allclose(got, reference)


@pytest.mark.parametrize("unit", UNITS)
def test_epoch_nanoseconds_is_identical_at_every_resolution(unit):
    index = pd.date_range("2024-05-01", periods=500, freq="30s")
    reference = to_epoch_nanoseconds(pd.Series(index.as_unit("ns")))
    got = to_epoch_nanoseconds(pd.Series(index.as_unit(unit)))
    np.testing.assert_array_equal(got, reference)


def test_the_naive_conversion_is_wrong_and_this_is_why_the_helper_exists():
    """Pins the bug itself, so the helper's reason for existing stays visible."""
    index = pd.date_range("2024-05-01", periods=100, freq="1s")
    naive_us = pd.Series(index.as_unit("us")).astype("int64").to_numpy() / 1e9
    correct = to_epoch_seconds(pd.Series(index.as_unit("us")))
    # Microsecond counts divided by 1e9 come out a thousand times too small.
    assert not np.allclose(naive_us, correct)
    np.testing.assert_allclose(naive_us * 1000.0, correct)


@pytest.mark.parametrize("unit", UNITS)
def test_aliasing_guard_answers_the_same_at_every_resolution(unit):
    """The guard inverted its answer on CI. It must not depend on the unit."""
    coarse = pd.Series(pd.date_range("2024-05-01", periods=400, freq="30s").as_unit(unit))
    dense = pd.Series(pd.date_range("2024-05-01", periods=2000, freq="1s").as_unit(unit))
    # 30 s against a 15 s period puts every sample at the same phase.
    assert sampling_aliases_period(coarse)
    assert not sampling_aliases_period(dense)


@pytest.mark.parametrize("unit", UNITS)
def test_phase_recovery_survives_every_resolution(unit):
    """The end-to-end symptom: edge detection found one edge instead of hundreds."""
    frame = generate_frame(SyntheticSpec(n_seconds=9000, phase_offset=12, seed=12))
    frame["timestamp"] = frame["timestamp"].dt.as_unit(unit)
    reference = recover_phase(frame)
    assert reference.is_recovered, f"fell back to {reference.method} at unit {unit}"
    error = abs((reference.offset_seconds - 12 + 7.5) % 15 - 7.5)
    assert error < 1.0
