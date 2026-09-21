"""Shared fixtures. Everything here is synthetic and none of it is a result."""

from __future__ import annotations

import pytest

from thalweg.config import FEATURE_COLUMNS, FeatureConfig
from thalweg.ingest.synthetic import SyntheticSpec, generate_frame
from thalweg.state.features import (
    Standardizer,
    build_features,
    make_sequences,
    standardize_sequences,
)


@pytest.fixture(scope="session")
def raw_frame():
    return generate_frame(SyntheticSpec(n_seconds=6000, phase_offset=12, seed=11))


@pytest.fixture(scope="session")
def feature_frame(raw_frame):
    frame, reference, encoder = build_features(raw_frame)
    return frame, reference, encoder


@pytest.fixture(scope="session")
def raw_sequences(feature_frame):
    """Unstandardised windows, which is what the runner splits before scaling."""
    frame, _, _ = feature_frame
    return make_sequences(frame, config=FeatureConfig(lookback=30, horizon=5))


@pytest.fixture(scope="session")
def sequences(raw_sequences):
    """Scaled the way the runner scales: standardiser fit on windows, target
    left in Mbps."""
    standardizer = Standardizer().fit_windows(raw_sequences.x, FEATURE_COLUMNS)
    return standardize_sequences(raw_sequences, standardizer)


@pytest.fixture(scope="session")
def fitted_standardizer(raw_sequences):
    return Standardizer().fit_windows(raw_sequences.x, FEATURE_COLUMNS)
