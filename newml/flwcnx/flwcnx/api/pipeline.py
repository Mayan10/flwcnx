"""Build a ready-to-stream `DemoEngine` for the API server.

This is `scripts/run_demo.py` with the disk output taken off, split into the
part that is expensive and done once (ingest, train, seed the calibrator) and
the part that is cheap and done per client (a fresh engine over the test
split).

Two sources feed it:

  synthetic   `ingest.synthetic.SyntheticSource`. The generated trace carries
              the 15 s period, the handover dip and the geometry effects, so
              the forecaster, regime assigner and calibrator all run for real.
              The *data* is fake; nothing downstream of it is.
  replay      `ingest.replay.ReplaySource` over `data/starnet/<location>`.

As everywhere else in this repository, synthetic output is for demos and
development and is never a result.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from flwcnx.calibrate.adaptive import AdaptiveRegimeCalibrator
from flwcnx.config import (
    CalibrationConfig,
    DataConfig,
    DecisionConfig,
    ExperimentConfig,
    FeatureConfig,
    RegimeConfig,
    SplitConfig,
    StarNetConfig,
)
from flwcnx.demo import DecisionRecord, DemoEngine
from flwcnx.device import resolve_device
from flwcnx.eval.runner import build_backbone, prepare
from flwcnx.forecast.train import train_model
from flwcnx.ingest.base import Source
from flwcnx.state.regime import RegimeAssigner, presets_for

logger = logging.getLogger("flwcnx.api")


@dataclass(frozen=True)
class PipelineSpec:
    mode: str = "synthetic"          # synthetic | real
    location: str = "canada"
    epsilon: float = 0.35
    granularity: str = "level"
    stride: int = 2
    epochs: int = 5
    commitment_mbps: float = 150.0
    synthetic_seconds: int = 6000
    seed: int = 1337


@dataclass
class TrainedPipeline:
    """Everything a stream needs, computed once and shared across clients."""

    spec: PipelineSpec
    config: ExperimentConfig
    assigner: RegimeAssigner | None
    predicted_cal: np.ndarray
    actual_cal: np.ndarray
    regime_cal: pd.DataFrame
    predicted_test: np.ndarray
    actual_test: np.ndarray
    regime_test: pd.DataFrame
    feature_set: str

    def new_engine(self) -> DemoEngine:
        """A fresh calibrator and engine, so every client starts from the same
        seeded state instead of inheriting another client's online updates."""
        calibrator = AdaptiveRegimeCalibrator(config=self.config.calibration,
                                              assigner=self.assigner,
                                              direction=self.config.direction)
        calibrator.fit(self.predicted_cal, self.actual_cal, self.regime_cal)
        return DemoEngine(calibrator=calibrator, decision=self.config.decision,
                          assigner=self.assigner, config=self.config.calibration)

    def init_message(self) -> dict:
        return {
            "type": "init",
            "mode": self.spec.mode,
            "location": self.spec.location,
            "epsilon": self.spec.epsilon,
            "direction": self.config.direction,
            "feature_set": self.feature_set,
            "n_test_samples": int(len(self.predicted_test)),
        }

    def stream(self, steps: int = 0) -> Iterator[DecisionRecord]:
        """Replay the test split through one engine.

        `steps == 0` means forever: when the split runs out it is replayed
        again through the *same* engine, so the calibrator keeps learning and
        the realised risk rate stays one running number rather than resetting
        every lap. Timestamps are wall clock at 1 s per decision, because the
        synthetic trace is dated 2024 and a live monitor showing that is
        confusing.
        """
        n = len(self.predicted_test)
        engine = self.new_engine()
        start = datetime.now(UTC)
        step = 0
        while not steps or step < steps:
            t = step % n
            yield engine.step(
                step, (start + timedelta(seconds=step)).isoformat(),
                float(self.predicted_test[t]), float(self.actual_test[t]),
                self.regime_test.iloc[[t]].reset_index(drop=True))
            step += 1


def _source(spec: PipelineSpec, data: DataConfig) -> Source:
    if spec.mode == "synthetic":
        from flwcnx.ingest.synthetic import SyntheticSource, SyntheticSpec

        return SyntheticSource(SyntheticSpec(n_seconds=spec.synthetic_seconds,
                                             seed=spec.seed),
                               location=spec.location)
    from flwcnx.ingest.replay import ReplaySource

    return ReplaySource(data)


def build_pipeline(spec: PipelineSpec) -> TrainedPipeline:
    """Ingest, train the forecaster and seed the calibrator. Blocking."""
    axes = presets_for("starnet")[spec.granularity]
    config = ExperimentConfig(
        name=f"api-{spec.mode}-{spec.location}", dataset="starnet", seed=spec.seed,
        data=DataConfig(location=spec.location),
        features=FeatureConfig(lookback=30, horizon=5, stride=spec.stride,
                               recover_phase=True),
        model=StarNetConfig(epochs=spec.epochs),
        split=SplitConfig(scheme="temporal"),
        calibration=CalibrationConfig(epsilon=spec.epsilon,
                                      regime=RegimeConfig(axes=axes, min_samples=200)),
        decision=DecisionConfig(commitment_mbps=spec.commitment_mbps),
    )

    logger.info("[%s/%s] preparing, regime axes = %s", spec.mode, spec.location,
                axes or ("global",))
    prepared = prepare(_source(spec, config.data), config)
    train, calibration, test = prepared["train"], prepared["calibration"], prepared["test"]
    logger.info("[%s/%s] windows: train=%d calibration=%d test=%d", spec.mode,
                spec.location, len(train.x), len(calibration.x), len(test.x))

    device = resolve_device("auto")
    logger.info("[%s/%s] training StarNet for up to %d epochs on %s", spec.mode,
                spec.location, spec.epochs, device)
    model = build_backbone("starnet", train, config)
    forecaster, _ = train_model(model, train, calibration, prepared["standardizer"],
                                config=config.model, device=device,
                                seed=spec.seed, verbose=False)

    assigner = RegimeAssigner(config.calibration.regime).fit(train.regime) if axes else None
    pipeline = TrainedPipeline(
        spec=spec, config=config, assigner=assigner,
        predicted_cal=forecaster.predict_horizon_mean(calibration),
        actual_cal=calibration.y.mean(axis=1),
        regime_cal=calibration.regime,
        predicted_test=forecaster.predict_horizon_mean(test),
        actual_test=test.y.mean(axis=1),
        regime_test=test.regime.reset_index(drop=True),
        feature_set=",".join(config.feature_columns),
    )
    logger.info("[%s/%s] ready, %d test decisions per replay", spec.mode,
                spec.location, len(pipeline.predicted_test))
    return pipeline
