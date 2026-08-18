"""Experiment grid.

One entry point that walks every layer in order and writes results with a
config snapshot beside them, so any figure can be traced back to the exact
settings that produced it (CLAUDE.md section 11).

What a single run does:

    ingest -> features -> sequences -> split -> fit forecaster on train
           -> calibrate on calibration -> evaluate on test
           -> admission control and congestion on the same test bounds

Everything downstream of the split reads the split, so the calibration set is
never fit on and the test set is never touched until the end. `check_no_leak`
runs on every experiment rather than on request.

The grid covers what CLAUDE.md section 8 asks for: the epsilon sweep from 0.05
to 0.35, the regime granularity ablation, and every calibration method against
the same point forecaster so that a difference between them is a difference in
calibration and nothing else.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from flwcnx.calibrate.conformal import fit_conformal
from flwcnx.calibrate.regime_cal import RegimeCalibrator, global_calibrator
from flwcnx.config import (
    FEATURE_COLUMNS,
    CalibrationConfig,
    ExperimentConfig,
    RegimeConfig,
    resolve_device,
    seed_everything,
)
from flwcnx.decide.admission import evaluate_admission
from flwcnx.decide.congestion import detect_congestion
from flwcnx.eval.metrics import (
    RISK_SLICES,
    conditional_metrics,
    per_regime_metrics,
    summarise,
    worst_regime_over_rate,
)
from flwcnx.eval.splits import check_no_leak, contiguous_82, temporal_split
from flwcnx.forecast.baselines import build_baseline
from flwcnx.forecast.starnet import StarNet, StarNetNoAttention, StarNetNoPeriodicalEmbedding
from flwcnx.forecast.train import train_model
from flwcnx.ingest.base import Source
from flwcnx.ingest.replay import ReplaySource
from flwcnx.state.features import (
    Standardizer,
    build_features,
    make_sequences,
    standardize_sequences,
)
from flwcnx.state.regime import RegimeAssigner, regime_granularity_presets

BACKBONES = ("starnet", "starnet_no_pe", "starnet_no_attn", "dlinear", "patchtst", "timesnet")
EPSILON_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)


@dataclass
class RunResult:
    """Everything one run produced, in a serialisable shape."""

    name: str
    config: dict
    split_summary: dict
    leak_check: dict
    point_metrics: dict
    calibration: dict = field(default_factory=dict)
    admission: dict = field(default_factory=dict)
    congestion: dict = field(default_factory=dict)
    regime_tables: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in asdict(self).items() if k != "regime_tables"}
        (directory / "result.json").write_text(json.dumps(payload, indent=2, default=str))
        (directory / "config.json").write_text(json.dumps(self.config, indent=2, default=str))
        for name, table in self.regime_tables.items():
            if isinstance(table, pd.DataFrame) and not table.empty:
                table.to_csv(directory / f"{name}.csv", index=False)
        return directory


def _environment() -> dict:
    import torch

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def build_backbone(name: str, sequences, config: ExperimentConfig):
    """Instantiate a forecaster by name against a sequence set's shape."""
    name = name.lower()
    horizon = config.features.horizon
    if name == "starnet":
        return StarNet(sequences.class_slices(), horizon, config.model)
    if name == "starnet_no_pe":
        return StarNetNoPeriodicalEmbedding(sequences.class_slices(), horizon, config.model)
    if name == "starnet_no_attn":
        return StarNetNoAttention(sequences.class_slices(), horizon, config.model)
    return build_baseline(
        name, lookback=config.features.lookback, horizon=horizon,
        n_features=len(sequences.feature_names), target_index=sequences.target_index,
    )


def prepare(source: Source, config: ExperimentConfig):
    """Ingest through to split sequence sets, with the standardiser fit on train.

    Sequences are built unstandardised, split, and only then scaled, because
    the standardiser must see training windows only. Building scaled sequences
    first and splitting afterwards is the easy version of this function and it
    leaks the test distribution's scale into training.
    """
    frame = source.load_frame()
    features, phase_reference, encoder = build_features(frame, config=config.features)
    raw = make_sequences(features, config=config.features)

    if config.split.scheme == "contiguous_82":
        split = contiguous_82(raw, lookback=config.features.lookback,
                              horizon=config.features.horizon, stride=config.features.stride)
    else:
        split = temporal_split(raw, config.split, lookback=config.features.lookback,
                               horizon=config.features.horizon, stride=config.features.stride)

    leak = check_no_leak(raw, split, lookback=config.features.lookback,
                         horizon=config.features.horizon)

    standardizer = Standardizer()
    if config.features.standardize:
        standardizer.fit_windows(raw.x[split.train], FEATURE_COLUMNS)
        scaled = standardize_sequences(raw, standardizer)
    else:
        standardizer.fit_windows(np.zeros((1, 1, raw.x.shape[-1])), FEATURE_COLUMNS)
        standardizer.mean = np.zeros(raw.x.shape[-1])
        standardizer.std = np.ones(raw.x.shape[-1])
        scaled = raw

    train, calibration, test = split.apply(scaled)
    return {
        "train": train, "calibration": calibration, "test": test,
        "split": split, "leak": leak, "standardizer": standardizer,
        "phase_reference": phase_reference, "encoder": encoder, "raw": raw,
    }


def run_experiment(
    source: Source,
    config: ExperimentConfig,
    *,
    backbone: str = "starnet",
    calibration_methods: tuple[str, ...] = ("global_conformal", "regime_conformal",
                                            "regime_bgcfqs"),
    epsilons: tuple[float, ...] | None = None,
    granularities: tuple[str, ...] = ("full",),
    verbose: bool = True,
) -> RunResult:
    """One data source, one backbone, and a sweep over the calibration axis.

    The point forecaster is trained once and reused across every calibration
    setting. That is not just a saving: it means every calibration number in
    the result differs only in calibration.
    """
    seed_everything(config.seed)
    device = resolve_device(config.device)
    epsilons = epsilons or (config.calibration.epsilon,)

    prepared = prepare(source, config)
    train, calibration, test = prepared["train"], prepared["calibration"], prepared["test"]
    if verbose:
        print(f"[{config.name}] {prepared['split'].summary()}  leak_clean="
              f"{prepared['leak']['clean']}  device={device}")

    model = build_backbone(backbone, train, config)
    forecaster, history = train_model(
        model, train, calibration, prepared["standardizer"],
        config=config.model, device=device, seed=config.seed, verbose=verbose,
    )

    # One number per window: the horizon mean is what an allocation decision
    # acts on over the next H seconds.
    predicted_cal = forecaster.predict_horizon_mean(calibration)
    predicted_test = forecaster.predict_horizon_mean(test)
    actual_cal = calibration.y.mean(axis=1)
    actual_test = test.y.mean(axis=1)

    point = summarise(conditional_metrics(predicted_test, actual_test)).to_dict(orient="records")

    result = RunResult(
        name=config.name,
        config=config.to_dict(),
        split_summary=prepared["split"].summary(),
        leak_check=prepared["leak"],
        point_metrics={"backbone": backbone, "slices": point},
        training=history.to_dict(),
        environment=_environment(),
    )
    result.config["phase_reference"] = {
        "offset_seconds": prepared["phase_reference"].offset_seconds,
        "method": prepared["phase_reference"].method,
        "confidence": prepared["phase_reference"].confidence,
    }

    presets = regime_granularity_presets()
    for epsilon in epsilons:
        for granularity in granularities:
            axes = presets[granularity]
            calibration_config = CalibrationConfig(
                epsilon=epsilon,
                regime=replace(config.calibration.regime, axes=axes),
            )
            for method in calibration_methods:
                key = f"{method}|eps={epsilon:.2f}|regime={granularity}"
                result.calibration[key] = _calibrate_and_score(
                    method, calibration_config, predicted_cal, actual_cal,
                    predicted_test, actual_test, calibration, test, train,
                    config, result, key,
                )
                if verbose:
                    entry = result.calibration[key]
                    print(f"  {key:52s} OverRate glob={entry['global']['OverRate']:.3f} "
                          f"P30={entry['P30']['OverRate']:.3f} P10={entry['P10']['OverRate']:.3f} "
                          f"MAE={entry['global']['MAE']:.2f}")
    return result


def _calibrate_and_score(method: str, calibration_config: CalibrationConfig,
                         predicted_cal, actual_cal, predicted_test, actual_test,
                         calibration_set, test_set, train_set,
                         config: ExperimentConfig, result: RunResult, key: str) -> dict:
    """Fit one calibration method and score its bounds all the way downstream."""
    if method == "global_conformal":
        bound = fit_conformal(predicted_cal, actual_cal, calibration_config.epsilon)
        lower = bound.apply(predicted_test)
        detail = {"offset_mbps": round(bound.offset, 4), "rank": bound.rank,
                  "degenerate": bound.degenerate}
        calibrator = None
    elif method in ("regime_conformal", "regime_bgcfqs"):
        selector = "conformal" if method == "regime_conformal" else "bgcfqs"
        if calibration_config.regime.axes:
            assigner = RegimeAssigner(calibration_config.regime).fit(train_set.regime)
            calibrator = RegimeCalibrator(config=calibration_config, assigner=assigner,
                                          selector=selector)
        else:
            calibrator = global_calibrator(calibration_config, selector)
        calibrator.fit(predicted_cal, actual_cal, calibration_set.regime)
        lower = calibrator.transform(predicted_test, test_set.regime)
        detail = calibrator.summary()
    elif method == "point":
        # The uncalibrated forecaster, so the risk metrics have a floor to be
        # measured against. StarNet-point in the BG-CFQS table is this.
        lower = predicted_test
        detail = {"note": "uncalibrated point forecast"}
        calibrator = None
    else:
        raise ValueError(f"unknown calibration method {method!r}")

    metrics = conditional_metrics(lower, actual_test)
    entry: dict = {"method": method, "epsilon": calibration_config.epsilon,
                   "axes": list(calibration_config.regime.axes), "detail": detail}
    for name, metric in metrics.items():
        entry[name] = metric.to_dict()
        entry[name]["risk_pass"] = metric.risk_pass(calibration_config.epsilon)

    # Per regime breakdown, always at the full granularity so that different
    # calibration granularities are judged against the same partition.
    full_assigner = RegimeAssigner(RegimeConfig()).fit(train_set.regime)
    labels = full_assigner.assign(test_set.regime)
    entry["worst_regime_over_rate"] = worst_regime_over_rate(lower, actual_test, labels)
    result.regime_tables[f"regime_{key.replace('|', '_').replace('=', '')}"] = (
        per_regime_metrics(lower, actual_test, labels, min_samples=30)
    )

    admission = evaluate_admission(lower, actual_test,
                                   config.decision.bandwidth_per_session_mbps)
    result.admission[key] = admission.to_dict(orient="records")

    congestion = detect_congestion(lower, actual_test,
                                   config.decision.commitment_mbps,
                                   config.decision.congestion_window)
    result.congestion[key] = {**congestion.scores(), **congestion.confusion()}
    return entry


def run_grid(source: Source, config: ExperimentConfig, output: str | Path,
             *, backbones: tuple[str, ...] = ("starnet",),
             epsilons: tuple[float, ...] = EPSILON_SWEEP,
             granularities: tuple[str, ...] = ("global", "phase", "phase+elevation", "full"),
             verbose: bool = True) -> list[Path]:
    """The full grid from CLAUDE.md section 8, one directory per backbone."""
    output = Path(output)
    written = []
    for backbone in backbones:
        run_config = replace(config, name=f"{config.name}-{backbone}")
        result = run_experiment(
            source, run_config, backbone=backbone,
            calibration_methods=("point", "global_conformal", "regime_conformal",
                                 "regime_bgcfqs"),
            epsilons=epsilons, granularities=granularities, verbose=verbose,
        )
        written.append(result.save(output / run_config.name))
    return written


def comparison_table(result: RunResult, slice_name: str = "global") -> pd.DataFrame:
    """Flatten a run's calibration results into one reportable table."""
    rows = []
    for key, entry in result.calibration.items():
        metrics = entry.get(slice_name, {})
        rows.append({
            "setting": key,
            "method": entry["method"],
            "epsilon": entry["epsilon"],
            "axes": "+".join(entry["axes"]) or "global",
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "OverRate": metrics.get("OverRate"),
            "MPE": metrics.get("MPE"),
            "P95+Err": metrics.get("P95+Err"),
            "risk_pass": metrics.get("risk_pass"),
            "worst_regime_OverRate": entry.get("worst_regime_over_rate"),
        })
    return pd.DataFrame(rows)


def risk_table(result: RunResult) -> pd.DataFrame:
    """The motivation table: OverRate globally and on each risk slice."""
    rows = []
    for key, entry in result.calibration.items():
        row = {"setting": key, "method": entry["method"], "epsilon": entry["epsilon"]}
        for name in ("global", *RISK_SLICES):
            row[f"OverRate_{name}"] = entry.get(name, {}).get("OverRate")
        row["MAE"] = entry.get("global", {}).get("MAE")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the flwcnx experiment grid.")
    parser.add_argument("--data", type=Path, default=Path("data/starnet"),
                        help="root directory holding the StarNet traces")
    parser.add_argument("--location", default="usa", choices=["usa", "canada", "germany"])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--backbones", nargs="+", default=["starnet"], choices=list(BACKBONES))
    parser.add_argument("--epsilons", nargs="+", type=float, default=list(EPSILON_SWEEP))
    parser.add_argument("--granularities", nargs="+",
                        default=["global", "phase", "phase+elevation", "full"])
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="temporal", choices=["temporal", "contiguous_82"])
    parser.add_argument("--synthetic", action="store_true",
                        help="run on generated data, for smoke testing only")
    args = parser.parse_args(argv)

    from flwcnx.config import DataConfig, FeatureConfig, SplitConfig, StarNetConfig

    config = ExperimentConfig(
        name=f"{args.location}",
        seed=args.seed,
        device=args.device,
        data=DataConfig(root=args.data, location=args.location),
        features=FeatureConfig(lookback=args.lookback, horizon=args.horizon),
        model=StarNetConfig(epochs=args.epochs),
        split=SplitConfig(scheme=args.split),
    )

    if args.synthetic:
        from flwcnx.ingest.synthetic import SyntheticSource, SyntheticSpec

        print("WARNING: synthetic data. Smoke test only, never a reported result.")
        source: Source = SyntheticSource(SyntheticSpec(n_seconds=20000))
    else:
        source = ReplaySource(config.data)

    written = run_grid(source, config, args.output, backbones=tuple(args.backbones),
                       epsilons=tuple(args.epsilons), granularities=tuple(args.granularities))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
