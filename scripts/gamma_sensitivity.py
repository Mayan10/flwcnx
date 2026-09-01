#!/usr/bin/env python3
"""Does the conditioning result depend on the ACI learning rate?

Raised by the independent novelty review (`docs/novelty-review.md`, final
section). The online layer carries one hyperparameter, `gamma`, and POGO's whole
motivating argument is that it cannot be tuned on a non-stationary stream.

Two separate questions, and only the first is settled by the git history:

1. **Was gamma tuned on calibration or test?** No. It was set to 0.02 in the
   commit that created `calibrate/adaptive.py`, with an a priori justification,
   and never changed, swept or overridden by any run. There is no validity
   problem.

2. **Would a different gamma change the conclusion?** Unknown until measured.
   The headline negative is that per-regime conditioning is worse than no
   conditioning on the StarNet traces. If that flips at some other gamma, the
   negative is a statement about one hyperparameter rather than about
   conditioning, and it has to be reported that way.

This script answers (2). The forecaster is trained once per location and reused
across every gamma, so the only thing varying is the online update rate.

    python scripts/gamma_sensitivity.py --locations usa canada
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from flwcnx.calibrate.adaptive import AdaptiveRegimeCalibrator
from flwcnx.config import (
    CalibrationConfig,
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    RegimeConfig,
    SplitConfig,
    StarNetConfig,
)
from flwcnx.eval.metrics import conditional_metrics
from flwcnx.eval.runner import build_backbone, prepare, resolve_device
from flwcnx.forecast.train import train_model
from flwcnx.ingest.replay import ReplaySource
from flwcnx.state.regime import RegimeAssigner

GAMMAS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
GRANULARITIES = ("global", "level", "full")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locations", nargs="+", default=["usa", "canada"])
    parser.add_argument("--gammas", nargs="+", type=float, default=list(GAMMAS))
    parser.add_argument("--granularities", nargs="+", default=list(GRANULARITIES))
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=Path("results/gamma_sensitivity"))
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for location in args.locations:
        config = ExperimentConfig(
            name=f"gamma-{location}", dataset="starnet", seed=args.seed,
            data=DataConfig(location=location),
            features=FeatureConfig(lookback=30, horizon=5, stride=args.stride,
                                   recover_phase=True),
            model=StarNetConfig(epochs=args.epochs),
            split=SplitConfig(scheme="temporal"),
            calibration=CalibrationConfig(epsilon=args.epsilon,
                                          regime=RegimeConfig(min_samples=200)),
        )
        prepared = prepare(ReplaySource(config.data), config)
        train, calibration, test = (prepared["train"], prepared["calibration"],
                                    prepared["test"])
        model = build_backbone("starnet", train, config)
        forecaster, _ = train_model(model, train, calibration, prepared["standardizer"],
                                    config=config.model, device=resolve_device("auto"),
                                    seed=args.seed, verbose=False)
        predicted_cal = forecaster.predict_horizon_mean(calibration)
        predicted_test = forecaster.predict_horizon_mean(test)
        actual_cal, actual_test = calibration.y.mean(axis=1), test.y.mean(axis=1)
        print(f"\n=== {location} === train {len(train)} cal {len(calibration)} "
              f"test {len(test)}")

        from flwcnx.state.regime import presets_for
        presets = presets_for(config.dataset)
        for granularity in args.granularities:
            axes = presets[granularity]
            calibration_config = replace(
                config.calibration, epsilon=args.epsilon,
                regime=replace(config.calibration.regime, axes=axes),
            )
            assigner = (RegimeAssigner(calibration_config.regime).fit(train.regime)
                        if axes else None)
            for gamma in args.gammas:
                online = AdaptiveRegimeCalibrator(config=calibration_config,
                                                  assigner=assigner, direction="lower",
                                                  gamma=gamma)
                online.fit(predicted_cal, actual_cal, calibration.regime)
                bound = online.transform_online(predicted_test, actual_test, test.regime)
                m = conditional_metrics(bound, actual_test, direction="lower")
                row = {"location": location, "granularity": granularity,
                       "axes": "+".join(axes) or "global", "gamma": gamma,
                       "OverRate": round(m["global"].over_rate, 4),
                       "P30": round(m["P30"].over_rate, 4),
                       "P10": round(m["P10"].over_rate, 4),
                       "MAE": round(m["global"].mae, 3)}
                rows.append(row)
                print(f"  {granularity:8s} gamma={gamma:<6.3f} "
                      f"OverRate={row['OverRate']:.4f} P10={row['P10']:.4f} "
                      f"MAE={row['MAE']:.2f}")

    # The question is whether conditioning ever beats global, at any gamma.
    print("\n=== does conditioning ever win, at any gamma? ===")
    verdict = []
    for location in args.locations:
        for granularity in args.granularities:
            if granularity == "global":
                continue
            for gamma in args.gammas:
                g = next(r for r in rows if r["location"] == location
                         and r["granularity"] == "global" and r["gamma"] == gamma)
                c = next(r for r in rows if r["location"] == location
                         and r["granularity"] == granularity and r["gamma"] == gamma)
                delta = 100 * (c["P10"] - g["P10"]) / g["P10"]
                verdict.append({"location": location, "granularity": granularity,
                                "gamma": gamma, "p10_delta_pct": round(delta, 2)})
    wins = [v for v in verdict if v["p10_delta_pct"] < 0]
    for v in verdict:
        mark = "  WIN" if v["p10_delta_pct"] < 0 else ""
        print(f"  {v['location']:8s} {v['granularity']:8s} gamma={v['gamma']:<6.3f} "
              f"P10 delta {v['p10_delta_pct']:+6.2f}%{mark}")
    print(f"\nconditioning beat global in {len(wins)} of {len(verdict)} "
          f"(location, granularity, gamma) cells")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict,
                    "n_wins": len(wins), "n_cells": len(verdict)},
                   indent=2, default=str))
    print(f"wrote {args.output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
