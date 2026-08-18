#!/usr/bin/env python3
"""Run the system on the supplied WetLinks data.

Target is latency, because `downlink` in this dataset is offered load for 90.7%
of samples and only the periodic speedtest bursts measure capacity
(docs/supplied-dataset.md). The bound is therefore an upper bound and the risk
metric is UnderRate: the fraction of slots where we promised a delay the link
did not meet.

The regime axes are the ones this dataset actually has. Phase is aliased dead
by the 30 s grid and there is no satellite geometry, so conditioning is on
obstruction, hour of day and dish azimuth sector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flwcnx.config import (
    CalibrationConfig,
    DecisionConfig,
    ExperimentConfig,
    FeatureConfig,
    RegimeConfig,
    SplitConfig,
    StarNetConfig,
)
from flwcnx.eval.runner import comparison_table, risk_table, run_experiment
from flwcnx.ingest.wetlinks import WetLinksConfig, WetLinksSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/supplied"))
    parser.add_argument("--site", default=None, help="uos-rz or utwente; default both")
    parser.add_argument("--target", default="latency",
                        choices=["latency", "latency_pop", "capacity"])
    parser.add_argument("--backbone", default="starnet")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    parser.add_argument("--granularities", nargs="+",
                        default=["global", "obstruction", "obstruction+hour", "full"])
    parser.add_argument("--commitment-ms", type=float, default=45.0,
                        help="delay budget for the congestion rule")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/wetlinks"))
    args = parser.parse_args(argv)

    config = ExperimentConfig(
        name=f"wetlinks-{args.site or 'both'}-{args.target}",
        dataset="wetlinks",
        seed=args.seed,
        device=args.device,
        features=FeatureConfig(lookback=args.lookback, horizon=args.horizon,
                               stride=args.stride, recover_phase=False),
        model=StarNetConfig(epochs=args.epochs),
        split=SplitConfig(scheme="temporal"),
        calibration=CalibrationConfig(epsilon=0.35,
                                      regime=RegimeConfig(min_samples=200)),
        decision=DecisionConfig(commitment_mbps=args.commitment_ms),
    )
    source = WetLinksSource(WetLinksConfig(root=args.root, site=args.site,
                                           target=args.target))
    print(json.dumps(source.describe(), indent=2))

    result = run_experiment(
        source, config, backbone=args.backbone,
        calibration_methods=("point", "global_conformal", "regime_conformal",
                             "regime_bgcfqs"),
        epsilons=tuple(args.epsilons), granularities=tuple(args.granularities),
    )
    written = result.save(args.output / config.name)

    print("\n=== risk table ===")
    print(risk_table(result).round(4).to_string(index=False))
    print("\n=== comparison ===")
    print(comparison_table(result).round(4).to_string(index=False))
    print(f"\nwrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
