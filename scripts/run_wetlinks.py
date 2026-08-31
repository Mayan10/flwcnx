#!/usr/bin/env python3
"""Run the system on WetLinks.

Two releases, two different problems, one calibration layer.

  --release status   the 30 s dish status stream (the CSVs supplied with the
                     brief). Its `downlink` column is offered load for 91% of
                     samples, so the usable target is latency, the bound is an
                     upper bound, and the risk metric is UnderRate. The 30 s
                     grid aliases the 15 s scheduling phase to a constant, so
                     the regime axes are obstruction, hour and dish azimuth.

  --release seconds  the per-second iperf release. Real measured capacity at
                     1 Hz, so the bound is a lower bound and OverRate is the
                     risk, exactly as in the StarNet and BG-CFQS setting. The
                     phase axis is alive here, and satellite geometry can be
                     reconstructed from propagated elements with --geometry.

The hard constraint on the seconds release: iperf runs are exactly 15 samples,
so lookback + horizon must be at most 15. StarNet's 30/5 and BG-CFQS's 75/15
cannot be run on this data. Every number produced here must carry its sequence
length, and none of them is like-for-like against those papers' tables.
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
from flwcnx.ingest.wetlinks import SITE_COORDINATES, WetLinksConfig, WetLinksSource

STATUS_GRANULARITIES = ["global", "obstruction", "obstruction+hour", "full"]
SECONDS_GRANULARITIES = ["global", "phase", "candidates", "level",
                         "level+volatility", "level+phase", "level+candidates",
                         "full"]


def build_source(args, root: Path, seconds: bool):
    if not seconds:
        return WetLinksSource(WetLinksConfig(root=root, site=args.site,
                                             target=args.target))

    from flwcnx.ingest.wetlinks_full import (
        WetLinksFullConfig,
        WetLinksSecondsSource,
        attach_geometry,
        load_cached_elements,
    )

    site = args.site or "Osnabruck"
    source = WetLinksSecondsSource(WetLinksFullConfig(root=root, site=site))
    if args.geometry:
        frame = source.load_frame()
        key = "uos-rz" if site.lower().startswith("osnabr") else "utwente"
        latitude, longitude = SITE_COORDINATES[key]
        source._frame = attach_geometry(frame, load_cached_elements(),
                                        latitude=latitude, longitude=longitude)
        print("geometry: RECONSTRUCTED from propagated elements, not measured")
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default="status", choices=["status", "seconds"])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--site", default=None)
    parser.add_argument("--target", default="latency",
                        choices=["latency", "latency_pop", "capacity"])
    parser.add_argument("--geometry", action="store_true",
                        help="attach reconstructed candidate count and best-in-view "
                             "geometry (seconds release only)")
    parser.add_argument("--backbone", default="starnet")
    parser.add_argument("--methods", nargs="+", default=[
        "point", "global_conformal", "regime_conformal", "regime_bgcfqs",
        "adaptive_global_conformal", "adaptive_regime_conformal",
    ])
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    parser.add_argument("--granularities", nargs="+", default=None)
    parser.add_argument("--commitment", type=float, default=None,
                        help="Mbps on the seconds release, ms on the status release")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/wetlinks"))
    args = parser.parse_args(argv)

    seconds = args.release == "seconds"
    root = args.root or Path("data/wetlinks_full" if seconds else "data/supplied")
    lookback = args.lookback if args.lookback is not None else (10 if seconds else 30)
    if seconds and lookback + args.horizon > 15:
        parser.error(
            f"lookback {lookback} + horizon {args.horizon} = {lookback + args.horizon} "
            "exceeds the 15 sample iperf run. See docs/wetlinks-full.md."
        )

    target = "capacity" if seconds else args.target
    commitment = args.commitment if args.commitment is not None else (
        100.0 if seconds else 45.0
    )
    config = ExperimentConfig(
        name=f"wetlinks-{args.release}-{args.site or 'both'}-{target}",
        dataset="wetlinks_seconds" if seconds else "wetlinks",
        geometry=bool(args.geometry),
        seed=args.seed,
        device=args.device,
        features=FeatureConfig(lookback=lookback, horizon=args.horizon,
                               stride=args.stride, recover_phase=seconds),
        model=StarNetConfig(epochs=args.epochs),
        split=SplitConfig(scheme="temporal"),
        calibration=CalibrationConfig(epsilon=0.35, regime=RegimeConfig(min_samples=200)),
        decision=DecisionConfig(commitment_mbps=commitment),
    )

    source = build_source(args, root, seconds)
    print(json.dumps(source.describe(), indent=2))
    print(f"\nsequence: lookback {lookback}, horizon {args.horizon}, "
          f"direction {config.direction}, features {len(config.feature_columns)}"
          f"{' (with reconstructed geometry)' if args.geometry else ' (no geometry)'}")

    result = run_experiment(
        source, config, backbone=args.backbone,
        calibration_methods=tuple(args.methods),
        epsilons=tuple(args.epsilons),
        granularities=tuple(args.granularities or
                            (SECONDS_GRANULARITIES if seconds else STATUS_GRANULARITIES)),
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
