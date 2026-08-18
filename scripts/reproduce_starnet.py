#!/usr/bin/env python3
"""Phase 1 gate: reproduce the StarNet throughput results.

The target, from CLAUDE.md section 6 (look-back 30, output 5):

    USA      RMSE 40.33  MAE 29.88
    Canada   RMSE 41.08  MAE 30.84
    Germany  RMSE 36.48  MAE 27.11
    Average  RMSE 39.30  MAE 29.28

plus median error 33.57 Mbps, and the two ablations: 38.00 without the
periodical embedding, 37.01 without attention.

If these do not land within a few percent, something is wrong with the data
loader and everything downstream is meaningless. Do not proceed past this
script until it passes, and do not report a published number as if we obtained
it.

Two known divergences from their released code, both recorded in docs/data.md
and both expected to cost us a little rather than gain us anything:

  - their loader fits its MinMaxScaler on the whole trace before splitting,
    which leaks the validation scale into training; we fit on training windows;
  - their main_pred.py defaults to hidden_size 60 while the paper text says
    128. We follow the paper. If the gap is small and stubborn, try --hidden 60.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flwcnx.config import (
    FEATURE_COLUMNS,
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    SplitConfig,
    StarNetConfig,
    resolve_device,
    seed_everything,
)
from flwcnx.eval.metrics import compute_metrics
from flwcnx.eval.runner import build_backbone, prepare
from flwcnx.forecast.train import inference_latency_ms, train_model
from flwcnx.ingest.replay import ReplaySource

PUBLISHED = {
    "usa": {"RMSE": 40.33, "MAE": 29.88},
    "canada": {"RMSE": 41.08, "MAE": 30.84},
    "germany": {"RMSE": 36.48, "MAE": 27.11},
}
PUBLISHED_MEDIAN_ERROR = 33.57
PUBLISHED_ABLATIONS = {"starnet_no_pe": 38.00, "starnet_no_attn": 37.01}


def run_one(location: str, args) -> dict:
    config = ExperimentConfig(
        name=f"reproduce-starnet-{location}",
        seed=args.seed,
        device=args.device,
        data=DataConfig(root=args.data, location=location),
        features=FeatureConfig(lookback=args.lookback, horizon=args.horizon,
                               stride=args.stride),
        model=StarNetConfig(epochs=args.epochs, hidden_size=args.hidden,
                            batch_size=args.batch_size),
        # StarNet split 8:2 into two contiguous blocks, not interleaved samples.
        split=SplitConfig(scheme="contiguous_82"),
    )
    seed_everything(config.seed)

    source = ReplaySource(config.data)
    verification = source.verify_against_published()
    if not verification["passed"]:
        print(f"  WARNING: loader does not match published statistics for {location}.")
        print("  " + json.dumps(verification["checks"], indent=2).replace("\n", "\n  "))
        print("  Investigate before believing anything below.")

    prepared = prepare(source, config)
    train, calibration, test = prepared["train"], prepared["calibration"], prepared["test"]
    print(f"  {prepared['split'].summary()}  leak_clean={prepared['leak']['clean']}")
    print(f"  phase reference {prepared['phase_reference'].offset_seconds:.2f} s "
          f"({prepared['phase_reference'].method}, "
          f"confidence {prepared['phase_reference'].confidence:.2f})")

    results: dict = {"location": location, "verification": verification,
                     "n_features": len(FEATURE_COLUMNS), "backbones": {}}

    backbones = ["starnet"] + (list(PUBLISHED_ABLATIONS) if args.ablations else [])
    for backbone in backbones:
        print(f"\n  --- {backbone} ---")
        model = build_backbone(backbone, train, config)
        forecaster, history = train_model(
            model, train, calibration, prepared["standardizer"],
            config=config.model, device=resolve_device(args.device), seed=config.seed,
        )
        predicted = forecaster.predict(test)
        metrics = compute_metrics(predicted.ravel(), test.y.ravel())
        median_error = float(np.median(np.abs(predicted.ravel() - test.y.ravel())))

        entry = {
            **metrics.to_dict(),
            "median_abs_error": median_error,
            "training": history.to_dict(),
            "inference_ms_per_batch": inference_latency_ms(
                forecaster.model, test, device=resolve_device(args.device)
            ),
        }
        if backbone == "starnet" and location in PUBLISHED:
            for key, published in PUBLISHED[location].items():
                entry[f"published_{key}"] = published
                entry[f"gap_{key}_pct"] = round(
                    100.0 * (entry[key] - published) / published, 2
                )
            entry["published_median_abs_error"] = PUBLISHED_MEDIAN_ERROR
        elif backbone in PUBLISHED_ABLATIONS:
            entry["published_median_abs_error"] = PUBLISHED_ABLATIONS[backbone]

        results["backbones"][backbone] = entry
        print(f"  RMSE {metrics.rmse:7.2f}  MAE {metrics.mae:7.2f}  "
              f"median |err| {median_error:6.2f}")
        if backbone == "starnet" and location in PUBLISHED:
            print(f"  published        RMSE {PUBLISHED[location]['RMSE']:.2f}  "
                  f"MAE {PUBLISHED[location]['MAE']:.2f}   "
                  f"gap {entry['gap_RMSE_pct']:+.1f}% / {entry['gap_MAE_pct']:+.1f}%")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/starnet"))
    parser.add_argument("--location", default="usa",
                        choices=["usa", "canada", "germany", "all"])
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128,
                        help="128 per the paper text; their code defaults to 60")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ablations", action="store_true",
                        help="also run the no-periodical-embedding and no-attention variants")
    parser.add_argument("--output", type=Path, default=Path("results/reproduce_starnet"))
    args = parser.parse_args(argv)

    locations = list(PUBLISHED) if args.location == "all" else [args.location]
    all_results = {}
    for location in locations:
        print(f"\n=== {location} ===")
        all_results[location] = run_one(location, args)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nwrote {path}")

    if len(locations) > 1:
        rmse = [all_results[loc]["backbones"]["starnet"]["RMSE"] for loc in locations]
        mae = [all_results[loc]["backbones"]["starnet"]["MAE"] for loc in locations]
        print(f"\naverage RMSE {np.mean(rmse):.2f} (published 39.30), "
              f"MAE {np.mean(mae):.2f} (published 29.28)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
