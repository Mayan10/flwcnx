#!/usr/bin/env python3
"""Phase 2 gate: reproduce BG-CFQS and expose the conditional risk gap.

Two things have to come out of this script, and the second one matters more.

First, the published average table (Xie et al., over CHI, OSN, VIC):

    T3P-point      MAE 37.775  RMSE 49.699  OverRate 0.512  MPE 20.182  P95+Err 87.984  0/3
    StarNet-point  MAE 39.806  RMSE 52.102  OverRate 0.433  MPE 17.030  P95+Err 82.842  0/3
    BG-CFQS        MAE 40.364  RMSE 52.480  OverRate 0.349  MPE 11.745  P95+Err 65.834  3/3

with selected quantiles CHI 0.314, OSN 0.244, VIC 0.306.

Second, and this is the motivation figure for the whole project, the
conditional OverRate on the low throughput subsets:

    Dataset  High-risk P30   Severe-risk P10
    CHI      0.65            0.83
    OSN      0.71            0.86
    VIC      0.71            0.86

Risk held on average and lost precisely where capacity is lowest. Commit the
figure as soon as it exists.

Configuration follows the paper: L = 75, H = 15, epsilon = 0.35, candidate set
[0.15, 0.40], coarse delta 0.05, fine grid M = 5, XGBoost with pinball loss,
and the calibration split used only for quantile selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from flwcnx.calibrate.bgcfqs import BGCFQS
from flwcnx.config import (
    BGCFQS_ALIASES,
    BGCFQSConfig,
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    SplitConfig,
    seed_everything,
)
from flwcnx.eval.figures import conditional_over_rate
from flwcnx.eval.metrics import conditional_metrics
from flwcnx.eval.runner import prepare
from flwcnx.forecast.baselines import XGBoostForecaster
from flwcnx.ingest.replay import ReplaySource
from flwcnx.state.features import flatten_for_tabular

# The date ranges BG-CFQS process, so the comparison is apples to apples.
BGCFQS_RANGES = {
    "usa": ("2024-04-26", "2024-05-28"),
    "germany": ("2024-07-13", "2024-07-31"),
    "canada": ("2024-07-11", "2024-07-28"),
}
PUBLISHED_QUANTILES = {"usa": 0.314, "germany": 0.244, "canada": 0.306}
PUBLISHED_CONDITIONAL = {
    "usa": {"P30": 0.65, "P10": 0.83},
    "germany": {"P30": 0.71, "P10": 0.86},
    "canada": {"P30": 0.71, "P10": 0.86},
}
PUBLISHED_AVERAGES = {
    "T3P-point": {"MAE": 37.775, "RMSE": 49.699, "OverRate": 0.512,
                  "MPE": 20.182, "P95+Err": 87.984},
    "StarNet-point": {"MAE": 39.806, "RMSE": 52.102, "OverRate": 0.433,
                      "MPE": 17.030, "P95+Err": 82.842},
    "BG-CFQS": {"MAE": 40.364, "RMSE": 52.480, "OverRate": 0.349,
                "MPE": 11.745, "P95+Err": 65.834},
}


def run_one(location: str, args) -> dict:
    start, end = BGCFQS_RANGES[location]
    config = ExperimentConfig(
        name=f"reproduce-bgcfqs-{location}",
        seed=args.seed,
        data=DataConfig(root=args.data, location=location,
                        date_start=start, date_end=end),
        # BG-CFQS use history 75 and horizon 15, not StarNet's 30/5.
        features=FeatureConfig(lookback=args.lookback, horizon=args.horizon,
                               stride=args.stride),
        split=SplitConfig(scheme="temporal"),
    )
    seed_everything(config.seed)

    prepared = prepare(ReplaySource(config.data), config)
    train, calibration, test = prepared["train"], prepared["calibration"], prepared["test"]
    print(f"  {prepared['split'].summary()}  leak_clean={prepared['leak']['clean']}")

    x_train, y_train = flatten_for_tabular(train)
    x_cal, y_cal = flatten_for_tabular(calibration)
    x_test, y_test = flatten_for_tabular(test)

    bgcfqs_config = BGCFQSConfig(lookback=args.lookback, horizon=args.horizon,
                                 epsilon=args.epsilon)
    model = BGCFQS(bgcfqs_config, seed=config.seed).fit(x_train, y_train)
    tau = model.select(x_cal, y_cal, args.epsilon)
    lower = model.predict(x_test)

    # The point baseline: the same backbone with a squared error objective, so
    # the difference against BG-CFQS is the quantile selection and nothing else.
    point = XGBoostForecaster(
        n_estimators=bgcfqs_config.n_estimators, max_depth=bgcfqs_config.max_depth,
        learning_rate=bgcfqs_config.learning_rate, seed=config.seed,
    ).fit(x_train, y_train)
    point_prediction = point.predict(x_test)

    results = {
        "location": location,
        "alias": BGCFQS_ALIASES[location],
        "date_range": [start, end],
        "selected_tau": tau,
        "published_tau": PUBLISHED_QUANTILES[location],
        "tau_gap": round(tau - PUBLISHED_QUANTILES[location], 4),
        "selection_trace": model.trace.to_dict(),
        "split": prepared["split"].summary(),
        "methods": {},
    }
    for name, prediction in (("point", point_prediction), ("bgcfqs", lower)):
        metrics = conditional_metrics(prediction, y_test)
        results["methods"][name] = {
            slice_name: metric.to_dict() for slice_name, metric in metrics.items()
        }
        results["methods"][name]["risk_pass"] = metrics["global"].risk_pass(args.epsilon)

    published = PUBLISHED_CONDITIONAL[location]
    observed = results["methods"]["bgcfqs"]
    print(f"  selected tau {tau:.3f} (published {PUBLISHED_QUANTILES[location]:.3f})")
    print(f"  OverRate  global {observed['global']['OverRate']:.3f}  "
          f"P30 {observed['P30']['OverRate']:.3f} (published {published['P30']:.2f})  "
          f"P10 {observed['P10']['OverRate']:.3f} (published {published['P10']:.2f})")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/starnet"))
    parser.add_argument("--location", default="usa",
                        choices=["usa", "canada", "germany"])
    parser.add_argument("--all", action="store_true", help="run all three locations")
    parser.add_argument("--lookback", type=int, default=75)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=Path("results/reproduce_bgcfqs"))
    args = parser.parse_args(argv)

    locations = list(BGCFQS_RANGES) if args.all else [args.location]
    all_results = {}
    for location in locations:
        print(f"\n=== {location} ({BGCFQS_ALIASES[location]}) ===")
        all_results[location] = run_one(location, args)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(all_results, indent=2, default=str))

    # The motivation table. Averaged across whatever locations were run, with
    # the count stated so a single location average is never mistaken for the
    # three location average the paper reports.
    rows = []
    for location, result in all_results.items():
        for method, entry in result["methods"].items():
            if method == "risk_pass":
                continue
            rows.append({
                "location": location, "alias": result["alias"], "method": method,
                "OverRate_global": entry["global"]["OverRate"],
                "OverRate_P30": entry["P30"]["OverRate"],
                "OverRate_P10": entry["P10"]["OverRate"],
                "MAE": entry["global"]["MAE"], "RMSE": entry["global"]["RMSE"],
                "MPE": entry["global"]["MPE"], "P95+Err": entry["global"]["P95+Err"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "conditional_over_rate.csv", index=False)
    print("\n" + table.round(3).to_string(index=False))
    print(f"\naveraged over {len(all_results)} location(s); the published table "
          "averages three")

    if not table.empty:
        rates = {
            method: {
                "global": float(part["OverRate_global"].mean()),
                "P30": float(part["OverRate_P30"].mean()),
                "P10": float(part["OverRate_P10"].mean()),
            }
            for method, part in table.groupby("method")
        }
        figure = conditional_over_rate(
            rates, args.epsilon, args.output / "motivation.png",
            subtitle=f"reproduction over {len(all_results)} StarNet location(s), "
                     f"L={args.lookback} H={args.horizon}",
        )
        print(f"wrote {figure}")

    print(f"wrote {args.output / 'results.json'}")
    print("\nPublished averages for comparison (three locations):")
    print(pd.DataFrame(PUBLISHED_AVERAGES).T.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
