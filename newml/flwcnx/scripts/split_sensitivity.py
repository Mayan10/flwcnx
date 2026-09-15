#!/usr/bin/env python3
"""Is BG-CFQS's risk guarantee a property of the method or of the split?

Phase 2 reproduced their conditional failure but not their risk-pass column:
they report 3/3 within budget, we got 0/3 under a temporal split. Before
reporting that as a failed reproduction, the obvious confound has to be ruled
out, because our splits are contiguous temporal blocks and theirs are not
specified.

The experiment holds everything constant except the split. Same data, same
XGBoost backbone, same boundary search, same budget. One arm splits into
contiguous temporal blocks, which is what a deployed terminal faces and what
`eval/splits.py` does everywhere else in this project. The other permutes the
same windows before splitting, which makes calibration and test exchangeable,
and exchangeability is exactly the assumption the boundary search inherits from
split conformal.

If the risk-pass column comes back under the random split, the reimplementation
is correct and their guarantee is conditional on an assumption that does not
hold in deployment. If it does not come back, the reimplementation is wrong and
nothing built on it should be trusted.

    python scripts/split_sensitivity.py --output results/split_sensitivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flwcnx.calibrate.bgcfqs import bgcfqs_on_residuals
from flwcnx.config import FEATURE_COLUMNS, BGCFQSConfig, DataConfig, FeatureConfig
from flwcnx.eval.metrics import conditional_metrics
from flwcnx.forecast.baselines import XGBoostForecaster
from flwcnx.ingest.replay import ReplaySource
from flwcnx.state.features import build_features, make_sequences

ALIAS = {"usa": "CHI", "germany": "OSN", "canada": "VIC"}


def run_one(location: str, epsilon: float, seed: int, lookback: int, horizon: int,
            stride: int) -> list[dict]:
    frame = ReplaySource(DataConfig(location=location)).load_frame()
    config = FeatureConfig(lookback=lookback, horizon=horizon, stride=stride,
                           recover_phase=True)
    features, _, _ = build_features(frame, config=config)
    sequences = make_sequences(features, config=config, feature_names=FEATURE_COLUMNS)

    n = len(sequences)
    x = sequences.x.reshape(n, -1)
    y = sequences.y.mean(axis=1)
    rng = np.random.default_rng(seed)
    orders = {"temporal": np.arange(n), "random": rng.permutation(n)}

    rows = []
    for scheme, order in orders.items():
        cut_train, cut_cal = int(0.6 * n), int(0.8 * n)
        i_train, i_cal, i_test = order[:cut_train], order[cut_train:cut_cal], order[cut_cal:]

        model = XGBoostForecaster(n_estimators=300, max_depth=6,
                                  learning_rate=0.05, seed=seed).fit(x[i_train], y[i_train])
        predicted_cal, predicted_test = model.predict(x[i_cal]), model.predict(x[i_test])
        offset = bgcfqs_on_residuals(predicted_cal, y[i_cal], epsilon, BGCFQSConfig(),
                                     direction="lower")
        metrics = conditional_metrics(predicted_test + offset, y[i_test], direction="lower")
        rows.append({
            "location": location, "alias": ALIAS[location], "split": scheme,
            "n_test": int(len(i_test)), "offset_mbps": round(float(offset), 3),
            "OverRate_global": round(metrics["global"].over_rate, 4),
            "OverRate_P30": round(metrics["P30"].over_rate, 4),
            "OverRate_P10": round(metrics["P10"].over_rate, 4),
            "MAE": round(metrics["global"].mae, 3),
            "risk_pass": bool(metrics["global"].over_rate <= epsilon),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locations", nargs="+",
                        default=["usa", "germany", "canada"])
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--lookback", type=int, default=75)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=Path("results/split_sensitivity"))
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for location in args.locations:
        rows += run_one(location, args.epsilon, args.seed, args.lookback,
                        args.horizon, args.stride)
        for row in rows[-2:]:
            print(f"{row['alias']:4s} {row['split']:9s} "
                  f"global={row['OverRate_global']:.3f} P30={row['OverRate_P30']:.3f} "
                  f"P10={row['OverRate_P10']:.3f} MAE={row['MAE']:6.2f} "
                  f"pass={row['risk_pass']}")

    by_split: dict[str, list[dict]] = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
    summary = {
        scheme: {
            "mean_OverRate_global": round(float(np.mean([r["OverRate_global"] for r in part])), 4),
            "mean_OverRate_P30": round(float(np.mean([r["OverRate_P30"] for r in part])), 4),
            "mean_OverRate_P10": round(float(np.mean([r["OverRate_P10"] for r in part])), 4),
            "risk_pass": f"{sum(r['risk_pass'] for r in part)}/{len(part)}",
        }
        for scheme, part in by_split.items()
    }
    print("\n=== averages ===")
    for scheme, s in summary.items():
        print(f"{scheme:9s} global={s['mean_OverRate_global']:.3f} "
              f"P30={s['mean_OverRate_P30']:.3f} P10={s['mean_OverRate_P10']:.3f} "
              f"risk_pass={s['risk_pass']}")
    print("\npublished BG-CFQS: OverRate 0.349, risk pass 3/3, "
          "P30 0.65-0.71, P10 0.83-0.86")

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(args) | {"output": str(args.output)},
               "rows": rows, "summary": summary}
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {args.output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
