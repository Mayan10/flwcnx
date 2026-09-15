#!/usr/bin/env python3
"""Turn a saved run into the markdown tables that go into `results/summary/`.

`results/` is gitignored except `results/summary/`, so this is the step that
decides what enters the history. It writes markdown rather than a pickle for
the same reason: a summary that cannot be read in a diff is not a summary.

Nothing is recomputed. Every number is read from `result.json`, so a table can
always be traced to the run directory and the config snapshot beside it.

    python scripts/make_summary.py results/final/wetlinks-seconds-Osnabruck-capacity \\
        --out results/summary/osnabruck-capacity.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Same directory, and running this file directly puts `scripts/` on sys.path
# rather than the repo root, so a package-qualified import would not resolve.
from make_figures import best_granularity, calibration_frame, headline_epsilon

#: Display order and display names. Order is the argument: each row should be
#: strictly more capable than the one above it, so the table reads as a
#: progression rather than a lookup.
METHOD_LABELS = {
    "point": "point forecast (uncalibrated)",
    "global_conformal": "split conformal, global",
    "regime_bgcfqs": "BG-CFQS boundary search, per regime",
    "regime_conformal": "regime conformal (ours, static)",
    "adaptive_global_conformal": "adaptive conformal, global",
    "adaptive_regime_conformal": "adaptive regime conformal (ours)",
}
GLOBAL_ONLY = ("point", "global_conformal", "adaptive_global_conformal")


def fmt(value, digits: int = 4) -> str:
    # numpy scalars arrive here after a pandas round trip, so the bool check has
    # to come first and cover np.bool_ as well as bool. Otherwise risk_pass
    # renders as "True" in a column captioned "within budget".
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float | np.floating):
        return f"{float(value):.{digits}f}"
    return str(value)


def headline_table(frame: pd.DataFrame, epsilon: float, granularity: str) -> str:
    """The one table a reviewer reads: risk globally and on each risk slice."""
    lines = [
        "| method | regime axes | OverRate | P30 | P10 | worst regime | MAE | MPE | P95+Err | within budget |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method, label in METHOD_LABELS.items():
        wanted = "global" if method in GLOBAL_ONLY else granularity
        row = frame[(frame["method"] == method) & (frame["epsilon"] == epsilon)
                    & (frame["granularity"] == wanted)]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"| {label} | {r['axes']} | {fmt(r['OverRate_global'])} | "
            f"{fmt(r['OverRate_P30'])} | {fmt(r['OverRate_P10'])} | "
            f"{fmt(r['worst_regime_OverRate'])} | {fmt(r['MAE'], 2)} | "
            f"{fmt(r['MPE'], 2)} | {fmt(r['P95+Err'], 2)} | {fmt(r['risk_pass'])} |"
        )
    return "\n".join(lines)


def ablation_table(frame: pd.DataFrame, epsilon: float,
                   method: str = "adaptive_regime_conformal") -> str:
    part = frame[(frame["method"] == method) & (frame["epsilon"] == epsilon)]
    part = part.sort_values("OverRate_P10")
    lines = ["| regime axes | OverRate | P30 | P10 | MAE |", "|---|---|---|---|---|"]
    for _, r in part.iterrows():
        lines.append(
            f"| {r['axes']} | {fmt(r['OverRate_global'])} | {fmt(r['OverRate_P30'])} | "
            f"{fmt(r['OverRate_P10'])} | {fmt(r['MAE'], 2)} |"
        )
    return "\n".join(lines)


def sweep_table(frame: pd.DataFrame, granularity: str) -> str:
    """How each method tracks the budget as the budget tightens.

    The column that matters is the gap between the achieved global rate and the
    budget. A method that only holds its budget at 0.35 is not a risk aware
    method, it is a method that happened to be tuned at one point.
    """
    methods = ["global_conformal", "regime_conformal", "adaptive_regime_conformal"]
    epsilons = sorted(frame["epsilon"].unique())
    lines = ["| budget | " + " | ".join(METHOD_LABELS[m] for m in methods) + " |",
             "|---" * (len(methods) + 1) + "|"]
    for epsilon in epsilons:
        cells = []
        for method in methods:
            wanted = "global" if method in GLOBAL_ONLY else granularity
            row = frame[(frame["method"] == method) & (frame["epsilon"] == epsilon)
                        & (frame["granularity"] == wanted)]
            if row.empty:
                cells.append("-")
                continue
            r = row.iloc[0]
            cells.append(f"{fmt(r['OverRate_global'])} / {fmt(r['OverRate_P10'])}")
        lines.append(f"| {epsilon:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def admission_table(payload: dict, epsilon: float, granularity: str) -> str:
    admission = payload.get("admission") or {}
    if not admission:
        return "_No admission results in this run (latency targets have no session count)._"
    lines = ["| method | slice | mean dropped | violation rate | P95 dropped | utilisation |",
             "|---|---|---|---|---|---|"]
    for method, label in METHOD_LABELS.items():
        wanted = "global" if method in GLOBAL_ONLY else granularity
        key = f"{method}|eps={epsilon:.2f}|regime={wanted}"
        rows = admission.get(key)
        if not rows:
            continue
        for entry in rows:
            lines.append(
                f"| {label} | {entry['slice']} | {fmt(entry['mean_dropped'], 3)} | "
                f"{fmt(entry['violation_rate'])} | {fmt(entry['p95_dropped'], 1)} | "
                f"{fmt(entry.get('utilisation'), 4)} |"
            )
    return "\n".join(lines)


def adaptation_note(payload: dict, epsilon: float, granularity: str) -> str:
    key = f"adaptive_regime_conformal|eps={epsilon:.2f}|regime={granularity}"
    detail = (payload.get("calibration", {}).get(key, {}) or {}).get("detail", {})
    if not detail:
        return ""
    keep = ("reference", "gamma", "window", "alpha_start", "alpha_end", "alpha_mean",
            "realised_risk_rate", "fallback_rate", "n_regimes_active")
    lines = ["| quantity | value |", "|---|---|"]
    for name in keep:
        if name in detail:
            value = detail[name]
            lines.append(f"| {name} | "
                         f"{fmt(value) if isinstance(value, float) else value} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    payload = json.loads((args.directory / "result.json").read_text())
    frame = calibration_frame(payload)
    epsilon = headline_epsilon(frame, args.epsilon)
    granularity = best_granularity(frame, epsilon)

    config = payload.get("config", {})
    features = config.get("features", {})
    split = payload.get("split_summary", {})
    training = payload.get("training", {})
    environment = payload.get("environment", {})

    title = args.title or payload["name"]
    body = f"""# {title}

Generated by `scripts/make_summary.py` from `{args.directory}`. Every number is
read from that run's `result.json`; nothing here is recomputed or rounded from
memory.

## Setup

| | |
|---|---|
| dataset | {config.get('dataset')} |
| geometry | {'reconstructed from propagated elements' if config.get('geometry') else 'none'} |
| look-back / horizon | {features.get('lookback')} / {features.get('horizon')} s |
| split | {split.get('scheme')}, train {split.get('train')} / calib {split.get('calibration')} / test {split.get('test')} |
| leak check | {'clean' if payload.get('leak_check', {}).get('clean') else 'FAILED'} |
| backbone | {payload.get('point_metrics', {}).get('backbone')} |
| epochs run / best | {training.get('epochs_run')} / {training.get('best_epoch')} |
| seed | {config.get('seed')} |
| device | {training.get('device')} |
| run at | {environment.get('timestamp')} |

## Headline, budget {epsilon:.2f}, regime axes `{granularity}`

OverRate is the fraction of decisions where the bound promised capacity the
link did not deliver. P30 and P10 are the lowest 30% and 10% of true
throughput: the slices where over-allocation actually drops sessions.

{headline_table(frame, epsilon, granularity)}

## Which axes carry the gain

Ordered by the severe-risk slice, so the top row is the best conditioning set.

{ablation_table(frame, epsilon)}

## Budget sweep, cells are `global OverRate / P10 OverRate`

{sweep_table(frame, granularity)}

## Downstream: admission control

{admission_table(payload, epsilon, granularity)}

## The online layer

{adaptation_note(payload, epsilon, granularity)}
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body)
    print(f"wrote {args.out} (epsilon {epsilon}, granularity {granularity!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
