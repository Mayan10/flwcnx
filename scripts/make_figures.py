#!/usr/bin/env python3
"""Render every figure and summary table from a saved run.

Reads a result directory produced by `RunResult.save` and writes figures next
to it. Nothing here recomputes a metric: if a number is in a figure it came out
of `result.json`, so a figure can always be traced back to the run and the
config snapshot sitting beside it (CLAUDE.md section 11).

    python scripts/make_figures.py results/final/wetlinks-seconds-Osnabruck-capacity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thalweg.eval import figures

#: The methods the headline figure compares, in argument order. Restricted on
#: purpose: the motivation figure has to be readable, and the whole argument is
#: carried by the distance between these five.
HEADLINE_METHODS = (
    "point",
    "global_conformal",
    "regime_conformal",
    "adaptive_global_conformal",
    "adaptive_regime_conformal",
)


def load(directory: Path) -> dict:
    payload = json.loads((directory / "result.json").read_text())
    if not payload.get("calibration"):
        raise ValueError(f"{directory} has no calibration results to plot")
    return payload


def parse_key(key: str) -> tuple[str, float, str]:
    """`method|eps=0.35|regime=level` -> (method, 0.35, 'level')."""
    method, eps, regime = key.split("|")
    return method, float(eps.split("=")[1]), regime.split("=")[1]


def calibration_frame(payload: dict) -> pd.DataFrame:
    """One row per grid cell, with each risk slice's OverRate as a column."""
    rows = []
    for key, entry in payload["calibration"].items():
        method, epsilon, regime = parse_key(key)
        row = {"setting": key, "method": method, "epsilon": epsilon,
               "granularity": regime, "axes": "+".join(entry["axes"]) or "global"}
        for slice_name in ("global", "P30", "P10"):
            metrics = entry.get(slice_name, {})
            row[f"OverRate_{slice_name}"] = metrics.get("OverRate")
            row[f"MAE_{slice_name}"] = metrics.get("MAE")
        row["MAE"] = entry.get("global", {}).get("MAE")
        row["MPE"] = entry.get("global", {}).get("MPE")
        row["P95+Err"] = entry.get("global", {}).get("P95+Err")
        row["risk_pass"] = entry.get("global", {}).get("risk_pass")
        row["worst_regime_OverRate"] = entry.get("worst_regime_over_rate")
        rows.append(row)
    return pd.DataFrame(rows)


def headline_epsilon(frame: pd.DataFrame, requested: float | None) -> float:
    if requested is not None:
        return requested
    # 0.35 is BG-CFQS's published budget, so it is the comparable one.
    available = sorted(frame["epsilon"].unique())
    return 0.35 if 0.35 in available else available[-1]


def best_granularity(frame: pd.DataFrame, epsilon: float) -> str:
    """The regime granularity that best controls the severe-risk slice.

    Chosen on the P10 OverRate of our own method rather than fixed to "full",
    because which axes carry the gain is an empirical question the ablation
    exists to answer, and hardcoding an answer here would prejudge it.
    """
    part = frame[(frame["epsilon"] == epsilon)
                 & (frame["method"] == "adaptive_regime_conformal")
                 & (frame["granularity"] != "global")]
    if part.empty or part["OverRate_P10"].isna().all():
        return "full"
    return str(part.loc[part["OverRate_P10"].idxmin(), "granularity"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args(argv)

    directory = args.directory
    outdir = args.outdir or (directory / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    payload = load(directory)
    frame = calibration_frame(payload)
    epsilon = headline_epsilon(frame, args.epsilon)
    granularity = best_granularity(frame, epsilon)
    written: list[Path] = []

    frame.to_csv(outdir / "calibration_grid.csv", index=False)

    # -- 1. the motivation figure -----------------------------------------
    # Each method at its own best granularity is not a fair picture, so every
    # regime-conditioned method is read at the same one.
    rates: dict[str, dict[str, float]] = {}
    for method in HEADLINE_METHODS:
        wanted = "global" if method in ("point", "global_conformal",
                                        "adaptive_global_conformal") else granularity
        row = frame[(frame["method"] == method) & (frame["epsilon"] == epsilon)
                    & (frame["granularity"] == wanted)]
        if row.empty:
            continue
        row = row.iloc[0]
        rates[method] = {s: row[f"OverRate_{s}"] for s in ("global", "P30", "P10")}

    if rates:
        written.append(figures.conditional_over_rate(
            rates, epsilon, outdir / "motivation_conditional_overrate.pdf",
            title="Overestimation rate by risk slice",
            subtitle=(f"{payload['name']}, regime axes = {granularity}, "
                      f"n = {payload['split_summary']['test']} test decisions"),
        ))

    # -- 2. the budget sweep ----------------------------------------------
    sweep = frame[frame["granularity"].isin([granularity, "global"])].copy()
    sweep = sweep[~((sweep["granularity"] == "global")
                    & sweep["method"].isin(["regime_conformal", "regime_bgcfqs",
                                            "adaptive_regime_conformal"]))]
    for slice_name in ("P30", "P10"):
        try:
            written.append(figures.epsilon_sweep(
                sweep, outdir / f"epsilon_sweep_{slice_name}.pdf", slice_name=slice_name,
                title=f"Risk budget sweep, {slice_name} slice",
            ))
        except (KeyError, ValueError) as exc:
            print(f"  skipped epsilon_sweep {slice_name}: {exc}")

    # -- 3. the ablation ---------------------------------------------------
    ablation = frame[(frame["epsilon"] == epsilon)
                     & (frame["method"] == "adaptive_regime_conformal")]
    if not ablation.empty:
        for slice_name in ("P30", "P10"):
            try:
                written.append(figures.granularity_ablation(
                    ablation, outdir / f"granularity_ablation_{slice_name}.pdf",
                    slice_name=slice_name,
                ))
            except (KeyError, ValueError) as exc:
                print(f"  skipped granularity_ablation {slice_name}: {exc}")

    # -- 4. the adaptation trace ------------------------------------------
    trace_name = (f"trace_adaptive_regime_conformal_eps{epsilon:.2f}"
                  f"_regime{granularity}.csv")
    trace_path = directory / trace_name
    if trace_path.exists():
        written.append(figures.adaptation_trace(
            pd.read_csv(trace_path), epsilon, outdir / "adaptation_trace.pdf",
            title=f"Online recalibration, regime axes = {granularity}",
        ))
    else:
        print(f"  no adaptation trace at {trace_path.name}")

    # -- 5. dropped sessions, which is what a user would actually notice ---
    admission = payload.get("admission") or {}
    tables = {}
    for method in HEADLINE_METHODS:
        wanted = "global" if method in ("point", "global_conformal",
                                        "adaptive_global_conformal") else granularity
        rows = admission.get(f"{method}|eps={epsilon:.2f}|regime={wanted}")
        if rows:
            tables[method] = pd.DataFrame(rows)
    if tables:
        try:
            written.append(figures.admission_comparison(
                tables, outdir / "admission_dropped_sessions.pdf"))
        except (KeyError, ValueError) as exc:
            print(f"  skipped admission_comparison: {exc}")

    # -- 6. the spread the headline number hides ---------------------------
    for method in ("global_conformal", "adaptive_regime_conformal"):
        name = f"regime_{method}_eps{epsilon:.2f}_regime{granularity}.csv"
        path = directory / name
        if not path.exists():
            continue
        written.append(figures.regime_over_rate_spread(
            pd.read_csv(path), epsilon, outdir / f"regime_spread_{method}.pdf",
            title=f"OverRate across regimes, {method.replace('_', ' ')}",
        ))

    print(f"headline epsilon {epsilon}, granularity {granularity!r}")
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
