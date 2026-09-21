#!/usr/bin/env python3
"""Turn a protection run into the committed markdown table.

`results/` is gitignored except `results/summary/`, so this is how a protection
run leaves a trace that survives. It recomputes nothing: every figure in the
output is read from `result.json`, averaged over the workload seeds that run
used, and the header records which run produced it.

    python scripts/make_protection_summary.py results/protection/canada \
        --out results/summary/protection-canada.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POLICY_LABEL = {
    "oracle": "oracle (perfect classification)",
    "protected": "**criticality-weighted, with floors (ours)**",
    "by_class": "class priority (DiffServ)",
    "equal_share": "equal share (fair queue)",
    "shed_largest": "throttle the largest first",
}
ORDER = ["oracle", "protected", "by_class", "equal_share", "shed_largest"]


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if column in frame and len(frame) else float("nan")


def _spread(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].std(ddof=0)) if column in frame and len(frame) else float("nan")


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def policy_table(rows: list[dict], load: float) -> str:
    frame = pd.DataFrame(rows)
    frame = frame[frame["load_multiplier"] == load]
    lines = [
        "| policy | critical violation (allocated) | critical violation (delivered) "
        "| critical transfers abandoned | critical goodput | ordinary goodput "
        "| link utilisation | ordinary slowdown |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for policy in ORDER:
        group = frame[frame["policy"] == policy]
        if group.empty:
            continue
        lines.append(
            f"| {POLICY_LABEL[policy]} "
            f"| {_fmt(_mean(group, 'critical_violation_allocated'))} "
            f"+/- {_fmt(_spread(group, 'critical_violation_allocated'))} "
            f"| {_fmt(_mean(group, 'critical_violation_delivered'))} "
            f"| {_fmt(_mean(group, 'critical_abandon_rate'))} "
            f"| {_fmt(_mean(group, 'critical_goodput_ratio'))} "
            f"| {_fmt(_mean(group, 'ordinary_goodput_ratio'))} "
            f"| {_fmt(_mean(group, 'utilisation'))} "
            f"| {_fmt(_mean(group, 'completion_inflation_ordinary'), 2)}x |"
        )
    return "\n".join(lines)


def sweep_table(rows: list[dict]) -> str:
    frame = pd.DataFrame(rows)
    loads = sorted(frame["load_multiplier"].unique())
    header = "| offered load | " + " | ".join(
        POLICY_LABEL[p].replace("**", "") for p in ORDER if p in set(frame["policy"])) + " |"
    lines = [header, "|---" * (len([p for p in ORDER if p in set(frame['policy'])]) + 1) + "|"]
    for load in loads:
        at_load = frame[frame["load_multiplier"] == load]
        over = _mean(at_load, "oversubscription")
        cells = [f"{load:g}x nominal ({over:.1f}x capacity)"]
        for policy in ORDER:
            group = at_load[at_load["policy"] == policy]
            if group.empty:
                continue
            cells.append(_fmt(_mean(group, "critical_violation_allocated")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def simple_table(rows: list[dict], key: str, label: str) -> str:
    frame = pd.DataFrame(rows)
    baseline_key = "full" if key == "variant" else "none"
    grouped = frame.groupby(key)
    baseline = float(grouped["critical_violation_allocated"].mean().get(baseline_key, np.nan))
    # Goodput and flap rate are here because two variants do not move the
    # violation rate at all and are not therefore no-ops: hysteresis acts on
    # flapping and the weight exponent acts on the residual sharing.
    lines = [f"| {label} | critical violation | change | critical goodput "
             f"| scorer AP | flap rate |",
             "|---|---|---|---|---|---|"]
    for name, group in grouped:
        value = _mean(group, "critical_violation_allocated")
        delta = value - baseline
        marker = "" if name == baseline_key else f"{delta:+.3f}"
        lines.append(f"| {str(name).replace('_', ' ')} | {_fmt(value)} +/- "
                     f"{_fmt(_spread(group, 'critical_violation_allocated'))} | {marker} "
                     f"| {_fmt(_mean(group, 'critical_goodput_ratio'))} "
                     f"| {_fmt(_mean(group, 'scorer_ap'))} "
                     f"| {_fmt(_mean(group, 'scorer_flap_rate'), 4)} |")
    return "\n".join(lines)


def archetype_table(rows: list[dict]) -> str:
    frame = pd.DataFrame(rows)
    frame = frame[frame["policy"] == "protected"].sort_values(
        ["truly_critical", "violation_allocated"], ascending=[False, False])
    lines = ["| archetype | must not be shed | mean criticality | violation rate | goodput |",
             "|---|---|---|---|---|"]
    for _, row in frame.iterrows():
        lines.append(
            f"| {str(row['archetype']).replace('_', ' ')} "
            f"| {'yes' if row['truly_critical'] else 'no'} "
            f"| {_fmt(float(row['mean_criticality']), 2)} "
            f"| {_fmt(float(row['violation_allocated']))} "
            f"| {_fmt(float(row['goodput_ratio']), 2)} |")
    return "\n".join(lines)


def build(result: dict, run: Path) -> str:
    args = result.get("args", {})
    series = result.get("capacity_series", {})
    load = float(args.get("reference_load", 2.0))
    parts: list[str] = []

    location = str(args.get("location", "unknown"))
    parts.append(f"# Protection layer: {location}\n")
    parts.append(
        f"Run `{run}`, {series.get('n_slots', 0):,} decisions, "
        f"{args.get('n_seeds', 1)} workload seeds, seed {args.get('seed')}. "
        f"Config snapshot and environment are in `result.json`.\n")
    parts.append(
        "**The capacity is measured and the flows are a model.** No public LEO dataset "
        "carries a flow table, process attribution or a user-attention signal, so the "
        "capacity series below is the calibrated bound and the realised throughput from a "
        "trained pipeline over the real traces, and the flows contending for it come from "
        "`thalweg/eval/workload.py`. Every number here is a number about that workload.\n")
    parts.append(
        f"| capacity series | |\n|---|---|\n"
        f"| decisions | {series.get('n_slots', 0):,} |\n"
        f"| calibrated bound, mean | {_fmt(series.get('bound_mean_mbps', float('nan')), 1)} Mbps |\n"
        f"| realised throughput, mean | {_fmt(series.get('realised_mean_mbps', float('nan')), 1)} Mbps |\n"
        f"| realised risk rate | {_fmt(series.get('realised_risk_rate', float('nan')))} "
        f"against a budget of {series.get('budget')} |\n"
        f"| point forecast risk rate | {_fmt(series.get('point_risk_rate', float('nan')))} |\n")

    if result.get("policies"):
        parts.append(f"\n## Policies at {load:g}x offered load\n")
        parts.append(policy_table(result["policies"], load))
        parts.append("\n## The load sweep\n")
        parts.append(
            "Under light load nothing has to be shed and every policy looks the same. "
            "Under heavy enough load the floors stop fitting and the oracle fails too. "
            "Either end on its own would be mistaken for the general case.\n")
        parts.append(sweep_table(result["policies"]))

    if result.get("by_archetype"):
        parts.append("\n## Where the violations land\n")
        parts.append(archetype_table(result["by_archetype"]))

    if result.get("ablation"):
        parts.append("\n## Removing a mechanism\n")
        parts.append(simple_table(result["ablation"], "variant", "variant"))

    if result.get("channels"):
        parts.append("\n## Removing an evidence channel\n")
        parts.append(
            "A channel whose removal improves the outcome is not carrying its weight. "
            "The defaults were not retuned on the strength of this table, because fitting "
            "the weights on the evaluation workload and then evaluating on the same "
            "generator is the circularity the module docstring warns about.\n")
        parts.append(simple_table(result["channels"], "dropped_channel", "channel removed"))

    if result.get("capacity_arm"):
        frame = pd.DataFrame(result["capacity_arm"])
        frame = frame[frame["load_multiplier"] == load]
        parts.append("\n## Dividing the bound against dividing the forecast\n")
        parts.append(
            "The allocator honours every feasible floor in both arms. Against the point "
            "forecast it writes *more* floors, because the number is larger, and the link "
            "then fails them.\n")
        parts.append("| capacity divided | violation (allocated) | violation (delivered) |")
        parts.append("|---|---|---|")
        for source in ("bound", "point_forecast"):
            group = frame[frame["capacity_source"] == source]
            parts.append(f"| {source.replace('_', ' ')} "
                         f"| {_fmt(_mean(group, 'critical_violation_allocated'))} "
                         f"| {_fmt(_mean(group, 'critical_violation_delivered'))} |")

    if result.get("sensitivity"):
        frame = pd.DataFrame(result["sensitivity"])
        values = frame["critical_violation_allocated"]
        baseline = pd.DataFrame(result.get("policies", []))
        reference = float("nan")
        if not baseline.empty:
            reference = _mean(baseline[(baseline["policy"] == "protected")
                                       & (baseline["load_multiplier"] == load)],
                              "critical_violation_allocated")
        beaten = pd.DataFrame(result.get("policies", []))
        rival = float("nan")
        if not beaten.empty:
            rival = beaten[(beaten["policy"].isin(["by_class", "equal_share",
                                                   "shed_largest"]))
                           & (beaten["load_multiplier"] == load)][
                "critical_violation_allocated"].min()
        parts.append("\n## The weights are hand set, so they were perturbed\n")
        parts.append(
            f"{len(frame)} draws, each channel weight multiplied by a log-uniform factor "
            f"spanning a factor of four either way. Critical violation ranged "
            f"{_fmt(values.min())} to {_fmt(values.max())}, median {_fmt(values.median())}, "
            f"against {_fmt(reference)} at the hand-set values. "
            f"The best deployable baseline at this load is {_fmt(rival)}, and "
            f"{int((values < rival).sum())} of {len(frame)} draws beat it.\n")

    parts.append("\n## The workload\n")
    parts.append("| archetype | declared | must not be shed | nominal | floor | elastic | deadline |")
    parts.append("|---|---|---|---|---|---|---|")
    for row in result.get("workload_archetypes", []):
        parts.append(
            f"| {row['archetype'].replace('_', ' ')} | {row['declared']} "
            f"| {'yes' if row['truly_critical'] else 'no'} | {row['nominal_mbps']:g} Mbps "
            f"| {row['floor_mbps']:g} Mbps | {'yes' if row['elastic'] else 'no'} "
            f"| {row['deadline']} |")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = json.loads((args.run / "result.json").read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(result, args.run))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
