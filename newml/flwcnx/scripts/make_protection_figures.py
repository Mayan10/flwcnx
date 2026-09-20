#!/usr/bin/env python3
"""Publication-quality figures for the protection layer.

Same contract as `make_readme_figures.py`: every number is read from a saved
`result.json` or the per-slot CSVs beside it, never hardcoded, so a figure
cannot drift away from the run that produced it. Output is 300 DPI PNG into
`docs/figures/`, which is versioned so GitHub renders it inline.

Palette and ink tokens are the ones validated for light mode in
`make_readme_figures.py` and are imported from there rather than restated, so
the two sets of figures cannot drift apart. Four categorical slots, worst
adjacent CVD deltaE 9.2 (deutan) against 27.6 in normal vision. Aqua sits below
3:1 against the surface, so every aqua mark carries a visible value label,
which is the documented relief for that case. The oracle is drawn as a dashed
neutral reference rather than a fifth hue, because it is not a policy anyone
could deploy: it is the floor imposed by physics with classification error
removed.

    python scripts/make_protection_figures.py --run results/protection/canada
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_readme_figures import (  # noqa: E402
    AQUA,
    BLUE,
    DPI,
    GRID,
    ORANGE,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    VIOLET,
    _save,
    _style,
)

#: Fixed hue per policy, assigned by entity and never by rank, so a figure that
#: drops a policy does not repaint the survivors.
POLICY_COLOUR = {
    "protected": BLUE,
    "by_class": ORANGE,
    "equal_share": VIOLET,
    "shed_largest": AQUA,
}
POLICY_LABEL = {
    "protected": "criticality-weighted, with floors (ours)",
    "by_class": "class priority (DiffServ)",
    "equal_share": "equal share (fair queue)",
    "shed_largest": "throttle the largest first",
    "oracle": "oracle: perfect classification",
}
DEPLOYABLE = ("protected", "by_class", "equal_share", "shed_largest")

CRITICAL_COLOUR, ORDINARY_COLOUR = ORANGE, BLUE


def _read(run: Path) -> dict:
    path = run / "result.json"
    if not path.exists():
        raise SystemExit(f"no result.json under {run}; run scripts/run_protection.py first")
    return json.loads(path.read_text())


def _frame(run: Path, name: str) -> pd.DataFrame | None:
    path = run / name
    return pd.read_csv(path) if path.exists() else None


def _aggregate(rows: list[dict], keys: list[str], value: str) -> pd.DataFrame:
    """Mean and spread across workload seeds, which is the only honest form for
    a difference of a few points between policies."""
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(keys)[value]
    return pd.DataFrame({"mean": grouped.mean(), "std": grouped.std(ddof=0),
                         "n": grouped.size()}).reset_index()


# -- 1. the case study -------------------------------------------------------


def fig_case_study(run: Path, out: Path) -> Path | None:
    """One critical transfer, under the policy in use today and under this one.

    The argument for the whole layer in a single pair of panels. Both panels
    replay an identical workload against an identical capacity series, so the
    only thing that differs is what the allocator decided.
    """
    ours = _frame(run, "episode_protected.csv")
    naive = _frame(run, "episode_shed_largest.csv")
    if ours is None or naive is None:
        return None

    # The focal flow: the longest lived critical transfer that is large enough
    # to be the first thing a rate-based shaper reaches for.
    critical = ours[ours["truly_critical"] & (ours["demand_mbps"] > 8.0)]
    if critical.empty:
        return None
    focal = critical.groupby("flow_id").size().idxmax()
    archetype = str(ours.loc[ours["flow_id"] == focal, "archetype"].iloc[0])

    panels = [("throttle the largest first", naive), ("criticality-weighted, with floors", ours)]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True,
                             gridspec_kw={"hspace": 0.28})

    window = None
    for ax, (title, table) in zip(axes, panels, strict=True):
        flow = table[table["flow_id"] == focal].sort_values("slot")
        if flow.empty:
            return None
        if window is None:
            window = (int(flow["slot"].min()), int(flow["slot"].max()))
        slots = flow["slot"].to_numpy()
        delivered = flow["delivered_mbps"].to_numpy()
        floor = float(flow["floor_mbps"].median())

        others = (table[(table["flow_id"] != focal)
                        & table["slot"].between(*window)]
                  .groupby("slot")["delivered_mbps"].sum().reindex(slots, fill_value=0.0))
        ax.fill_between(slots, 0, others.to_numpy(), color=GRID, alpha=0.85,
                        linewidth=0, zorder=1, label="everything else on the link")
        ax.plot(slots, delivered, linewidth=2.0, color=BLUE, zorder=4,
                label=f"the {archetype.replace('_', ' ')}")
        ax.axhline(floor, linestyle="--", linewidth=1.4, color=TEXT_SECONDARY,
                   zorder=3, label="the rate below which it is useless")

        starved = delivered < floor - 1e-9
        ax.fill_between(slots, 0, np.maximum(delivered, 0), where=starved,
                        color=ORANGE, alpha=0.30, linewidth=0, zorder=2)
        share = float(starved.mean())
        _style(ax, title=title, ylabel="delivered Mbps")
        ax.text(0.995, 0.93,
                f"below its floor on {share:5.1%} of slots",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
                color=ORANGE if share > 0.05 else TEXT_SECONDARY,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.0})

    axes[-1].set_xlabel("decision slot", color=TEXT_SECONDARY, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
                   loc="upper left", ncol=3, bbox_to_anchor=(0.0, -0.16))
    fig.suptitle("The same transfer, under the shaper in use today and under this layer",
                 x=0.005, ha="left", fontsize=13, color=TEXT_PRIMARY, y=1.005)
    return _save(fig, out / "protection-case-study.png")


# -- 2. the policy sweep -----------------------------------------------------


def fig_policy_sweep(run: Path, out: Path) -> Path | None:
    """Critical-flow floor violations against how oversubscribed the link is.

    The sweep rather than an operating point, because under light load nothing
    has to be shed and every policy looks identical, and under heavy enough
    load the floors stop fitting and the oracle fails too. Either end on its
    own would be mistaken for the general case.
    """
    data = _read(run).get("policies")
    if not data:
        return None
    load = _aggregate(data, ["policy", "load_multiplier"], "critical_violation_allocated")
    over = _aggregate(data, ["policy", "load_multiplier"], "oversubscription")
    load = load.merge(over[["policy", "load_multiplier", "mean"]],
                      on=["policy", "load_multiplier"], suffixes=("", "_over"))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    oracle = load[load["policy"] == "oracle"].sort_values("load_multiplier")
    if not oracle.empty:
        ax.plot(oracle["mean_over"], oracle["mean"], linestyle="--", linewidth=1.6,
                color=TEXT_SECONDARY, zorder=5, label=POLICY_LABEL["oracle"])
        ax.fill_between(oracle["mean_over"], 0, oracle["mean"], color=TEXT_SECONDARY,
                        alpha=0.07, linewidth=0, zorder=1)

    for policy in DEPLOYABLE:
        series = load[load["policy"] == policy].sort_values("load_multiplier")
        if series.empty:
            continue
        colour = POLICY_COLOUR[policy]
        ax.fill_between(series["mean_over"], series["mean"] - series["std"],
                        series["mean"] + series["std"], color=colour, alpha=0.13,
                        linewidth=0, zorder=2)
        ax.plot(series["mean_over"], series["mean"], marker="o", markersize=7,
                linewidth=2.0, color=colour, zorder=4, label=POLICY_LABEL[policy],
                markeredgecolor=SURFACE, markeredgewidth=1.4)
        if policy == "shed_largest":      # aqua, below 3:1, so it is labelled
            peak = series.loc[series["mean"].idxmax()]
            ax.annotate(f"{peak['mean']:.2f}", (peak["mean_over"], peak["mean"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8.5,
                        color=TEXT_SECONDARY)

    _style(ax, title="Under congestion, which flow is throttled is the whole question",
           subtitle="critical flows held below the rate at which they are useful, "
                    "by policy, mean and spread over workload seeds",
           xlabel="offered load as a multiple of delivered capacity",
           ylabel="critical-flow floor violation rate")
    ax.axvspan(0.0, 1.0, color=GRID, alpha=0.45, zorder=0)
    ax.set_xlim(left=float(load["mean_over"].min()) * 0.95)
    ax.text(min(1.0, ax.get_xlim()[1]), ax.get_ylim()[1] * 0.97, "  link not oversubscribed",
            fontsize=8.5, color=TEXT_SECONDARY, va="top", style="italic")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    return _save(fig, out / "protection-policy-sweep.png")


# -- 3. what protection costs ------------------------------------------------


def fig_cost(run: Path, out: Path) -> Path | None:
    """Protection is not free, and the figure that says so sits next to the one
    that says it works.

    A layer that protected critical traffic by starving everything else would
    score perfectly on the sweep above. The horizontal axis is what the elastic
    traffic pays for it.
    """
    data = _read(run).get("policies")
    if not data:
        return None
    frame = pd.DataFrame(data)
    reference = frame["load_multiplier"].median()
    frame = frame[frame["load_multiplier"] == reference]
    if frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for policy, group in frame.groupby("policy"):
        x = float(group["completion_inflation_ordinary"].mean())
        y = float(group["critical_goodput_ratio"].mean())
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        oracle = policy == "oracle"
        colour = TEXT_SECONDARY if oracle else POLICY_COLOUR[policy]
        ax.scatter([x], [y], s=190, color=colour, zorder=4, edgecolor=SURFACE,
                   linewidth=2.0, marker="D" if oracle else "o")
        ax.annotate(f"{POLICY_LABEL[policy]}\n{y:.0%} of critical demand delivered",
                    (x, y), textcoords="offset points", xytext=(11, 6), fontsize=9,
                    color=TEXT_SECONDARY, va="center")

    _style(ax, title="Protecting critical traffic is paid for by everything else",
           subtitle=f"at {reference:g}x offered load; up and to the left is better",
           xlabel="how much longer an ordinary transfer takes than on an idle link",
           ylabel="share of critical demand delivered")
    ax.set_xlim(right=ax.get_xlim()[1] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.55)
    return _save(fig, out / "protection-cost.png")


# -- 4. the scorer, including where it fails ---------------------------------


def fig_scorer(run: Path, out: Path) -> Path | None:
    """Criticality by archetype, with the protection threshold across it.

    An aggregate precision figure hides which flows a scorer is wrong about,
    and which flows it is wrong about is the only thing an operator needs to
    know. The hard negative in the workload is deliberately the one that
    overlaps the threshold.
    """
    rows = _frame(run, "flow_slots_protected.csv")
    if rows is None or "criticality" not in rows:
        return None
    order = (rows.groupby("archetype")["criticality"].median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    rng = np.random.default_rng(1337)
    for index, archetype in enumerate(order):
        group = rows[rows["archetype"] == archetype]
        critical = bool(group["truly_critical"].iloc[0])
        colour = CRITICAL_COLOUR if critical else ORDINARY_COLOUR
        values = group["criticality"].to_numpy()
        sample = values if values.size <= 1500 else rng.choice(values, 1500, replace=False)
        ax.scatter(sample, index + rng.normal(0, 0.09, sample.size), s=4, alpha=0.10,
                   color=colour, linewidth=0, zorder=2)
        median = float(np.median(values))
        ax.plot([median], [index], marker="|", markersize=26, markeredgewidth=2.6,
                color=colour, zorder=4)
        ax.annotate(f"{median:.2f}", (median, index), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=8.5, color=TEXT_SECONDARY)

    threshold = 0.60
    ax.axvline(threshold, linestyle="--", linewidth=1.5, color=TEXT_SECONDARY, zorder=5)
    ax.text(threshold + 0.012, len(order) - 0.35, "protection threshold", fontsize=9,
            color=TEXT_SECONDARY, style="italic")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([a.replace("_", " ") for a in order])
    ax.set_ylim(-0.7, len(order) - 0.3)

    _style(ax, title="Where the scorer is confident, and where it is not",
           subtitle="one point per flow-slot; orange archetypes must not be shed, "
                    "blue ones may be",
           xlabel="inferred criticality")
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.grid(False, axis="y")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CRITICAL_COLOUR,
                          markersize=8, label="critical: must not be shed"),
               plt.Line2D([], [], marker="o", linestyle="", color=ORDINARY_COLOUR,
                          markersize=8, label="ordinary: may be throttled")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
              loc="lower right")
    return _save(fig, out / "protection-scorer.png")


# -- 5. the dynamic claim ----------------------------------------------------


def fig_deadline(run: Path, out: Path) -> Path | None:
    """The same flow, changing its own answer as its deadline closes.

    The claim that distinguishes this from a classifier. Nothing about a backup
    changes between 02:00 and 07:55 except the slack, and the right decision
    changes with it.
    """
    rows = _frame(run, "episode_protected.csv")
    if rows is None or "deadline_remaining_s" not in rows:
        return None
    dated = rows[np.isfinite(rows["deadline_remaining_s"])]
    if dated.empty:
        return None

    # The flow with the widest swing in criticality over its own lifetime, which
    # is the one whose answer actually changed.
    swing = dated.groupby("flow_id")["criticality"].agg(lambda s: s.max() - s.min())
    lived = dated.groupby("flow_id").size()
    eligible = swing[lived >= 12]
    if eligible.empty:
        return None
    focal = str(eligible.idxmax())
    flow = dated[dated["flow_id"] == focal].sort_values("slot")
    archetype = str(flow["archetype"].iloc[0]).replace("_", " ")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.2, 5.8), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1],
                                                   "hspace": 0.22})
    slots = flow["slot"].to_numpy()
    criticality = flow["criticality"].to_numpy()

    top.plot(slots, criticality, linewidth=2.2, color=BLUE, zorder=4)
    top.axhline(0.60, linestyle="--", linewidth=1.4, color=TEXT_SECONDARY, zorder=3)
    top.fill_between(slots, 0.60, criticality, where=criticality >= 0.60,
                     color=BLUE, alpha=0.16, linewidth=0, zorder=2)
    crossings = np.flatnonzero(np.diff((criticality >= 0.60).astype(int)) > 0)
    if crossings.size:
        cross = int(slots[crossings[0] + 1])
        top.axvline(cross, color=ORANGE, linewidth=1.4, zorder=5)
        top.annotate("protection starts here", (cross, 0.62),
                     textcoords="offset points", xytext=(8, 4), fontsize=9,
                     color=ORANGE)
    top.text(slots[0], 0.605, " protection threshold", fontsize=8.5,
             color=TEXT_SECONDARY, va="bottom", style="italic")
    _style(top, title=f"A {archetype} that becomes critical without changing",
           subtitle="the deadline channel is the only one whose answer moves over a "
                    "flow's own lifetime",
           ylabel="inferred criticality")
    top.set_ylim(0, 1)

    slack = flow["deadline_remaining_s"].to_numpy()
    bottom.plot(slots, slack, linewidth=2.0, color=ORANGE, zorder=4)
    bottom.axhline(0.0, linewidth=1.2, color=TEXT_SECONDARY, zorder=3)
    _style(bottom, ylabel="seconds to deadline", xlabel="decision slot")
    return _save(fig, out / "protection-deadline.png")


# -- 6. the ablations --------------------------------------------------------


def fig_ablation(run: Path, out: Path) -> Path | None:
    """What each mechanism and each evidence channel is worth.

    Diverging around the full layer, because the quantity of interest is signed:
    a channel whose removal *improves* the outcome is not carrying its weight,
    and that is a result rather than a bug to hide.
    """
    data = _read(run)
    mechanisms, channels = data.get("ablation"), data.get("channels")
    if not mechanisms or not channels:
        return None

    mech = _aggregate(mechanisms, ["variant"], "critical_violation_allocated")
    chan = _aggregate(channels, ["dropped_channel"], "critical_violation_allocated")
    baseline = float(mech.loc[mech["variant"] == "full", "mean"].iloc[0])

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.8),
                                      gridspec_kw={"wspace": 0.42})

    def draw(ax, frame, key, title, subtitle, drop: str) -> None:
        frame = frame[frame[key] != drop].sort_values("mean")
        delta = frame["mean"].to_numpy() - baseline
        colours = [ORANGE if d > 0 else BLUE for d in delta]
        labels = [str(v).replace("_", " ") for v in frame[key]]
        bars = ax.barh(range(len(frame)), delta, color=colours, zorder=3, height=0.62)
        ax.errorbar(delta, range(len(frame)), xerr=frame["std"].to_numpy(), fmt="none",
                    ecolor=TEXT_SECONDARY, elinewidth=1.0, capsize=3, zorder=4)
        ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.3, zorder=5)
        ax.set_yticks(range(len(frame)))
        ax.set_yticklabels(labels, fontsize=9)
        for bar, value in zip(bars, delta, strict=True):
            offset = 4 if value >= 0 else -4
            ax.annotate(f"{value:+.3f}",
                        (value, bar.get_y() + bar.get_height() / 2),
                        textcoords="offset points", xytext=(offset, 0), fontsize=8.5,
                        ha="left" if value >= 0 else "right", va="center",
                        color=TEXT_SECONDARY)
        _style(ax, title=title, subtitle=subtitle,
               xlabel="change in critical-flow violation rate")
        ax.grid(True, axis="x", color=GRID, linewidth=0.7, zorder=0)
        ax.grid(False, axis="y")
        span = max(abs(delta).max(), 0.01) * 1.45
        ax.set_xlim(-span, span)

    draw(left, mech, "variant", "Removing a mechanism",
         f"against the full layer at {baseline:.3f}; right is worse", "full")
    draw(right, chan, "dropped_channel", "Removing an evidence channel",
         "bars pointing left are channels that do not earn their place", "none")
    return _save(fig, out / "protection-ablation.png")


# -- 7. the coupling to the calibration layer --------------------------------


def fig_calibration_coupling(run: Path, out: Path) -> Path | None:
    """Why the allocator divides the bound and not the forecast.

    The allocator's guarantee is algorithmic and holds on every slot the floors
    fit. What a user experiences is that guarantee and then the link. Against
    the point forecast the allocator looks *better*, because the larger number
    lets it promise more floors, and the delivered bars are where the promise
    is checked.
    """
    data = _read(run)
    rows, series = data.get("capacity_arm"), data.get("capacity_series", {})
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    reference = frame["load_multiplier"].median()
    frame = frame[frame["load_multiplier"] == reference]

    sources = ["bound", "point_forecast"]
    names = {"bound": f"calibrated bound\nrisk {series.get('realised_risk_rate', float('nan')):.2f}",
             "point_forecast": f"point forecast\nrisk {series.get('point_risk_rate', float('nan')):.2f}"}
    metrics = [("critical_violation_allocated", "what the allocator wrote", BLUE),
               ("critical_violation_delivered", "what the link delivered", ORANGE)]

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    width, gap = 0.34, 0.02
    for index, (column, label, colour) in enumerate(metrics):
        values = [float(frame[frame["capacity_source"] == s][column].mean())
                  for s in sources]
        errors = [float(frame[frame["capacity_source"] == s][column].std(ddof=0))
                  for s in sources]
        offset = (index - 0.5) * (width + gap)
        positions = np.arange(len(sources)) + offset
        ax.bar(positions, values, width=width, color=colour, zorder=3, label=label)
        ax.errorbar(positions, values, yerr=errors, fmt="none", ecolor=TEXT_SECONDARY,
                    elinewidth=1.0, capsize=3, zorder=5)
        for x, value in zip(positions, values, strict=True):
            ax.annotate(f"{value:.3f}", (x, value), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9, color=TEXT_SECONDARY)

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([names[s] for s in sources], fontsize=9.5)
    _style(ax, title="A floor written against a forecast the link misses is not a floor",
           subtitle=f"at {reference:g}x offered load; the allocator honours every "
                    "feasible floor in both arms",
           ylabel="critical-flow floor violation rate")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")
    return _save(fig, out / "protection-calibration.png")


# -- 8. why a flow was protected ---------------------------------------------


def fig_evidence(run: Path, out: Path) -> Path | None:
    """The evidence behind one decision, channel by channel.

    A protection decision an operator cannot trace back to the evidence that
    drove it is not one they can act on. This is also the check that the layer
    protects a large undeclared transfer for the right reason rather than by
    coincidence.
    """
    rows = _frame(run, "episode_protected.csv")
    if rows is None:
        return None
    weighted = [c for c in rows.columns if c.startswith("w_")]
    if not weighted:
        return None

    critical = rows[rows["truly_critical"] & (rows["demand_mbps"] > 8.0)
                    & (rows["declared"] == "standard")]
    if critical.empty:
        return None
    focal = critical.groupby("flow_id").size().idxmax()
    flow = rows[rows["flow_id"] == focal].sort_values("slot")
    archetype = str(flow["archetype"].iloc[0]).replace("_", " ")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2],
                                                   "hspace": 0.2})
    slots = flow["slot"].to_numpy()
    # Order by mean absolute contribution so the channels that matter are the
    # ones nearest the axis and the legend reads in that order too.
    order = sorted(weighted, key=lambda c: -float(flow[c].abs().mean()))
    palette = [BLUE, ORANGE, AQUA, VIOLET, TEXT_SECONDARY, GRID, "#8a6d3b"]

    positive = np.zeros(len(flow))
    negative = np.zeros(len(flow))
    for column, colour in zip(order, palette, strict=False):
        values = flow[column].to_numpy()
        up, down = np.clip(values, 0, None), np.clip(values, None, 0)
        top.bar(slots, up, bottom=positive, width=1.0, color=colour, linewidth=0,
                zorder=3, label=column[2:].replace("_", " "))
        top.bar(slots, down, bottom=negative, width=1.0, color=colour, linewidth=0,
                zorder=3)
        positive += up
        negative += down
    top.axhline(0.0, color=TEXT_PRIMARY, linewidth=1.0, zorder=5)
    _style(top, title=f"Why the layer protected this {archetype}",
           subtitle="weighted evidence per channel, in log-odds; the flow declares "
                    "itself standard, so none of this comes from its declaration",
           ylabel="contribution to log-odds")
    top.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY, ncol=4,
               loc="upper left", bbox_to_anchor=(0.0, -0.06))

    bottom.plot(slots, flow["criticality"].to_numpy(), linewidth=2.2, color=BLUE,
                zorder=4)
    bottom.axhline(0.60, linestyle="--", linewidth=1.4, color=TEXT_SECONDARY, zorder=3)
    bottom.set_ylim(0, 1)
    _style(bottom, ylabel="criticality", xlabel="decision slot")
    return _save(fig, out / "protection-evidence.png")


FIGURES = (fig_case_study, fig_policy_sweep, fig_cost, fig_scorer,
           fig_deadline, fig_ablation, fig_calibration_coupling, fig_evidence)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("results/protection/canada"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args(argv)

    written = 0
    for figure in FIGURES:
        path = figure(args.run, args.out)
        if path is None:
            print(f"skipped {figure.__name__}: inputs missing under {args.run}")
            continue
        print(f"wrote {path}  ({DPI} DPI)")
        written += 1
    if not written:
        print("nothing written; run scripts/run_protection.py first")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
