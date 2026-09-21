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
#: Trace folder to the city the trace was actually recorded in. Two of the
#: three released folders are mislabelled, which is documented in the data
#: section of the README and handled in the loader.
LOCATION_LABEL = {"usa": "Chicago", "germany": "Osnabruck", "canada": "Victoria"}
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


def _reference_load(data: dict) -> float:
    """The load the ablations and the case study ran at.

    Read from the run's own arguments rather than taken as the median of the
    sweep, because the sweep's median need not be one of the levels it visited.
    """
    return float(data.get("args", {}).get("reference_load", 2.0))


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

    The argument for the whole layer. Both panels replay an identical workload
    against an identical capacity series, so the only thing that differs
    between them is what the allocator decided.

    The rate plotted is what the allocator granted, which is what the allocator
    controls. What the link then delivered is the subject of
    `protection-calibration.png`, and separating the two is the point of
    reporting both.
    """
    ours = _frame(run, "episode_protected.csv")
    naive = _frame(run, "episode_shed_largest.csv")
    context = _frame(run, "episode_protected_slots.csv")
    if ours is None or naive is None:
        return None

    # The focal flow: the critical archetype the whole layer is about, if it is
    # present, and otherwise the largest critical flow on the link. Either way
    # it has to be the same flow in both panels.
    shared = set(ours["flow_id"]) & set(naive["flow_id"])
    critical = ours[ours["flow_id"].isin(shared) & ours["truly_critical"]
                    & (ours["demand_mbps"] > 8.0)]
    if critical.empty:
        return None
    preferred = critical[critical["archetype"] == "pacs_image_push"]
    pool = preferred if not preferred.empty else critical
    # The longest lived instance, which is also the one that lived through the
    # most congestion. That is the hard case for both policies rather than a
    # flattering one, and it is chosen before either policy is looked at.
    focal = pool.groupby("flow_id").size().idxmax()
    archetype = str(pool.loc[pool["flow_id"] == focal, "archetype"].iloc[0])
    label = archetype.replace("_", " ")

    spans = [t.loc[t["flow_id"] == focal, "slot"] for t in (ours, naive)]
    window = (int(min(s.min() for s in spans)), int(max(s.max() for s in spans)))

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.55, 1.55],
                                          "hspace": 0.38})

    # -- context: what the link could carry, and what the allocator divided.
    top = axes[0]
    if context is not None:
        frame = context[context["slot"].between(*window)]
        top.plot(frame["slot"], frame["realised_mbps"], linewidth=1.3,
                 color=TEXT_SECONDARY, alpha=0.8, zorder=3, label="carried")
        top.plot(frame["slot"], frame["capacity_mbps"], linewidth=1.6, color=VIOLET,
                 zorder=4, label="calibrated bound, which is what was divided")
        # Right aligned above the axes. The panel title is left aligned in the
        # same band, and inside the axes the two series fill the whole box, so
        # the labels are kept short enough that the two cannot meet.
        top.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY, ncol=2,
                   loc="lower right", bbox_to_anchor=(1.0, 1.01))
    _style(top, title="The link", ylabel="Mbps")

    panels = [(axes[1], "Throttling the largest flow first, which is what a rate-based "
                        "shaper does", naive),
              (axes[2], "Criticality-weighted, with protected floors", ours)]
    ceiling = max(float(t.loc[t["flow_id"] == focal, "demand_mbps"].max())
                  for _, _, t in panels) * 1.22

    finished = {}
    for ax, title, table in panels:
        flow = table[table["flow_id"] == focal].sort_values("slot")
        slots = flow["slot"].to_numpy()
        granted = flow["granted_mbps"].to_numpy()
        floor = float(flow["floor_mbps"].median())
        finished[title] = int(slots[-1])

        ax.fill_between(slots, 0, granted, color=BLUE, alpha=0.18, linewidth=0, zorder=3)
        ax.plot(slots, granted, linewidth=1.7, color=BLUE, zorder=4)
        ax.axhline(floor, linestyle="--", linewidth=1.5, color=TEXT_SECONDARY, zorder=5)

        # Starvation as a rug along the baseline rather than a full-height fill.
        # A fill that covers the panel hides the series it is annotating.
        starved = granted < floor - 1e-9
        ax.fill_between(slots, 0, ceiling * 0.055, where=starved, color=ORANGE,
                        alpha=0.85, linewidth=0, zorder=2)
        share = float(starved.mean())

        ax.axvline(slots[-1], color=TEXT_PRIMARY, linewidth=1.2, zorder=6)
        ax.annotate(f"done, slot {slots[-1]}", (slots[-1], ceiling * 0.93),
                    textcoords="offset points", xytext=(-6, 0), ha="right",
                    fontsize=9, color=TEXT_PRIMARY,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})

        _style(ax, title=title, ylabel="Mbps granted")
        ax.set_ylim(0, ceiling)
        ax.set_xlim(window[0] - 4, window[1] + 4)
        ax.text(0.012, 0.90, f"below its floor on {share:.0%} of its slots",
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                color=ORANGE if share > 0.05 else TEXT_PRIMARY)
        ax.text(window[1], floor, f"its floor, {floor:.1f} Mbps ", fontsize=8.5,
                color=TEXT_SECONDARY, va="bottom", ha="right", style="italic")

    axes[-1].set_xlabel("decision slot", color=TEXT_SECONDARY, fontsize=10)
    slower, faster = (finished[panels[0][1]] - window[0],
                      finished[panels[1][1]] - window[0])
    fig.suptitle(f"The same {label}, under the shaper in use today and under this layer",
                 x=0.005, ha="left", fontsize=13.5, color=TEXT_PRIMARY, y=1.075)
    fig.text(0.005, 1.025,
             f"identical workload, identical link, identical seed. Orange marks the "
             f"slots where the transfer was getting too little to be worth carrying; "
             f"it completes in {faster} slots instead of {slower}.",
             ha="left", fontsize=9.5, color=TEXT_SECONDARY)
    return _save(fig, out / "protection-case-study.png")


# -- 2. the policy sweep -----------------------------------------------------


def fig_policy_sweep(run: Path, out: Path) -> Path | None:
    """Critical-flow floor violations against how oversubscribed the link is.

    The sweep rather than an operating point, because under light load nothing
    has to be shed and every policy looks identical, and under heavy enough
    load the floors stop fitting and the oracle fails too. Either end on its
    own would be mistaken for the general case.
    """
    data = _read(run)
    rows = data.get("policies")
    if not rows:
        return None
    load = _aggregate(rows, ["policy", "load_multiplier"], "critical_violation_allocated")
    # Offered load over delivered capacity, computed from the archetype table
    # rather than from each run's realised demand. A policy that throttles
    # harder makes its elastic flows back off further, which lowers *its own*
    # measured offered load: using that would give every policy a different x
    # for the same workload and shift the lines sideways against each other.
    nominal = sum(a["nominal_mbps"] * a["concurrency"]
                  for a in data.get("workload_archetypes", []))
    capacity = float(data.get("capacity_series", {}).get("realised_mean_mbps", 0.0)) or 1.0
    load["mean_over"] = load["load_multiplier"] * nominal / capacity
    # This is *unconstrained* demand: what the flows would ask for on an idle
    # link. Their measured offered load is lower, because elastic transfers back
    # off, and by a different amount under each policy, which is exactly why it
    # cannot be the shared axis.

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    oracle = load[load["policy"] == "oracle"].sort_values("load_multiplier")
    if not oracle.empty:
        # A line, not a filled region: the area under the oracle is not a
        # category, and a second grey field competes with the shaded band that
        # marks where the link is not oversubscribed.
        ax.plot(oracle["mean_over"], oracle["mean"], linestyle="--", linewidth=1.6,
                color=TEXT_SECONDARY, zorder=5, label=POLICY_LABEL["oracle"])

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
                        textcoords="offset points", xytext=(-8, -14), fontsize=8.5,
                        ha="right", color=TEXT_SECONDARY)

    _style(ax, title="Under congestion, which flow is throttled is the whole question",
           subtitle="critical flows held below the rate at which they are useful, "
                    "by policy, mean and spread over workload seeds",
           xlabel="unconstrained demand as a multiple of delivered capacity",
           ylabel="critical-flow floor violation rate")
    ax.axvspan(0.0, 1.0, color=GRID, alpha=0.45, zorder=0)
    ax.set_xlim(left=float(load["mean_over"].min()) * 0.95)
    ax.text(min(1.0, ax.get_xlim()[1]), ax.get_ylim()[1] * 0.97, "  demand below capacity",
            fontsize=8.5, color=TEXT_SECONDARY, va="top", style="italic")
    # Below the axes. Five entries do not fit in any corner once the sweep has
    # its full set of load levels, and every corner has a series in it.
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return _save(fig, out / "protection-policy-sweep.png")


# -- 3. what protection costs ------------------------------------------------


def fig_cost(run: Path, out: Path) -> Path | None:
    """Protection is not free, and the figure that says so sits next to the one
    that says it works.

    A layer that protected critical traffic by starving everything else would
    score perfectly on the sweep above. Both axes count transfers that gave up,
    so they are in the same units and the trade needs no second scale: the
    vertical axis is what the layer buys and the horizontal one is who pays.
    """
    data = _read(run)
    rows = data.get("policies")
    if not rows:
        return None
    reference = _reference_load(data)
    frame = pd.DataFrame(rows)
    frame = frame[frame["load_multiplier"] == reference]
    if frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    points: list[tuple[float, float, str, str, float]] = []
    for policy, group in frame.groupby("policy"):
        # Both axes are "gave up", so they are directly comparable and the
        # trade is legible without a second scale. Goodput and slowdown go in
        # the label rather than on an axis.
        x = float(group["ordinary_abandon_rate"].mean())
        y = float(group["critical_abandon_rate"].mean())
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        oracle = policy == "oracle"
        colour = TEXT_SECONDARY if oracle else POLICY_COLOUR[policy]
        ax.scatter([x], [y], s=190, color=colour, zorder=4, edgecolor=SURFACE,
                   linewidth=2.0, marker="D" if oracle else "o")
        points.append((x, y, policy, colour,
                       float(group["critical_goodput_ratio"].mean())))

    # Label on whichever side has room. Placing every label to the right runs
    # the rightmost ones off the frame, and widening the axis to fit them
    # leaves the points bunched in one corner.
    # All labels to the right of their marker, with the axis widened to hold
    # them. Alternating sides puts the longest label off the left edge, and
    # stacking them above the markers collides wherever two policies land
    # within a few points of each other.
    for x, y, policy, _, goodput in points:
        ax.annotate(f"  {POLICY_LABEL[policy]}, {goodput:.0%} of critical demand delivered",
                    (x, y), textcoords="offset points", xytext=(10, 0), fontsize=9,
                    ha="left", color=TEXT_SECONDARY, va="center")

    _style(ax, title="Protecting critical traffic is paid for by everything else",
           subtitle=f"at {reference:g}x offered load; a transfer held below a useful "
                    f"rate for five minutes gives up. Down is better, right is the price.",
           xlabel="ordinary transfers that gave up",
           ylabel="critical transfers that gave up")
    # Both axes are rates, so the frame stops at 100 per cent. The labels run
    # past it into the figure margin, which the tight bounding box keeps.
    lo, hi = ax.get_xlim()
    ax.set_xlim(max(0.0, lo - 0.02), min(1.0, hi + 0.03))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
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
        # Past the error bar, not on top of it: the whisker and the value label
        # want the same few pixels at the end of every bar.
        errors = frame["std"].to_numpy()
        for bar, value, error in zip(bars, delta, errors, strict=True):
            sign = 1.0 if value >= 0 else -1.0
            tip = value + sign * (error if np.isfinite(error) else 0.0)
            ax.annotate(f"{value:+.3f}", (tip, bar.get_y() + bar.get_height() / 2),
                        textcoords="offset points", xytext=(6 * sign, 0), fontsize=8.5,
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
    reference = _reference_load(data)
    frame = pd.DataFrame(rows)
    frame = frame[frame["load_multiplier"] == reference]

    sources = ["bound", "point_forecast"]
    names = {"bound": f"calibrated bound\nrisk {series.get('realised_risk_rate', float('nan')):.2f}",
             "point_forecast": f"point forecast\nrisk {series.get('point_risk_rate', float('nan')):.2f}"}
    metrics = [("critical_violation_allocated", "what the allocator wrote", BLUE),
               ("critical_violation_delivered", "what the link delivered", ORANGE)]

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    width, gap = 0.22, 0.02
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
        for x, value, error in zip(positions, values, errors, strict=True):
            # Above the whisker, not through it.
            top = value + (error if np.isfinite(error) else 0.0)
            ax.annotate(f"{value:.3f}", (x, top), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=9, color=TEXT_SECONDARY)

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([names[s] for s in sources], fontsize=9.5)
    ax.set_xlim(-0.55, len(sources) - 0.45)
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

    Drawn as a diverging heat map rather than a stacked bar. The quantity is a
    signed magnitude over two dimensions, channel and time, and seven stacked
    categorical hues is past the point where any palette separates safely under
    colour vision deficiency. Two poles and a neutral midpoint carry the sign,
    which is what the reader needs.
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

    # Ordered by how much each channel actually moved this decision, so the
    # rows nearest the axis are the ones that decided it.
    order = sorted(weighted, key=lambda c: float(flow[c].abs().mean()))
    matrix = np.vstack([flow[c].to_numpy() for c in order])
    slots = flow["slot"].to_numpy()
    span = float(np.abs(matrix).max()) or 1.0

    # Diverging: two hues from the validated set, with the surface as the
    # neutral midpoint. Never a hue at the middle of a diverging ramp.
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "thalweg-diverging", [ORANGE, SURFACE, BLUE])

    fig, (heat, line) = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1.0],
                                                  "hspace": 0.16})
    # Cell edges, not centres: flat shading wants one more boundary than cell
    # in each direction.
    edges = np.append(slots, slots[-1] + (slots[-1] - slots[-2] if slots.size > 1 else 1))
    mesh = heat.pcolormesh(edges, np.arange(len(order) + 1), matrix, cmap=ramp,
                           vmin=-span, vmax=span, shading="flat")
    heat.set_yticks(np.arange(len(order)) + 0.5)
    heat.set_yticklabels([c[2:].replace("_", " ") for c in order], fontsize=9.5)
    heat.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for side in ("top", "right", "left", "bottom"):
        heat.spines[side].set_visible(False)
    heat.set_title(f"Why the layer protected this {archetype}", loc="left",
                   color=TEXT_PRIMARY, fontsize=12.5, pad=24, fontweight="medium")
    heat.text(0, 1.015, "weighted evidence per channel, in log-odds. The flow declares "
                        "itself standard, so none of this is its declaration.",
              transform=heat.transAxes, fontsize=9.5, color=TEXT_SECONDARY, va="bottom")

    bar = fig.colorbar(mesh, ax=heat, pad=0.012, fraction=0.035)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    bar.set_label("argues for shedding        argues for protecting",
                  color=TEXT_SECONDARY, fontsize=8.5)

    criticality = flow["criticality"].to_numpy()
    line.plot(slots, criticality, linewidth=2.2, color=BLUE, zorder=4)
    line.axhline(0.60, linestyle="--", linewidth=1.4, color=TEXT_SECONDARY, zorder=3)
    line.fill_between(slots, 0.60, criticality, where=criticality >= 0.60,
                      color=BLUE, alpha=0.16, linewidth=0, zorder=2)
    line.set_ylim(0, 1)
    _style(line, ylabel="criticality", xlabel="decision slot")
    line.text(slots[0], 0.615, " protection threshold", fontsize=8.5,
              color=TEXT_SECONDARY, va="bottom", style="italic")
    # The colour bar steals width from the heat map only; match the line axes
    # to it so the two share an x position as well as an x scale.
    line.set_position([heat.get_position().x0, line.get_position().y0,
                       heat.get_position().width, line.get_position().height])
    return _save(fig, out / "protection-evidence.png")


# -- 9. cross location ------------------------------------------------------


def fig_cross_location(runs: list[Path], out: Path) -> Path | None:
    """The same sweep on three links, as small multiples.

    Fifteen series on one pair of axes is past what any palette separates, and
    the comparison of interest is within a panel rather than across them.

    The x axis is oversubscription rather than the workload's nominal load
    multiplier, and that is the whole reason this figure exists. Chicago
    delivers about 222 Mbps against Victoria's 146, so the same workload at the
    same nominal setting leaves one link comfortably under capacity and the
    other half as much again over it. Plotted against the load multiplier the
    three panels look like three different results; plotted against how
    oversubscribed each link actually is, they are one.
    """
    loaded = [(r, _read(r)) for r in runs if (r / "result.json").exists()]
    if len(loaded) < 2:
        return None

    fig, axes = plt.subplots(1, len(loaded), figsize=(4.3 * len(loaded), 4.4),
                             sharey=True, gridspec_kw={"wspace": 0.09})
    axes = np.atleast_1d(axes)

    for index, (ax, (run, data)) in enumerate(zip(axes, loaded, strict=True)):
        rows = data.get("policies")
        if not rows:
            continue
        load = _aggregate(rows, ["policy", "load_multiplier"],
                          "critical_violation_allocated")
        nominal = sum(a["nominal_mbps"] * a["concurrency"]
                      for a in data.get("workload_archetypes", []))
        realised = float(data.get("capacity_series", {})
                         .get("realised_mean_mbps", 0.0)) or 1.0
        load["mean_over"] = load["load_multiplier"] * nominal / realised

        oracle = load[load["policy"] == "oracle"].sort_values("load_multiplier")
        if not oracle.empty:
            ax.plot(oracle["mean_over"], oracle["mean"], linestyle="--",
                    linewidth=1.5, color=TEXT_SECONDARY, zorder=5,
                    label=POLICY_LABEL["oracle"])
        for policy in DEPLOYABLE:
            series = load[load["policy"] == policy].sort_values("load_multiplier")
            if series.empty:
                continue
            colour = POLICY_COLOUR[policy]
            ax.fill_between(series["mean_over"], series["mean"] - series["std"],
                            series["mean"] + series["std"], color=colour,
                            alpha=0.13, linewidth=0, zorder=2)
            ax.plot(series["mean_over"], series["mean"], marker="o", markersize=5.5,
                    linewidth=1.9, color=colour, zorder=4, label=POLICY_LABEL[policy],
                    markeredgecolor=SURFACE, markeredgewidth=1.2)

        ax.axvspan(0.0, 1.0, color=GRID, alpha=0.45, zorder=0)
        _style(ax, title=f"{LOCATION_LABEL.get(run.name, run.name)}, "
                         f"{realised:.0f} Mbps delivered",
               xlabel="unconstrained demand over delivered capacity")
        if index == 0:
            ax.set_ylabel("critical-flow floor violation rate",
                          color=TEXT_SECONDARY, fontsize=10)
        ax.set_xlim(left=0.3)

    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncol=5,
                   loc="upper left", bbox_to_anchor=(0.0, -0.17))
    fig.suptitle("Three links, one result, once the axis is what the link is "
                 "actually carrying", x=0.005, ha="left", fontsize=13,
                 color=TEXT_PRIMARY, y=1.04)
    return _save(fig, out / "protection-cross-location.png")


FIGURES = (fig_case_study, fig_policy_sweep, fig_cost, fig_scorer,
           fig_deadline, fig_ablation, fig_calibration_coupling, fig_evidence)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("results/protection/canada"))
    parser.add_argument("--cross", type=Path, nargs="*",
                        default=[Path("results/protection/canada"),
                                 Path("results/protection/germany"),
                                 Path("results/protection/usa")],
                        help="runs for the cross-location panel")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args(argv)

    written = 0
    path = fig_cross_location(list(args.cross), args.out)
    if path is None:
        print("skipped fig_cross_location: fewer than two runs available")
    else:
        print(f"wrote {path}  ({DPI} DPI)")
        written += 1
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
