"""Figures.

Light mode only and deliberately so: these are paper figures, rendered to PDF
and printed on white.

The categorical palette is the four slot subset blue / orange / aqua / violet,
validated for the adjacent pairlist in light mode (worst adjacent CVD deltaE
9.2 deutan, normal vision 27.6). Aqua sits below 3:1 against the surface, so
every bar carries a visible value label rather than relying on fill alone.
Colour follows the method, never its rank, so a figure that drops a method does
not repaint the others.

No figure in this module uses two y scales. Where MAE and OverRate need to
appear together they get their own panels, because a shared frame with two
scales can be made to show any relationship you like.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Method identity, fixed. Never cycled, never reassigned by rank.
SERIES_COLORS: dict[str, str] = {
    "point": "#4a3aa7",
    "global_conformal": "#2a78d6",
    "bgcfqs": "#eb6834",
    "regime_conformal": "#1baf7a",
    "regime_bgcfqs": "#eb6834",
}
FALLBACK_ORDER = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"

SLICE_LABELS = {
    "global": "All",
    "P30": "High-risk P30\n(lowest 30% throughput)",
    "P10": "Severe-risk P10\n(lowest 10% throughput)",
}


def _style(ax: plt.Axes, *, ylabel: str = "", xlabel: str = "", title: str = "") -> None:
    """Recessive frame. The data should be the only thing with weight."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left", pad=12)


def _color(method: str, index: int) -> str:
    return SERIES_COLORS.get(method, FALLBACK_ORDER[index % len(FALLBACK_ORDER)])


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def conditional_over_rate(
    rates: dict[str, dict[str, float]],
    epsilon: float,
    path: str | Path,
    *,
    title: str = "Overestimation rate by risk slice",
    subtitle: str | None = None,
) -> Path:
    """The motivation figure.

    `rates` maps method name to slice name to OverRate. The budget is drawn as
    a reference line, because the entire argument is the distance between a
    global rate that sits on it and a conditional rate that does not.

    Every bar is labelled. The palette warns on aqua contrast and a value label
    is the relief, and in a paper figure the reader wants the number anyway.
    """
    slices = ["global", "P30", "P10"]
    methods = list(rates)
    n = len(methods)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    positions = np.arange(len(slices))
    # A 2px surface gap between adjacent bars, expressed as a width fraction.
    width = 0.78 / max(n, 1)

    for index, method in enumerate(methods):
        offset = (index - (n - 1) / 2) * width
        values = [rates[method].get(s, np.nan) for s in slices]
        bars = ax.bar(positions + offset, values, width * 0.94,
                      label=method.replace("_", " "), color=_color(method, index), zorder=3)
        for bar, value in zip(bars, values, strict=True):
            if not np.isfinite(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)

    # The budget label gets its own margin past the last group, and the line
    # stops short of it. Anywhere inside the plot the label eventually collides
    # with a bar or a value label, and which one depends on the data, so
    # reserving space is the only fix that holds for every input.
    left, right = -0.6, len(slices) - 1 + 0.5
    ax.hlines(epsilon, left, right, color=TEXT_PRIMARY, linewidth=1.4,
              linestyle="--", zorder=4)
    ax.set_xlim(left, right + 0.5)
    ax.text(right + 0.06, epsilon, f"$\\epsilon$ = {epsilon:.2f}",
            fontsize=9, color=TEXT_PRIMARY, ha="left", va="center")

    ax.set_xticks(positions)
    ax.set_xticklabels([SLICE_LABELS[s] for s in slices])
    _style(ax, ylabel="OverRate", title=title)
    if subtitle:
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left", pad=26)
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                color=TEXT_SECONDARY, va="bottom")
    ax.set_ylim(0, max(1.0, max(
        (v for method in rates.values() for v in method.values() if np.isfinite(v)),
        default=1.0) * 1.18))
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncols=min(n, 4),
              loc="upper left", bbox_to_anchor=(0, -0.16))
    return _save(fig, path)


def epsilon_sweep(table: pd.DataFrame, path: str | Path,
                  *, slice_name: str = "P30",
                  title: str = "Risk budget sweep") -> Path:
    """OverRate and MAE against the budget, in separate panels.

    Two panels rather than two y axes. The question the sweep answers is
    whether the advantage widens as the budget tightens, and that is read off
    the left panel; the right panel exists so a risk gain bought by a
    conservative collapse in accuracy cannot be hidden.

    Expects columns: method, epsilon, OverRate_<slice>, MAE.
    """
    column = f"OverRate_{slice_name}"
    if column not in table.columns:
        raise KeyError(f"{column!r} not in table; columns are {list(table.columns)}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for index, (method, part) in enumerate(table.groupby("method", sort=False)):
        part = part.sort_values("epsilon")
        color = _color(str(method), index)
        label = str(method).replace("_", " ")
        axes[0].plot(part["epsilon"], part[column], marker="o", markersize=6,
                     linewidth=2.0, color=color, label=label, zorder=3)
        axes[1].plot(part["epsilon"], part["MAE"], marker="o", markersize=6,
                     linewidth=2.0, color=color, label=label, zorder=3)

    # y = x is the budget the method was asked to hold. Anything above the line
    # is a method failing its own contract on this slice.
    limits = [float(table["epsilon"].min()), float(table["epsilon"].max())]
    axes[0].plot(limits, limits, color=TEXT_SECONDARY, linewidth=1.2, linestyle="--",
                 zorder=2)
    axes[0].text(limits[1], limits[1], "  budget", fontsize=9, color=TEXT_SECONDARY,
                 va="center")

    _style(axes[0], xlabel="risk budget $\\epsilon$",
           ylabel=f"OverRate ({SLICE_LABELS.get(slice_name, slice_name).split(chr(10))[0]})",
           title=title)
    _style(axes[1], xlabel="risk budget $\\epsilon$", ylabel="MAE (Mbps)",
           title="Accuracy cost")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def granularity_ablation(table: pd.DataFrame, path: str | Path,
                         *, slice_name: str = "P30",
                         title: str = "Which conditioning variables carry the gain") -> Path:
    """OverRate by regime granularity.

    A flat bar chart here is a clean negative result: if phase alone gets most
    of the gain and satellite geometry adds nothing, the figure says so and the
    writeup reports it.

    Expects columns: axes, OverRate_<slice>, MAE.
    """
    column = f"OverRate_{slice_name}"
    order = ["global", "phase", "phase+elevation", "phase+elevation+distance+candidates"]
    table = table.copy()
    table["axes"] = table["axes"].fillna("global").replace("", "global")
    present = [a for a in order if a in set(table["axes"])]
    present += [a for a in table["axes"].unique() if a not in present]
    table = table.set_index("axes").reindex(present).reset_index()

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    positions = np.arange(len(table))
    bars = ax.bar(positions, table[column], 0.62, color=_color("regime_conformal", 2), zorder=3)
    for bar, value in zip(bars, table[column], strict=True):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY)

    ax.set_xticks(positions)
    ax.set_xticklabels([a.replace("+", "\n+") for a in table["axes"]], fontsize=9)
    _style(ax, ylabel=f"OverRate ({slice_name})", xlabel="regime axes", title=title)
    return _save(fig, path)


def regime_over_rate_spread(per_regime: pd.DataFrame, epsilon: float, path: str | Path,
                            *, title: str = "OverRate across regimes") -> Path:
    """Every regime's OverRate against the budget, sized by sample count.

    The picture the headline number hides. A global rate sitting exactly on the
    budget is consistent with half the regimes at double it.

    Expects the output of `eval.metrics.per_regime_metrics`.
    """
    if per_regime.empty:
        raise ValueError("no per regime metrics to plot")
    table = per_regime.sort_values("OverRate").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    sizes = 20 + 180 * table["n"] / max(table["n"].max(), 1)
    over = table["OverRate"] > epsilon
    ax.scatter(np.arange(len(table))[~over], table["OverRate"][~over], s=sizes[~over],
               color=_color("regime_conformal", 2), zorder=3,
               edgecolors=SURFACE, linewidths=1.2, label="within budget")
    ax.scatter(np.arange(len(table))[over], table["OverRate"][over], s=sizes[over],
               color="#e34948", zorder=3, edgecolors=SURFACE, linewidths=1.2,
               label="over budget")
    # Same reserved margin as the motivation figure, for the same reason.
    left, right = -0.6, len(table) - 0.4
    ax.hlines(epsilon, left, right, color=TEXT_PRIMARY, linewidth=1.4,
              linestyle="--", zorder=4)
    ax.set_xlim(left, right + max(len(table) * 0.06, 1.2))
    ax.text(right + 0.15, epsilon, f"$\\epsilon$ = {epsilon:.2f}", fontsize=9,
            color=TEXT_PRIMARY, ha="left", va="center")

    _style(ax, ylabel="OverRate", xlabel="regime, ordered by OverRate (marker area = n)",
           title=title)
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")
    return _save(fig, path)


def phase_histogram(reference, path: str | Path,
                    *, title: str = "Recovered 15 s scheduling phase") -> Path:
    """Evidence for the phase recovery.

    Publishing the histogram alongside the recovered offset is what separates
    a recovered reference from an assumed one. A flat histogram with a
    confident-looking offset printed on it is the failure this figure exposes.
    """
    if reference.histogram.size == 0:
        raise ValueError("phase reference carries no histogram")

    centres = 0.5 * (reference.bin_edges[:-1] + reference.bin_edges[1:])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(centres, reference.histogram, width=(centres[1] - centres[0]) * 0.88,
           color=_color("global_conformal", 0), zorder=3)
    ax.axvline(reference.offset_seconds, color="#eb6834", linewidth=2.0, zorder=4)
    ax.text(reference.offset_seconds, ax.get_ylim()[1] * 0.96,
            f"  offset {reference.offset_seconds:.2f} s ({reference.method})",
            color="#eb6834", fontsize=9, va="top")

    _style(ax, ylabel="detected edges", xlabel="phase within the 15 s period (s)", title=title)
    ax.text(0, 1.02, f"{reference.n_edges} edges, peak/uniform = {reference.confidence:.2f}",
            transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY)
    return _save(fig, path)


def admission_comparison(tables: dict[str, pd.DataFrame], path: str | Path,
                         *, title: str = "Dropped sessions by policy") -> Path:
    """Mean dropped sessions per slice, one group per policy.

    The metric that a user would notice. `tables` maps policy name to the
    output of `decide.admission.evaluate_admission`.
    """
    slices = ["all", "P30", "P10"]
    methods = list(tables)
    n = len(methods)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    positions = np.arange(len(slices))
    width = 0.78 / max(n, 1)

    for index, (method, table) in enumerate(tables.items()):
        lookup = table.set_index("slice")["mean_dropped"].to_dict()
        values = [lookup.get(s, np.nan) for s in slices]
        offset = (index - (n - 1) / 2) * width
        bars = ax.bar(positions + offset, values, width * 0.94, color=_color(method, index),
                      label=method.replace("_", " "), zorder=3)
        for bar, value in zip(bars, values, strict=True):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                        ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)

    ax.set_xticks(positions)
    ax.set_xticklabels(["All decisions", "High-risk P30", "Severe-risk P10"])
    _style(ax, ylabel="mean dropped sessions", title=title)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncols=min(n, 4),
              loc="upper left", bbox_to_anchor=(0, -0.14))
    return _save(fig, path)
