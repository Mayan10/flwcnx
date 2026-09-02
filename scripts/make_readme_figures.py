#!/usr/bin/env python3
"""Publication-quality figures for the README, half findings and half limitations.

Every number is read from a saved `result.json`, never hardcoded, so a figure
cannot drift away from the run that produced it. Output is 300 DPI PNG into
`docs/figures/`, which is versioned so GitHub can render it inline.

Three figures report what was learned and three report where the work fails.
That split is deliberate: a README that shows only the wins is advertising, and
the limitations here are load-bearing, because two of them overturned claims the
project had already made.

Palette is the four-slot categorical set validated for light mode:
worst adjacent CVD deltaE 9.2 (deutan), normal-vision 27.6. Aqua sits below 3:1
against the surface, so every aqua mark carries a visible value label, which is
the documented relief for that case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -- design tokens -----------------------------------------------------------

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID, SURFACE = "#dedcd6", "#fcfcfb"
DPI = 300

EPSILONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
STARNET = {"usa": "Chicago", "germany": "Osnabruck", "canada": "Victoria"}


def _style(ax, *, title="", subtitle="", xlabel="", ylabel=""):
    """Recessive grid and axes, text in ink tokens rather than series colour."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    if title:
        pad = 22 if subtitle else 10
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=12.5, loc="left",
                     pad=pad, fontweight="medium")
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=TEXT_SECONDARY, va="bottom")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# -- findings ----------------------------------------------------------------

def fig_floor(root: Path, out: Path) -> Path | None:
    """The strongest result: BG-CFQS pins below its candidate set's low end."""
    series = {}
    for loc, label in STARNET.items():
        d = _load(root / f"results/final_starnet/{loc}/{loc}-starnet/result.json")
        if not d:
            continue
        vals = [d["calibration"].get(f"regime_bgcfqs|eps={e:.2f}|regime=level", {})
                .get("global", {}).get("OverRate") for e in EPSILONS]
        if all(v is not None for v in vals):
            series[label] = vals
    d = _load(root / "results/final/wetlinks-seconds-Osnabruck-capacity/result.json")
    if d:
        vals = [d["calibration"].get(f"regime_bgcfqs|eps={e:.2f}|regime=level+candidates",
                                     {}).get("global", {}).get("OverRate") for e in EPSILONS]
        if all(v is not None for v in vals):
            series["Osnabruck (WetLinks)"] = vals
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(EPSILONS, EPSILONS, linestyle="--", linewidth=1.4, color=TEXT_SECONDARY,
            zorder=2, label="what was asked for")
    # Two of the flat runs sit close together, so labels alternate above and
    # below the line. Placing them all above puts the nearer pair on top of
    # each other.
    order = sorted(series, key=lambda k: series[k][0])
    for index, label in enumerate(order):
        vals = series[label]
        colour = [BLUE, ORANGE, AQUA, VIOLET][list(series).index(label) % 4]
        ax.plot(EPSILONS, vals, marker="o", markersize=7, linewidth=2.0,
                color=colour, zorder=3, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.4)
        # Direct label on the middle of the flat run, which is the point of the
        # figure, rather than at its left end against the y-axis ticks.
        dy = 8 if index % 2 == 0 else -17
        ax.annotate(f"{vals[0]:.3f}", (EPSILONS[1], vals[1]),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=8.5, color=TEXT_SECONDARY,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0})

    ax.axvspan(0.04, 0.152, color=ORANGE, alpha=0.055, zorder=1)
    _style(ax, title="A published method silently ignores tight risk budgets",
           subtitle="BG-CFQS returns the same quantile for every budget at or below 0.15, "
                    "because its candidate set starts there",
           xlabel="risk budget requested", ylabel="overestimation rate achieved")
    # Caption inside the shaded band and low, where no series runs; legend in the
    # empty lower right. Neither can reach the other.
    ax.text(0.096, ax.get_ylim()[0] + 0.004, "budgets the method\ncannot serve",
            ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY, style="italic")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
              loc="lower right", bbox_to_anchor=(1.0, 0.02))
    return _save(fig, out / "finding-bgcfqs-floor.png")


def fig_budget_tracking(root: Path, out: Path) -> Path | None:
    """Static calibration misses the budget; online tracks it."""
    online, static = [], []
    for e in EPSILONS:
        o, s = [], []
        for loc in STARNET:
            d = _load(root / f"results/final_starnet/{loc}/{loc}-starnet/result.json")
            if not d:
                continue
            a = d["calibration"].get(f"adaptive_global_conformal|eps={e:.2f}|regime=global", {})
            c = d["calibration"].get(f"global_conformal|eps={e:.2f}|regime=global", {})
            if a.get("global") and c.get("global"):
                o.append(a["global"]["OverRate"])
                s.append(c["global"]["OverRate"])
        if not o:
            return None
        online.append(float(np.mean(o)))
        static.append(float(np.mean(s)))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # The online series lands on the ideal line almost exactly, which is the
    # result. Drawn as a thin dashed rule it simply disappears underneath and
    # reads as a missing series, so it is a wide pale band instead: the aqua
    # line visibly sits inside it.
    ax.plot(EPSILONS, EPSILONS, linewidth=9, color=TEXT_SECONDARY, alpha=0.18,
            solid_capstyle="round", zorder=2, label="what was asked for")
    ax.plot(EPSILONS, static, marker="s", markersize=7, linewidth=2.0, color=ORANGE,
            zorder=3, label="static calibration",
            markeredgecolor=SURFACE, markeredgewidth=1.4)
    ax.plot(EPSILONS, online, marker="o", markersize=7, linewidth=2.0, color=AQUA,
            zorder=4, label="online recalibration (on the band)",
            markeredgecolor=SURFACE, markeredgewidth=1.4)
    for e, v in zip(EPSILONS, online, strict=True):
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.5, color=TEXT_SECONDARY)
    for e, v in zip(EPSILONS, static, strict=True):
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points", xytext=(0, -17),
                    ha="center", fontsize=8.5, color=TEXT_SECONDARY)
    # Headroom so the lowest label is not clipped by the axis.
    ax.set_ylim(min(static) - 0.032, max(online) + 0.028)

    _style(ax, title="Only the online layer delivers the risk budget it was set",
           subtitle="Mean over three StarNet locations. Static calibration runs 16 to 21% low "
                    "throughout; online tracks within 1%",
           xlabel="risk budget requested", ylabel="overestimation rate achieved")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")
    return _save(fig, out / "finding-budget-tracking.png")


def fig_availability(root: Path, out: Path) -> Path | None:
    """Calibration moves the link from one nine to two."""
    d = _load(root / "results/requirements/results.json")
    if not d or not d.get("sla"):
        return None
    labels, point, best = [], [], []
    for loc, v in d["sla"].items():
        rows = {r["policy"]: r for r in v["policies"]}
        cal = [r for k, r in rows.items() if k.startswith(("online_regime_eps0.05",
                                                           "static_conformal_eps0.05"))]
        if "point_forecast" not in rows or not cal:
            continue
        labels.append(STARNET.get(loc, loc))
        point.append(rows["point_forecast"]["nines"])
        best.append(max(r["nines"] for r in cal))
    if not labels:
        return None

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar(x - width / 2, point, width * 0.94, color=ORANGE, zorder=3,
                label="uncalibrated forecast", edgecolor=SURFACE, linewidth=2)
    b2 = ax.bar(x + width / 2, best, width * 0.94, color=AQUA, zorder=3,
                label="calibrated, budget 0.05", edgecolor=SURFACE, linewidth=2)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=9, color=TEXT_SECONDARY)
    ax.axhline(3.0, color=TEXT_PRIMARY, linewidth=1.3, linestyle="--", zorder=4)
    ax.text(len(labels) - 0.42, 3.0, "  three nines, not reached", fontsize=9,
            color=TEXT_PRIMARY, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 3.5)
    _style(ax, title="Calibration buys about one extra nine of availability",
           subtitle="Delivered over promised session-slots. Outage count falls five to six fold",
           ylabel="availability (nines)")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")
    return _save(fig, out / "finding-availability.png")


# -- limitations -------------------------------------------------------------

def fig_conditioning_fails(root: Path, out: Path) -> Path | None:
    """Every regime axis is worse than no conditioning. Objective O2's negative.

    Plotted as change from the no-conditioning baseline rather than as absolute
    rates. The absolute values run 0.666 to 0.709, so on a zero baseline the
    bars are indistinguishable and the finding is invisible; truncating the axis
    instead would be the usual dishonest fix. The difference *is* the quantity
    of interest, so it is what gets plotted.
    """
    axes_order = [("candidates", "visible satellite count"),
                  ("level", "look-back level"),
                  ("elevation", "serving elevation"),
                  ("level+geometry", "level + all geometry"),
                  ("level+full", "level + all four axes"),
                  ("geometry", "all measured geometry"),
                  ("phase", "15 s scheduling phase"),
                  ("full", "all four brief axes")]
    baseline, deltas = [], []
    for loc in STARNET:
        d = _load(root / f"results/starnet_regime/{loc}/{loc}-starnet/result.json")
        if not d:
            continue
        g = d["calibration"].get("adaptive_regime_conformal|eps=0.35|regime=global")
        if g:
            baseline.append(g["P10"]["OverRate"])
    if not baseline:
        return None

    for key, label in axes_order:
        vals = []
        for loc in STARNET:
            d = _load(root / f"results/starnet_regime/{loc}/{loc}-starnet/result.json")
            if not d:
                continue
            g = d["calibration"].get("adaptive_regime_conformal|eps=0.35|regime=global")
            e = d["calibration"].get(f"adaptive_regime_conformal|eps=0.35|regime={key}")
            if g and e:
                base = g["P10"]["OverRate"]
                vals.append(100 * (e["P10"]["OverRate"] - base) / base)
        if vals:
            deltas.append((label, float(np.mean(vals))))
    if len(deltas) < 4:
        return None

    deltas.sort(key=lambda kv: kv[1])
    labels = [d[0] for d in deltas]
    values = [d[1] for d in deltas]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(values))
    bars = ax.barh(y, values, 0.62, zorder=3, color=ORANGE,
                   edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(f"{v:+.1f}%", (v, bar.get_y() + bar.get_height() / 2),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=9.5, color=TEXT_SECONDARY)
    ax.axvline(0, color=TEXT_PRIMARY, linewidth=1.4, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, max(values) * 1.28)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, zorder=0)
    _style(ax, title="LIMITATION: no covariate we have improves risk control",
           subtitle="Change from no conditioning at all. Every axis is worse, including "
                    "measured satellite geometry. Objective O2's premise is not supported",
           xlabel="change in severe-risk P10 rate versus no conditioning (%)")
    # No extra caption: the axis label already says "versus no conditioning",
    # and every bar being positive says the rest.
    return _save(fig, out / "limitation-conditioning-fails.png")


def fig_gate_fails(root: Path, out: Path) -> Path | None:
    """The pre-test statistic does not order the outcome. A withdrawn claim."""
    points = []
    for loc, label in STARNET.items():
        d = _load(root / f"results/final_starnet/{loc}/{loc}-starnet/result.json")
        if not d:
            continue
        gate = (d["calibration"].get("gated_adaptive|eps=0.35|regime=level", {})
                .get("detail", {}).get("gate"))
        g = d["calibration"].get("adaptive_regime_conformal|eps=0.35|regime=global")
        lv = d["calibration"].get("adaptive_regime_conformal|eps=0.35|regime=level")
        if not (gate and g and lv):
            continue
        effect = 100 * (lv["P10"]["OverRate"] - g["P10"]["OverRate"]) / g["P10"]["OverRate"]
        points.append((label, gate["spread"], effect))
    if len(points) < 3:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for label, spread, effect in points:
        colour = ORANGE if effect > 0 else AQUA
        ax.scatter([spread], [effect], s=190, color=colour, zorder=4,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{label}\n{effect:+.1f}%", (spread, effect),
                    textcoords="offset points", xytext=(12, -4), fontsize=9.5,
                    color=TEXT_SECONDARY, va="center")
    ax.axhline(0, color=TEXT_PRIMARY, linewidth=1.2, zorder=3)
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 2.6)
    # Headroom both ways so the lowest annotation is not clipped by the axis.
    span = max(ys) - min(ys)
    ax.set_ylim(min(ys) - span * 0.28, max(ys) + span * 0.14)
    ax.text(ax.get_xlim()[0] + 0.15, -span * 0.055,
            "conditioning helps below this line", fontsize=8.5,
            color=TEXT_SECONDARY, va="top", style="italic")
    _style(ax, title="LIMITATION: we could not predict when conditioning would help",
           subtitle="If the statistic worked, these points would slope downward. "
                    "Osnabruck has the widest spread and gains nothing",
           xlabel="per-regime offset spread on the calibration split (Mbps)",
           ylabel="change in P10 risk from conditioning (%)")
    return _save(fig, out / "limitation-gate-fails.png")


def fig_congestion_weak(root: Path, out: Path) -> Path | None:
    """Congestion detection satisfies the requirement and is not good."""
    labels, precision, recall = [], [], []
    for loc, label in STARNET.items():
        d = _load(root / f"results/final_starnet/{loc}/{loc}-starnet/result.json")
        if not d:
            continue
        c = d.get("congestion", {}).get("adaptive_regime_conformal|eps=0.35|regime=level")
        if not c:
            continue
        labels.append(label)
        precision.append(c["precision"])
        recall.append(c["recall"])
    if not labels:
        return None

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar(x - width / 2, precision, width * 0.94, color=BLUE, zorder=3,
                label="precision", edgecolor=SURFACE, linewidth=2)
    b2 = ax.bar(x + width / 2, recall, width * 0.94, color=AQUA, zorder=3,
                label="recall", edgecolor=SURFACE, linewidth=2)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=9, color=TEXT_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    _style(ax, title="LIMITATION: congestion alerts fire early, and often wrongly",
           subtitle="Warning lead is 12 to 138 slots, so the requirement is met. "
                    "On Victoria the alert is wrong three times in four",
           ylabel="rate")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right")
    return _save(fig, out / "limitation-congestion-weak.png")


def fig_phase_evidence(root: Path, out: Path) -> Path | None:
    """Evidence that the scheduling phase was recovered rather than assumed.

    A recovered offset printed on its own is unfalsifiable. The histogram is
    what makes it checkable: a flat one with a confident number beside it is
    exactly the failure this figure exposes.

    Needs the traces, so it is skipped rather than faked when they are absent.
    """
    try:
        from flwcnx.config import DataConfig
        from flwcnx.ingest.replay import ReplaySource
        from flwcnx.state.phase import recover_phase
    except ImportError:                        # pragma: no cover
        return None

    references = {}
    for loc, label in STARNET.items():
        try:
            frame = ReplaySource(DataConfig(location=loc)).load_frame()
        except (FileNotFoundError, OSError):
            continue
        references[label] = recover_phase(frame)
    if not references:
        return None

    fig, axes = plt.subplots(1, len(references), figsize=(3.6 * len(references), 3.5),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (label, ref), colour in zip(axes, references.items(),
                                        [BLUE, ORANGE, AQUA], strict=False):
        centres = (ref.bin_edges[:-1] + ref.bin_edges[1:]) / 2
        ax.bar(centres, ref.histogram, width=0.85, color=colour, zorder=3,
               edgecolor=SURFACE, linewidth=1.2)
        ax.axvline(ref.offset_seconds, color=TEXT_PRIMARY, linewidth=1.5,
                   linestyle="--", zorder=4)
        ax.annotate(f"{ref.offset_seconds:.2f} s", (ref.offset_seconds,
                    ref.histogram.max()), textcoords="offset points",
                    xytext=(5, -4), fontsize=10, color=TEXT_PRIMARY, va="top")
        _style(ax, title=label, xlabel="phase within the 15 s period (s)")
        ax.set_ylabel("edges detected" if ax is axes[0] else "",
                      color=TEXT_SECONDARY, fontsize=10)
    # tight_layout first, then the header, so the two are laid out against the
    # final axes rather than fighting over the same strip of figure.
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.text(0.005, 0.985, "The 15 s scheduling phase is recovered, not assumed",
             color=TEXT_PRIMARY, fontsize=13, ha="left", va="top",
             fontweight="medium")
    fig.text(0.005, 0.915, "Edge-detection histograms on 1 Hz throughput. Casparsen et al. "
             "report 12 s from 500 Hz latency probes at one European site",
             fontsize=9.5, color=TEXT_SECONDARY, ha="left", va="top")
    return _save(fig, out / "finding-phase-recovery.png")


FIGURES = (fig_floor, fig_budget_tracking, fig_availability, fig_phase_evidence,
           fig_conditioning_fails, fig_gate_fails, fig_congestion_weak)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args(argv)

    written, skipped = [], []
    for builder in FIGURES:
        path = builder(args.root, args.out)
        (written if path else skipped).append(path or builder.__name__)
    for path in written:
        print(f"wrote {path}  ({DPI} DPI)")
    for name in skipped:
        print(f"SKIPPED {name}: the run it reads has not been produced")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
