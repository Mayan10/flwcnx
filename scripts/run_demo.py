#!/usr/bin/env python3
"""Run the whole system as a system, one decision at a time.

Trains the forecaster, seeds the calibrator, then replays the test split
decision by decision through forecast -> regime -> safe bound -> admission ->
congestion, revealing each outcome only after the decision that depended on it.

Writes three things:

  decisions.csv    one row per decision, the full audit trail
  summary.json     realised risk against the budget, and the allocation outcome
  demo.html        a self contained page: the bound tracking capacity, the
                   online alpha, and the admission decisions

    python scripts/run_demo.py --location canada --steps 2000
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from flwcnx.calibrate.adaptive import AdaptiveRegimeCalibrator
from flwcnx.config import (
    CalibrationConfig,
    DataConfig,
    DecisionConfig,
    ExperimentConfig,
    FeatureConfig,
    RegimeConfig,
    SplitConfig,
    StarNetConfig,
)
from flwcnx.demo import DemoEngine, records_to_frame
from flwcnx.eval.runner import build_backbone, prepare, resolve_device
from flwcnx.forecast.train import train_model
from flwcnx.ingest.replay import ReplaySource
from flwcnx.state.regime import RegimeAssigner, presets_for


def build_page(frame, summary: dict, title: str) -> str:
    """A self contained page. No CDN, no fonts, no network at all."""
    step = frame["step"].tolist()
    actual = frame["actual_mbps"].tolist()
    predicted = frame["predicted_mbps"].tolist()
    bound = frame["bound_mbps"].tolist()
    alpha = frame["alpha"].tolist()
    risk = frame["risk_event"].tolist()
    dropped = frame["dropped"].tolist()

    def path(values, lo, hi, width, height):
        if hi <= lo:
            hi = lo + 1.0
        n = max(len(values) - 1, 1)
        pts = []
        for i, v in enumerate(values):
            x = i / n * width
            y = height - (float(v) - lo) / (hi - lo) * height
            pts.append(f"{x:.2f},{y:.2f}")
        return " ".join(pts)

    w, h = 1120, 260
    lo = min(min(actual), min(bound)) * 0.95
    hi = max(max(actual), max(predicted)) * 1.02
    risk_marks = "".join(
        f'<rect x="{i / max(len(step) - 1, 1) * w:.2f}" y="0" width="1.5" '
        f'height="{h}" fill="#eb6834" opacity="0.30"/>'
        for i, r in enumerate(risk) if r
    )
    a_lo, a_hi = min(alpha) * 0.9, max(alpha) * 1.1
    budget_y = h - (summary["budget"] - a_lo) / max(a_hi - a_lo, 1e-9) * h

    rate = summary["realised_risk_rate"]
    verdict = "within budget" if summary["within_budget"] else "OVER BUDGET"
    verdict_color = "#1baf7a" if summary["within_budget"] else "#eb6834"

    return f"""<title>{html.escape(title)}</title>
<style>
  :root {{ --bg:#fcfcfb; --fg:#0b0b0b; --dim:#52514e; --grid:#dedcd6; --card:#fff; }}
  :root:not([data-theme="light"]) {{ }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{ --bg:#131313; --fg:#f2f2f0; --dim:#a3a19c;
      --grid:#333; --card:#1c1c1b; }}
  }}
  :root[data-theme="dark"] {{ --bg:#131313; --fg:#f2f2f0; --dim:#a3a19c;
    --grid:#333; --card:#1c1c1b; }}
  body {{ background:var(--bg); color:var(--fg); margin:0; padding:2rem 1.5rem;
    font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:1.45rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
  .sub {{ color:var(--dim); margin:0 0 1.75rem; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:.75rem; margin-bottom:1.75rem; }}
  .tile {{ background:var(--card); border:1px solid var(--grid); border-radius:10px;
    padding:.85rem 1rem; }}
  .tile .k {{ color:var(--dim); font-size:.78rem; text-transform:uppercase;
    letter-spacing:.05em; }}
  .tile .v {{ font-size:1.5rem; font-variant-numeric:tabular-nums; margin-top:.15rem; }}
  .panel {{ background:var(--card); border:1px solid var(--grid); border-radius:10px;
    padding:1rem 1.1rem; margin-bottom:1.25rem; overflow-x:auto; }}
  .panel h2 {{ font-size:.95rem; margin:0 0 .1rem; }}
  .panel p {{ color:var(--dim); font-size:.85rem; margin:0 0 .8rem; }}
  svg {{ display:block; max-width:100%; height:auto; }}
  .legend {{ display:flex; gap:1.1rem; flex-wrap:wrap; color:var(--dim);
    font-size:.82rem; margin-top:.5rem; }}
  .swatch {{ display:inline-block; width:11px; height:11px; border-radius:2px;
    margin-right:.35rem; vertical-align:-1px; }}
  footer {{ color:var(--dim); font-size:.82rem; border-top:1px solid var(--grid);
    padding-top:1rem; }}
</style>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">Replay of a held-out trace, one decision at a time. Each bound is
emitted before its outcome is revealed.</p>

<div class="tiles">
  <div class="tile"><div class="k">Decisions</div><div class="v">{summary['n_decisions']:,}</div></div>
  <div class="tile"><div class="k">Risk budget</div><div class="v">{summary['budget']:.2f}</div></div>
  <div class="tile"><div class="k">Realised risk</div>
    <div class="v" style="color:{verdict_color}">{rate:.3f}</div></div>
  <div class="tile"><div class="k">Verdict</div>
    <div class="v" style="color:{verdict_color};font-size:1.05rem">{verdict}</div></div>
  <div class="tile"><div class="k">Mean dropped</div><div class="v">{np.mean(dropped):.2f}</div></div>
  <div class="tile"><div class="k">Congested slots</div>
    <div class="v">{int(frame['congested'].sum()):,}</div></div>
</div>

<div class="panel">
  <h2>Capacity, forecast and safe bound</h2>
  <p>Orange bars mark decisions where the bound promised more than the link
     delivered. The budget allows {summary['budget']:.0%} of them.</p>
  <svg viewBox="0 0 {w} {h}" role="img" aria-label="capacity and bound over time">
    {risk_marks}
    <polyline fill="none" stroke="#9a9894" stroke-width="1" opacity="0.85"
      points="{path(actual, lo, hi, w, h)}"/>
    <polyline fill="none" stroke="#4a3aa7" stroke-width="1.3"
      points="{path(predicted, lo, hi, w, h)}"/>
    <polyline fill="none" stroke="#1baf7a" stroke-width="1.6"
      points="{path(bound, lo, hi, w, h)}"/>
  </svg>
  <div class="legend">
    <span><i class="swatch" style="background:#9a9894"></i>actual capacity</span>
    <span><i class="swatch" style="background:#4a3aa7"></i>point forecast</span>
    <span><i class="swatch" style="background:#1baf7a"></i>safe bound</span>
    <span><i class="swatch" style="background:#eb6834"></i>over-promise</span>
  </div>
</div>

<div class="panel">
  <h2>The calibrator adapting</h2>
  <p>The operating point per regime, updated from each outcome after it is
     observed. The dashed line is the budget it is steering toward.</p>
  <svg viewBox="0 0 {w} 150" role="img" aria-label="alpha over time">
    <line x1="0" y1="{budget_y * 150 / h:.1f}" x2="{w}" y2="{budget_y * 150 / h:.1f}"
      stroke="#0b0b0b" stroke-dasharray="4 3" stroke-width="1" opacity=".55"/>
    <polyline fill="none" stroke="#2a78d6" stroke-width="1.3"
      points="{path(alpha, a_lo, a_hi, w, 150)}"/>
  </svg>
</div>

<footer>
Generated by <code>scripts/run_demo.py</code>. Every number here comes from a
single causal pass: at each step the engine sees the look-back window, the
regime covariates, and the outcomes of earlier steps only.
</footer>
</div>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default="canada",
                        choices=["usa", "canada", "germany"])
    parser.add_argument("--granularity", default="level")
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--commitment", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=Path("results/demo"))
    args = parser.parse_args(argv)

    axes = presets_for("starnet")[args.granularity]
    config = ExperimentConfig(
        name=f"demo-{args.location}", dataset="starnet", seed=args.seed,
        data=DataConfig(location=args.location),
        features=FeatureConfig(lookback=30, horizon=5, stride=args.stride,
                               recover_phase=True),
        model=StarNetConfig(epochs=args.epochs),
        split=SplitConfig(scheme="temporal"),
        calibration=CalibrationConfig(epsilon=args.epsilon,
                                      regime=RegimeConfig(axes=axes, min_samples=200)),
        decision=DecisionConfig(commitment_mbps=args.commitment),
    )

    print(f"[{args.location}] preparing, regime axes = {axes or ('global',)}")
    prepared = prepare(ReplaySource(config.data), config)
    train, calibration, test = (prepared["train"], prepared["calibration"],
                                prepared["test"])
    model = build_backbone("starnet", train, config)
    forecaster, _ = train_model(model, train, calibration, prepared["standardizer"],
                                config=config.model, device=resolve_device("auto"),
                                seed=args.seed, verbose=False)

    predicted_cal = forecaster.predict_horizon_mean(calibration)
    predicted_test = forecaster.predict_horizon_mean(test)
    actual_cal, actual_test = calibration.y.mean(axis=1), test.y.mean(axis=1)

    assigner = RegimeAssigner(config.calibration.regime).fit(train.regime) if axes else None
    calibrator = AdaptiveRegimeCalibrator(config=config.calibration, assigner=assigner,
                                          direction="lower")
    calibrator.fit(predicted_cal, actual_cal, calibration.regime)

    engine = DemoEngine(calibrator=calibrator, decision=config.decision,
                        assigner=assigner, config=config.calibration)
    records = list(engine.run(predicted_test, actual_test, test.regime,
                              timestamps=test.origin_time, limit=args.steps))
    frame = records_to_frame(records)
    summary = engine.summary() | {
        "location": args.location, "granularity": args.granularity,
        "mean_dropped": float(frame["dropped"].mean()),
        "mean_admitted": float(frame["admitted"].mean()),
        "mean_oracle": float(frame["oracle"].mean()),
        "utilisation": float(frame["admitted"].sum() / max(frame["oracle"].sum(), 1)),
        "congested_slots": int(frame["congested"].sum()),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "decisions.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    title = f"flwcnx live replay: {args.location}, budget {args.epsilon:.2f}"
    (args.output / "demo.html").write_text(build_page(frame, summary, title))

    print(f"\n  decisions        {summary['n_decisions']:,}")
    print(f"  realised risk    {summary['realised_risk_rate']:.4f} "
          f"against budget {summary['budget']:.2f} "
          f"({'within' if summary['within_budget'] else 'OVER'})")
    print(f"  mean dropped     {summary['mean_dropped']:.3f} sessions")
    print(f"  utilisation      {summary['utilisation']:.4f}")
    print(f"  congested slots  {summary['congested_slots']}")
    for name in ("decisions.csv", "summary.json", "demo.html"):
        print(f"  wrote {args.output / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
