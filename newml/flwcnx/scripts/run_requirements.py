#!/usr/bin/env python3
"""The three requirements the throughput pipeline did not answer.

The brief names six industry needs and permits addressing three. Three were
addressed by the throughput work: predict throughput degradation, detect
congestion before users are affected, and optimize bandwidth allocation. This
script closes the other three, on the same traces and through the same
calibration layer.

  1. **Predict latency spikes.** The StarNet traces carry latency, which the
     brief did not expect. Casparsen's period-level Good/Degraded framing, with
     the spike decision read off a risk-controlled *upper* bound rather than a
     second trained classifier, so the operating point is set by the risk budget
     instead of by tuning a threshold.

  5. **Improve service availability.** Availability in the form a carrier SLA is
     written in, with MTBF and MTTR, per allocation policy.

  6. **Reduce operational costs.** SLA credits for over-allocation against
     foregone revenue for under-allocation, so the two failure directions land
     on one scale and policies can be ranked by money.

    python scripts/run_requirements.py --location all
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from flwcnx.calibrate.adaptive import AdaptiveRegimeCalibrator
from flwcnx.calibrate.conformal import fit_conformal
from flwcnx.config import (
    CalibrationConfig,
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    RegimeConfig,
    SplitConfig,
    StarNetConfig,
)
from flwcnx.decide.sla import (
    ServiceLevelAgreement,
    break_even_credit_ratio,
    evaluate_policy,
)
from flwcnx.eval.runner import build_backbone, prepare, resolve_device
from flwcnx.forecast.train import train_model
from flwcnx.ingest.replay import ReplaySource
from flwcnx.state.latency import (
    DEFAULT_LATENCY_THRESHOLD_MS,
    degraded_from_bound,
    label_periods,
    spike_metrics,
)
from flwcnx.state.phase import recover_phase
from flwcnx.state.regime import RegimeAssigner

LOCATIONS = ("usa", "germany", "canada")


# --------------------------------------------------------------------------
# Requirement 1: latency spikes
# --------------------------------------------------------------------------

def run_latency(location: str, args) -> dict:
    """Risk-controlled latency upper bound, and the spike decision from it."""
    config = ExperimentConfig(
        name=f"latency-{location}", dataset="starnet", seed=args.seed,
        data=DataConfig(location=location),
        features=FeatureConfig(lookback=30, horizon=args.horizon, stride=args.stride,
                               recover_phase=True, target_column="latency_ms"),
        model=StarNetConfig(epochs=args.epochs),
        split=SplitConfig(scheme="temporal"),
        calibration=CalibrationConfig(epsilon=args.epsilon,
                                      regime=RegimeConfig(axes=("level",),
                                                          min_samples=200)),
    )
    frame = ReplaySource(config.data).load_frame()

    # Casparsen's period labels on the whole trace, aligned to the recovered
    # scheduling phase rather than to arbitrary 15 s boundaries.
    reference = recover_phase(frame)
    labels = label_periods(frame, phase_offset=reference.offset_seconds,
                           threshold_ms=args.threshold_ms)

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

    # Latency flips the bound: the risk is promising a delay the link will not
    # meet, so the bound is an upper one and the risk event is bound < actual.
    assigner = RegimeAssigner(config.calibration.regime).fit(train.regime)
    online = AdaptiveRegimeCalibrator(config=config.calibration, assigner=assigner,
                                      direction="upper")
    online.fit(predicted_cal, actual_cal, calibration.regime)
    bound = online.transform_online(predicted_test, actual_test, test.regime)

    static = fit_conformal(predicted_cal, actual_cal, args.epsilon, direction="upper")
    static_bound = static.apply(predicted_test, floor=0.0)

    actual_degraded = actual_test > args.threshold_ms
    out = {
        "location": location,
        "period_labels": labels.summary(),
        "n_test": int(actual_test.size),
        "actual_degraded_rate": float(actual_degraded.mean()),
        "under_rate_online": float(np.mean(bound < actual_test)),
        "under_rate_static": float(np.mean(static_bound < actual_test)),
        "budget": args.epsilon,
        "mae_ms": float(np.mean(np.abs(predicted_test - actual_test))),
        "policies": {},
    }
    for name, scores in (("point_forecast", predicted_test),
                         ("static_conformal", static_bound),
                         ("online_regime", bound)):
        m = spike_metrics(degraded_from_bound(scores, args.threshold_ms),
                          actual_degraded, scores=scores)
        out["policies"][name] = m.to_dict()
    return out


# --------------------------------------------------------------------------
# Requirements 5 and 6: availability and cost
# --------------------------------------------------------------------------

def run_availability_and_cost(location: str, args) -> dict:
    """Every throughput policy, priced."""
    config = ExperimentConfig(
        name=f"sla-{location}", dataset="starnet", seed=args.seed,
        data=DataConfig(location=location),
        features=FeatureConfig(lookback=30, horizon=args.horizon, stride=args.stride,
                               recover_phase=True),
        model=StarNetConfig(epochs=args.epochs),
        split=SplitConfig(scheme="temporal"),
        calibration=CalibrationConfig(epsilon=args.epsilon,
                                      regime=RegimeConfig(axes=("level",),
                                                          min_samples=200)),
    )
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

    sla = ServiceLevelAgreement(
        committed_rate_mbps=args.commitment,
        availability_target=args.availability_target,
        bandwidth_per_session_mbps=args.session_mbps,
        seconds_per_slot=float(args.horizon),
    )

    policies: dict[str, np.ndarray] = {"point_forecast": predicted_test}
    for epsilon in args.epsilons:
        cfg = replace(config.calibration, epsilon=epsilon)
        static = fit_conformal(predicted_cal, actual_cal, epsilon, direction="lower")
        policies[f"static_conformal_eps{epsilon:.2f}"] = static.apply(predicted_test)

        assigner = RegimeAssigner(cfg.regime).fit(train.regime)
        online = AdaptiveRegimeCalibrator(config=cfg, assigner=assigner,
                                          direction="lower")
        online.fit(predicted_cal, actual_cal, calibration.regime)
        policies[f"online_regime_eps{epsilon:.2f}"] = online.transform_online(
            predicted_test, actual_test, test.regime)

    # The oracle admits exactly what the link could carry. Not achievable, and
    # the right reference for how much of the gap a policy closes.
    policies["oracle"] = actual_test

    rows = [evaluate_policy(name, bound, actual_test, sla).to_dict()
            for name, bound in policies.items()]

    # The ranking above depends on our price guesses. The break-even ratio does
    # not: it is how much more a violated session-hour must cost than a sold one
    # before risk control pays, which an operator checks against their contract.
    tightest = f"online_regime_eps{min(args.epsilons):.2f}"
    break_even = break_even_credit_ratio(policies[tightest], policies["point_forecast"],
                                         actual_test, sla)
    break_even["conservative_policy"] = tightest
    break_even["aggressive_policy"] = "point_forecast"
    return {"location": location, "sla": vars(sla) | {}, "policies": rows,
            "break_even": break_even}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default="all",
                        choices=[*LOCATIONS, "all"])
    parser.add_argument("--epsilon", type=float, default=0.10,
                        help="headline risk budget for the latency bound")
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.05, 0.10, 0.20, 0.35],
                        help="budgets to price for availability and cost")
    parser.add_argument("--threshold-ms", type=float,
                        default=DEFAULT_LATENCY_THRESHOLD_MS)
    parser.add_argument("--commitment", type=float, default=100.0)
    parser.add_argument("--session-mbps", type=float, default=10.0)
    parser.add_argument("--availability-target", type=float, default=0.999)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/requirements"))
    args = parser.parse_args(argv)

    locations = LOCATIONS if args.location == "all" else (args.location,)
    results: dict = {"latency": {}, "sla": {}, "config": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}

    for location in locations:
        if not args.skip_latency:
            print(f"\n=== requirement 1: latency spikes, {location} ===")
            latency = run_latency(location, args)
            results["latency"][location] = latency
            lab = latency["period_labels"]
            print(f"  periods {lab['n_periods']:,}, degraded {lab['degraded_rate']:.1%}"
                  f"  (threshold {lab['threshold_ms']:.0f} ms)")
            print(f"  latency MAE {latency['mae_ms']:.2f} ms; UnderRate "
                  f"online {latency['under_rate_online']:.3f} vs static "
                  f"{latency['under_rate_static']:.3f} against budget {args.epsilon}")
            for name, m in latency["policies"].items():
                print(f"    {name:18s} P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f} AUPRC={m['auprc']:.3f} "
                      f"(chance {m['baseline_auprc']:.3f}, lift {m['lift']:.2f}x)")

        print(f"\n=== requirements 5 and 6: availability and cost, {location} ===")
        sla = run_availability_and_cost(location, args)
        results["sla"][location] = sla
        frame = pd.DataFrame(sla["policies"])
        show = frame[["policy", "availability", "nines", "n_outages",
                      "mttr_seconds", "cost_per_hour", "mean_admitted"]]
        print(show.to_string(index=False))
        # The oracle is not a policy, it is the unreachable reference, so it is
        # excluded from "best" rather than trivially winning at zero cost.
        real = frame[frame["policy"] != "oracle"]
        best = real.loc[real["cost_per_hour"].idxmin()]
        print(f"  cheapest deployable policy: {best['policy']} at "
              f"{best['cost_per_hour']:.5f} per hour")
        be = sla["break_even"]
        print(f"  break-even: a violated session-hour must cost "
              f"{be['break_even_credit_ratio']:.2f}x a sold one before "
              f"{be['conservative_policy']} beats {be['aggressive_policy']}")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
