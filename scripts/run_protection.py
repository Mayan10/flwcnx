#!/usr/bin/env python3
"""Evaluate the protection layer against a measured capacity series.

Trains the forecaster, seeds the calibrator and streams the test split through
the decision layer exactly as `run_demo.py` does, then hands the per-slot
calibrated bound and realised throughput to the protection layer and contends
for them with the labelled flow workload in `flwcnx/eval/workload.py`.

**The capacity is measured and the flows are a model.** No public LEO dataset
carries a flow table, process attribution or a user-attention signal, so every
number this script produces about the protection layer is a number about that
workload. It is stated that way in `results/summary/` and in the README.

Six arms, all written to one `result.json` beside a config snapshot:

  policies     every policy against the calibrated bound, over a load sweep
  capacity     the same policy against the point forecast instead of the bound,
               which is the only arm that isolates what calibration buys here
  ablation     the layer with one mechanism removed at a time
  channels     the layer with one evidence channel removed at a time
  sensitivity  the hand-set channel weights perturbed, since they are not
               learned and the conclusions should not rest on their exact values
  episode      a per-slot trace of one congestion episode under two policies,
               which is what the case-study figure is drawn from

    python scripts/run_protection.py --location canada --steps 3000
    python scripts/run_protection.py --from-decisions results/demo/decisions.csv
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

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
from flwcnx.decide.flows import CHANNEL_WEIGHTS, ScorerConfig
from flwcnx.decide.protect import ProtectionConfig
from flwcnx.demo import DemoEngine, records_to_frame
from flwcnx.eval.protection import ORACLE, PolicyRun, run_policy
from flwcnx.eval.runner import build_backbone, prepare, resolve_device
from flwcnx.eval.workload import WorkloadSpec, archetype_table
from flwcnx.forecast.train import train_model
from flwcnx.ingest.replay import ReplaySource
from flwcnx.state.regime import RegimeAssigner, presets_for

ALL_POLICIES = [ORACLE, "protected", "by_class", "equal_share", "shed_largest"]

#: Offered-load levels for the sweep. It stops at 4x deliberately. The workload
#: is not a steady-state queueing model: arrivals continue at a fixed rate while
#: starved transfers take proportionally longer to finish, so above capacity the
#: active set grows over the run rather than settling. At 4x that backlog is
#: still small next to the flows in flight over three thousand decisions; past
#: it the run measures an accumulating queue more than it measures a policy, and
#: the load level stops meaning what its label says. Documented in
#: docs/limitations.md section 4c.
DEFAULT_LOADS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


# -- the capacity series -----------------------------------------------------


def decisions_from_pipeline(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Train, calibrate and stream, returning the same frame run_demo.py writes."""
    axes = presets_for("starnet")[args.granularity]
    config = ExperimentConfig(
        name=f"protection-{args.location}", dataset="starnet", seed=args.seed,
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
    print(f"[{args.location}] windows: train={len(train.x)} "
          f"calibration={len(calibration.x)} test={len(test.x)}")

    model = build_backbone("starnet", train, config)
    forecaster, _ = train_model(model, train, calibration, prepared["standardizer"],
                                config=config.model, device=resolve_device("auto"),
                                seed=args.seed, verbose=False)

    assigner = RegimeAssigner(config.calibration.regime).fit(train.regime) if axes else None
    calibrator = AdaptiveRegimeCalibrator(config=config.calibration, assigner=assigner,
                                          direction="lower")
    calibrator.fit(forecaster.predict_horizon_mean(calibration),
                   calibration.y.mean(axis=1), calibration.regime)

    engine = DemoEngine(calibrator=calibrator, decision=config.decision,
                        assigner=assigner, config=config.calibration)
    records = list(engine.run(forecaster.predict_horizon_mean(test),
                              test.y.mean(axis=1), test.regime,
                              timestamps=test.origin_time, limit=args.steps))
    return records_to_frame(records), _config_snapshot(config)


def _config_snapshot(config: ExperimentConfig) -> dict:
    snapshot = asdict(config)
    for key, value in snapshot.items():
        if hasattr(value, "value"):
            snapshot[key] = value.value
    return json.loads(json.dumps(snapshot, default=str))


# -- the arms ----------------------------------------------------------------


def _spec(args: argparse.Namespace, load: float, seed: int | None = None) -> WorkloadSpec:
    return WorkloadSpec(seed=args.seed if seed is None else seed,
                        seconds_per_slot=float(args.seconds_per_slot),
                        load_multiplier=load, declaration_loss=args.declaration_loss)


def _seeds(args: argparse.Namespace) -> list[int]:
    """Workload seeds to average an arm over.

    One seed is one realisation of the arrival process, and the differences
    between policies here are a few points. Reporting a single draw would not
    distinguish those from the sampling noise, and the channel ablation in
    particular has differences small enough that it matters.
    """
    return [args.seed + i for i in range(args.n_seeds)]


def arm_policies(bound: np.ndarray, actual: np.ndarray, args: argparse.Namespace
                 ) -> tuple[list[dict], dict[str, PolicyRun]]:
    """Every policy against the calibrated bound, across the load sweep.

    The sweep is not decoration. Under light load nothing separates the
    policies because nothing has to be shed, and under heavy enough load
    nothing separates them either because the floors stop fitting and the
    oracle fails too. A single operating point would let either of those be
    mistaken for the general case.
    """
    rows: list[dict] = []
    at_reference: dict[str, PolicyRun] = {}
    for load in args.loads:
        for policy in ALL_POLICIES:
            violations = []
            for seed in _seeds(args):
                run = run_policy(policy, bound, actual,
                                 workload_spec=_spec(args, load, seed),
                                 record_channels=(load == args.reference_load
                                                  and seed == args.seed))
                rows.append({"load_multiplier": load, "capacity_source": "bound",
                             "seed": seed} | run.summary())
                violations.append(rows[-1]["critical_violation_allocated"])
                if load == args.reference_load and seed == args.seed:
                    at_reference[policy] = run
            print(f"  load {load:>4} {policy:<13} violation "
                  f"{np.mean(violations):.3f} +/- {np.std(violations):.3f}")
    return rows, at_reference


def arm_capacity(bound: np.ndarray, point: np.ndarray, actual: np.ndarray,
                 args: argparse.Namespace) -> list[dict]:
    """The same policy fed the point forecast instead of the calibrated bound.

    The arm that ties this layer to the rest of the repository. A floor written
    against a forecast the link does not meet is not a reservation, it is a
    number: the shortfall lands on whichever flow the transport starves first.
    The allocated-side violation rate is *better* against the point forecast,
    because the larger number lets the allocator promise more, and the delivered
    rate is what shows the promise was empty.
    """
    rows = []
    for source, series in (("bound", bound), ("point_forecast", point)):
        for load in args.loads:
            for seed in _seeds(args):
                run = run_policy("protected", series, actual,
                                 workload_spec=_spec(args, load, seed))
                rows.append({"load_multiplier": load, "capacity_source": source,
                             "seed": seed} | run.summary())
    return rows


def arm_ablation(bound: np.ndarray, actual: np.ndarray, args: argparse.Namespace
                 ) -> list[dict]:
    """One mechanism removed at a time, at the reference load.

    Each variant is a complete pair of configurations rather than a diff, so
    what any row actually ran is readable without tracing keyword defaults.
    """
    base_scorer, base_protection = ScorerConfig(), ProtectionConfig()
    neutral_priors = dict.fromkeys(base_scorer.priors, 0.0)
    silent_channels = dict.fromkeys(CHANNEL_WEIGHTS, 0.0)

    variants: dict[str, tuple[ScorerConfig, ProtectionConfig]] = {
        "full": (base_scorer, base_protection),
        # Weighted fair queueing: criticality still sets the weights, but a
        # protected flow no longer starts filled. Isolates the floors.
        "no_floors": (base_scorer, ProtectionConfig(honour_floors=False)),
        # One threshold instead of a band, so the protected set may flap. The
        # column to read for this row is the flap rate, not the violation rate.
        "no_hysteresis": (ScorerConfig(release_threshold=base_scorer.protect_threshold,
                                       min_protected_slots=0), base_protection),
        # Declarations ignored entirely: every flow starts neutral and has to
        # earn its score. Measures what the declarations are worth.
        "no_priors": (ScorerConfig(priors=neutral_priors), base_protection),
        # Evidence ignored entirely: the score is the declaration and nothing
        # else, which is class priority wearing this allocator.
        "priors_only": (ScorerConfig(weights=silent_channels), base_protection),
        # Sharpen the weight curve, which widens the gap between a confidently
        # critical flow and an uncertain one. Changes the residual sharing
        # rather than the floors, so read the goodput column for this row.
        "sharp_weights": (base_scorer, ProtectionConfig(weight_exponent=2.0)),
    }

    rows = []
    for name, (scorer, protection) in variants.items():
        violations = []
        for seed in _seeds(args):
            run = run_policy("protected", bound, actual,
                             workload_spec=_spec(args, args.reference_load, seed),
                             scorer_config=scorer, protection_config=protection)
            rows.append({"variant": name, "seed": seed} | run.summary())
            violations.append(rows[-1]["critical_violation_allocated"])
        print(f"  ablation {name:<14} violation "
              f"{np.mean(violations):.3f} +/- {np.std(violations):.3f}")
    return rows


def arm_channels(bound: np.ndarray, actual: np.ndarray, args: argparse.Namespace
                 ) -> list[dict]:
    """Each evidence channel zeroed in turn.

    A clean negative here is a result: if one channel carries everything, the
    other six are decoration and the module should say so.
    """
    rows = []
    for dropped in ["none", *CHANNEL_WEIGHTS]:
        weights = dict(CHANNEL_WEIGHTS)
        if dropped != "none":
            weights[dropped] = 0.0
        violations, aps = [], []
        for seed in _seeds(args):
            run = run_policy("protected", bound, actual,
                             workload_spec=_spec(args, args.reference_load, seed),
                             scorer_config=ScorerConfig(weights=weights))
            rows.append({"dropped_channel": dropped, "seed": seed} | run.summary())
            violations.append(rows[-1]["critical_violation_allocated"])
            aps.append(rows[-1].get("scorer_ap", float("nan")))
        print(f"  without {dropped:<16} violation "
              f"{np.mean(violations):.3f} +/- {np.std(violations):.3f}  "
              f"AP {np.mean(aps):.3f}")
    return rows


def arm_sensitivity(bound: np.ndarray, actual: np.ndarray, args: argparse.Namespace,
                    n_draws: int = 24) -> list[dict]:
    """Random perturbations of the hand-set channel weights.

    The weights are not learned, because fitting them on this workload and
    evaluating on the same generator would be circular. What can be checked is
    whether the ranking against the baselines survives getting them wrong, and
    that is what this measures: each weight is multiplied by a log-uniform
    factor spanning a factor of four either way.
    """
    rng = np.random.default_rng(args.seed)
    spec = _spec(args, args.reference_load)
    rows = []
    for draw in range(n_draws):
        factors = np.exp(rng.uniform(np.log(0.25), np.log(4.0), len(CHANNEL_WEIGHTS)))
        weights = {k: float(v * f) for (k, v), f in zip(CHANNEL_WEIGHTS.items(),
                                                        factors, strict=True)}
        run = run_policy("protected", bound, actual, workload_spec=spec,
                         scorer_config=ScorerConfig(weights=weights))
        rows.append({"draw": draw, "weights": weights} | run.summary())
    return rows


def arm_episode(bound: np.ndarray, actual: np.ndarray, args: argparse.Namespace
                ) -> dict[str, pd.DataFrame]:
    """Per-slot traces under two policies, for the case-study figure.

    Both runs see an identical workload, so the two panels differ only in what
    the allocator did.
    """
    spec = _spec(args, args.reference_load)
    out = {}
    for policy in ("protected", "shed_largest"):
        run = run_policy(policy, bound, actual, workload_spec=spec,
                         record_channels=(policy == "protected"))
        out[policy] = run.records
        out[f"{policy}_slots"] = run.slots
    return out


# -- entry point -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default="canada",
                        choices=["usa", "canada", "germany"])
    parser.add_argument("--from-decisions", type=Path, default=None,
                        help="reuse a decisions.csv from run_demo.py instead of training")
    parser.add_argument("--granularity", default="level")
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--commitment", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--seconds-per-slot", type=float, default=5.0,
                        help="wall clock a decision commits capacity for, the horizon")
    parser.add_argument("--loads", type=float, nargs="+", default=list(DEFAULT_LOADS))
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="workload realisations to average each arm over")
    parser.add_argument("--reference-load", type=float, default=2.0,
                        help="the load level the ablations and the case study run at")
    parser.add_argument("--declaration-loss", type=float, default=0.0,
                        help="fraction of declarations dropped to STANDARD")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["policies", "capacity", "ablation", "channels",
                                 "sensitivity", "episode"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    output = args.output or Path(f"results/protection/{args.location}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.from_decisions:
        frame = pd.read_csv(args.from_decisions)
        snapshot = {"source": str(args.from_decisions)}
        print(f"reusing {len(frame)} decisions from {args.from_decisions}")
    else:
        frame, snapshot = decisions_from_pipeline(args)
    frame = frame.iloc[: args.steps]

    bound = frame["bound_mbps"].to_numpy(dtype=float)
    point = frame["predicted_mbps"].to_numpy(dtype=float)
    actual = frame["actual_mbps"].to_numpy(dtype=float)
    print(f"\n{len(bound)} decisions: bound mean {bound.mean():.1f} Mbps, "
          f"realised mean {actual.mean():.1f} Mbps, "
          f"risk rate {(bound > actual).mean():.4f}\n")

    result: dict = {
        "name": f"protection-{args.location}",
        "config": snapshot,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "capacity_series": {
            "n_slots": int(len(bound)),
            "bound_mean_mbps": float(bound.mean()),
            "point_mean_mbps": float(point.mean()),
            "realised_mean_mbps": float(actual.mean()),
            "realised_risk_rate": float((bound > actual).mean()),
            "point_risk_rate": float((point > actual).mean()),
            "budget": args.epsilon,
        },
        "workload_archetypes": archetype_table(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }

    def checkpoint() -> None:
        """Write what exists so far.

        The grid takes tens of minutes and each arm is independent. Writing only
        at the end means an interruption three arms in loses all three, which is
        how the first attempt at this run was lost.
        """
        result["seconds"] = round(time.time() - started, 1)
        (output / "result.json").write_text(json.dumps(result, indent=2, default=str))

    checkpoint()
    if "policies" not in args.skip:
        print("policy sweep")
        rows, reference = arm_policies(bound, actual, args)
        result["policies"] = rows
        result["by_archetype"] = pd.concat(
            [run.by_archetype() for run in reference.values()]
        ).to_dict("records")
        for policy, run in reference.items():
            run.records.to_csv(output / f"flow_slots_{policy}.csv", index=False)
            run.flows.to_csv(output / f"flows_{policy}.csv", index=False)
        checkpoint()

    if "capacity" not in args.skip:
        print("\ncapacity source arm")
        result["capacity_arm"] = arm_capacity(bound, point, actual, args)
        checkpoint()

    if "ablation" not in args.skip:
        print("\nablation")
        result["ablation"] = arm_ablation(bound, actual, args)
        checkpoint()

    if "channels" not in args.skip:
        print("\nchannel ablation")
        result["channels"] = arm_channels(bound, actual, args)
        checkpoint()

    if "sensitivity" not in args.skip:
        print("\nweight sensitivity")
        result["sensitivity"] = arm_sensitivity(bound, actual, args)
        violations = [r["critical_violation_allocated"] for r in result["sensitivity"]]
        print(f"  {len(violations)} draws, violation "
              f"{np.min(violations):.3f} to {np.max(violations):.3f}")
        checkpoint()

    if "episode" not in args.skip:
        print("\ncase study traces")
        for name, table in arm_episode(bound, actual, args).items():
            table.to_csv(output / f"episode_{name}.csv", index=False)

    frame.to_csv(output / "decisions.csv", index=False)
    checkpoint()
    print(f"\nwrote {output / 'result.json'}  ({result['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
