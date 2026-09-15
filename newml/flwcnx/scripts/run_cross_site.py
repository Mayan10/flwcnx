#!/usr/bin/env python3
"""Leave-one-location-out across the two WetLinks sites. Objective O1.

Osnabruck and Enschede are 150 km apart, on the same continent, under the same
constellation shell, measured by the same instrument over the same months. That
makes this a weak version of the cross-location question, and it is stated that
way rather than dressed up: two European sites are not the three continents the
StarNet traces would have given.

What it can still answer is the one that matters for the calibration layer. The
regime bucket edges, the residual distribution and the operating point are all
fit at one site and then applied at the other. If the layer only works when the
calibration data comes from the same dish, it is a per-terminal tuning trick
rather than a method, and this run is what distinguishes those two cases.

The held out site contributes calibration data from its own earlier half, which
is realistic: a terminal deployed somewhere new accumulates its own residuals
within hours. The forecaster never sees a single window from the held out site.

    python scripts/run_cross_site.py --held-out Enschede --geometry
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flwcnx.config import (
    CalibrationConfig,
    DecisionConfig,
    ExperimentConfig,
    FeatureConfig,
    RegimeConfig,
    SplitConfig,
    StarNetConfig,
)
from flwcnx.eval.runner import comparison_table, risk_table, run_experiment
from flwcnx.eval.splits import check_no_leak, leave_one_location_out
from flwcnx.ingest.wetlinks import SITE_COORDINATES
from flwcnx.ingest.wetlinks_full import (
    WetLinksFullConfig,
    WetLinksSecondsSource,
    attach_geometry,
    load_cached_elements,
)
from flwcnx.state.features import (
    Standardizer,
    build_features,
    make_sequences,
    standardize_sequences,
)

SITES = ("Osnabruck", "Enschede")
SITE_KEYS = {"Osnabruck": "uos-rz", "Enschede": "utwente"}

METHODS = ("point", "global_conformal", "regime_conformal", "regime_bgcfqs",
           "adaptive_global_conformal", "adaptive_regime_conformal")
GRANULARITIES = ("global", "phase", "candidates", "level", "level+volatility",
                 "level+phase", "level+candidates", "full")


def build_sequences(site: str, config: ExperimentConfig, root: Path, geometry: bool,
                    elements=None):
    """One site's windows, unstandardised. Scaling happens after the split."""
    source = WetLinksSecondsSource(WetLinksFullConfig(root=root, site=site))
    frame = source.load_frame()
    if geometry:
        latitude, longitude = SITE_COORDINATES[SITE_KEYS[site]]
        frame = attach_geometry(frame, elements, latitude=latitude, longitude=longitude)
    columns = config.feature_columns
    features, phase_reference, _ = build_features(frame, config=config.features,
                                                  feature_columns=columns)
    sequences = make_sequences(features, config=config.features, feature_names=columns)
    return sequences, phase_reference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held-out", default="Enschede", choices=list(SITES))
    parser.add_argument("--root", type=Path, default=Path("data/wetlinks_full"))
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    parser.add_argument("--granularities", nargs="+", default=list(GRANULARITIES))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/cross_site"))
    args = parser.parse_args(argv)

    if args.lookback + args.horizon > 15:
        parser.error("iperf runs are exactly 15 samples; lookback + horizon must fit")

    config = ExperimentConfig(
        name=f"cross-site-holdout-{args.held_out}",
        dataset="wetlinks_seconds",
        geometry=bool(args.geometry),
        seed=args.seed,
        device=args.device,
        features=FeatureConfig(lookback=args.lookback, horizon=args.horizon,
                               stride=1, recover_phase=True),
        model=StarNetConfig(epochs=args.epochs),
        split=SplitConfig(scheme="leave_one_location_out"),
        calibration=CalibrationConfig(epsilon=0.35, regime=RegimeConfig(min_samples=200)),
        decision=DecisionConfig(commitment_mbps=100.0),
    )

    elements = load_cached_elements() if args.geometry else None
    if args.geometry:
        print("geometry: RECONSTRUCTED from propagated elements, not measured")

    sets, references = {}, {}
    for site in SITES:
        sequences, reference = build_sequences(site, config, args.root, args.geometry,
                                               elements)
        sets[site] = sequences
        references[site] = reference
        print(f"{site}: {len(sequences)} windows, phase offset "
              f"{reference.offset_seconds:.2f}s ({reference.method})")

    train, calibration, test = leave_one_location_out(
        sets, args.held_out, lookback=args.lookback, horizon=args.horizon, stride=1,
    )
    trained_on = [s for s in SITES if s != args.held_out]
    print(f"\ntrain on {trained_on} ({len(train)} windows), "
          f"calibrate and test on {args.held_out} "
          f"({len(calibration)} / {len(test)} windows)")

    # Standardise on training windows only, exactly as `prepare` does. Here it
    # carries extra weight: the scaler is fit at one site and applied at
    # another, so any site level scale difference stays visible to the model
    # instead of being normalised away into a flattering result.
    standardizer = Standardizer().fit_windows(train.x, config.feature_columns)
    train, calibration, test = (standardize_sequences(s, standardizer)
                                for s in (train, calibration, test))

    held_leak = check_no_leak(
        sets[args.held_out], _held_out_split(sets[args.held_out], calibration, test),
        lookback=args.lookback, horizon=args.horizon,
    )
    prepared = {
        "train": train, "calibration": calibration, "test": test,
        "split": _CrossSiteSplit(trained_on, args.held_out, train, calibration, test),
        "leak": held_leak, "standardizer": standardizer,
        "phase_reference": references[args.held_out], "encoder": None, "raw": None,
    }

    result = run_experiment(
        None, config, prepared=prepared, backbone="starnet",
        calibration_methods=METHODS, epsilons=tuple(args.epsilons),
        granularities=tuple(args.granularities),
    )
    result.config["cross_site"] = {
        "trained_on": trained_on, "held_out": args.held_out,
        "train_windows": len(train), "calibration_windows": len(calibration),
        "test_windows": len(test),
        "phase_offsets": {s: references[s].offset_seconds for s in SITES},
    }
    written = result.save(args.output / config.name)

    print("\n=== risk table ===")
    print(risk_table(result).round(4).to_string(index=False))
    print("\n=== comparison ===")
    print(comparison_table(result).round(4).to_string(index=False))
    print(f"\nwrote {written}")
    return 0


class _CrossSiteSplit:
    """Minimal stand-in for `Split`, carrying only what the runner reads."""

    def __init__(self, trained_on, held_out, train, calibration, test):
        self._summary = {
            "scheme": "leave_one_location_out",
            "trained_on": "+".join(trained_on),
            "held_out": held_out,
            "train": len(train), "calibration": len(calibration), "test": len(test),
        }

    def summary(self) -> dict:
        return dict(self._summary)


def _held_out_split(target, calibration, test):
    """Rebuild the held out site's index split so the leak check can run on it.

    The train side lives at another site and cannot overlap by construction, so
    the only ordering that needs checking is calibration before test within the
    held out site.
    """
    from flwcnx.eval.splits import Split

    order = np.argsort(target.origin_time, kind="stable")
    n_cal, n_test = len(calibration), len(test)
    return Split(train=np.array([], dtype=int),
                 calibration=order[:n_cal],
                 test=order[len(order) - n_test:],
                 purged=len(order) - n_cal - n_test,
                 scheme="leave_one_location_out")


if __name__ == "__main__":
    raise SystemExit(main())
