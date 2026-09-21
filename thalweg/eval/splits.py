"""Splits. Temporal only.

Random splits flatter these models. Horizon's window length asymmetry is the
evidence: latency prediction is best with a two month training window while
throughput improves monotonically out to eleven months, which only makes sense
if the two signals are non-stationary on different timescales. A random split
hands the model samples from either side of every change point and hides all of
that.

One thing that is easy to get wrong here and fatal if you do. Consecutive
windows overlap: with a look-back of 30 and a stride of 1, window i and window
i+1 share 29 of their 30 input steps. Splitting the window index contiguously
still leaks, because the last training window's horizon overlaps the first
calibration window's look-back. Every split in this module therefore purges a
band of windows at each boundary, wide enough that no training window shares a
single time step with a calibration or test window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thalweg.config import SplitConfig
from thalweg.state.features import SequenceSet


@dataclass(frozen=True)
class Split:
    """Index arrays into a SequenceSet, plus what was thrown away."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    purged: int
    scheme: str

    def summary(self) -> dict[str, int | str]:
        return {
            "scheme": self.scheme,
            "train": int(len(self.train)),
            "calibration": int(len(self.calibration)),
            "test": int(len(self.test)),
            "purged": int(self.purged),
        }

    def apply(self, sequences: SequenceSet) -> tuple[SequenceSet, SequenceSet, SequenceSet]:
        return (sequences.subset(self.train),
                sequences.subset(self.calibration),
                sequences.subset(self.test))


def purge_width(lookback: int, horizon: int, stride: int = 1) -> int:
    """Windows to drop at a boundary so no time step is shared across it."""
    return int(np.ceil((lookback + horizon) / max(stride, 1)))


def temporal_split(sequences: SequenceSet, config: SplitConfig | None = None,
                   *, lookback: int, horizon: int, stride: int = 1) -> Split:
    """Contiguous train, calibration and test blocks in time order.

    Order is train, then calibration, then test, oldest first. Calibrating on
    data newer than the test set would let the operating point see the future,
    which is a subtler leak than training on test but just as invalidating.
    """
    config = config or SplitConfig()
    total = config.train_frac + config.calib_frac + config.test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")

    order = np.argsort(sequences.origin_time, kind="stable")
    n = len(order)
    purge = purge_width(lookback, horizon, stride)

    train_end = int(n * config.train_frac)
    calib_end = train_end + int(n * config.calib_frac)

    train = order[: max(train_end - purge, 0)]
    calibration = order[train_end : max(calib_end - purge, train_end)]
    test = order[calib_end:]

    dropped = n - (len(train) + len(calibration) + len(test))
    return Split(train, calibration, test, dropped, "temporal")


def contiguous_82(sequences: SequenceSet, *, lookback: int, horizon: int,
                  stride: int = 1, calibration_from_train: float = 0.25) -> Split:
    """StarNet's protocol: two contiguous 8:2 blocks, not interleaved samples.

    Their split has no calibration set because they only fit a point model. Our
    calibration layer needs one, and it has to come out of the training 80% or
    the test block stops being held out. The tail of the training block is used,
    so the calibration data is the closest in time to the test block, which is
    what a deployed system would have.
    """
    order = np.argsort(sequences.origin_time, kind="stable")
    n = len(order)
    purge = purge_width(lookback, horizon, stride)

    train_end = int(n * 0.8)
    block = order[:train_end]
    calib_start = int(len(block) * (1.0 - calibration_from_train))

    train = block[: max(calib_start - purge, 0)]
    calibration = block[calib_start:]
    test = order[train_end + purge :]

    dropped = n - (len(train) + len(calibration) + len(test))
    return Split(train, calibration, test, dropped, "contiguous_82")


def leave_one_location_out(sets: dict[str, SequenceSet], held_out: str, *,
                           lookback: int, horizon: int, stride: int = 1,
                           calibration_frac: float = 0.5) -> tuple[SequenceSet, SequenceSet, SequenceSet]:
    """Train on every other location, test on the held out one.

    The held out location is split in time: the earlier half calibrates and the
    later half tests. Calibrating on the target location is the realistic
    setting, since a terminal deployed somewhere new accumulates its own
    residuals within hours, and it is also the setting our layer is for: the
    regimes are defined by covariates that location shifts.
    """
    if held_out not in sets:
        raise KeyError(f"{held_out!r} not among {sorted(sets)}")
    others = [s for name, s in sets.items() if name != held_out]
    if not others:
        raise ValueError("leave one location out needs at least two locations")

    train = _concatenate(others)
    target = sets[held_out]
    order = np.argsort(target.origin_time, kind="stable")
    purge = purge_width(lookback, horizon, stride)
    cut = int(len(order) * calibration_frac)

    calibration = target.subset(order[: max(cut - purge, 0)])
    test = target.subset(order[cut:])
    return train, calibration, test


def _concatenate(sets: list[SequenceSet]) -> SequenceSet:
    """Join sequence sets from different locations.

    Segments are renumbered so two locations cannot collide on a segment id,
    which would silently merge windows from different continents.
    """
    import pandas as pd

    offset = 0
    segments = []
    for s in sets:
        segments.append(s.segment + offset)
        offset += int(s.segment.max()) + 1 if len(s.segment) else 0

    return SequenceSet(
        x=np.concatenate([s.x for s in sets], axis=0),
        y=np.concatenate([s.y for s in sets], axis=0),
        phase=np.concatenate([s.phase for s in sets], axis=0),
        regime=pd.concat([s.regime for s in sets], ignore_index=True),
        origin_time=np.concatenate([s.origin_time for s in sets], axis=0),
        segment=np.concatenate(segments, axis=0),
        feature_names=sets[0].feature_names,
        target_index=sets[0].target_index,
    )


def check_no_leak(sequences: SequenceSet, split: Split, *, lookback: int,
                  horizon: int) -> dict[str, bool | int]:
    """Assert that no time step appears on both sides of a split boundary.

    Cheap to run and worth running on every experiment. CLAUDE.md section 11
    says to stop and check when a leak is suspected; this makes checking the
    default rather than the reaction.
    """
    def span(index: np.ndarray) -> tuple[np.datetime64, np.datetime64] | None:
        if len(index) == 0:
            return None
        times = sequences.origin_time[index]
        return times.min(), times.max()

    train, calibration, test = span(split.train), span(split.calibration), span(split.test)
    result: dict[str, bool | int] = {"n_train": len(split.train),
                                     "n_calibration": len(split.calibration),
                                     "n_test": len(split.test)}
    # Origin times are the last observed step, so a train window reaches
    # `horizon` steps past its origin and a test window reaches `lookback`
    # steps before its own.
    gap = np.timedelta64(lookback + horizon, "s")
    result["train_before_calibration"] = bool(
        train is None or calibration is None or calibration[0] - train[1] >= gap
    )
    result["calibration_before_test"] = bool(
        calibration is None or test is None or test[0] - calibration[1] >= gap
    )
    result["clean"] = bool(result["train_before_calibration"] and
                           result["calibration_before_test"])
    return result
