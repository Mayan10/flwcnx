"""Converting timestamps to numbers, once, correctly.

This module exists because of a bug that only appeared in CI.

pandas 2 supports datetime64 at second, millisecond, microsecond and nanosecond
resolution, and `Series.astype("int64")` returns the count in **whatever unit
the column happens to carry**. Code that writes

    times.astype("int64") / 1e9

is therefore only correct when the column is `datetime64[ns]`, and silently
returns a number a million or a billion times wrong otherwise. Which unit you
get depends on the pandas version and on how the column was constructed, so the
same code produced correct results locally and nonsense on a runner with a
different pandas.

The symptom was not a crash. `state/phase.py` computes the scheduling phase from
these seconds, so the aliasing guard inverted its answer, edge detection found
one edge instead of hundreds, and thirteen tests failed on CI while all of them
passed locally.

Pin the unit first, always.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_epoch_seconds(times) -> np.ndarray:
    """Seconds since the Unix epoch, whatever datetime resolution came in.

    Accepts a Series, an Index, or a datetime64 array. 60 is a multiple of 15,
    so a phase measured against the epoch is the same phase as second-of-minute
    modulo 15, which is what the scheduling recovery needs.
    """
    values = pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(times)).values))
    return values.as_unit("ns").asi8.astype("float64") / 1e9


def to_epoch_nanoseconds(times) -> np.ndarray:
    """Integer nanoseconds since the epoch. Same unit hazard, same fix.

    Used where a bucket index is wanted rather than a float, so that the
    division stays exact.
    """
    values = pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(times)).values))
    return values.as_unit("ns").asi8
