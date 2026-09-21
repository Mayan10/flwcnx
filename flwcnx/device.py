"""Choosing a compute device, and refusing to start a run that cannot finish.

Two portability problems, both found by trying to run this on machines other
than the one it was written on.

**Device detection was Mac-shaped.** The original check called
`torch.backends.mps.is_available()` unguarded. On a torch build without the MPS
backend that attribute does not exist and the call raises `AttributeError`
during setup, before any useful error message. It also treated an explicit
`--device cuda` as a promise rather than a request, so asking for CUDA on a
machine without it failed later and less clearly.

**Memory was unbounded.** Sequence windowing materialises
`(n_windows, lookback, n_features)` as float32. On the US trace at stride 1 that
is 1.1M windows of 30 x 13, about 1.7 GB for the inputs alone, before the model,
the optimiser, or the copy the split makes. On a laptop that is an OOM kill with
no explanation. The estimate is cheap and exact, so it is checked up front and
reported with the two knobs that fix it.
"""

from __future__ import annotations

import os
import warnings


def available_devices() -> list[str]:
    """Every device this machine can actually use, best first."""
    devices = []
    try:
        import torch
    except ImportError:
        return ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    # `torch.backends.mps` is absent on builds compiled without it, so the
    # attribute has to be probed rather than assumed. `is_built` guards the
    # case where the backend exists but this machine has no Metal device.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        try:
            if mps.is_built() and mps.is_available():
                devices.append("mps")
        except Exception:              # pragma: no cover - defensive
            pass
    devices.append("cpu")
    return devices


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device, falling back with a warning rather than failing.

    `FLWCNX_DEVICE` overrides everything, which is what makes a constrained CI
    runner or a shared GPU box reproducible without editing a config.

    An explicit request for a device this machine does not have is a warning and
    a fallback, not an error: the run is still valid on CPU, only slower, and
    dying at setup time on a borrowed machine helps nobody.
    """
    override = os.environ.get("FLWCNX_DEVICE")
    if override:
        preference = override

    devices = available_devices()
    if preference == "auto":
        return devices[0]
    if preference in devices:
        return preference
    warnings.warn(
        f"device {preference!r} was requested but this machine offers "
        f"{devices}. Falling back to {devices[0]!r}. Set FLWCNX_DEVICE to pin "
        "a device explicitly.",
        RuntimeWarning, stacklevel=2,
    )
    return devices[0]


def describe_device(device: str) -> dict:
    """What the run actually landed on, for the config snapshot."""
    info: dict = {"device": device, "available": available_devices()}
    try:
        import torch
    except ImportError:
        return info | {"torch": None}
    info["torch"] = torch.__version__
    if device == "cuda" and torch.cuda.is_available():
        info["name"] = torch.cuda.get_device_name(0)
        info["total_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def estimate_window_bytes(n_windows: int, lookback: int, n_features: int,
                          horizon: int = 0, itemsize: int = 4) -> int:
    """Bytes the windowed inputs and targets will occupy.

    The split copies its slices rather than viewing them, so the peak is roughly
    twice the base array. That factor is included, because the number a caller
    wants is "will this fit", not "how big is one array".
    """
    per_window = (lookback * n_features + horizon) * itemsize
    return int(n_windows * per_window * 2)


def check_memory_budget(n_windows: int, lookback: int, n_features: int,
                        horizon: int = 0, *, limit_gb: float | None = None,
                        stride: int = 1) -> None:
    """Refuse a run that will not fit, and say which knob fixes it.

    Raises before anything is allocated. `limit_gb` defaults to the
    `FLWCNX_MEMORY_LIMIT_GB` environment variable, or 8 GB, which is a
    conservative laptop figure rather than a measurement of this machine.
    """
    if limit_gb is None:
        limit_gb = float(os.environ.get("FLWCNX_MEMORY_LIMIT_GB", "8"))
    needed = estimate_window_bytes(n_windows, lookback, n_features, horizon)
    needed_gb = needed / 1e9
    if needed_gb <= limit_gb:
        return
    suggested = max(int(stride * needed_gb / (limit_gb * 0.6)), stride + 1)
    raise MemoryError(
        f"windowing {n_windows:,} sequences of {lookback} x {n_features} needs "
        f"about {needed_gb:.1f} GB, over the {limit_gb:.1f} GB limit.\n"
        f"  Raise the limit:  FLWCNX_MEMORY_LIMIT_GB={needed_gb:.0f}\n"
        f"  Or window less:   --stride {suggested} "
        f"(currently {stride})\n"
        "Stride above the horizon also makes scored decisions disjoint, which "
        "is the honest denominator for a risk rate."
    )
