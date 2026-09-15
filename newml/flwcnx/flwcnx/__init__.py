"""Predictive bandwidth allocation for Starlink access links.

Layer order, strictly downward:

    ingest -> state -> forecast -> calibrate -> decide

`eval` sits beside all of them and imports whatever it needs.
"""

__version__ = "0.1.0"
