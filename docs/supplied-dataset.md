# The supplied dataset

Inspected 2026-08-18, before any feature code was written against it, per the
brief's instruction.

**It is WetLinks.** The two site names, `uos-rz` and `utwente`, are the
University of Osnabrück computing centre and the University of Twente: exactly
the two European vantage points in Laniewski et al., TMA 2024, which is already
entry 9 in `docs/references.bib`. The lat/lon that is populated (52.2853,
8.0220) is Osnabrück.

| | uos-rz | utwente |
|---|---|---|
| Rows | 482,599 | 325,554 |
| Span | 2023-09-14 to 2024-03-12 (179 d) | 2023-09-28 to 2024-03-09 (162 d) |
| Sampling | 30 s, very regular | 30 s, very regular |
| lat/lon populated | 29% | 0% (geocodable from the site name) |
| State | 99.97% CONNECTED | 99.99% CONNECTED |

Columns are `starlink-grpc-tools` dish status fields: `state`, `uplink`,
`downlink`, `pop_ping_latency`, `ping_drop`, `mean_ping_latency`,
`ping_stdvar`, `fraction_obstructed`, `obstruction_duration`,
`obstruction_interval`, `direction_azimuth`, `direction_elevation`, `lat`,
`lon`.

## Three findings that change the plan

### 1. The 15 second phase axis is not sparse, it is dead

Sampling is 30 s and 30 is an exact multiple of 15, so **every sample lands at
the same point in the scheduling period**. Measured across 482,463 connected
samples the phase standard deviation is 0.156 s.

This is not a resolution problem that more data would fix. It is aliasing. The
Casparsen phase recovery, the phase regime axis, and the periodical embedding's
phase channel all have nothing to act on here. `state/phase.py` still works and
is still correct; there is simply no phase information in this dataset.

### 2. `direction_azimuth` / `direction_elevation` are the dish, not the satellite

`direction_elevation` has a standard deviation of 0.43 degrees around 70, which
is the dish's own boresight tilt, not a satellite track. There is no serving
satellite ID, no satellite elevation, distance, or candidate count anywhere in
the file.

So three of our four regime axes (phase, elevation, distance) are unavailable,
and the fourth (candidate count) is only recoverable by propagating TLEs
ourselves against the known site coordinates.

`direction_azimuth` does move (std 53.7 degrees, 22 distinct values at 1 degree
resolution), so the dish does re-orient. It is a weak proxy for serving
direction and is worth testing as a regime axis, but it is not satellite
geometry and must not be labelled as such.

### 3. `downlink` is offered load, not link capacity, except in bursts

This is the most important finding and the easiest one to get wrong.

- 90.7% of samples sit below 0.1 Mbps. The median is 0.008 Mbps.
- 8.7% of samples exceed 50 Mbps, with a median of **230.8 Mbps**.
- Those high samples are isolated, one every ~6 minutes, never consecutive.

They are periodic speedtests. The 230.8 Mbps median is a striking match for
StarNet's reported "throughput holds around 230 Mbps", which is good evidence
they are genuine capacity measurements rather than an artefact.

The consequence: the `downlink` column is **91% idle telemetry**. Training a
throughput forecaster on it as-is would be forecasting whether someone happened
to be using the link, not what the link could carry. Any result from that would
be meaningless, and it would look fine in a MAE table.

Usable capacity data: about 42,000 isolated measurements per site over six
months, spaced ~6 minutes apart. That supports forecasting at 6 minute
granularity from a history of previous speedtests. It does not support a 30 s
look-back sequence model.

## What this dataset does support, well

Latency is continuous, dense, and well behaved, and "predict latency spikes" is
the **first** industry need named in the problem statement.

- `mean_ping_latency`: median 31.6 ms, p99 40.4 ms, clean diurnal cycle
  (29.8 ms at 08:00 rising to 34.9 ms at 02:00).
- `pop_ping_latency`: median 30.7 ms with a heavy tail to 1197 ms. Against the
  Casparsen threshold of l_t = 50 ms, **3.15% of samples are exceedances**,
  15,199 events at uos-rz alone. That is a well posed spike prediction problem
  with enough positives to evaluate honestly.

## The re-framing, and what survives

**Our contribution transfers intact.** The regime conditioned calibration layer
is signal agnostic: it converts a point forecast into a risk controlled bound
and holds the budget within each regime. On throughput the bound is a lower
bound and the risk is over-allocation. On latency the bound is an **upper**
bound and the risk is under-promising delay. The failure it fixes, risk held on
average and lost in the worst regime, is the same failure.

What does not survive is the reproduction plan. StarNet and BG-CFQS both need
1 Hz throughput with satellite features, and neither can be run on this data.

Revised source roles:

| Source | Role | Status |
|---|---|---|
| WetLinks (supplied) | **primary**: latency forecasting, spike prediction, the calibration layer, allocation | in hand |
| StarNet | reproduction gates and the satellite geometry story | blocked, links expired |
| Horizon | O1 cross location analysis | public, unblocked |

Regime axes available here: obstruction fraction, dish azimuth sector, hour of
day, day of week, weather (joined via Open-Meteo, `ingest/weather.py`), and
TLE-derived candidate count once historical elements are obtained.
