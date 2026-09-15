# The 15 second scheduling phase, recovered independently on three continents

Run 2026-09-01, immediately after the StarNet traces arrived. This is the
cheapest result in the project and one of the more interesting ones, because it
is a crossover neither source paper performed.

## What was done

Casparsen et al. (arXiv 2601.08439) report that Starlink reschedules at the
12th, 27th, 42nd and 57th second of each minute, an offset of 12 s on a 15 s
period. They establish this on their own 500 Hz latency probes at a single
European vantage point.

CLAUDE.md section 6 says explicitly not to hardcode that offset, and
`state/phase.py` does not: it edge-detects on the first difference of the
signal, thresholds, enforces a minimum spacing of 15 s between accepted edges,
histograms the candidate times over phase bins, takes the dominant bin, and
finishes with a weighted circular mean over the top-k bins in its
neighbourhood. The fixed offset is a fallback used only when the histogram peak
falls below threshold, and it was not used here.

Applied to StarNet's 1 Hz **throughput** traces, which is a different signal, a
different sampling rate and three different continents:

| location | recovered offset | method | confidence | edges accepted |
|---|---|---|---|---|
| USA (Chicago) | 11.98 s | recovered | 12.61 | 30,400 |
| Canada (Victoria) | 12.25 s | recovered | 7.92 | 3,140 |
| Germany (Osnabruck) | 12.09 s | recovered | 10.69 | 16,733 |

## Why it is worth recording

All three land within 0.25 s of 12, and none of them was told to. Casparsen's
scheduling reference is confirmed on data they never used, from a signal they
never measured, at 1/500th of their sampling rate, at three sites spanning two
continents.

The two source papers are disjoint here: StarNet has the traces and does not
analyse the scheduling phase as a recoverable quantity; Casparsen recovers the
phase and has no throughput traces or satellite identifiers. Running one
paper's method on the other's data is a small piece of work and it is genuinely
new.

## What it does not show

Casparsen's boundary regions (the first 140 ms and last 75 ms of each period,
carrying a handover spike averaging 74 ms above the period mean) **cannot** be
checked at 1 Hz. One sample per second cannot resolve a 140 ms window. Nothing
here supports or contradicts that part of their work, and CLAUDE.md section 6
already said not to claim those numbers on 1 Hz data.

Nor does this say the phase is *useful*. On the WetLinks per-second release,
conditioning the calibration layer on phase alone was worse than not
conditioning at all (`results/summary/osnabruck-capacity.md`). Recovering a
real periodic structure and that structure carrying risk-relevant information
are two different claims, and only the first is made here.
