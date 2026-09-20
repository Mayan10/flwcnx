# Committed summaries

`results/` is gitignored except this directory. Anything here is a real run
with its config snapshot inside the JSON or named in the markdown.

## Read these first

| file | what it is |
|---|---|
| `starnet-regime-grid.md` | **the main result.** Three locations, measured satellite geometry, budget sweep, nine granularities. Online recalibration replicates and dominates; regime conditioning does not. |
| `phase1-starnet-gate.md` | Phase 1 gate, passed. Average within 1.1% RMSE and 2.4% MAE of published. |
| `phase2-bgcfqs-gate.md` | Phase 2 gate, passed, and the split sensitivity finding that was needed to pass it. |
| `requirements-1-5-6.md` | Latency spike prediction, availability and operational cost: the three industry needs the throughput pipeline did not answer. Contains the break-even ratio and the AUPRC argument for online calibration. |
| `protection-canada.md`, `protection-germany.md`, `protection-usa.md` | The protection layer: which flow gets throttled when the bound falls. **The capacity is measured and the flows are a model**, for the reason given in `docs/limitations.md` section 4c, and every table there says so. |

`starnet-regime-grid.md` and `osnabruck-capacity.md` **disagree about the
regime layer**. That is the honest state of the project and both are kept.
Neither is the "real" one.

## StarNet runs

| file | what it is |
|---|---|
| `starnet-usa.md`, `starnet-germany.md`, `starnet-canada.md` | per location detail behind the grid summary |
| `phase-recovery-crossover.md` | Casparsen's 15 s offset recovered independently on three continents |

## WetLinks runs

Still valid, and no longer the primary path. They were the only way to make
progress while the traces were unavailable.

| file | what it is |
|---|---|
| `osnabruck-capacity.md` | full budget sweep on the per-second iperf release, 13,530 test decisions |
| `cross-site-enschede.md` | leave-one-location-out, forecaster trained on Osnabruck only |
| `backbones.md` | the same calibration layer over six forecast backbones |

### Resolved: the worst-regime discrepancy from the pilots

The pilot note below flagged that `worst_regime_OverRate` was identical for the
global and the regime conditioned bound, and said it was unexplained and
therefore not reported as anything. It is now explained.

The per regime breakdown is deliberately computed against one fixed partition
so that different calibration granularities are judged on the same footing. The
runner chose that partition by risk *direction* rather than by dataset, so a
capacity run got the StarNet axes (phase, elevation, distance, candidates).
Without geometry attached, elevation and distance were entirely null, every
sample landed in one bucket, and "worst regime" was the global rate by
construction. `state.regime.presets_for` now keys the reference partition on
the dataset. In the final runs the numbers separate as they should: 0.7273
global conformal against 0.6667 for ours at Osnabruck, and 0.800 against 0.4255
at Enschede.

This was a measurement bug, not a modelling one, and it only ever made our
layer look worse than it was.

## Superseded pilots

Kept because the reasoning in them is still the reasoning that got here, and
because one of them recorded a limitation that the final runs then fixed.

## pilot-wetlinks-uos-rz-latency.json

**Pilot, not a final result.** 4 epochs, stride 30, one site, one budget. Run
to prove the pipeline end to end on real data, not to report numbers.

Site uos-rz, target latency, upper bound, epsilon 0.35. UnderRate is the
fraction of slots where the bound promised a delay the link did not meet.

| method | axes | UnderRate global | P30 | P10 | MAE (ms) |
|---|---|---|---|---|---|
| point | - | 0.3189 | 0.5010 | 0.6180 | 0.5195 |
| global conformal | global | 0.3030 | 0.4855 | 0.6056 | 0.5293 |
| regime conformal | obstruction+hour+azimuth | 0.2965 | 0.4554 | 0.5559 | 0.5284 |
| regime BG-CFQS | obstruction+hour+azimuth | 0.2915 | 0.4512 | 0.5528 | 0.5322 |

Two things worth recording.

**The conditional failure is real on real data.** A global UnderRate of 0.303
against a 0.35 budget sits comfortably inside it, while the same bound runs at
0.606 on the worst decile. That is the BG-CFQS pattern reproduced on a
different dataset, a different target and a different bound direction, which is
better evidence that the problem is structural than reproducing it on their own
setup would have been.

**The layer helps, modestly, and does not cost accuracy.** P30 0.4855 to
0.4554 and P10 0.6056 to 0.5559, with MAE unchanged at 0.529 to 0.528. Much
smaller than the synthetic demonstration, which was built to contain the
effect. This is what it looks like on data that was not.

**Open discrepancy.** `worst_regime_OverRate` is 0.4407 for both the global and
the regime conditioned bound. If the layer is working, the worst regime should
move. Either the worst regime falls back to global anyway under the 200 sample
guard, or the per regime breakdown is being computed against the wrong
partition. Not yet explained, so not yet reported as anything.

## pilot-wetlinks-seconds-capacity.json

**Pilot, not a final result.** 6 epochs, one site, one budget, no geometry.

Osnabrück, per-second iperf release, **capacity** target, lower bound,
epsilon 0.35, lookback 10 / horizon 5 (the 15 sample iperf run forbids more).
OverRate is the fraction of slots where the bound promised capacity the link
did not deliver.

| method | axes | OverRate | P30 | P10 | worst regime | MAE (Mbps) |
|---|---|---|---|---|---|---|
| point | - | 0.5319 | 0.7788 | 0.8707 | 0.5609 | 24.61 |
| global conformal | global | 0.4174 | 0.6721 | 0.7642 | 0.5113 | 24.99 |
| regime conformal | phase | 0.4166 | 0.6755 | 0.7724 | **0.4622** | 24.98 |
| regime BG-CFQS | phase | 0.4086 | 0.6669 | 0.7620 | **0.4532** | 25.06 |

**The phase axis carries real signal.** Conditioning on nothing but the 15 s
scheduling phase moves the worst regime's OverRate from 0.511 to 0.453, and
P95+Err from 57.3 to 54.6, at unchanged MAE. This is the axis the supplied 30 s
subset could not test at all, and the first direct evidence that the regime
idea works on capacity rather than only on latency.

> **Wrong, corrected 2026-08-31.** This claim does not survive the full run.
> It rested on `worst_regime_OverRate`, which was the miscomputed metric
> described at the top of this file, so the 0.511 to 0.453 movement was an
> artefact of the reference partition rather than a property of the phase axis.
>
> On the full grid, conditioning on phase alone gives a P10 OverRate of 0.6711
> against 0.6393 for no conditioning at all: phase is *worse* than nothing. The
> axis that carries the gain is the observed look-back level, which did not
> exist when this pilot was run. Left in place rather than deleted, because a
> summary that quietly drops its wrong claims is not a record.

**The conditional failure reproduces, strongly.** Global 0.417 against P10
0.762. BG-CFQS report 0.349 against 0.83 to 0.86 on their data. Same shape,
independent dataset, and here it is measured against a lower bound on real
capacity rather than reconstructed from their tables.

**Nothing passes the budget, including the global baseline.** Every risk_pass
is False: global conformal lands at 0.417 against a 0.35 budget. This is the
temporal-drift limitation written up in docs/limitations.md, showing up exactly
where that document predicted. Split conformal's guarantee needs exchangeability
and a temporal split deliberately breaks it. Until a rolling calibration is
implemented, no absolute risk claim can be made from these runs, only relative
ones between methods that all share the drift.

**Not comparable to published tables.** MAE 24.6 Mbps looks favourable against
StarNet's 27 to 30, and the comparison is meaningless: different link,
different country, 10/5 rather than 30/5, and a horizon mean rather than their
per-step error. It is recorded here only so nobody later mistakes it for a win.
