# Progress

One entry per phase. Every entry names the commit range and the artifacts it
produced. Nothing is recorded here that was not actually run.

## Phase 0. Scaffold

Status: done.

- Repo structure, `pyproject.toml`, `.gitignore`, `.claude/settings.json` with
  `includeCoAuthoredBy: false`, `README.md`, `docs/references.bib`.
- Git identity verified as `Mayan10 <mayan25sharma@gmail.com>` before the first
  commit.
- Full package skeleton implemented ahead of the phase order so that every
  later phase has a place to land. Implemented does not mean validated: see the
  gates below.

## Phase 1. Reproduce StarNet

Status: code complete, **not yet run**. Blocked on the traces.

The gate is the RMSE/MAE table in CLAUDE.md section 6 (look-back 30, output 5)
within a few percent:

| | RMSE | MAE |
|---|---|---|
| USA | 40.33 | 29.88 |
| Canada | 41.08 | 30.84 |
| Germany | 36.48 | 27.11 |

To run: `python scripts/download_data.py --dataset starnet --dest data/starnet`
then `python scripts/reproduce_starnet.py --data data/starnet --location usa`.
Until that table is reproduced no number from any later phase means anything.

## Phase 2. Reproduce BG-CFQS and expose the gap

Status: code complete, **not yet run**. Blocked on the traces.

Two gates. First the published average table (MAE 40.364, RMSE 52.480,
OverRate 0.349, MPE 11.745, P95+Err 65.834, risk pass 3/3) and the selected
quantiles (CHI 0.314, OSN 0.244, VIC 0.306). Then the conditional OverRate on
the P30 and P10 subsets, which should land in 0.65 to 0.71 and 0.83 to 0.86.
That second table is the motivation figure for the whole project.

To run: `python scripts/reproduce_bgcfqs.py --data data/starnet --all`.

## Phase 3. Evaluation harness

Status: code complete, unit tested on synthetic fixtures.

`eval/metrics.py`, `eval/splits.py`, `eval/runner.py`, `eval/figures.py`.
Metric definitions follow BG-CFQS exactly so the numbers are comparable.

## Phase 4. Regime conditioned calibration

Status: code complete, unit tested on synthetic fixtures, **not yet evaluated
on real traces**.

`state/regime.py` and `calibrate/regime_cal.py`, then `decide/admission.py` and
`decide/congestion.py`. The epsilon sweep and the regime granularity ablation
are wired into the runner grid and have not been run.

## Phase 5. Cross location analysis for O1

Status: not started. Needs the Horizon dataset pull.

## Phase 6. Hardening and writeup

Status: not started.

## Open questions still outstanding

Carried from CLAUDE.md section 12, none answered yet:

1. What is the dataset supplied with the problem statement?
2. Is there a GPU available? The machine this was scaffolded on has Apple MPS
   and no CUDA device, so `reproduce_starnet.py` defaults to MPS.
3. What is the submission deadline?
4. Is the deliverable a report, a demo, a presentation, or all three?
