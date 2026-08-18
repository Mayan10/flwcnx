# Progress

One entry per phase. Every entry names the commit range and the artifacts it
produced. Nothing is recorded here that was not actually run.

## Phase 0. Scaffold

Status: done.

- Repo structure, `pyproject.toml`, `.gitignore`, `.claude/settings.json` with
  `includeCoAuthoredBy: false`, `README.md`, `docs/references.bib`.
- Git identity verified as `Mayan10 <mayan25sharma@gmail.com>` before the first
  commit.
- Full package implemented ahead of the phase order so that every later phase
  has a place to land. Implemented does not mean validated: see the gates below.
- 110 tests, ruff clean, package installable with `pip install -e ".[dev]"`.
- Attribution check returns nothing:
  `git log --format='%an <%ae>%n%b' | grep -i -E 'claude|anthropic|co-authored'`

## Phase 1. Reproduce StarNet

Status: code complete, **not yet run**. Blocked on the traces, which cannot be
downloaded programmatically (see `docs/data.md`).

What was learned from their released code while building the loader, all of it
recorded in `docs/data.md`:

- the traces are pickled DataFrames on OneDrive, not CSVs in the repo;
- the columns are `alt`, `az`, `sat_name`, `n_candidates`, `clouds`, `humidity`,
  not the names assumed in the brief;
- the traces **do** carry latency, which makes the Casparsen crossover cheaper
  than treating it as a stretch implied;
- the weather variable is humidity, not precipitation;
- there are thirteen model inputs, not eleven;
- their loader fits its scaler on the whole trace before splitting, so their
  published numbers carry a scale leak we deliberately do not reproduce.

The gate is the RMSE/MAE table in CLAUDE.md section 6 (look-back 30, output 5)
within a few percent:

| | RMSE | MAE |
|---|---|---|
| USA | 40.33 | 29.88 |
| Canada | 41.08 | 30.84 |
| Germany | 36.48 | 27.11 |

To run:

```
python scripts/download_data.py --dataset starnet --dest data/starnet
# download the three OneDrive folders by hand, then:
python scripts/download_data.py --inspect data/starnet/usa
python scripts/reproduce_starnet.py --location all --ablations
```

Until that table is reproduced no number from any later phase means anything.
If the gap is small and stubborn, the two things to try first are
`--hidden 60` (their code default, against the paper text's 128) and their
whole-trace scaler.

## Phase 2. Reproduce BG-CFQS and expose the gap

Status: code complete, **not yet run**. Blocked on the traces.

Two gates. First the published average table (MAE 40.364, RMSE 52.480,
OverRate 0.349, MPE 11.745, P95+Err 65.834, risk pass 3/3) and the selected
quantiles (CHI 0.314, OSN 0.244, VIC 0.306). Then the conditional OverRate on
the P30 and P10 subsets, which should land in 0.65 to 0.71 and 0.83 to 0.86.
That second table is the motivation figure for the whole project.

To run: `python scripts/reproduce_bgcfqs.py --all`. The motivation figure is
written on every run.

## Phase 3. Evaluation harness

Status: code complete, unit tested on synthetic fixtures.

`eval/metrics.py`, `eval/splits.py`, `eval/runner.py`, `eval/figures.py`.
Metric definitions follow BG-CFQS exactly so the numbers are comparable.

## Phase 4. Regime conditioned calibration

Status: code complete, unit tested on synthetic fixtures, **not yet evaluated
on real traces**.

The mechanism is confirmed on constructed heteroscedastic data, where low
elevation means low capacity and wide residuals at the same time. There the
layer moves P30 OverRate from 0.408 to 0.356 and P10 from 0.404 to 0.355 with
the global rate and MAE unchanged. That is a demonstration that the code does
what it claims, on data built to contain the effect. It is **not a result** and
must never be quoted as one.

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

And one raised by the build:

5. The brief says the StarNet traces are throughput only and treats the
   Casparsen latency crossover as a stretch. They carry latency. Is that
   crossover worth promoting now that it is cheap, or does it stay behind the
   core system?
