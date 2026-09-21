# CLAUDE.md

Project instructions for Claude Code. Read this file fully before touching anything in the repo.

Repository: https://github.com/Mayan10/thalweg (private)
Owner: Mayan Sharma (GitHub: Mayan10)

---

## 0. Git and attribution rules (read first, apply always)

These are non-negotiable and apply from the very first commit.

### Attribution

Claude must not appear anywhere in the commit history, contributor list, or repository metadata.

1. Create or update `.claude/settings.json` in the repo root with:
   ```json
   {
     "includeCoAuthoredBy": false
   }
   ```
2. Never write a `Co-Authored-By: Claude` trailer, a `Co-Authored-By: ... @anthropic.com` trailer, or a "Generated with Claude Code" line into any commit message, PR body, changelog, or file header.
3. Confirm `git config user.name` and `git config user.email` resolve to Mayan's own GitHub identity before the first commit. If they are unset or wrong, stop and ask. Do not guess an email.
4. After the first three commits, verify with:
   ```
   git log --format='%an <%ae>%n%b' | grep -i -E 'claude|anthropic|co-authored'
   ```
   This must return nothing. If it returns anything, stop and report it rather than rewriting history unilaterally.
5. Do not add Claude, Anthropic, or any AI tool to `AUTHORS`, `CONTRIBUTORS`, `package.json`, `pyproject.toml`, README credits, or docstrings.

### Commit discipline

The point of this is that nothing is ever lost. Commit far more often than feels necessary.

- Commit after every meaningful unit of work: a module that imports cleanly, a function with a passing test, a config file, a completed data loader, a figure that renders, a fixed bug.
- Also commit on a periodic basis regardless of milestones. If more than roughly 30 minutes of work or more than about 150 lines have accumulated uncommitted, commit them as a work-in-progress with an honest message.
- Never leave the working tree dirty at the end of a session. If work is incomplete, commit it as `wip: <what is half-done and what is next>`.
- Push to `origin` after every commit or at minimum at the end of every session. Local commits are not a backup.
- Before starting any large refactor or structural move, commit the current state first so there is a clean rollback point.

Commit message format (Conventional Commits, no emoji, no AI attribution):

```
feat(features): recover 15s phase reference via edge detection

Implements the phase-recovery procedure from Casparsen et al. rather than
assuming the 12/27/42/57 offset. Falls back to the fixed offset when the
histogram peak is below threshold.
```

Prefixes: `feat`, `fix`, `refactor`, `test`, `data`, `docs`, `chore`, `exp` (for experiment runs), `wip`.

### Things never to commit

Add these to `.gitignore` on the first commit: raw datasets, model checkpoints, `.env`, API keys, `__pycache__`, `.ipynb_checkpoints`, `outputs/`, `wandb/`, anything over 50 MB. Data lives in `data/` and is gitignored; only the download and preprocessing scripts are versioned.

---

## 1. What we are building

A predictive bandwidth allocation system for Starlink (LEO satellite) access links.

The problem statement given to Mayan:

> Industry needs predictive network management. Industries require systems that can: predict latency spikes; predict throughput degradation; detect congestion before users are affected; optimize bandwidth allocation automatically; improve service availability; reduce operational costs through autonomous network management. We can concentrate on any 3 issues with our dataset.

The three issues selected, chosen because they chain into one system rather than three disconnected models:

1. **Predict throughput degradation** (the forecaster)
2. **Detect congestion before users are affected** (derived from the calibrated bound, not a separate model)
3. **Optimize bandwidth allocation automatically** (the decision layer)

Objectives this maps onto:

- **O1** Analyze Starlink network performance across different time periods and locations. Falls out of the feature layer and the leave-one-location-out evaluation.
- **O2** Study the influence of obstruction and satellite parameters on connectivity. Falls out of serving-satellite resolution and the feature ablation.
- **O3** Develop and evaluate ML models for network-performance prediction. The forecast backbone plus baselines.
- **O4** Design a predictive framework for assessing Starlink connectivity quality. The calibration and allocation layers.

---

## 2. The contribution boundary

This matters more than anything else in this document. Be scrupulous about which parts are reproduction of published work and which part is ours. Never blur this in code comments, README text, or commit messages.

**Reproduction (not ours):**
- The StarNet GRU seq2seq backbone with periodical embedding and attention (Liu et al., CoNEXT 2025).
- The 2D-to-3D obstruction map projection and DTW-based TLE matching for serving-satellite identification (same paper).
- The BG-CFQS budget-guided coarse-to-fine quantile selection baseline (Xie et al., 2026).
- The 15-second period segmentation and boundary isolation (Casparsen et al., 2026).

**Ours (the novel layer):**

> A regime-conditioned calibration layer that converts a point throughput forecast into a safe lower bound whose overestimation rate is controlled *within each operating regime*, where regime is defined by covariates the terminal can compute at prediction time: 15-second phase bucket, serving satellite elevation and distance, and candidate satellite count.

**Why this is a real gap.** BG-CFQS selects a single global quantile to hold the overestimation rate at the budget. Their own results show this fails conditionally. Against a budget of 0.35 they achieve a global OverRate of 0.349, but on the lowest-30% throughput subset the rate rises to 0.65 to 0.71 across their three datasets, and on the lowest-10% subset it reaches 0.83 to 0.86. Risk is controlled on average and lost precisely in the low-capacity regime where over-allocation actually drops sessions. One knob for a system with several distinct operating regimes.

**Optional second contribution (stretch, only if Phase 4 finishes early).** Casparsen's period-level Good/Degraded latency classifier is deliberately provider-agnostic and uses no satellite identifiers, elevation angles, or routing information. Conditioning that same classifier on the serving-satellite geometry that StarNet's tool recovers is a crossover neither paper attempted. Treat as stretch. Do not start it before the core system is evaluated.

---

## 3. Working mode with Mayan

Mayan's stated preference is to read primary documentation and implement things himself, with Claude pointing to what to read and giving guidance rather than being the main driver writing all the code. Respect this by default:

- Before writing a new module, briefly state what it needs to do, what the key design decision is, and which paper section or doc page it comes from. Then write it.
- Comment the *reasoning*, not the syntax. `# 15s phase, not t mod 15, because the scheduler offset is 12/27/42/57` is useful. `# increment counter` is not.
- Work one module at a time. Do not generate five files in one turn.
- Ask before large structural moves (renaming packages, changing the data schema, swapping a framework).
- Leave the final small extension of each module for Mayan to write himself, and say explicitly which piece you left.

If Mayan says to just build it, drop this mode and build it. He moves fast and this should not slow him down.

**Style:** never use em dashes in any code comment, docstring, commit message, README, or file this project produces.

---

## 4. System architecture

Five layers. Each layer only talks to the one below it. This separation is what makes the thing testable.

```
ingest/     raw sources to a normalized frame     (no ML, no features)
state/      frame to feature vectors + regime id  (no model)
forecast/   feature vectors to point prediction   (StarNet backbone)
calibrate/  point prediction to safe lower bound  (OUR LAYER)
decide/     safe bound to allocation + alerts     (no ML)
eval/       harness, splits, metrics, figures
```

Proposed package layout:

```
thalweg/
  __init__.py
  config.py              dataclass config, no globals
  ingest/
    base.py              Source interface: iter_windows() -> RawWindow
    replay.py            reads StarNet CSV traces (primary path)
    live.py              starlink-grpc + CelesTrak + OpenMeteo (stub is fine)
    tle.py               SGP4 propagation, candidate satellite count
    weather.py           OpenMeteo join by lat/lon and timestamp
  state/
    phase.py             15s phase recovery via edge detection
    satellite.py         serving satellite resolution (2D-3D + DTW match)
    regime.py            regime assignment (OURS - see section 7)
    features.py          assembles the 11-feature vector
  forecast/
    starnet.py           GRU seq2seq + periodical embedding + attention
    baselines.py         XGBoost/T3P, DLinear, PatchTST, TimesNet
    train.py
  calibrate/
    conformal.py         one-sided split conformal lower bound
    regime_cal.py        OUR regime-conditioned calibration
    bgcfqs.py            BG-CFQS baseline reimplementation
  decide/
    admission.py         admission control from the safe bound
    congestion.py        congestion flag from bound vs commitment
  eval/
    splits.py            temporal holdout, leave-one-location-out
    metrics.py           MAE, RMSE, OverRate, MPE, P95+Err, conditional variants
    runner.py            experiment grid
    figures.py
tests/
scripts/
  download_data.py
  reproduce_starnet.py
  reproduce_bgcfqs.py
data/                    gitignored
results/                 gitignored except committed summary tables
```

**This layout is the repository root**, as of the 2026-09-21 restructure. The
package spent part of the project nested at `newml/thalweg/`, which put the
README that carries every figure two directories down from the front page and
gave the repository a second, thinner README at the top. The web console, the
terminal console and the accounts service now live under `services/`, and the
Dockerfile that builds the ML service is at the root and builds from it.

### Two ingestion modes

Build `LiveSource` and `ReplaySource` behind one interface from day one. Mayan almost certainly does not have a Starlink dish, so `ReplaySource` over the published StarNet CSVs is the real path. `LiveSource` can stay a stub that raises `NotImplementedError` on `connect()`, but the interface must exist so the architecture is honest rather than a notebook wearing a diagram.

### Congestion detection is derived, not modeled

Do not train a separate congestion classifier. Congestion is: the calibrated lower bound falls below the currently committed allocation for a sustained window of W slots. This gets issue 2 nearly free and keeps the system coherent.

---

## 5. Datasets

### Primary: StarNet traces (use this one)

- Repo and data: https://github.com/ConnectedSystemsLab/StarNet
- Three locations: USA, Canada, Germany. Gen 3 dishy, rev3_proto2, collected via own setup plus LEOScope.
- Contains serving satellite ID, distance, azimuth, elevation, candidate satellite count, time of day, day of week, precipitation, cloudiness, pressure, and throughput at 1 Hz.

Published dataset statistics (use these to verify your loader is reading the data correctly):

| | USA | Canada | Germany |
|---|---|---|---|
| Duration | 6 months | 1 month | 1 month |
| Cumulative trace minutes | 41,252 | 2,417 | 10,221 |
| Total throughput samples | 2,475,163 | 145,053 | 613,295 |
| Unique serving satellites | 6,052 | 3,166 | 3,956 |
| Satellite handovers | 86,808 | 7,257 | 26,782 |

The BG-CFQS paper renames these CHI (US), OSN (Germany), VIC (Canada) and reports processed ranges 2024-04-26 to 2024-05-28 (CHI, 1,123,832 samples used), 2024-07-13 to 2024-07-31 (OSN), 2024-07-11 to 2024-07-28 (VIC). Match this processing so the baseline comparison is apples to apples.

### Secondary: Horizon

- Code: https://github.com/spear-lab/Horizon-Predicting-Starlink-Performance
- Dataset DOI: https://doi.org/10.4121/0bf59468-e5cb-433f-aeb2-e04cf694b65c
- Crowdsourced M-Lab NDT7 + Cloudflare AIM, 11 months (Jan to Nov 2025), 90+ countries, ~15.6M NDT7 speedtests and ~157K AIM measurements, Starlink identified by AS14593. Both are publicly accessible via BigQuery.
- Use this only for the O1 cross-location analysis. It is hourly-aggregated and has no terminal telemetry, so it cannot support the calibration work.

### Live-source dependencies (for the stub, and for O2)

- Terminal telemetry: https://github.com/sparky8512/starlink-grpc-tools
- Orbital elements: https://celestrak.org (daily TLE, propagate with SGP4 via the `sgp4` and `skyfield` Python packages)
- Weather: https://open-meteo.com (free hourly, ~9 km resolution)

### Optional, for cross-validation of findings only

- WetLinks (longitudinal Starlink with contiguous weather): arXiv 2402.16448
- LENS LEO measurement dataset (Zhao and Pan, MMSys 2024)
- IRTT for high-rate one-way delay if a dish ever becomes available: https://github.com/heistp/irtt

### Mayan's supplied dataset

**OPEN QUESTION. Do not assume.** Mayan was handed a dataset with the problem statement and has not yet confirmed its columns or sampling rate. If a dataset appears under `data/supplied/`, inspect it and report the schema before building any feature code against it. If the deliverable is required to use it, the StarNet traces become supporting evidence rather than the primary result and the feature layer must be rebuilt around the available columns. Ask before restructuring on this basis.

---

## 6. Reference implementations and reproduction targets

Reproduce these numbers before writing any of our own code. If they do not reproduce within a few percent, something is wrong with the data loader and everything downstream is meaningless.

### StarNet (Phase 1 target)

Reference code: https://github.com/ConnectedSystemsLab/StarNet (includes the model, the TimesNet baseline, and the measurement tooling)

Published configuration:
- Encoder: 2-layer GRU, hidden size 128. Decoder: 2-layer GRU, hidden 128, input size 128 + 11.
- Projection head: 2 linear layers, 128 hidden units, LeakyReLU, output dim 1.
- Attention: three 1-layer MLPs, input size 128.
- Periodical embedding: 1D conv over each of the four feature classes to L x 48, plus an L x 1 vector holding the second within the current 15s interval, values in [0, 15).
- Input 11 features, look-back 30 s, output 15 s (5 s in some comparisons).
- AdamW, lr 0.001, decay 0.99 per step, batch 512.
- Data points: 24,430 (USA), 24,168 (Canada), 16,916 (Germany). Sequence length 45 (30 in, 15 out). Step sizes 46, 6, 29 respectively.
- Split 8:2 into two **contiguous blocks**, not interleaved samples. This matters.
- Converges in about an hour, 50 epochs, on an RTX 3070. Inference 3.9 ms per batch.

Published results to hit (look-back 30, output 5):

| | RMSE | MAE |
|---|---|---|
| USA | 40.33 | 29.88 |
| Canada | 41.08 | 30.84 |
| Germany | 36.48 | 27.11 |
| Average | 39.30 | 29.28 |

Also: median error 33.57 Mbps vs T3P 43.63 (30.3% better). Ablations: without periodical embedding 38.00, without attention 37.01. Latency prediction median error 2.5 ms.

### BG-CFQS (Phase 2 target, this is our baseline)

Paper: Xie et al., *Risk-Aware Safe Throughput Forecasting for Starlink Networks*, arXiv 2605.09508. No public code, so reimplement from the paper. The algorithm is fully specified (Algorithm 1).

Configuration:
- History L = 75, horizon H = 15
- Default risk budget epsilon = 0.35
- Candidate quantile set T = [0.15, 0.40]
- Coarse tolerance delta = 0.05, fine grid size M = 5
- XGBoost backbone trained with pinball loss
- Train / calibration / test protocol: calibration set used only for quantile selection, never for fitting
- Excluded from model inputs: raw timestamp, latency
- Included auxiliary variables: elevation, azimuth, satellite distance, encoded satellite ID, candidate count, cloud coverage, pressure, humidity, 15-second phase, minute, hour, day of week

Published averages over CHI, OSN, VIC:

| Method | MAE | RMSE | OverRate | MPE | P95+Err | Risk pass |
|---|---|---|---|---|---|---|
| T3P-point | 37.775 | 49.699 | 0.512 | 20.182 | 87.984 | 0/3 |
| StarNet-point | 39.806 | 52.102 | 0.433 | 17.030 | 82.842 | 0/3 |
| BG-CFQS | 40.364 | 52.480 | 0.349 | 11.745 | 65.834 | 3/3 |

Selected quantiles: CHI 0.314, OSN 0.244, VIC 0.306.

**The failure we exploit.** Conditional OverRate on low-throughput subsets:

| Dataset | High-risk P30 | Severe-risk P10 |
|---|---|---|
| CHI | 0.65 | 0.83 |
| OSN | 0.71 | 0.86 |
| VIC | 0.71 | 0.86 |

Reproducing this table is the single most important intermediate result in the project. It is the motivation figure. Commit it as soon as it exists.

### Casparsen phase segmentation (Phase 1, feature layer)

Paper: *Statistical Characterization and Prediction of E2E Latency over LEO Satellite Networks*, arXiv 2601.08439. No public code.

What to implement:
- Rescheduling points occur at the 12th, 27th, 42nd, and 57th second of each minute. **Do not hardcode this.** Implement the phase recovery: edge-detect on the first difference of the signal, threshold, enforce minimum spacing of T = 15 s between accepted edges, histogram candidate times over phase bins, take the dominant bin, then a weighted circular mean over the top-k bins in its neighbourhood. The paper reports this reference stays stable across months. Fall back to the fixed offset only if the histogram peak is below threshold.
- Boundary regions: the first 140 ms and last 75 ms of each period carry the handover spike (average 74 ms above the period mean). Exclude these from intra-period statistics.
- Period-level class: Good if at least 99% of packets meet the latency threshold l_t, Degraded otherwise. Their threshold l_t = 50 ms on the 99th latency quantile, which is typical for cyber-physical systems and 5G NR satellite access.
- Their models: uniform, Gaussian, GMM with 2 and 3 kernels, quantile regression, and EVT with a Generalized Pareto tail using a fixed top-k of 25. GMM-3 reaches AUPRC 0.95 in 1.6 s; EVT is slower to converge but reaches near-perfect classification after 3.5 s.

Note their sampling rate is 500 Hz (2 ms probes, S = 7500 samples per 15 s period) and the StarNet traces are 1 Hz. The boundary-region analysis will not transfer directly. Use the phase recovery and the period-level framing; do not claim the 140 ms / 75 ms numbers on 1 Hz data.

### Horizon (Phase 5, for O1 only)

Code and dataset linked in section 5. Key findings to cite and to test against:
- Latency prediction is best with a **two-month** training window (MAE 24.12 ms); throughput improves monotonically out to **eleven months** (MAE 39.13 Mbps). This asymmetry is direct evidence that these signals are non-stationary on different timescales and is why our splits must be temporal, not random.
- Best anomaly filtering: percentile with k = 0.75 for latency, Isolation Forest with contamination 0.25 for throughput.
- Model: Random Forest (100 trees) + Gradient Boosting (100 trees, lr 0.1) ensemble, RobustScaler, weights grid-searched in 0.05 increments.
- Feature importance: latitude alone 42% (latency) and 46% (throughput); Weather Index 14% and 15.3%; satellite density 10.3% and 6.4%; client-server distance 8.3% for latency and 1.1% for throughput.

---

## 7. The novel layer, specified

`calibrate/regime_cal.py`.

### Regime definition

A regime is a bucket over covariates available at prediction time. Start with this and ablate:

- `phase_bucket`: the 15 s interval split into buckets. Start with 3: opening (first ~2 s, where StarNet's attention concentrates and throughput drops are largest), middle, closing.
- `elevation_bucket`: serving satellite elevation. StarNet found throughput rises with elevation and plateaus above 60 degrees, and all serving satellites are above 25 degrees per the FCC filing. Suggest bins at 25 to 45, 45 to 60, above 60.
- `distance_bucket`: serving satellite distance. StarNet found throughput holds around 230 Mbps until 645 km then declines. Suggest below 645 km and above 645 km.
- `candidate_bucket`: number of visible candidate satellites. StarNet found average throughput rises 26% going from 15 to 45 candidates. Suggest terciles.

Full cross product is too sparse. Design for a configurable regime function with a minimum-sample guard: if a regime has fewer than N calibration points, fall back up a hierarchy to a coarser regime, and ultimately to the global calibration. Make the fallback explicit and logged, not silent.

### Calibration procedure

1. Split into train / calibration / test. Calibration is used only for selecting the operating point, never for fitting. Splits must be temporal.
2. Fit the point forecaster on train.
3. On calibration, compute residuals grouped by regime.
4. Per regime, select the lower bound that satisfies the overestimation budget epsilon within that regime, using either a per-regime quantile of the residual distribution (one-sided split conformal style) or a per-regime run of the BG-CFQS boundary search. Implement both; compare.
5. At test time, assign the regime, apply that regime's bound.

### What must be true for this to be a result

The global OverRate should stay at or below epsilon (matching BG-CFQS), while the **conditional** OverRate on the P30 and P10 subsets drops substantially below the 0.65 to 0.86 range BG-CFQS reports, at comparable or better MAE. If MAE collapses because the bounds became uselessly conservative, that is a negative result and must be reported honestly, not hidden by only reporting the risk metrics.

---

## 8. Evaluation protocol

The headline table is not MAE. Build `eval/runner.py` to produce all of this as a grid.

**Metrics** (implement in `eval/metrics.py`, following the BG-CFQS definitions exactly so numbers are comparable):
- MAE, RMSE
- OverRate: fraction of predictions where predicted > actual
- MPE: mean of max(predicted - actual, 0)
- P95+Err: 95th percentile of the positive error component
- All of the above computed **globally and per regime**, and on the High-risk P30 (lowest 30% of true throughput) and Severe-risk P10 (lowest 10%) subsets

**Splits:**
- Temporal holdout, not random. Random splits flatter these models; Horizon's window-length asymmetry is the evidence.
- Leave-one-location-out across the three StarNet countries.
- Contiguous 8:2 blocks, matching StarNet's protocol, for the reproduction runs.

**Sweeps:**
- Risk budget epsilon from 0.05 to 0.35. BG-CFQS only reports 0.35, which permits a 35% overestimation rate. No real allocator would accept that. Our advantage should widen as epsilon tightens. If it does not, find that out in Phase 4, not at the end.
- Regime granularity: global, phase only, phase + elevation, full cross product.

**Downstream evaluation:**
Admission control with b = 10 Mbps per service. Admit floor(safe_forecast / b) sessions against an oracle capacity of floor(actual / b). Report mean dropped sessions, violation rate, and P95 dropped sessions, on all decisions and on the P30 and P10 subsets. BG-CFQS reports 6.8% / 11.0% / 12.6% relative reduction in dropped sessions on those three slices against budget-scale baselines. Beat that or explain why not.

**Ablation:**
Which conditioning variables carry the gain. If phase alone gets most of it and satellite geometry adds nothing, report that plainly. A clean negative ablation is a result.

---

## 9. Bibliography

Maintain this as `docs/references.bib`. Twenty-four entries, grouped by why they are here. Do not pad it further; every entry should be cited somewhere in the writeup.

**Measurement and characterization (O1, O2)**
1. Michel, Trevisan, Giordano, Bonaventure. A First Look at Starlink Performance. IMC 2022.
2. Kassem, Raman, Perino, Sastry. A Browser-side View of Starlink Connectivity. IMC 2022.
3. Ma, Chou, Zhao, Chen, Ma, Liu. Network Characteristics of LEO Satellite Constellations: A Starlink-Based Measurement from End Users. INFOCOM 2023.
4. Mohan, Ferguson, Cech, Bose, Renatin, Marina, Ott. A Multifaceted Look at Starlink Performance. WWW 2024.
5. Garcia, Sundberg, Caso, Brunstrom. Multi-timescale Evaluation of Starlink Throughput. LEO-NET 2023.
6. Tanveer, Puchol, Singh, Bianchi, Nithyanand. Making Sense of Constellations: Methodologies for Understanding Starlink's Scheduling Algorithms. CoNEXT Companion 2023.
7. Izhikevich, Tran, Izhikevich, Akiwate, Durumeric. Democratizing LEO Satellite Network Measurement. SIGMETRICS 2024.
8. Pan, Zhao, Cai. Measuring a Low-Earth-Orbit Satellite Network. 2023.

**Datasets and weather**
9. Laniewski, Lanfer, Meijerink, van Rijswijk-Deij, Aschenbruck. WetLinks: A Large-Scale Longitudinal Starlink Dataset with Contiguous Weather Data. TMA 2024.
10. Lanfer, Laniewski, Otten, Aschenbruck. Weather-Based Link Prediction for LEO-Satellite Networks using the WetLinks Dataset. IFIP Networking 2024.
11. Zhao, Pan. LENS: A LEO Satellite Network Measurement Dataset. MMSys 2024.

**Prediction (O3, O4)**
12. Liu, Reidys, Tanveer, Vasisht. Vivisecting Starlink Throughput: Measurement and Prediction (StarNet). Proc. ACM Netw. 3, CoNEXT4, 2025. **Base paper.**
13. Tiwari et al. T3P: Demystifying Low-Earth Orbit Satellite Broadband. arXiv 2310.11835, 2023.
14. Benghe, Graure, Shreedhar, Mohan. Horizon: Understanding and Predicting Global Starlink Performance. Proc. ACM Meas. Anal. Comput. Syst. 10(2), SIGMETRICS 2026.
15. Casparsen, Jakobsen, Nielsen, Popovski, Leyva Mayorga. Statistical Characterization and Prediction of E2E Latency over LEO Satellite Networks. arXiv 2601.08439, 2026.
16. Narayanan et al. Lumos5G: Mapping and Predicting Commercial mmWave 5G Throughput. IMC 2020.

**Forecasting backbones (baselines)**
17. Zeng, Chen, Zhang, Xu. Are Transformers Effective for Time Series Forecasting? (DLinear). AAAI 2023.
18. Nie, Nguyen, Sinthong, Kalagnanam. A Time Series is Worth 64 Words (PatchTST). arXiv 2211.14730.
19. Wu, Hu, Liu, Zhou, Wang, Long. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. arXiv 2210.02186.
20. Chen, Guestrin. XGBoost: A Scalable Tree Boosting System. KDD 2016.

**Risk, calibration, and the decision layer**
21. Xie, Zhang, Luo, Zhang, Yang, Zhang, Soong. Risk-Aware Safe Throughput Forecasting for Starlink Networks (BG-CFQS). arXiv 2605.09508, 2026. **Direct baseline.**
22. Koenker, Hallock. Quantile Regression. Cambridge University Press.
23. Yin, Jindal, Sekar, Sinopoli. A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP (MPC / RobustMPC). SIGCOMM 2015.
24. Vovk, Gammerman, Shafer. Algorithmic Learning in a Random World.

---

## 10. Build phases

Do not skip ahead. Each phase ends with a commit, a pushed branch, and a short note in `docs/progress.md`.

**Phase 0. Scaffold (half a day)**
Repo structure, `pyproject.toml`, `.gitignore`, `.claude/settings.json` with the attribution setting, `README.md` stating what this is, `docs/references.bib`. Verify git identity. Commit and push.

**Phase 1. Reproduce StarNet (weeks 1 to 2)**
Get their repo running on their released traces. Hit the RMSE/MAE table in section 6 within a few percent. Build `ingest/replay.py` and `state/features.py` around their data format. Implement `state/phase.py`. Do not write any of our own model code until this reproduces. This is the gate.

**Phase 2. Reproduce BG-CFQS and expose the gap (weeks 3 to 4)**
Reimplement from the paper with their exact config. Match their average table. Then compute the conditional OverRate on P30 and P10 and reproduce the 0.65 to 0.86 numbers. This is the motivation figure and the point at which we know the project has a spine. Commit the figure and the table.

**Phase 3. Evaluation harness (week 5)**
`eval/metrics.py`, `eval/splits.py`, `eval/runner.py`. Temporal and leave-one-location-out splits. Build this before the novel layer so the novel layer is measured properly from its first run.

**Phase 4. The regime-conditioned calibration layer (weeks 5 to 8)**
`state/regime.py` and `calibrate/regime_cal.py`. Then `decide/admission.py` and `decide/congestion.py`. Run the full grid including the epsilon sweep and the regime-granularity ablation.

**Phase 5. Cross-location analysis for O1 (week 9)**
Horizon dataset, leave-one-location-out, the training-window-length question.

**Phase 6. Hardening and writeup (remaining)**
`ingest/live.py` fleshed out, tests, figures, README with the architecture diagram, honest limitations section. Latency head only if there is genuine time left.

---

## 11. Standing rules

- Python 3.11+. Type hints on public functions. `ruff` and `pytest`.
- Never fabricate a number. If a reproduction does not match the published value, report the gap and investigate; do not report the published number as if we obtained it.
- Never train on the calibration set or the test set. If a leak is suspected, stop and check before running more experiments.
- Every experiment run writes a config snapshot alongside its results so any figure can be traced back to the exact settings that produced it.
- Seed everything and record the seed.
- When a design decision comes from a paper, cite the paper and section in the code comment.
- If a published method turns out not to reproduce, that is a finding worth writing down, not a failure to hide.

## 12. Open questions to raise with Mayan

> **Status as of 2026-09-03.** Questions 1, 2 and 4 are answered and the answers
> are recorded where the work needed them: the supplied dataset is WetLinks
> (`docs/supplied-dataset.md`), compute is Apple MPS with no CUDA and the project
> now runs on CPU and CUDA too (`thalweg/device.py`), and the deliverable is a
> report, a paper and a demonstration, all three of which exist. Question 3, the
> deadline, is still unanswered. Questions 5 to 7 were raised by the build and
> are tracked in `docs/progress.md`.


1. What is the dataset supplied with the problem statement? Columns, sampling rate, and whether the deliverable must use it.
2. Is there a GPU available, and what kind? StarNet trains in about an hour on an RTX 3070.
3. What is the actual submission deadline for the project?
4. Is the deliverable a report, a demo, a presentation, or all three? This changes how much goes into `decide/` and the live path.
