# Review 1 content pack

Self-contained source material for the four rubric sections. Written to be
pasted whole into an assistant with no other context, to generate slides and a
report.

Project: **Predictive bandwidth allocation for Starlink (LEO satellite) access
links.** Student: Mayan Sharma. Repository: `flwcnx`.

Rubric covered: Domain/Problem Statement (5), Literature Review (5, min. 15
recent papers), Design of Proposed Methodology (5), Module Description /
System Design (5).

> **Verify before submitting.** The bibliography below was assembled from the
> project brief. One entry in that brief had incorrect author names (StarNet),
> found and corrected against the official repository and the ACM DL record.
> Others have not been independently checked. Confirm authors, venue and year
> for every entry you cite.

---

# SECTION 1, Knowledge on Domain / Problem Statement (5 marks)

## 1.1 The domain in one paragraph

Starlink is a Low-Earth-Orbit (LEO) satellite broadband constellation. Unlike
geostationary satellites, which sit at ~35,786 km and stay fixed over one point
of the Earth, LEO satellites orbit at roughly 550 km and move at about 7.5 km/s.
A given satellite is only usable from a fixed ground terminal for a few minutes
before the terminal must be handed over to another. This buys a large latency
reduction (roughly 20-50 ms round trip rather than 600+ ms) at the cost of a
link whose capacity is **non-stationary by construction**.

## 1.2 What makes the link hard to predict

Five mechanisms, each of which is a feature the system exploits:

1. **A 15-second scheduling cycle.** Starlink reassigns terminals to satellites
   on a fixed 15 s cadence. Published measurement work places the rescheduling
   instants at the 12th, 27th, 42nd and 57th second of each minute, which all
   reduce to a single phase offset of 12 s modulo 15. Throughput and latency
   both show a characteristic disturbance at these boundaries.
2. **Satellite handovers.** At each boundary the terminal may switch serving
   satellite, producing a measurable throughput dip and latency spike.
3. **Serving-satellite geometry.** Throughput rises with the serving
   satellite's elevation angle and plateaus above roughly 60 degrees. It holds
   roughly flat with distance until about 645 km and then declines. All serving
   satellites are above 25 degrees elevation, per the operator's FCC filing.
4. **Candidate satellite density.** More visible satellites means more
   scheduling freedom. Published work reports average throughput rising about
   26% going from 15 to 45 visible candidates.
5. **Weather and obstruction.** Rain fade, cloud cover and physical obstruction
   of the terminal's field of view all degrade the link.

**The key statistical property:** the error of any throughput forecaster on this
link is *heteroscedastic*. It is not merely noisy; it is noisier in some
operating conditions than others, and specifically noisiest when capacity is
lowest. That single fact is what the proposed contribution addresses.

## 1.3 The industry problem statement (as given)

> Industry needs predictive network management. Industries require systems that
> can: predict latency spikes; predict throughput degradation; detect congestion
> before users are affected; optimize bandwidth allocation automatically;
> improve service availability; reduce operational costs through autonomous
> network management.

Three of the six are addressed, chosen because they **chain into one system**
rather than being three disconnected models:

| # | Issue addressed | How |
|---|---|---|
| 1 | Predict throughput degradation | The forecasting backbone |
| 2 | Detect congestion before users are affected | Derived from the calibrated bound, not a separate model |
| 3 | Optimize bandwidth allocation automatically | The decision layer acting on that bound |

The design point worth stating explicitly in a viva: **congestion detection is
not a separate trained classifier.** Congestion is defined as the calibrated
lower bound falling below the committed allocation for a sustained window of W
slots. A second model would have its own errors and its own calibration, and no
guarantee of agreeing with the allocator about when the link is in trouble.

## 1.4 Objectives

- **O1** Analyse Starlink network performance across different time periods and
  locations. (Falls out of the feature layer and leave-one-location-out
  evaluation.)
- **O2** Study the influence of obstruction and satellite parameters on
  connectivity. (Falls out of serving-satellite resolution and feature
  ablation.)
- **O3** Develop and evaluate ML models for network-performance prediction.
  (The forecasting backbone plus baselines.)
- **O4** Design a predictive framework for assessing Starlink connectivity
  quality. (The calibration and allocation layers.)

## 1.5 The gap, stated precisely

A point forecast is not directly usable for allocation. If a scheduler allocates
against a prediction of 200 Mbps and the link delivers 150, sessions are
dropped. What an allocator needs is a **safe lower bound**: a number the link
will beat with a controlled probability.

Existing work produces such a bound by selecting a single global quantile so
that the *overall* overestimation rate meets a budget. The published evidence
shows this fails conditionally: against a budget of 0.35 the reported global
overestimation rate is 0.349, but on the lowest-30% throughput subset it rises
to 0.65-0.71, and on the lowest-10% subset to 0.83-0.86.

**Risk is controlled on average and lost precisely in the low-capacity regime,
which is exactly where over-allocation actually drops sessions.** One knob is
being used to control a system with several distinct operating regimes.

---

# SECTION 2, Literature Review (5 marks, minimum 15 recent papers)

Twenty entries, grouped by the role each plays. For each: what they did, the
finding that matters, and what it leaves open.

## Group A, Measurement and characterisation of LEO links (O1, O2)

**1. Michel, Trevisan, Giordano, Bonaventure. "A First Look at Starlink
Performance." ACM IMC, 2022.**
First systematic academic measurement of Starlink from an end user. Established
baseline latency, throughput and loss characteristics and the presence of
periodic performance variation. *Leaves open:* no prediction, single vantage
point.

**2. Kassem, Raman, Perino, Sastry. "A Browser-side View of Starlink
Connectivity." ACM IMC, 2022.**
Measured Starlink from the application layer via browser-based tests across many
users. Showed application-visible performance differs substantially from
link-layer measurements. *Leaves open:* no terminal telemetry.

**3. Ma, Chou, Zhao, Chen, Ma, Liu. "Network Characteristics of LEO Satellite
Constellations: A Starlink-Based Measurement from End Users." IEEE INFOCOM,
2023.**
Characterised throughput, latency and loss and connected them to constellation
dynamics. Among the first to link observed variation to satellite motion.

**4. Mohan, Ferguson, Cech, Bose, Renatin, Marina, Ott. "A Multifaceted Look at
Starlink Performance." ACM Web Conference (WWW), 2024.**
Large multi-vantage-point study covering geography, time of day, weather and
application performance. Strong evidence that performance is location- and
condition-dependent, motivating conditioning rather than global modelling.

**5. Garcia, Sundberg, Caso, Brunstrom. "Multi-timescale Evaluation of Starlink
Throughput." ACM LEO-NET Workshop, 2023.**
Showed Starlink throughput has structure at several distinct timescales
simultaneously. Directly motivates a model with an explicit periodic component.

**6. Tanveer, Puchol, Singh, Bianchi, Nithyanand. "Making Sense of
Constellations: Methodologies for Understanding Starlink's Scheduling
Algorithms." ACM CoNEXT Companion, 2023.**
Reverse-engineered the scheduling behaviour, establishing the **15-second
reconfiguration interval**. This is the origin of the phase structure the
proposed method conditions on.

**7. Izhikevich, Tran, Izhikevich, Akiwate, Durumeric. "Democratizing LEO
Satellite Network Measurement." ACM SIGMETRICS, 2024.**
Methods for measuring LEO networks at scale without privileged access.
Methodological grounding for measurement-driven study.

**8. Pan, Zhao, Cai. "Measuring a Low-Earth-Orbit Satellite Network." 2023.**
Longitudinal measurement including the relationship between satellite
visibility and observed performance.

## Group B, Datasets and weather

**9. Laniewski, Lanfer, Meijerink, van Rijswijk-Deij, Aschenbruck. "WetLinks: A
Large-Scale Longitudinal Starlink Dataset with Contiguous Weather Data." IFIP
TMA, 2024.**
**This project's primary dataset.** Two European terminals (Osnabrück, Enschede)
measured continuously for six months, with co-located professional weather
stations. Provides per-second iperf throughput, terminal status, latency and
weather. *Leaves open:* no serving-satellite identification.

**10. Lanfer, Laniewski, Otten, Aschenbruck. "Weather-Based Link Prediction for
LEO-Satellite Networks using the WetLinks Dataset." IFIP Networking, 2024.**
Predicts link degradation from weather on the same dataset. A direct comparison
point, and evidence weather is a usable predictive covariate.

**11. Zhao, Pan. "LENS: A LEO Satellite Network Measurement Dataset." ACM
MMSys, 2024.**
Twenty measurement locations, high-rate latency and obstruction maps.
Cross-validation source for the latency findings.

## Group C, Prediction (O3, O4)

**12. Liu, Reidys, Tanveer, Vasisht. "Vivisecting Starlink Throughput:
Measurement and Prediction" (StarNet). Proc. ACM Networking, 3(CoNEXT4), 2025.
DOI 10.1145/3768971., BASE PAPER.**
The forecasting backbone this project reproduces. A GRU sequence-to-sequence
model with two innovations: a **periodical embedding** that convolves each
feature class separately and carries an explicit within-period phase channel,
and an **attention** mechanism over the look-back. Also contributes a tool that
recovers *which satellite is currently serving* the terminal by projecting the
dish's 2D obstruction map into 3D and DTW-matching against propagated orbital
elements. Reported errors: RMSE 40.33 / MAE 29.88 (USA), 41.08 / 30.84
(Canada), 36.48 / 27.11 (Germany); median error 33.57 Mbps, about 30% better
than T3P. Ablations: 38.00 without periodical embedding, 37.01 without
attention. *Leaves open:* produces a point forecast with no risk control.

**13. Tiwari et al. "T3P: Demystifying Low-Earth Orbit Satellite Broadband."
arXiv:2310.11835, 2023.**
Earlier learned throughput predictor for LEO, used as the standard comparison
baseline.

**14. Benghe, Graure, Shreedhar, Mohan. "Horizon: Understanding and Predicting
Global Starlink Performance." Proc. ACM Meas. Anal. Comput. Syst. (SIGMETRICS),
2026.**
Global crowdsourced study, 11 months, 90+ countries. **Key finding used
directly:** latency prediction is best with a two-month training window, while
throughput improves monotonically out to eleven months. The two signals are
non-stationary on *different* timescales, which is the argument for temporal
rather than random splits. Also reports latitude as the single strongest
feature (42-46% importance).

**15. Casparsen, Jakobsen, Nielsen, Popovski, Leyva Mayorga. "Statistical
Characterization and Prediction of E2E Latency over LEO Satellite Networks."
arXiv:2601.08439, 2026.**
Characterises latency within the 15 s period at 500 Hz. Contributes a
**phase-recovery procedure** (edge-detect the signal, enforce minimum spacing,
histogram over phase bins, circular mean over the dominant bins) that recovers
the scheduling reference from data rather than assuming it. Also a period-level
Good/Degraded classification. *Leaves open:* deliberately provider-agnostic,
uses no satellite identifiers or geometry.

**16. Narayanan et al. "Lumos5G: Mapping and Predicting Commercial mmWave 5G
Throughput." ACM IMC, 2020.**
The methodological template for throughput prediction on a highly variable
wireless link, from the 5G domain. Establishes that context features
(geometry, orientation) matter more than history alone.

## Group D, Forecasting backbones used as baselines

**17. Zeng, Chen, Zhang, Xu. "Are Transformers Effective for Time Series
Forecasting?" (DLinear). AAAI, 2023.**
Shows a simple decomposition-plus-linear model matches or beats transformers on
many forecasting benchmarks. Included as the "is complexity justified?" control.

**18. Nie, Nguyen, Sinthong, Kalagnanam. "A Time Series is Worth 64 Words"
(PatchTST). arXiv:2211.14730, 2022.**
Patching plus channel independence for transformer forecasting. Strong modern
baseline.

**19. Wu, Hu, Liu, Zhou, Wang, Long. "TimesNet: Temporal 2D-Variation Modeling
for General Time Series Analysis." arXiv:2210.02186, 2022.**
Detects dominant periods by FFT and reshapes the series into 2D to capture
intra- and inter-period variation jointly. Especially relevant given the known
15 s periodicity; also the baseline shipped in the StarNet repository.

## Group E, Risk, calibration and decision-making

**20. Xie, Zhang, Luo, Zhang, Yang, Zhang, Soong. "Risk-Aware Safe Throughput
Forecasting for Starlink Networks" (BG-CFQS). arXiv:2605.09508, 2026., DIRECT
BASELINE.**
The closest prior work and the one this project improves on. Proposes
budget-guided coarse-to-fine quantile selection: train quantile regressors, then
search a candidate quantile set for the largest quantile whose overestimation
rate on a held-out calibration set stays within a risk budget. Reported averages
against a 0.35 budget: MAE 40.364, RMSE 52.480, OverRate 0.349, MPE 11.745,
P95+Err 65.834, risk budget met on 3/3 datasets. **The critical result this
project builds on is their own conditional analysis:** on the lowest-30%
throughput subset the overestimation rate is 0.65-0.71, and on the lowest-10%
subset 0.83-0.86. *Leaves open:* one global quantile cannot control risk within
regimes.

**21. Vovk, Gammerman, Shafer. "Algorithmic Learning in a Random World."
Springer.** (Foundational, not recent, cite for method not novelty.)
Split conformal prediction: the theory giving distribution-free finite-sample
coverage guarantees, on which the calibration layer is built. Important caveat
used explicitly in this project: the guarantee is **marginal**, holding on
average over the whole distribution, and says nothing about any subpopulation.

**22. Koenker, Hallock. "Quantile Regression." Cambridge University Press.**
The pinball loss and quantile estimation underlying the baseline.

**23. Yin, Jindal, Sekar, Sinopoli. "A Control-Theoretic Approach for Dynamic
Adaptive Video Streaming over HTTP" (MPC/RobustMPC). ACM SIGCOMM, 2015.**
The canonical demonstration that a *conservative* throughput estimate beats a
point estimate for a downstream decision. Precedent for the entire framing.

**24. Chen, Guestrin. "XGBoost: A Scalable Tree Boosting System." ACM KDD,
2016.**
Gradient-boosted trees; the backbone BG-CFQS trains with pinball loss, hence
reimplemented here.

### Literature review synthesis (say this out loud in the presentation)

The field has moved through three phases. **Measurement** (2022-2024,
entries 1-8) established what LEO links do. **Prediction** (2023-2026,
entries 12-16) established that throughput and latency are learnable from
terminal and geometry features. **Risk-aware prediction** (2026, entry 20) is
the current frontier and has taken exactly one step: controlling the
overestimation rate globally.

The gap is one step further on. Risk control that holds *on average* is not
risk control where it matters, and the direct baseline's own numbers prove it.

---

# SECTION 3, Design of Proposed Methodology (5 marks)

## 3.1 The contribution, in one sentence

> A **regime-conditioned calibration layer** that converts a point throughput
> forecast into a safe lower bound whose overestimation rate is controlled
> *within each operating regime*, where a regime is defined by covariates the
> terminal can compute at prediction time.

## 3.2 Why a global operating point cannot work (the core argument)

A conformal or quantile bound guarantees `P(bound > actual) <= epsilon`
**marginally**, averaged over everything.

If the residual distribution is wider in one operating condition than another,
the single offset that hits the budget on average sits **too high in the wide
condition and too low in the narrow one**. Because on this link the residuals
are widest exactly when capacity is lowest, risk concentrates precisely where
over-allocation is most costly. The baseline's conditional numbers (0.35 global
against 0.86 on the worst decile) are this effect measured.

## 3.3 Regime definition

A regime is a bucket over covariates available **at prediction time** (read at
the forecast origin, never from the horizon, otherwise the layer would be
using the future):

| Axis | Buckets | Justification from the literature |
|---|---|---|
| 15 s phase | opening / middle / closing | Attention concentrates on the period opening; drops are largest there |
| Serving satellite elevation | 25-45, 45-60, >60 deg | Throughput rises with elevation, plateaus above 60 |
| Serving satellite distance | <645 km, >=645 km | Throughput holds flat until ~645 km then declines |
| Candidate count | terciles (data-fit) | Throughput rises ~26% from 15 to 45 candidates |

**Sparsity handling.** The full cross-product is too sparse to calibrate
directly. A **coarsening hierarchy** is defined: if a regime has fewer than N
calibration points, it falls back to a coarser regime, ultimately to global
calibration. The fallback is **explicit and reported**, the fraction of
calibration mass resolved at each level is published alongside every result,
because if most mass falls back to global then the layer is doing nothing and
the ablation must say so.

## 3.4 The calibration procedure

1. Split the data **temporally** into train / calibration / test. Random splits
   are invalid here (Horizon's window-length asymmetry is the evidence).
2. Fit the point forecaster on **train only**.
3. Compute residuals on the **calibration** split, grouped by regime. The
   calibration set is used *only* to select the operating point, never to fit.
4. Per regime, select the bound satisfying the budget within that regime, by
   either (a) a per-regime one-sided conformal order statistic, or (b) a
   per-regime run of the baseline's coarse-to-fine search. Both are implemented
   and compared.
5. At test time, assign the regime and apply that regime's offset.

**One-sided split conformal, formally.** With calibration residuals
`r_i = y_i - yhat_i` sorted ascending, take `q` as the k-th smallest with
`k = floor(epsilon * (n+1))`. The bound is `L = yhat + q`, and by exchangeability
`P(L > y) = P(r < q) <= k/(n+1) <= epsilon`. The upper-bound form (for latency)
is the mirror image, taking the k-th largest.

## 3.5 Evaluation protocol

**Metrics** (defined identically to the baseline so numbers are comparable):
MAE, RMSE, **OverRate** (fraction where predicted > actual), **MPE** (mean of
the positive error component), **P95+Err** (95th percentile of that component).
Every metric is computed globally, per regime, and on two risk slices:
High-risk P30 (lowest 30% of true throughput) and Severe-risk P10 (lowest 10%).

**Splits:** temporal holdout; leave-one-location-out; contiguous 8:2 blocks for
reproduction runs. A **purge band** is applied at every split boundary, because
consecutive windows overlap (a 30-step look-back with stride 1 shares 29 steps
with its neighbour) and a naive contiguous split still leaks.

**Sweeps:** risk budget from 0.05 to 0.35 (the baseline reports only 0.35,
which permits a 35% overestimation rate that no real allocator would accept;
the advantage should widen as the budget tightens). Regime granularity ablation:
global / phase only / phase+elevation / full cross-product.

**Downstream:** admission control with b = 10 Mbps per session. Admit
`floor(safe_forecast / b)` against an oracle `floor(actual / b)`. Report mean
dropped sessions, violation rate, P95 dropped, **and utilisation**, without
utilisation a policy can win on dropped sessions by admitting nobody.

## 3.6 What must be true for this to be a result

Global OverRate stays at or below the budget (matching the baseline), while
**conditional** OverRate on P30 and P10 drops substantially below the baseline's
0.65-0.86 range, at comparable MAE. **If MAE collapses because the bounds became
uselessly conservative, that is a negative result and will be reported as one.**

## 3.7 Contribution boundary (state this explicitly, it earns marks)

| Component | Status |
|---|---|
| GRU seq2seq backbone, periodical embedding, attention | Reproduction (StarNet) |
| Obstruction-map-to-3D projection, DTW satellite matching | Reproduction (StarNet) |
| Budget-guided coarse-to-fine quantile selection | Reproduction (BG-CFQS) |
| 15 s phase segmentation and recovery | Reproduction (Casparsen) |
| **Regime-conditioned calibration layer** | **Original contribution** |
| **Regime hierarchy with reported fallback** | **Original contribution** |
| **Conditional risk evaluation protocol** | **Original contribution** |

---

# SECTION 4, Module Description / System Design (5 marks)

## 4.1 Architecture: five layers, strictly downward

```
   ingest/     raw sources  ->  normalised frame        (no ML, no features)
      |
   state/      frame        ->  feature vectors + regime id   (no model)
      |
   forecast/   features     ->  point prediction        (StarNet backbone)
      |
   calibrate/  prediction   ->  safe bound              (THE CONTRIBUTION)
      |
   decide/     bound        ->  allocation + alerts     (no ML)

   eval/  sits beside all of them: splits, metrics, experiment grid, figures
```

Each layer talks only to the one below it. This separation is what makes the
system testable and what lets the calibration layer be swapped or ablated
without touching anything else.

## 4.2 Module-by-module description

### `ingest/`, sources to a normalised frame
| Module | Responsibility |
|---|---|
| `base.py` | The `Source` interface; segment-aware windowing so no look-back ever spans a trace gap |
| `replay.py` | Reads published traces; maps columns by alias table and self-checks against published dataset statistics |
| `wetlinks.py`, `wetlinks_full.py` | The primary dataset; per-second capacity and weather |
| `live.py` | Live terminal over gRPC (interface complete; requires hardware) |
| `tle.py` | Orbital geometry: ECI to ECEF to topocentric look angles; vectorised propagation |
| `spacetrack.py` | Historical orbital elements; credentials from environment only |
| `weather.py` | Weather join by location and timestamp with an explicit tolerance |

*Design note:* two ingestion modes (replay and live) sit behind **one
interface** from day one, so the architecture is honest rather than a notebook
wearing a diagram.

### `state/`, frame to features and regimes
| Module | Responsibility |
|---|---|
| `phase.py` | Recovers the 15 s scheduling phase from the signal (edge detection, minimum spacing, phase histogram, circular mean); falls back to the published offset only when not confident |
| `satellite.py` | Serving-satellite resolution: obstruction projection, DTW matching with a Sakoe-Chiba band, handover and dwell derivation, leak-free ID encoding |
| `regime.py` | **Regime assignment, the coarsening hierarchy, and the fallback report** |
| `features.py` | Assembles the feature vector and windows it into look-back/horizon pairs; regime covariates read at the forecast origin |

### `forecast/`, features to a point prediction
| Module | Responsibility |
|---|---|
| `starnet.py` | GRU encoder-decoder (2 layers, hidden 128), periodical embedding (one 1D conv per feature class to L x 48 plus a raw phase channel), additive attention (three single-layer MLPs), 2-layer projection head. Both published ablations included |
| `baselines.py` | DLinear, PatchTST, TimesNet, XGBoost (with pinball loss for the quantile baseline) |
| `train.py` | AdamW, exponential LR decay, gradient clipping, early stopping restoring best weights |

### `calibrate/`, prediction to a safe bound (**the contribution**)
| Module | Responsibility |
|---|---|
| `conformal.py` | One-sided split conformal bound; both directions (lower for capacity, upper for latency) |
| `bgcfqs.py` | Reimplementation of the baseline's coarse-to-fine quantile search |
| `regime_cal.py` | **Per-regime operating points, hierarchy fallback, and the summary report pairing risk gain with MAE cost** |

### `decide/`, bound to actions
| Module | Responsibility |
|---|---|
| `admission.py` | Admit `floor(bound / 10 Mbps)` sessions; scores dropped sessions, violation rate and utilisation against an oracle |
| `congestion.py` | Congestion = bound below commitment for W sustained slots. Causal (never uses future samples). Reports lead time ahead of real episodes |

### `eval/`, measurement
| Module | Responsibility |
|---|---|
| `splits.py` | Temporal, contiguous 8:2, leave-one-location-out; purge bands; explicit leak check on every run |
| `metrics.py` | MAE, RMSE, OverRate, MPE, P95+Err, globally, per regime, and on the P30/P10 risk slices |
| `runner.py` | The experiment grid; one config snapshot written beside every result |
| `figures.py` | Publication figures |

## 4.3 Data flow (for a sequence/flow diagram)

```
Terminal telemetry ─┐
Orbital elements  ──┼─> ingest ─> normalised frame (1 Hz, segmented)
Weather station   ─┘                     │
                                         v
                          phase recovery + satellite geometry
                                         │
                                         v
                         feature windows  +  regime label
                                         │
                    ┌────────────────────┴────────────┐
                    v                                 v
            forecast backbone                  regime assigner
                    │                                 │
              point prediction ──────────> calibration layer <── per-regime
                                                      │            offsets
                                                 safe bound
                                                      │
                                ┌─────────────────────┴──────────────┐
                                v                                    v
                        admission control                    congestion alert
```

## 4.4 Diagrams to produce for the slides

1. **Layer stack**, the five boxes above, with one arrow down between each.
2. **The motivating failure**, grouped bar chart: x-axis All / P30 / P10;
   bars for point forecast, global quantile baseline, proposed method; a dashed
   horizontal line at the budget. The story is that the first two bars are fine
   on the left and terrible on the right.
3. **Regime hierarchy**, a tree: full cross-product at the bottom, coarsening
   upward to global, with the min-sample fallback arrows drawn.
4. **The 15 s period**, a throughput trace annotated with the recovered
   scheduling boundaries and the dip at each one.
5. **Data flow**, as in 4.3.

## 4.5 Engineering practices worth one slide

- Temporal splits with purge bands; automatic leak check on every experiment.
- Calibration set never used for fitting; encoders and scalers fit on train only.
- Every experiment writes a config snapshot beside its results, so any figure is
  traceable to the settings that produced it.
- Everything seeded and the seed recorded.
- Reproduction targets stated in advance; if a published method does not
  reproduce, that is recorded as a finding, not hidden.

---

# APPENDIX, Suggested slide deck (12-14 slides)

| # | Slide | Content source |
|---|---|---|
| 1 | Title | Project, name, guide |
| 2 | Domain: what LEO is and why it is different | 1.1 |
| 3 | Why the link is hard to predict (5 mechanisms) | 1.2 |
| 4 | The industry problem statement + 3 issues chosen | 1.3 |
| 5 | Objectives O1-O4 | 1.4 |
| 6 | Literature: measurement phase | Group A |
| 7 | Literature: prediction phase | Groups B, C |
| 8 | Literature: risk-aware prediction, and the gap | Group E + synthesis |
| 9 | **The gap, quantified** (0.35 global vs 0.86 worst decile) | 1.5 / 3.2 |
| 10 | Proposed method: regime-conditioned calibration | 3.1-3.3 |
| 11 | Calibration procedure (5 steps) | 3.4 |
| 12 | System architecture (5 layers) | 4.1 |
| 13 | Module description | 4.2 |
| 14 | Evaluation protocol + what counts as success | 3.5, 3.6 |

**The single most important slide is #9.** Everything before it motivates the
gap; everything after it addresses the gap. If the audience remembers one
number, make it *0.35 global versus 0.86 on the worst decile*.

---

# PART 2, PRESENTATION DELIVERY PACK

Everything below is for standing up and talking. Speaker notes are written as
what to *say*, not as bullet points to read off a slide.

## Timing (assume 12 minutes + questions)

| Slides | Minutes | Section |
|---|---|---|
| 1-5 | 3.0 | Domain and problem statement |
| 6-8 | 3.0 | Literature review |
| 9 | 1.5 | The gap (slow down here) |
| 10-11 | 2.5 | Proposed methodology |
| 12-13 | 1.5 | System design |
| 14 | 0.5 | Evaluation and success criteria |

If running short on time, cut slides 7 and 13. **Never cut slide 9.**

## Speaker notes, slide by slide

**Slide 2, What LEO is.**
"Starlink satellites orbit at about 550 kilometres, not 36,000 like traditional
geostationary satellites. That's a 65-fold reduction in distance, and it's why
latency drops from around 600 milliseconds to around 30. But the satellites are
moving at 7.5 kilometres per second, so any one of them is only usable for a
few minutes before your terminal has to be handed to another. The link is fast,
and it is non-stationary by construction. That trade is the entire domain."

**Slide 3, Why it's hard.**
"Five mechanisms drive the variation, and each one is a feature we use. First,
Starlink reschedules every 15 seconds, measurement work has pinned the
reconfiguration instants to the 12th, 27th, 42nd and 57th second of each
minute. Second, at those boundaries the serving satellite can change, and you
see a throughput dip and a latency spike. Third, throughput depends on the
serving satellite's geometry, it rises with elevation and plateaus above 60
degrees, and it falls off past about 645 kilometres of range. Fourth, more
visible satellites means more scheduling freedom; throughput rises about 26%
going from 15 to 45 candidates. Fifth, weather and physical obstruction."

Then the line that sets up everything: *"The property that matters for us is
that the prediction error isn't just noisy, it's noisier in some conditions
than others, and specifically noisiest when capacity is already lowest."*

**Slide 4, The problem statement.**
"We were given six industry needs. We address three, and we chose these three
because they chain into one system rather than being three disconnected models.
We predict throughput degradation, we detect congestion, and we allocate
bandwidth. And the important design decision is that congestion detection is
*not* a separate classifier. Congestion is defined as our calibrated bound
falling below the committed allocation for a sustained window. That gets us the
second problem almost for free, and it guarantees the alert and the allocator
never disagree about when the link is in trouble."

**Slide 6-8, Literature.**
Frame it as three phases, not a list:
"The field has moved through three phases. From 2022 to 2024 the work was
measurement, establishing what these links actually do. From 2023 to 2026 it
became prediction, showing throughput and latency are learnable from terminal
and geometry features. And in 2026 it became risk-aware prediction, which is
the current frontier, and it has taken exactly one step: controlling the
overestimation rate globally. Our work is the next step after that."

**Slide 9, THE GAP. Slow down. This is the slide that earns the marks.**
"Here is the problem with controlling risk globally. The direct baseline sets a
budget of 0.35, meaning they'll accept overestimating 35% of the time, and
they hit it, 0.349 globally. But look what happens when you split by how much
capacity the link actually had. On the lowest 30% of throughput samples, their
overestimation rate is 0.65 to 0.71. On the lowest 10%, it's 0.83 to 0.86.

So risk is controlled on average, and lost precisely in the low-capacity regime
,  which is exactly the regime where over-allocating actually drops user
sessions. When the link is healthy, being wrong is cheap. When the link is
struggling, being wrong is expensive, and that's exactly where the guarantee
stops holding.

These aren't our numbers criticising them. These are their own published
numbers. One knob is being used to control a system that has several distinct
operating regimes."

**Slide 10, The proposed method.**
"Our contribution is a regime-conditioned calibration layer. Instead of one
global operating point, we estimate a separate one inside each operating
regime, where a regime is a bucket over covariates the terminal already has at
prediction time, the 15-second phase, the serving satellite's elevation and
distance, and how many candidate satellites are visible.

The reason this works is mathematical, not empirical. A conformal bound
guarantees coverage *marginally*, averaged over everything. If residuals are
wider in one condition than another, the single offset that hits the budget on
average sits too high in the wide condition and too low in the narrow one.
Condition on the regime, and each one gets an offset sized for its own residual
spread."

**Slide 11, Procedure.**
"Five steps. Split temporally, never randomly, and there's evidence for that:
Horizon found latency prediction is best with a two-month training window while
throughput keeps improving out to eleven months, so the two signals are
non-stationary on different timescales and a random split hides all of it.
Then fit the forecaster on train only. Compute residuals on a separate
calibration split, grouped by regime. Select each regime's operating point.
Apply it at test time.

One practical problem: the full cross-product of regimes is too sparse to
calibrate directly. So we define a coarsening hierarchy, if a regime has too
few calibration points, it falls back to a coarser one, and ultimately to
global. And we *report* how much mass fell back, because if most of it lands on
global then our layer isn't doing anything and the ablation has to say so."

**Slide 12-13, Architecture.**
"Five layers, and each one only talks to the one below it. Ingest normalises
raw sources with no ML and no feature engineering. State turns that into
feature vectors and regime labels with no model. Forecast produces a point
prediction. Calibrate turns it into a safe bound, that's our layer. Decide
turns the bound into allocations and alerts, again with no ML.

The separation is what makes it testable, and it's what lets us ablate the
calibration layer without touching anything else."

**Slide 14, Success criteria. End on this, it shows rigour.**
"We've defined in advance what counts as success: global overestimation rate
stays at or below budget, matching the baseline, while the conditional rate on
the worst 30% and 10% drops substantially below their 0.65-to-0.86 range, at
comparable accuracy. And we've also defined what counts as failure, if the
bounds become so conservative that accuracy collapses, that's a negative result
and we report it as one rather than only showing the risk metrics."

## Key numbers to have memorised

| Number | What it is |
|---|---|
| **0.349 / 0.65-0.71 / 0.83-0.86** | Baseline OverRate: global / P30 / P10. **The core motivation.** |
| 15 s | Starlink scheduling period |
| 12, 27, 42, 57 s | Rescheduling instants (all = 12 mod 15) |
| 25 deg | Minimum serving elevation (FCC filing) |
| 60 deg / 645 km | Elevation plateau / distance knee |
| 26% | Throughput gain, 15 to 45 candidate satellites |
| 550 km | Starlink orbital altitude |
| RMSE 40.33 / MAE 29.88 | StarNet published, USA |
| 10 Mbps | Per-session bandwidth in the admission model |

## Anticipated questions, with answers

**"Why not just train a better forecaster instead of calibrating?"**
"A better point forecaster reduces average error but doesn't tell you how much
to trust any individual prediction. Even a perfect-on-average model is above
the truth roughly half the time, and each of those is a potential dropped
session. Calibration is orthogonal, it converts whatever forecaster you have
into a bound with a controlled failure rate. And it composes: a better backbone
makes our bounds tighter."

**"Isn't this just quantile regression?"**
"Quantile regression estimates a conditional quantile by fitting a model with
pinball loss, and it gives no finite-sample guarantee, if the model is
misspecified, the quantile is wrong. Conformal calibration gives a
distribution-free guarantee that holds for any underlying model. Our
contribution is that we make that guarantee hold *within* regimes rather than
only on average."

**"Why these four regime axes and not others?"**
"Every one comes from a measured relationship in the literature, not from
convenience, the elevation plateau at 60 degrees, the distance knee at 645
kilometres, the 26% candidate-count effect, and the attention concentration at
the period opening. And we run a granularity ablation precisely to test whether
they all earn their place. If phase alone gets most of the benefit and satellite
geometry adds nothing, we report that, a clean negative ablation is a result."

**"How do you know the regime split isn't just overfitting?"**
"Three protections. The regime definition is fixed before seeing calibration
data. The calibration split is used only to select the operating point and is
never fit on. And the minimum-sample guard means a regime that doesn't have
enough data to estimate a quantile doesn't get its own, it falls back, and we
report how often that happens."

**"What's your dataset?"**
"Primary is WetLinks, two European terminals measured continuously for six
months with co-located professional weather stations, giving per-second
throughput, latency and weather. That's about 1.65 million samples at 1 Hz
across both sites. We also target the StarNet traces from the base paper for
reproduction."

**"Have you got results yet?"**
Answer honestly and briefly: "We have the full pipeline running end to end on
real data and early pilot runs. The conditional failure we're targeting does
reproduce on our dataset, independently of the paper that first reported it.
The full evaluation grid, the risk budget sweep and the regime ablation, is
what we're running now."
*(Do not quote pilot numbers as final results. They are from short training
runs on one site.)*

**"Why is this hard? Isn't it just bucketing?"**
"The bucketing is the easy part. The hard parts are: the buckets have to be
computable at prediction time, so nothing can peek at the future; the
cross-product is too sparse to calibrate directly, so you need a principled
fallback rather than an arbitrary one; and you have to prove the fallback isn't
doing all the work, which is why we report the mass at each level."

**"What happens if it doesn't work?"**
"Then we report that. We've pre-specified the failure condition: if the
per-regime bounds become so conservative that accuracy collapses, that's a
negative result. We also have a limitations document tracking a known issue , 
that conformal's guarantee assumes exchangeability, and a temporal split
deliberately breaks it. That affects the baseline equally, and it's the next
thing we address."

## Things NOT to claim (you will get caught)

- Do **not** claim to have reproduced StarNet's published table. Their dataset
  links have expired and we are waiting on the authors.
- Do **not** compare our error figures to StarNet's published numbers. Different
  link, country, and sequence length, not like-for-like.
- Do **not** present pilot numbers as final results.
- Do **not** claim the risk budget is currently met. It is not, for us or for
  the baseline, because of temporal drift, say it is a known limitation being
  addressed if asked.
- Do **not** describe the satellite geometry on WetLinks as measured. It is
  reconstructed from propagated orbital elements.

## One-paragraph abstract (for the document front matter)

> Low-Earth-Orbit satellite broadband delivers order-of-magnitude latency
> improvements over geostationary links at the cost of capacity that varies on
> a 15-second scheduling cycle driven by satellite handovers, orbital geometry
> and weather. Allocating bandwidth against a point throughput forecast
> over-commits the link roughly half the time, dropping user sessions. Existing
> risk-aware forecasting selects a single global quantile so that the
> overestimation rate meets a budget, but this controls risk only on average:
> published results show the rate rising from 0.35 globally to 0.86 on the
> lowest-capacity decile, precisely where over-allocation is most costly. We
> propose a regime-conditioned calibration layer that converts a point forecast
> into a safe lower bound whose overestimation rate is controlled within each
> operating regime, where regimes are defined by covariates the terminal can
> compute at prediction time: scheduling phase, serving-satellite elevation and
> distance, and candidate satellite count. A coarsening hierarchy with an
> explicit, reported fallback handles regime sparsity. We evaluate on six months
> of measurements from two European terminals, reporting overestimation rate
> globally, per regime, and on low-capacity subsets, together with downstream
> admission-control outcomes.
