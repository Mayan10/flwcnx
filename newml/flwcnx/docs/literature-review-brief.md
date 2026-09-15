# Literature review brief

Hand this to a fresh Claude session with web access. It is written to be
self-contained: do not assume the reader has seen this repository.

---

## Your task

Establish, with primary sources, what is and is not novel in a student research
project on risk-controlled bandwidth allocation for Starlink (LEO satellite)
access links. The project is being written up as a research paper, so a novelty
claim has to survive a real search rather than the absence of one.

A preliminary check has already been done and it **withdrew the project's
methods claim**. Your job is partly to verify that withdrawal was correct, and
mainly to establish what is left.

**Be adversarial about novelty.** The failure mode to avoid is confirming that
something is new because you did not find it. If you cannot find prior work,
say what you searched and how confident you are, rather than concluding the
idea is new.

---

## What the project built

A five-layer system: ingest, feature/regime assignment, point forecast,
calibration, decision.

The part that matters here is the calibration layer. A point throughput
forecast is converted into a **safe lower bound** whose overestimation rate is
held at a stated budget `epsilon`. Overestimating means promising capacity the
link cannot deliver, which drops sessions; underestimating wastes capacity. The
bound is produced by one-sided split conformal prediction.

Two mechanisms were added on top:

1. **Per-regime calibration.** The covariate space is partitioned into
   "regimes" (buckets over 15-second scheduling phase, serving-satellite
   elevation, distance, visible-satellite count, and summaries of the observed
   look-back window). A separate operating point is fitted per regime, with a
   minimum-sample fallback up a hierarchy to coarser regimes and ultimately to
   a global one.

2. **Online recalibration.** Because the link is non-stationary, a static
   offset fitted on a calibration block misses its budget on a later test
   block. So one adaptive parameter `alpha` and one rolling residual window are
   maintained **per regime**, updated from outcomes as they become observable,
   using the Gibbs and Candes (2021) update
   `alpha_{t+1} = alpha_t + gamma (epsilon - err_t)`.

---

## What the preliminary check found, and what you must verify

It concluded the methods claim is dead, on the basis of three papers. **Verify
each against the primary source.** The provenance of finding (1) is weak and is
the single most important thing for you to check.

### (1) GCACI. Verify first, weak provenance.

Claim: per-group adaptive conformal is already published as **Group Conditional
ACI (GCACI)**, attributed to *Ramalingam, Gupta and Roth, "The Relationship
between No-Regret Learning and Online Conformal Prediction", arXiv:2502.10947,
2025*.

**How that was established, and why it needs checking:** the attribution was
not read from Ramalingam et al. It was read from a *description of* that paper
inside the related-work section of a different paper (arXiv:2606.00419), and
that description was retrieved through a summarising model rather than read
directly. The quoted sentence was:

> "Ramalingam et al. generalized ACI to provide group-conditional guarantees by
> using parameterized prediction sets and showing that minimizing the quantile
> loss with a 'follow the regularized leader' (FTRL) algorithm, which requires
> learning rates, achieves group coverage."

Please:
- confirm arXiv:2502.10947 exists, and that the authors and title are right;
- read the paper itself and confirm it does maintain group-conditional adaptive
  parameters in an online conformal setting;
- report how close it actually is to the description above. In particular,
  is the mechanism "one alpha per group updated by the ACI rule", or is it a
  more general FTRL formulation of which that is a special case?
- find the earliest paper that maintains a separate online-updated conformal
  parameter per covariate group. Ramalingam et al. may not be first.

### (2) POGO

Claim: *"Parameter-Free and Group Conditional Online Conformal Prediction",
arXiv:2606.00419*, is parameter-free group-conditional online conformal
prediction, removing the learning rate `gamma`.

Verify it exists, and report its relationship to GCACI and to this project's
naive per-group version.

### (3) AFCP

Claim: *"Conformal Classification with Equalized Coverage for Adaptively
Selected Groups", NeurIPS 2024, arXiv:2405.15106*, already performs data-driven
selection of which groups need conditioning.

Verify, and assess how close it really is to the idea in the next section,
given that AFCP is classification-and-fairness and this project is regression
under a one-sided risk budget with temporal drift.

**A general caveat on all of the above.** Several arXiv identifiers that
surfaced in the preliminary search were recent (2512.*, 2603.*, 2605.*, 2606.*,
2607.*). Confirm every citation resolves to a real paper that says what is
attributed to it. Discard anything you cannot verify and say so.

---

## The open question that actually matters

Given the methods claim is likely dead, the paper's remaining candidate
contribution is this empirical finding, and the question is whether it is
already known.

**The finding.** Per-regime conditioning helps *only when the regimes actually
differ*, and whether they differ is measurable before any test decision is
made. Across four datasets, the spread of the fitted per-regime offset on the
calibration split predicts the sign of the effect, monotonically:

| dataset | spread of per-regime offset | effect of conditioning on tail risk |
|---|---|---|
| StarNet USA | 1.85 Mbps | +6.1% (conditioning **hurts**) |
| StarNet Germany | 5.44 Mbps | +0.1% (neutral) |
| StarNet Canada | 11.30 Mbps | -1.7% (helps) |
| WetLinks Osnabruck | 9.28 Mbps | -5.6% (helps) |

On the US trace the four regimes want offsets of -10.69, -11.91, -10.06 and
-10.99 Mbps: nearly identical, so conditioning adds estimation noise and
nothing else.

The proposed use is a **gate**: measure the spread against its own estimation
noise on the calibration split, and fall back to global calibration when it
does not clear.

**Questions:**

1. Is there existing work that decides *whether* to condition, based on
   measured group heterogeneity, in conformal prediction or quantile
   regression? How does it differ from AFCP?
2. Is there a standard statistical test for this? It resembles a
   between-group versus within-group variance comparison (ANOVA-like), or a
   test for whether group-conditional quantiles differ. If a standard test
   exists, the gate should use it rather than an ad hoc threshold, and the
   contribution shrinks to the application.
3. Is the *observation* (group-conditional methods only help under sufficient
   heterogeneity) already stated somewhere as a known result? It seems close to
   folklore. Folklore that is written down somewhere kills the claim; folklore
   that nobody has stated or quantified may not.
4. Has anyone quantified a **predictor** of when conditioning will help, as
   opposed to observing after the fact that it did or did not?

---

## The application question

Independently of the method, is the *application* new?

1. Has conformal prediction or distribution-free risk control been applied to
   **network bandwidth allocation or admission control**? Look at networking
   venues (SIGCOMM, NSDI, CoNEXT, IMC, INFOCOM, MobiCom, MMSys) as well as ML
   venues.
2. Specifically for **LEO satellite / Starlink** links. Known adjacent work in
   this project's bibliography: StarNet (Liu et al., CoNEXT 2025) does
   throughput prediction with no risk control; BG-CFQS (Xie et al.,
   arXiv:2605.09508) does budget-guided quantile selection for safe Starlink
   throughput forecasting and is the direct baseline.
3. Is there work on **risk-aware or uncertainty-aware admission control** in
   networking generally, satellite or not, that would already cover the
   decision layer here?

---

## Findings that need a novelty check of their own

The paper's spine is now measurement. Please check whether each is already
known. Each is backed by a run in this project.

1. **BG-CFQS's risk guarantee is conditional on exchangeability.** Reproduced
   3/3 within budget under a random split of the same data, 1/3 under a
   contiguous temporal split. Is it already documented anywhere that
   budget-guided quantile selection, or split-conformal risk control generally,
   fails this way on network traces?
2. **BG-CFQS cannot serve a risk budget below 0.15**, because its published
   candidate quantile set is T = [0.15, 0.40] and the boundary search cannot
   select below its own range. Achieved rate pins at 0.185 for every tighter
   budget. Their paper reports only epsilon = 0.35, where the floor never
   binds. Has anyone noted this?
3. **Static calibration misses its budget in both directions**, over-shooting
   on one dataset and under-shooting on another, with the sign determined by
   the direction of the drift. Is the "undershoot" half documented? Most
   distribution-shift conformal work seems to emphasise under-coverage.
4. **Measured serving-satellite geometry (elevation, distance, visible count)
   does not improve conditional risk control.** StarNet reports these
   correlate with throughput *level*. This project finds they do not carry
   *residual* structure. Is that distinction made anywhere?
5. **The 15-second Starlink scheduling offset (~12 s) recovered independently**
   from 1 Hz throughput on three continents, confirming Casparsen et al.
   (arXiv:2601.08439) who established it from 500 Hz latency probes at one
   European site. Has anyone else recovered this offset from throughput, or
   cross-validated it across continents?

---

## What to produce

A markdown report with:

1. **Verdict on the withdrawal**: was the methods claim correctly withdrawn?
   Yes or no, with the primary sources.
2. **Prior-art table**: for each of the project's candidate claims (per-regime
   conformal, per-regime *adaptive* conformal, the heterogeneity gate, the
   application, and the five measurement findings), give status
   (clearly prior work / partially covered / no prior work found), the closest
   citations, and one sentence on how close they are.
3. **Closest competitors** to compare against experimentally, with enough
   detail to reimplement or cite. If GCACI or POGO should be baselines, say
   what implementing them would take.
4. **A recommended framing** for the paper in two or three sentences, given
   what survives.
5. **Search log**: queries, databases, and what you did *not* find. This is
   what lets a reader judge how much weight the negative results carry.

Prefer primary sources. Quote the sentence you are relying on and give the
section it came from. Where you are uncertain, say so explicitly rather than
resolving it in the project's favour: an overturned claim discovered now is
cheap, and one discovered by a reviewer is not.
