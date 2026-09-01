# Novelty review: risk-controlled bandwidth allocation for Starlink access links

Prepared against the brief `literature-review-brief.md`. All citations below were
resolved to a primary source and read unless explicitly flagged otherwise.

Date of search: 1 September 2026.

---

## 0. Summary

The withdrawal of the methods claim was correct. It was correct for a stronger
reason than the preliminary check found: the preliminary check identified the
right paper but got the author list wrong, and it missed the paper that actually
does the most damage, which is not GCACI but **Clustered Conformal Prediction**
(Ding et al., NeurIPS 2023).

The heterogeneity gate is in worse shape than the brief hopes. The observation is
written down, the mechanism is published in a more general form, and the "ad hoc
threshold" has a standard replacement that is a century-old idea in a different
literature. The gate should be reframed as an application, not a method.

The application is also not new. BG-CFQS, the direct baseline, already contains an
admission-control evaluation with the same arithmetic the project uses.

What survives is one clean, checkable, unreported defect in a published baseline,
plus a set of negative results. That is a real paper, but it is a measurement and
evaluation paper for a networking venue, not a methods paper.

---

## 1. Verdict on the withdrawal

**Yes, the methods claim was correctly withdrawn.** Both of the project's two
mechanisms are published.

### 1.1 GCACI: verified, with one attribution error

The paper exists and is correctly identified except for its author list.

- **Correct citation:** Ramya Ramalingam, **Shayan Kiyani**, Aaron Roth, "The
  Relationship between No-Regret Learning and Online Conformal Prediction,"
  arXiv:2502.10947 [cs.LG], submitted 16 February 2025, University of
  Pennsylvania. It appears in later bibliographies as ICML 2025 (see
  arXiv:2602.16537 reference list, which cites it as "In *International
  Conference on Machine Learning*").
- **The error:** the preliminary check attributed it to "Ramalingam, Gupta and
  Roth." The second author is Kiyani, not Gupta. Varun Gupta is a co-author on
  the adjacent Gupta et al. (ITCS 2022) and Bastani et al. (NeurIPS 2022) papers,
  which is the likely source of the confusion. Fix this before it reaches a
  bibliography.

**Does it maintain group-conditional adaptive parameters online?** Yes, and it
names them. Section 5 of the paper is titled "Group Conditional ACI" and
Algorithm 2 is labelled "Group Conditional ACI (GCACI)". The abstract states that
the authors "analyze and conduct experiments using a multi-group generalization
of the ACI algorithm of Gibbs and Candes [2021]."

**How close is it to "one alpha per group updated by the ACI rule"?** Closer than
the brief's framing suggests. Algorithm 2 maintains a parameter vector
`θ_t ∈ R^k`, one coordinate per group, predicts `τ̂_t = ⟨θ_t, g_t⟩` where `g_t` is
the group-membership vector, and updates

```
if ⟨θ_t, g_t⟩ < τ_t:  θ_{t+1} = θ_t + η·q·g_t        (undercover)
else:                 θ_{t+1} = θ_t − η·(1−q)·g_t     (cover)
```

This is FTRL with Euclidean regularizer `R(θ) = ‖θ‖²/2η`, i.e. online gradient
descent on the pinball loss, exactly as ACI is in one dimension. It is stated as
the general intersecting-group case with real-valued membership weights.

**The point that matters for this project:** the project's regimes are a
*partition* of covariate space (buckets over phase, elevation, distance, visible
count, look-back summaries, with a fallback hierarchy). For a partition, `g_t` is
one-hot, so `⟨θ_t, g_t⟩ = θ_{t,i}` and only coordinate `i` moves on any given
round. **GCACI on a partition reduces exactly to one adaptive parameter per
regime updated by the ACI rule.** That is not "a more general formulation of
which the project is a special case, so there is room" — the special case is the
published thing. Ramalingam et al.'s own Section 6 experiments use binary groups.

The one real difference is parameterization: GCACI updates the threshold `τ`
directly, whereas Gibbs–Candès (and the project) update `α`, the quantile level,
and then re-read the empirical quantile from a residual window. That is a
reparameterization with different finite-sample behaviour under a rolling window,
not a contribution. It is worth one sentence in a paper, not a claim.

**Verification of the quoted related-work sentence.** The sentence attributed to
arXiv:2606.00419 is verbatim accurate. It appears in POGO Section 1.1,
"Group-conditional coverage": Ramalingam et al. generalized ACI using
parameterized prediction sets and an FTRL algorithm requiring learning rates. The
description is faithful to Section 4 of the primary source, which is titled
"Coverage Guarantees for FTRL Algorithms." (Note a small inconsistency inside
Ramalingam et al. itself: its abstract says "follow the perturbed leader" while
its body consistently says "follow the regularized leader." The body is correct
and POGO's description matches the body. The summarising model's output was, in
this instance, more accurate than the abstract.)

**Earliest work maintaining a per-group online-updated conformal parameter.**
Ramalingam et al. are not first, and they say so.

| Work | Date | What it does |
|---|---|---|
| Vovk, Lindsay, Nouretdinov, Gammerman, "Mondrian confidence machine," RHUL tech. report | 2003 | Offline. Per-category (partition) conformal calibration. This is the origin of "one operating point per regime." 23 years old. |
| Gupta, Jung, Noarov, Pai, Roth, "Online multivalid learning," ITCS | 2022 | Online, adversarial, group-conditional. Quantile-calibration machinery rather than one ACI parameter per group. |
| Bastani, Gupta, Jung, Noarov, Ramalingam, Roth, "Practical adversarial multivalid conformal prediction" (MVP), NeurIPS | 2022 | Refinement of the above. The baseline GCACI is compared against. |
| Angelopoulos, Jordan, Tibshirani, "Gradient equilibrium in online learning," arXiv:2501.08330, JMLR 26(305) | Jan 2025 | Ramalingam et al. Section 1.2: "Our generalization of the 1-dimensional ACI bounds to group conditional coverage bounds was independently and concurrently discovered by Angelopoulos et al. [2025]." This is roughly one month earlier on arXiv. |
| Ramalingam, Kiyani, Roth (GCACI), arXiv:2502.10947 | Feb 2025 | Names and analyses the algorithm. |

For priority purposes, if the paper needs to say "this is prior work," the
honest citation is Vovk et al. 2003 for the offline per-regime idea, and
Angelopoulos et al. 2025 / Ramalingam et al. 2025 jointly for the online
per-group ACI parameter.

### 1.2 POGO: verified

- **Citation:** Beepul Bharti, Ambar Pal, Jacopo Teneggi, Jeremias Sulam,
  "Parameter-Free and Group Conditional Online Conformal Prediction,"
  arXiv:2606.00419 [stat.ML], v1 submitted 29 May 2026, v4 dated 7 July 2026.
  Johns Hopkins University and Amazon Responsible AI. Code at
  `github.com/beepulbharti/pogo`.
- It is what the brief says it is: "the first parameter-free algorithm for
  group-conditional online conformal prediction," removing the learning rate via
  universal-portfolio / coin-betting machinery (building on Liu, Dobriban,
  Orabona, UP-OCP, arXiv:2602.03168). Its Table 1 compares coverage rates
  directly: GCACI is `√(ηT(1−α)(ηk(1−α)+1))/(ηT)`, POGO is
  `(ln kT)/T + √(α ln(kT)/T)`.

**Relationship to the project's naive per-regime version.** POGO Section 4.2,
"Alternative Naive Approaches," addresses exactly this: why not run a
parameter-free marginal method separately per group? Their objection is that with
`k` intersecting groups you get `k` sequences of radii and no principled way to
pick one for a test point in several groups, plus a group-membership leakage
concern.

**That objection does not apply to a partition.** For disjoint regimes each point
is in exactly one group, so there is exactly one radius. So POGO's own section
concedes that the naive per-group scheme is well-defined in the project's
setting. That is a concession, not a novelty: it means the project's mechanism is
the case everyone agrees is trivial.

### 1.3 AFCP: verified, but it is not the same idea

- **Citation:** Yanfei Zhou, Matteo Sesia, "Conformal Classification with
  Equalized Coverage for Adaptively Selected Groups," arXiv:2405.15106
  [stat.ML], v1 23 May 2024, v2 30 Oct 2024, NeurIPS 2024. USC. Code at
  `github.com/FionaZ3696/Adaptively-Fair-Conformal-Prediction`.

It does perform data-driven selection of which groups to condition on. But the
statistic is not a heterogeneity statistic. From Section 2.1, AFCP computes, for
each candidate attribute `k`, the leave-one-out **worst-group miscoverage rate**

```
δ_{y,k} = max over groups m of  (Σ_i E_{y,i}·1{φ(X_i,{k})=m}) / (Σ_i 1{φ(X_i,{k})=m})
```

then sets `q̂_y = max_k δ_{y,k}` and runs "a one-sided t-test for the null
hypothesis `H_0 : q̂_y ≤ α` against `H_1 : q̂_y > α`." If `H_0` is rejected it
conditions on `argmax_k δ_{y,k}`; otherwise it selects nothing.

So AFCP asks **"is any group's error above budget?"** The project's gate asks
**"do the groups' fitted offsets differ from each other?"** These are different
null hypotheses. A set of regimes can all be well within budget and still differ
enormously (conditioning would then be safe but pointless), or all be equally out
of budget with zero spread (AFCP fires, the gate does not). AFCP's test is about
*level*; the gate's test is about *dispersion*.

Two further gaps in AFCP's favour, i.e. reasons it does not cover this project:

- It is classification. The paper's own Discussion lists "extending our method to
  accommodate regression tasks with continuous outcomes" as future work with
  "additional computational challenges."
- It is the exchangeable batch setting with a leave-one-out procedure over the
  calibration set, with no temporal drift and no online component.

**Assessment:** AFCP is genuinely adjacent and must be cited and distinguished,
but it is not the gate. It is not what kills the gate. The next section is.

### 1.4 The paper the preliminary check missed

**Ding, Angelopoulos, Bates, Jordan, Tibshirani, "Class-Conditional Conformal
Prediction with Many Classes," arXiv:2306.09335, NeurIPS 2023.** Clustered
Conformal Prediction (CCP).

This is the closest prior art to the heterogeneity gate and it was not in the
preliminary check. It does four things the project claims:

1. **It states the bias/variance framing explicitly.** From Section 2: the method
   "strikes a balance between the granularity of classwise and the data-pooling
   of standard by grouping together classes according to a clustering function."

2. **It states the project's core observation as a known result.** From Section
   1: "Somewhat paradoxically, the variance of the class-conditional coverage
   means that on a given realization of the calibration set, the CLASSWISE method
   can exhibit poor coverage on a substantial fraction of classes if the number
   of calibration data points per class is limited." That is the project's "on the
   US trace, conditioning adds estimation noise and nothing else," stated in 2023
   in a top-tier venue.

3. **It measures per-group calibration-split quantiles and pools where they are
   similar.** From Section 2.2: "we first summarize the empirical score
   distribution for each class via a vector of score quantiles evaluated at a
   discrete set of levels" and then cluster in that embedding space, because "we
   want to group together classes with similar quantiles." That is the gate's
   statistic (dispersion of per-group fitted quantiles on the calibration split)
   used for the gate's purpose (decide how much to condition), decided before any
   test point.

4. **It has the minimum-sample fallback.** A "null cluster" absorbs rare classes.

The differences that remain are real but narrow: CCP is classification, two-sided
coverage, exchangeable, offline, and it produces a *soft* answer (a clustering
with `k` between 1 and `K`) rather than a *binary* gate. The project's gate is the
degenerate case of CCP's clustering where the only options are `k=1` (global) and
`k=K` (fully conditional).

**Bottom line for Section 1:** the withdrawal was correct on GCACI, and the gate
that was supposed to survive it is itself substantially covered by CCP.

---

## 2. Prior-art table

| Claim | Status | Closest citations | How close |
|---|---|---|---|
| Per-regime (partition) conformal calibration with a min-sample fallback hierarchy | **Clearly prior work** | Vovk, Lindsay, Nouretdinov, Gammerman, Mondrian confidence machine, 2003; Romano, Barber, Sabatti, Candès, HDSR 2020 (equalized coverage, disjoint groups) | This is Mondrian conformal prediction. The hierarchy fallback is an implementation detail on it. No claim available. |
| Per-regime **adaptive** (online) conformal parameter, ACI update per regime | **Clearly prior work** | Ramalingam, Kiyani, Roth, arXiv:2502.10947 §5 Alg. 2 (GCACI); Angelopoulos, Jordan, Tibshirani, arXiv:2501.08330; earlier group-conditional online: Gupta et al. ITCS 2022, Bastani et al. NeurIPS 2022 | GCACI on a partition *is* this algorithm. Only difference is α-space vs τ-space update. No claim available. |
| Parameter-free version of the above | **Clearly prior work** | Bharti, Pal, Teneggi, Sulam, arXiv:2606.00419 (POGO) | Strictly stronger; removes γ entirely. Should be a baseline. |
| Online **one-sided risk** control under drift (as opposed to coverage) | **Clearly prior work** | Feldman, Ringel, Bates, Romano, "Achieving Risk Control in Online Learning Settings," arXiv:2205.09095, TMLR 2023 (Rolling RC) | Controls an arbitrary user-specified risk online under arbitrary/adversarial drift, with an additive calibration term and a stretching function. This is the canonical method for exactly what the project's calibration layer does, and it is currently missing from the project's bibliography. Code at `github.com/Shai128/rrc`. |
| Heterogeneity gate: decide *whether* to condition from measured group heterogeneity | **Partially covered, leaning prior work** | Ding, Angelopoulos, Bates, Jordan, Tibshirani, arXiv:2306.09335 (CCP), NeurIPS 2023; Zhou & Sesia, arXiv:2405.15106 (AFCP), NeurIPS 2024 | CCP does the same measurement for the same purpose with a finer output. AFCP does data-driven selection with a different statistic. What is left: the binary gate form, the regression / one-sided-risk / temporal-drift setting, and the specific pre-test predictor claim. |
| "Group-conditional methods help only under sufficient heterogeneity" as an **observation** | **Clearly prior work** | Ding et al. §1 (quoted in §1.4 above); Foygel Barber, Candès, Ramdas, Tibshirani, "The limits of distribution-free conditional predictive inference," Inf. Inference 10(2), 2021 | Written down, in print, in a NeurIPS paper. This is not unstated folklore. |
| A **quantitative pre-test predictor** of the sign of the conditioning effect | **No prior work found** — see caveat below | — | Nothing found that reports a scalar measured on the calibration split predicting the *sign* of the test-time tail-risk effect, monotonically across datasets. This is the thinnest surviving thread and the caveat in §2.1 is serious. |
| Application: conformal / distribution-free risk control for **network resource allocation** | **Clearly prior work** | Cohen, Park, Simeone, Shamai, "Calibrating AI Models for Wireless Communications via Conformal Prediction," arXiv:2212.07775; Cohen et al., "Guaranteed Dynamic Scheduling of Ultra-Reliable Low-Latency Traffic via Conformal Prediction," arXiv:2302.07675; Simeone et al., "Conformal Calibration," arXiv:2504.09310 | The URLLC scheduling paper is explicitly "a CP-based resource allocation scheme ... offering theoretical reliability guarantees that apply even when the predictor is poorly designed." Same shape as this project, different link layer. |
| Application: risk-budgeted forecasting for **LEO / Starlink** with admission control | **Clearly prior work** | Xie, Zhang, Luo, Zhang, Yang, Zhang, Soong, arXiv:2605.09508 (BG-CFQS); Xie et al., arXiv:2605.23560 (SafeSABR); *Aerospace* 13(5):442 (2026), spectral-aware risk-aware LEO allocation | BG-CFQS §V-G is an admission-control evaluation using `n_admit = ⌊ŷ_safe/b⌋`, `n_drop = max(n_admit − n_oracle, 0)` — the project's decision layer, published May 2026. |
| **F1.** Split-conformal / budget-guided risk control fails under temporal (non-exchangeable) splits | **Phenomenon is prior work; the specific instance is not** | Barber, Candès, Ramdas, Tibshirani, "Conformal prediction beyond exchangeability," *Ann. Statist.* 51(2):816–845, 2023; Oliveira, Orenstein, Ramos, Romano, "Split conformal prediction and non-exchangeable data," *JMLR* 25(225), 2024 | That split conformal degrades under non-exchangeability is textbook. That *BG-CFQS specifically* holds 3/3 under random split and 1/3 under contiguous temporal split on these traces is not published. Report it as a reproduction, not a discovery. |
| **F2.** BG-CFQS cannot serve ε below its candidate-set floor; achieved rate pins | **No prior work found. Strongest surviving finding.** | — | See §2.2. Mechanism verified directly from the primary source. |
| **F3.** Static calibration misses in **both** directions, sign set by drift direction | **Partially covered** | Barber et al. 2023 (the coverage-gap bound is two-sided); Gibbs & Candès, *JMLR* 25(162), 2024 | The overshoot half is under-emphasised in the literature, which does lean on under-coverage. But it follows immediately from the existing bounds, so a reviewer will call it an illustration rather than a finding. Weak. |
| **F4.** Serving-satellite geometry correlates with throughput *level* but carries no *residual* structure | **No prior work found, but it is F-the-gate restated** | StarNet (Liu et al. 2025) for the level correlation | Level-vs-residual is a fair distinction to draw and I found nobody drawing it for these features. But operationally this is "conditioning on geometry regimes does not help," i.e. the same negative result as the gate finding, on specific traces. Modest. |
| **F5.** 15 s scheduling offset (~12 s) recovered from 1 Hz throughput on three continents | **Largely prior work** | Casparsen et al. arXiv:2601.08439 (npj Wireless Technology, 2026); Mohan, Ferguson, Cech, Bose, Renatin, Marina, Ott, "A Multifaceted Look at Starlink Performance," WWW 2024, arXiv:2310.09242; Liu et al., StarNet, PACMNET 3(CoNEXT4), 2025; Garcia et al., LEO-NET 2023 | See §2.3. This one does not survive. |

### 2.1 The caveat that most threatens the gate

I did not find a conformal-prediction paper that quantifies a pre-test predictor
of when conditioning will help. But the gate is a restatement of a result that is
textbook in a different literature, and a statistics reviewer will say so.

The gate statistic — between-group spread of fitted per-regime offsets, compared
against its own estimation noise — is a **variance-components / shrinkage**
statistic. The claim "pooling beats per-group estimation when between-group
variance is small relative to within-group sampling variance" is the James–Stein
and empirical-Bayes result. Deciding whether to pool by testing `H_0 : τ² = 0`
before fitting is standard practice in meta-analysis and in multi-centre clinical
trial "poolability" analysis, where the rule is literally: treat centres as
heterogeneous if the variance of the random centre effect differs significantly
from zero.

I did **not** systematically search the shrinkage, empirical-Bayes, or
partial-pooling literature. That is the largest gap in this review, and it is the
gap most likely to produce an overturning citation. Before committing to the
gate as a contribution, someone should search that literature directly. My
expectation, stated so it can be held against me, is that it will find the
general result stated cleanly and the contribution will collapse to "we applied
a standard variance-components test to conformal offsets on Starlink traces."

### 2.2 Is there a standard test? Yes.

The brief asks whether a standard test exists and notes that if so, the
contribution shrinks to the application. It exists, in at least three usable
forms. Ordered by how easy they are to justify to a reviewer:

**(a) Cochran's Q and I².** Let `θ̂_i` be the fitted offset in regime `i` with
standard error `se_i`, and `w_i = 1/se_i²`. Then

```
Q = Σ_i w_i (θ̂_i − θ̄_w)²,   θ̄_w = Σ w_i θ̂_i / Σ w_i
```

Under `H_0` (all regimes share one offset), `Q ~ χ²_{k−1}`. `I² = (Q − (k−1))/Q`
is the fraction of observed spread not attributable to sampling error. This is
exactly "measure the spread against its own estimation noise," it has a null
distribution, and it is a hundred-year-old idea. The project's four-regime US
trace, with offsets −10.69, −11.91, −10.06, −10.99 Mbps, is precisely the case
`Q` is designed to declare non-significant.

**(b) Random-effects variance component test, `H_0 : τ² = 0`.** Equivalent in
spirit, better behaved with few groups, and it is what multi-site trials use to
decide poolability.

**(c) A Wald or F test on the quantile-regression coefficients.** If the
per-regime offset is fitted as quantile regression of the conformity score on
regime indicator variables — which is a standard formulation of group-conditional
conformal calibration (see Jung, Noarov, Ramalingam, Roth, "Batch multivalid
conformal prediction," ICLR 2023; and, for a worked applied example,
arXiv:2308.15094 §4.2.1, which regresses scores on group-membership indicators to
estimate group-conditional quantiles) — then "do the regimes differ" is a joint
test that the non-intercept coefficients are zero. Quantile-regression standard
errors depend on conditional densities and are awkward; Escanciano & Goh,
"Quantile-Regression Inference With Adaptive Control of Size," arXiv:1807.06977,
gives a variance estimator built for exactly this test.

**(d) If simultaneity matters** (many candidate regime definitions, selection
effects): Cherian & Candès, "Statistical inference for fairness auditing,"
arXiv:2305.03712, gives simultaneously valid confidence bounds on group-wise
disparities. AFCP explicitly names this as a drop-in for its own selection step.

**Recommendation:** use (a) or (c), report the p-value, drop the ad hoc
threshold. It costs nothing and removes the single most obvious reviewer
objection. And then be honest that the gate is now "a standard heterogeneity
test, applied in this setting" — which is a legitimate contribution to a
networking paper and not one to an ML methods paper.

### 2.3 Finding 5 does not survive

The brief asks whether anyone else has recovered the 15 s offset from throughput,
or cross-validated it across continents. Both, yes.

- Casparsen et al.'s own related work states that empirical studies have
  consistently identified a 15-second periodicity in Starlink latency **and
  throughput**, attributed to deterministic handovers, citing Mohan et al. 2024,
  Pan et al. 2024, and Garcia et al. 2024, and notes that these works observe
  latency stable within each period and shifting between periods, often with
  throughput drops. The paper positions its own contribution as resolving the
  *intra-period* structure at 500 Hz, not as discovering the period.
- Mohan et al. (WWW 2024, arXiv:2310.09242 §6.2) ran **simultaneous** iRTT
  measurements from Edinburgh and Munich, showed the reconfiguration intervals
  are time-aligned across sites, and concluded Starlink runs a globally
  coordinated schedule rather than a per-terminal one. They also explicitly note
  that "Previous studies have noticed drops in downlink throughput every 15 s but
  have not correlated these with the reconfiguration intervals." So the
  throughput signature predates even that.
- StarNet (Liu et al. 2025) builds "periodical embedding for the 15-second
  handover cycle" and is trained on the same three country traces (US, Germany,
  Canada) the project uses.

So: recovering 15 s periodicity from throughput is prior; multi-site alignment is
prior; the three-country scope is the scope of the dataset the project inherited
from StarNet, which already models the cycle. The only thing possibly left is the
specific ~12 s phase value agreeing with Casparsen's boundary location. That is a
corroboration of someone else's result using a coarser instrument. It belongs in
a sentence in the measurement section as a sanity check on the feature
construction, and nowhere near a contributions list.

### 2.4 Finding 2 is the best thing in the paper

This is the one item where I found nothing and where the mechanism is verifiable
from the primary source rather than inferred.

From BG-CFQS Algorithm 1 (arXiv:2605.09508 §IV-D), the boundary search is:

```
4:  if R(τ_max) ≤ ε then
5:      [τ⁻, τ⁺] = [τ_max, τ_max]
6:  else if R(τ_min) > ε then
7:      [τ⁻, τ⁺] = [τ_min, τ_min]
8:  else ... binary search ...
```

Line 6 is the floor. If the most conservative endpoint of the candidate interval
already violates the budget, the method collapses the search interval to a single
point at `τ_min` and returns it. The fine grid `LinSpace(τ⁻, τ⁺, M)` is then
degenerate, the feasible set is empty, and the penalized fallback
`J(τ) = A(τ) + λ·max(R(τ) − ε, 0)` selects `τ_min` regardless of how much tighter
`ε` gets. The achieved rate therefore pins at `R(τ_min)` for every budget below
it. This matches the reported behaviour (pinning at 0.185).

I also confirmed from Table II that the **default risk budget is ε = 0.35**,
which is where the paper reports results. At ε = 0.35 the floor never binds, so
the paper's own evaluation cannot expose it.

**One thing I could not verify.** The Table II row "Candidate quantile set" was
truncated at the end of my fetch of the HTML version. I did not independently
confirm the specific interval `T = [0.15, 0.40]`. The *mechanism* is confirmed
from the algorithm; the *numbers* need one direct look at Table II in the PDF
before they go into a paper. Do not take that bound from this report.

Nobody appears to have noted this. It is a concrete, reproducible, unreported
limitation of a specific published method, discovered by running it outside the
regime its authors evaluated. That is a legitimate finding, it is checkable by a
reviewer in ten minutes, and it is the kind of thing measurement venues reward.

---

## 3. Closest competitors, and what implementing them costs

If the paper claims anything about per-regime calibration, these have to be in the
table. Ordered by implementation cost.

**1. ACI (Gibbs & Candès 2021, arXiv:2106.00170).** Marginal baseline. Already in
the project. Trivial.

**2. GCACI (Ramalingam, Kiyani, Roth 2025, arXiv:2502.10947, Algorithm 2).**
About ten lines. Maintain `θ ∈ R^k`, one coordinate per regime; predict
`⟨θ_t, g_t⟩` with `g_t` one-hot; update `θ_{t+1} = θ_t + η q g_t` on undercover,
`θ_t − η(1−q) g_t` on cover. No code release found, but none is needed. Set
`η = 1` following the paper's own experiments (POGO also uses `η = 1` "following
[41]"). This is the direct methods baseline and its absence would be the first
thing a reviewer asks about.

**3. Rolling RC (Feldman, Ringel, Bates, Romano, TMLR 2023, arXiv:2205.09095).**
Code at `github.com/Shai128/rrc`, self-contained Python, with a
`regression-simple-example.ipynb`. This is the closest existing method to what the
calibration layer actually does: online control of a *user-specified risk* (not
just coverage) under arbitrary, even adversarial, drift, via an additive
calibration term with a stretching function for faster adaptation. It handles the
one-sided overestimation-rate risk natively. **This is the most important missing
baseline and the cheapest to add.** Its absence is currently a hole in the
project's related work, independent of any novelty question.

**4. Mondrian / classwise conformal, and Clustered Conformal Prediction (Ding et
al. 2023, arXiv:2306.09335).** The offline per-regime baselines and the soft
version of the gate. CCP is the right comparison for "should we condition, and
how much": the project's gate is binary, CCP chooses `k`. If the gate is the
claim, CCP is the baseline that has to lose. Reference implementation exists
(the paper is well-known and the method is k-means on quantile embeddings, a few
dozen lines).

**5. POGO (Bharti et al. 2026, arXiv:2606.00419).** Code at
`github.com/beepulbharti/pogo`. Universal Portfolio with Jeffreys prior over `k`
wealth processes — one integral per group per round, non-trivial but the code is
released. Worth including *only if* the paper makes a claim about learning-rate
sensitivity, since that is the axis on which POGO wins. If the paper does not
tune `γ` and does not claim robustness to it, POGO can be cited and not run.
Given that the project maintains one `γ` and one window per regime with no
principled way to set `γ`, a reviewer may well ask.

**6. BG-CFQS (Xie et al. 2026, arXiv:2605.09508).** Already the project's
baseline. No code found in my search.

---

## 4. Recommended framing

Given what survives, do not write a methods paper. Write an evaluation paper, and
say so in the first paragraph.

> Risk-budgeted throughput forecasting for LEO access links has recently been
> proposed as a way to make capacity estimates safe for admission control. We
> reproduce the state of the art on four traces and find that its risk guarantee
> is an artifact of a random calibration split: under a contiguous temporal
> split, matching deployment, it holds on one of three datasets. We further show
> that its published candidate quantile set imposes a hard floor on the
> achievable overestimation rate, so budgets tighter than that floor are silently
> unservable — a regime its own evaluation, conducted at a single loose budget,
> never enters. We then evaluate the natural remedy, per-regime online
> calibration, and find it is not free: it helps only when the regimes actually
> differ, and a standard heterogeneity test on the calibration split predicts
> which case a trace is in before any test-time decision is made.

Three deliberate choices in that framing:

- **Lead with the baseline's failure, not with the method.** The floor is the
  most defensible thing found. It is specific, reproducible, and unreported.
- **Present per-regime calibration as *evaluated*, not proposed.** Cite GCACI and
  Mondrian as the prior work it is. This converts the largest liability into a
  routine related-work paragraph.
- **Present the gate as a decision procedure using a standard test.** Cochran's Q
  or a quantile-regression Wald test. Cite CCP and AFCP as the prior art on
  data-driven conditioning and state plainly that the contribution is the
  application to one-sided risk control under temporal drift, not the idea.

**Venue.** This is a networking measurement and evaluation paper. IMC, CoNEXT,
MMSys, or the LEO-NET workshop, where reproducing and breaking a recent
system-adjacent result is a recognised contribution. Submitting it to an ML venue
as a conformal-prediction methods paper would be rejected on Section 1 of this
report.

**One honest risk to name in the paper itself.** BG-CFQS is four months old, from
a group at SJTU/UESTC/NTU, in eess.SY. Building a paper around defects in a very
recent preprint means the target may move: they may revise, widen the candidate
set, or switch splits. Check for a v2 before submission, and frame the finding as
about the method class (budget-guided selection over a bounded candidate set,
calibrated on an exchangeable split) rather than about the artifact, so it
survives a revision.

---

## 5. Search log

### Databases and tools

arXiv (listing pages, abstract pages, full-text HTML, and PDF extraction); ACM
Digital Library; NeurIPS proceedings; OpenReview; JMLR; Nature/npj; Semantic
Scholar and Google Scholar (via search snippets); GitHub. All primary sources
were fetched directly except where noted below.

### Sources read in full or substantially

| Source | Depth |
|---|---|
| arXiv:2502.10947 (GCACI) | Full text (HTML v1), including Algorithm 1, Algorithm 2, §5, §6, §1.2 related work, references |
| arXiv:2606.00419v4 (POGO) | Full text, incl. §1.1 related work, §4.1, §4.2, Table 1, Algorithm 1, references |
| arXiv:2405.15106 / NeurIPS 2024 (AFCP) | Full text of the conference PDF, incl. §2.1 selection procedure, Algorithms 1/2/A1–A9, Discussion |
| arXiv:2605.09508 (BG-CFQS) | HTML, §I–§V-A. Algorithm 1 and Table II read. **Table II "Candidate quantile set" row truncated.** §V-B onward not read. |
| arXiv:2306.09335 (CCP) | Abstract, §1, §2, §2.2 via search-result extracts and the HTML v1; not fetched in full |
| arXiv:2601.08439 (Casparsen) | Abstract page, plus §related-work extracts from HTML v1 and the npj version |
| arXiv:2310.09242 (Mohan et al., WWW 2024) | §6.2 extract only |
| arXiv:2205.09095 (Rolling RC) | Abstract, §4.1.4 extract, OpenReview record, README |

### Queries run

**GCACI / group-conditional online conformal**
- `arXiv 2502.10947 no-regret learning online conformal prediction`
- direct fetch of `arxiv.org/abs/2502.10947` and `arxiv.org/html/2502.10947v1`
- `arXiv 2606.00419 "Parameter-Free" group conditional online conformal prediction`
- direct fetch of `arxiv.org/html/2606.00419v4`

**AFCP / adaptive group selection**
- `arXiv 2405.15106 conformal classification equalized coverage adaptively selected groups`
- direct fetch of the NeurIPS 2024 camera-ready PDF

**Heterogeneity gate**
- `clustered conformal prediction Ding Angelopoulos class-conditional many classes when grouping helps quantile similarity`
- `test whether group-conditional quantiles differ heterogeneity between-group within-group variance decide pooling calibration`

**Application**
- `conformal prediction network bandwidth allocation admission control guarantees networking` (low yield: mostly US patents; discarded)
- `"conformal prediction" throughput prediction wireless network resource allocation coverage guarantee`
- `conformal prediction LEO satellite Starlink throughput uncertainty risk control 2026`

**Starlink measurement**
- `arXiv 2605.09508 budget-guided quantile selection Starlink throughput forecasting`
- `Casparsen arXiv 2601.08439 Starlink 15-second scheduling interval latency`
- `"Vivisecting Starlink Throughput" StarNet Liu CoNEXT 2025 venue authors`

**Online risk control**
- `Feldman Ringel Bates Romano "Achieving Risk Control in Online Learning Settings" rolling risk control`

### Citations checked and resolved

Every arXiv identifier flagged in the brief resolves to a real paper saying what
was attributed to it, with one correction:

| ID | Real? | Says what was claimed? |
|---|---|---|
| 2502.10947 | Yes | Yes. Author list wrong in the brief: Kiyani, not Gupta. |
| 2606.00419 | Yes | Yes. Bharti, Pal, Teneggi, Sulam, 29 May 2026. |
| 2405.15106 | Yes | Exists and does adaptive group selection, but by a different statistic than the gate. |
| 2605.09508 | Yes | Yes. Xie et al., 10 May 2026, eess.SY. |
| 2601.08439 | Yes | Yes. Casparsen et al., 13 Jan 2026; also npj Wireless Technology (2026). |

The brief's general caveat about recent identifiers (2512.\*, 2603.\*, 2605.\*,
2606.\*, 2607.\*) turned out to be unfounded for the five that mattered. Note in
passing that several *other* recent IDs surfaced incidentally and also resolved
cleanly (2602.03168 UP-OCP, 2602.16537, 2605.23560 SafeSABR, 2606.29324), so
recency alone is not a reason for suspicion here.

The one citation in the brief that needs correcting is StarNet's venue: it is
**Proc. ACM Netw. 3(CoNEXT4), Article 24, 1–23, November 2025**, doi
`10.1145/3768971`, by Zikun Liu, Fan-Xue Gabriella Reidys, Sarah Tanveer, Deepak
Vasisht. "CoNEXT 2025" is informally right (PACMNET is the CoNEXT proceedings
journal) but the bibliography entry should be the PACMNET form. Code at
`github.com/ConnectedSystemsLab/StarNet`.

### What I searched for and did not find

Stated as negatives so their weight can be judged. In each case I ran at least
one targeted query and read the top results; none is an exhaustive search.

- **No paper reporting a pre-test scalar that predicts the *sign* of the
  group-conditioning effect on tail risk.** Searched conformal and
  quantile-regression phrasings. Moderate confidence in the negative *within the
  conformal literature*; low confidence overall, because of §2.1.
- **No paper noting BG-CFQS's candidate-set floor.** BG-CFQS is four months old
  and lightly cited (I found it cited by SafeSABR, arXiv:2605.23560). With that
  little citation traffic, a negative here is weak evidence in general but
  reasonably safe in practice.
- **No conformal or distribution-free risk-control paper applied specifically to
  LEO/Starlink admission control.** BG-CFQS does the admission control but by
  quantile selection, not conformal calibration. So there is a narrow gap here —
  narrower than the project needs.
- **No paper distinguishing "geometry predicts level" from "geometry predicts
  residual structure"** for Starlink features. One query only. Low confidence.

### Gaps in this review, in order of how much they should worry you

1. **The shrinkage / empirical-Bayes / partial-pooling literature was not
   searched.** This is where an overturning citation for the gate is most likely
   to live. See §2.1. Do this before writing.
2. **Table II of BG-CFQS was truncated;** the `[0.15, 0.40]` bounds are not
   independently confirmed here. Check the PDF.
3. **Venue-specific databases were not searched directly.** SIGCOMM, NSDI, IMC,
   MobiCom, MMSys and INFOCOM were covered only through general web search and
   through the reference lists of the papers I read. A targeted DL search of IMC
   and CoNEXT proceedings for conformal/risk-control work would tighten the
   application negatives.
4. **CCP was not fetched in full.** The quotes in §1.4 come from search-result
   extracts of `arxiv.org/pdf/2306.09335` and `arxiv.org/html/2306.09335v1` and
   are consistent across three independent extracts, but the full method section
   and the clustering algorithm details were not read. Since CCP is now the
   central prior-art citation for the gate, read it properly before writing the
   related-work section.
5. **Non-English and non-arXiv preprint servers were not searched.**
6. **Statistics journals (JASA, Annals, JRSS-B, Biometrika) were searched only
   incidentally,** through the reference lists of the ML papers read. Given that
   the surviving claim is statistical, this is a real gap.

### One thing I would flag unprompted

The project's calibration layer maintains a `γ` (the ACI learning rate) per
regime. The brief does not say how `γ` is chosen. POGO's entire motivating
argument is that you cannot tune `γ` on a non-stationary stream, because a value
tuned during a stable period becomes inappropriate after a shift, and their
Figure 1b demonstrates exactly this. If `γ` was tuned on the calibration split or
selected by looking at test performance, that is a validity problem independent
of every novelty question in this report, and it would also partly explain the
dataset-to-dataset variation in the effect of conditioning that the gate is
built to predict. Worth checking against the run logs before anything else here
gets acted on.
