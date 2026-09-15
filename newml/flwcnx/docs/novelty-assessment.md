# Novelty assessment

Run 2026-09-01, before any paper was written, because the deliverable includes
a research paper and a claim of novelty has to survive a literature search
rather than an absence of one. CLAUDE.md open question 7 asked for exactly
this.

**Outcome: the method we built is not novel. The measurement work is.** The
paper has to be reframed from a methods contribution to a systems and
measurement contribution. Details below, with the prior work that closes each
door.

## 1. Per-regime adaptive conformal is already published

`calibrate/adaptive.py` maintains one adaptive alpha and one residual window
per regime, updating each from outcomes as they are observed. This was written
believing only the marginal single-alpha version (Gibbs and Candes 2021) was in
print.

That is wrong. **Group Conditional ACI (GCACI)**, Ramalingam, Kiyani and Roth 2025,
arXiv:2502.10947, "The Relationship between No-Regret Learning and Online
Conformal Prediction", generalises ACI to group-conditional guarantees. Quoting
the description in arXiv:2606.00419:

> "Ramalingam et al. generalized ACI to provide group-conditional guarantees by
> using parameterized prediction sets and showing that minimizing the quantile
> loss with a 'follow the regularized leader' (FTRL) algorithm, which requires
> learning rates, achieves group coverage."

Their mechanism is more general than ours: FTRL over parameterized sets against
our one alpha per bucket, and theirs carries a finite-time group-coverage
bound while ours carries none. Ours is the naive special case.

**POGO**, arXiv:2606.00419, 2026, goes further and removes the learning rate
entirely. It is the first parameter-free algorithm for group-conditional online
conformal prediction and reports the strongest known finite-time group-coverage
bounds. Our `gamma = 0.02` is exactly the hyperparameter POGO exists to
eliminate.

So the claim in the README, that maintaining one adaptive operating point per
regime "rather than the single global one the published rule assumes" is ours,
**is false and has to be withdrawn.**

## 2. The self-gating idea is also close to published work

The plan after the StarNet negative was to measure per-regime offset spread
against estimation noise and condition only when it clears. That direction is
occupied too.

**AFCP**, "Conformal Classification with Equalized Coverage for Adaptively
Selected Groups", NeurIPS 2024, arXiv:2405.15106, selects groups in a
data-driven way and states the motivating observation directly: "different
groups may not exhibit the same need for equalized coverage guarantees, as
standard prediction sets with marginal coverage may approximately satisfy
conditional coverage requirements for most groups." That is our finding, in a
classification and fairness setting.

Also relevant: **Kandinsky Conformal Prediction** (arXiv:2502.17264) for
overlapping and fractional group membership with minimax-optimal
group-conditional coverage error, and **Conditional Coverage Diagnostics for
Conformal Prediction** (arXiv:2512.11779).

The gate is not identical to AFCP: ours is a regression problem with a
one-sided risk budget under temporal drift, theirs is classification for
fairness, and ours keys on residual-offset spread rather than coverage
disparity. It is a reasonable engineering adaptation. It is not a new method,
and it should be presented as an application of a known idea.

## 3. What is actually ours

Everything that survives is measurement, and all of it is verified:

1. **First evaluation of risk-controlled capacity forecasting on LEO access
   links** across four datasets and two independent measurement campaigns.
2. **BG-CFQS's guarantee is conditional on exchangeability.** Reproduced 3/3
   under a random split (OverRate 0.340 against their 0.349, P30 0.671, P10
   0.848, all in range) and 1/3 under a temporal one.
   `scripts/split_sensitivity.py`.
3. **BG-CFQS cannot serve a budget below 0.15.** Their candidate set is
   T = [0.15, 0.40] (verified from their Table II), so the achieved OverRate is
   identical at budgets of 0.05, 0.10 and 0.15 on all four datasets, because the
   method returns the same quantile. Overshoot at 0.05 ranges 1.4x to 3.7x
   depending on the link. Their paper reports only 0.35, where the floor never
   binds.
4. **Static calibration misses its budget in both directions**, and the sign is
   a property of the drift rather than the method: it overshoots on WetLinks
   (0.423 against 0.35, test month slower) and undershoots on StarNet (0.293,
   test period easier). Either way an operator cannot set a budget and get it.
5. **Group conditioning helps only when the groups actually differ.** The
   attempt to turn that into a pre-test predictor failed and is reported as a
   negative: the spread figures originally quoted were post-hoc, and the
   calibration-split version does not order the effect.
   `results/summary/gate-negative-result.md`.
6. **The satellite covariates do not carry the signal.** With elevation,
   distance and candidate count *measured* rather than reconstructed, every
   axis is worse than no conditioning under the online layer. Objective O2's
   premise is not supported on this data.
7. **Casparsen's 15 s scheduling offset recovered independently** at 11.98,
   12.25 and 12.09 seconds on three continents, from 1 Hz throughput rather
   than 500 Hz latency. A crossover neither source paper performed.

## 4. What this changes

The paper is a measurement and systems paper, not a methods paper. That is a
weaker claim about method and a stronger claim about evidence, and for this
project it is the right trade: every item in section 3 is backed by a run with
a config snapshot beside it, whereas the methods claim was backed by not having
searched.

Concretely:

- `calibrate/adaptive.py` is **reproduction**, not contribution. Its module
  docstring and the README contribution boundary both need correcting, and the
  GCACI and POGO citations belong in `docs/references.bib`.
- The self-gating layer is still worth building, because it is the thing that
  makes the system usable on data like the US trace, but it ships as
  engineering rather than as a claim.
- The comparison against GCACI and POGO becomes a natural baseline section: we
  implemented the naive per-group version, and there are two better ones in
  print to compare against.
