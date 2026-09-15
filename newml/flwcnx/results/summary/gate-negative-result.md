# The heterogeneity gate does not work, and the predictor claim was wrong

Written 2026-09-01, correcting a claim made earlier the same day in
`README.md`, `docs/progress.md`, `docs/novelty-assessment.md`,
`results/summary/starnet-regime-grid.md` and the project memory.

## The claim that was made

That the spread of the fitted per-regime offset predicts whether conditioning
will help, monotonically across four datasets, from **a quantity computable on
the calibration split before any test decision is made**:

| dataset | offset spread | effect on P10 |
|---|---|---|
| StarNet USA | 1.85 Mbps | +6.1% |
| StarNet Germany | 5.44 Mbps | +0.1% |
| StarNet Canada | 11.30 Mbps | -1.7% |
| WetLinks | 9.28 Mbps | -5.6% |

## Why it was wrong

**Those numbers are not from the calibration split.** They were read from
`detail.per_regime[*].offset_mean` in the adaptive calibrator's summary, which
is the mean offset *actually applied during the test replay*. It is a
post-hoc quantity, computed from the test period, and it is not available
before a single decision is made.

The claim that it was pre-test was the specific thing this project is built to
prevent, and it survived several hours and five files because the number was
read out of a results file without checking which stage produced it.

## What the pre-test statistic actually does

`calibrate/heterogeneity.py` computes the right quantity: per-regime offsets
fitted on the calibration split only. Measured on the same three locations, at
the `level` partition:

| location | pre-test spread | Cochran Q | I-squared | measured effect |
|---|---|---|---|---|
| StarNet USA | 1.99 Mbps | 18.3 | 0.836 | +6.1% (hurts) |
| StarNet Germany | **5.83 Mbps** | 130.8 | 0.977 | +0.1% (neutral) |
| StarNet Canada | 5.17 Mbps | 21.6 | 0.861 | **-1.7% (helps)** |

Ordering by pre-test spread: USA < Canada < Germany.
Ordering by benefit: Canada > Germany > USA.

**Germany has the largest pre-test spread and gets no benefit; Canada has a
smaller spread and gets the only benefit.** The relationship is not monotonic
and the statistic does not predict the sign. The post-hoc version looked
monotonic because Canada's applied offsets diverge *during* the test period,
which is a consequence of the adaptation, not a predictor of it.

## And the gate fires everywhere

Independently of the ordering problem, the gate as built is unusable. It
returns `condition = True` on all three locations, including the US trace where
conditioning costs 6.1%:

    USA level:  Q = 18.27 on 3 dof, p = 0.00039, I2 = 0.836, spread 1.99 Mbps

This is textbook statistical-versus-practical significance. With roughly 1,200
calibration points per regime, a 2 Mbps difference in the fitted offset is
comfortably significant. `I-squared` does not rescue it, because I-squared is
scale free: it reports that the 2 Mbps difference is mostly real rather than
sampling noise, which is true and useless. Both hurdles are about whether the
difference *exists*, and neither is about whether it is *worth anything*.

`gated_adaptive` therefore returns results byte identical to
`adaptive_regime_conformal` at every granularity on every location, which is
how the problem was noticed.

## Status

**The gate is a negative result.** It is kept in the tree, with tests, because
the negative is worth reporting and because the machinery (per-regime quantile
standard errors, Cochran's Q against a hand-rolled chi-squared tail) is correct
and independently checked. But it should not be presented as a working
mechanism, and the predictor claim is withdrawn.

This removes the last surviving candidate for a methods contribution. The
independent review anticipated it exactly: it rated this thread "the thinnest
surviving" and warned that the shrinkage literature would likely collapse it.
It collapsed for a simpler reason before that literature was even searched.

## What would have to be true for a gate to work

Not attempted here, recorded so the next person does not restart from zero:

1. A **practical-significance** criterion, in units that matter. Neither p nor
   I-squared is one. A candidate is the spread relative to the magnitude of the
   global offset, but on these three locations that gives USA 23%, Canada 44%,
   Germany 49%, which still puts Germany top and still fails.
2. Something that separates Germany from Canada. Both have real, significant
   per-regime differences on the calibration split; only Canada's persist into
   the test period. That is a **stability** question, not a spread question,
   and it probably needs the calibration split cut in two and the per-regime
   offsets compared across the halves.
3. Honest accounting of how few points there are. Three locations and one
   WetLinks site is four observations. Any rule fitted on four points and
   validated on the same four is a story, not a predictor.
