"""Service availability and operational cost, in the terms an operator uses.

Requirements 5 and 6 of the brief, "improve service availability" and "reduce
operational costs through autonomous network management", are the two that a
research pipeline most easily leaves unmeasured. Dropped-session counts and
overestimation rates are the right quantities for evaluating a forecaster and
the wrong ones for deciding whether to deploy it, because nobody buys a link
with an OverRate in the contract. This module converts what the decision layer
already produces into availability and money.

**Nothing here is novel and it is not meant to be.** These are the standard
definitions:

- **Availability** as delivered-over-promised service time, reported both as a
  fraction and in "nines", which is how carrier SLAs are written.
- **MTBF and MTTR** over outage runs, from reliability engineering, because a
  single availability figure cannot distinguish one long outage from many short
  ones and operators care about the difference.
- **Tiered SLA credits**, the standard remedy structure in carrier and cloud
  contracts: miss the committed availability and a percentage of the fee is
  refunded, with the percentage stepping up as the miss widens.
- **Committed Information Rate (CIR)** as the contracted floor.

The one modelling choice worth arguing about is the cost of *under*-allocation.
An allocator that is too conservative never violates an SLA and still loses
money, because capacity it declined to sell is capacity nobody paid for. Pricing
that as foregone revenue is what makes the comparison two-sided; without it,
the optimal policy is trivially "admit nothing".

Default prices are illustrative and are stated as such. **The ranking of
policies is not invariant to them**, which is the first thing measurement
showed: under commodity pricing, where a session-hour is worth little and the
credit ladder only bites below three nines, the aggressive policy is cheapest,
because foregone revenue accrues on every unsold session-hour while credits are
a small fraction of a small prorated fee. Risk control starts paying only once
a violation costs enough relative to a sale.

So the model reports the **break-even ratio** rather than a single verdict:
`break_even_credit_ratio` is how much more expensive an SLA violation has to be,
relative to the revenue on a session-hour, before a given policy beats another.
That number is a property of the link and the policy pair, not of our price
guesses, and it is what an operator would actually check against their own
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Standard carrier SLA credit ladder: (availability floor, fraction of the fee
#: refunded when achieved availability falls below it). Ordered worst first so a
#: linear scan returns the largest applicable credit. These tiers mirror the
#: shape used in common enterprise connectivity and cloud contracts; the exact
#: numbers vary by provider and are a config knob, not a finding.
DEFAULT_CREDIT_TIERS: tuple[tuple[float, float], ...] = (
    (0.950, 0.50),
    (0.990, 0.25),
    (0.999, 0.10),
)


@dataclass(frozen=True)
class ServiceLevelAgreement:
    """What was promised, and what breaking the promise costs.

    `bandwidth_per_session_mbps` has to match the decision layer's own figure,
    because availability is counted in session-slots and the two must agree on
    what a session is.
    """

    committed_rate_mbps: float = 100.0        # CIR, the contracted floor
    availability_target: float = 0.999        # three nines
    bandwidth_per_session_mbps: float = 10.0
    #: Monthly fee per admitted session. Only the ratio to
    #: `revenue_per_session_hour` affects any conclusion.
    monthly_fee_per_session: float = 50.0
    #: What an unsold session-hour would have earned. This is what makes an
    #: over-conservative allocator expensive rather than free.
    revenue_per_session_hour: float = 0.10
    credit_tiers: tuple[tuple[float, float], ...] = DEFAULT_CREDIT_TIERS
    #: Wall-clock seconds represented by one decision. The horizon, since a
    #: decision commits capacity for exactly that long.
    seconds_per_slot: float = 5.0

    def credit_fraction(self, achieved_availability: float) -> float:
        """Fraction of the fee refunded at this availability."""
        for floor, fraction in self.credit_tiers:
            if achieved_availability < floor:
                return fraction
        return 0.0


@dataclass(frozen=True)
class AvailabilityReport:
    """Availability in the forms an operator and an engineer each want."""

    availability: float           # delivered session-slots over promised
    nines: float                  # -log10(1 - availability)
    meets_target: bool
    n_slots: int
    promised_session_slots: int
    delivered_session_slots: int
    n_outages: int                # runs of consecutive degraded slots
    mtbf_seconds: float           # mean time between outage onsets
    mttr_seconds: float           # mean outage duration
    longest_outage_seconds: float
    degraded_slot_fraction: float

    def to_dict(self) -> dict:
        return {
            "availability": round(self.availability, 6),
            "nines": round(self.nines, 3),
            "meets_target": self.meets_target,
            "n_slots": self.n_slots,
            "promised_session_slots": self.promised_session_slots,
            "delivered_session_slots": self.delivered_session_slots,
            "n_outages": self.n_outages,
            "mtbf_seconds": round(self.mtbf_seconds, 1),
            "mttr_seconds": round(self.mttr_seconds, 1),
            "longest_outage_seconds": round(self.longest_outage_seconds, 1),
            "degraded_slot_fraction": round(self.degraded_slot_fraction, 6),
        }


def _nines(availability: float) -> float:
    """Availability as nines. 0.999 is three nines."""
    if availability >= 1.0:
        return float("inf")
    if availability <= 0.0:
        return 0.0
    return float(-np.log10(1.0 - availability))


def _outage_runs(degraded: np.ndarray) -> list[int]:
    """Lengths of maximal runs of True. One long outage is not ten short ones."""
    if degraded.size == 0 or not degraded.any():
        return []
    padded = np.concatenate(([False], degraded, [False]))
    change = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return (ends - starts).tolist()


def availability_report(admitted: np.ndarray, actual_mbps: np.ndarray,
                        sla: ServiceLevelAgreement) -> AvailabilityReport:
    """Availability of the sessions the allocator actually admitted.

    A session-slot is *delivered* when the link carried enough capacity to serve
    it. With `n` sessions admitted and `y` Mbps realised, the number served is
    `min(n, floor(y / b))`; the rest are degraded, and they are degraded because
    the allocator promised capacity that was not there.

    Availability is delivered over promised, which is the quantity a carrier SLA
    is written against. It is deliberately *not* the fraction of slots with no
    degradation at all: an operator who admits 20 sessions and drops 1 has not
    had an outage, and counting it as one would make every policy look identical.
    """
    admitted = np.asarray(admitted, dtype=float)
    actual = np.asarray(actual_mbps, dtype=float)
    if admitted.size != actual.size:
        raise ValueError("admitted and actual must have the same length")
    if admitted.size == 0:
        return AvailabilityReport(1.0, float("inf"), True, 0, 0, 0, 0,
                                  float("nan"), float("nan"), 0.0, 0.0)

    servable = np.floor(np.maximum(actual, 0.0) / sla.bandwidth_per_session_mbps)
    delivered = np.minimum(admitted, servable)
    promised_total = float(admitted.sum())
    delivered_total = float(delivered.sum())
    availability = delivered_total / promised_total if promised_total > 0 else 1.0

    # An outage slot is one where any admitted session went unserved. MTBF and
    # MTTR are computed over runs of these, which is the reliability-engineering
    # convention and distinguishes one long outage from many brief ones.
    degraded = delivered < admitted
    runs = _outage_runs(degraded)
    seconds = sla.seconds_per_slot
    total_seconds = admitted.size * seconds
    mtbf = (total_seconds / len(runs)) if runs else float("inf")
    mttr = (float(np.mean(runs)) * seconds) if runs else 0.0

    return AvailabilityReport(
        availability=availability,
        nines=_nines(availability),
        meets_target=bool(availability >= sla.availability_target),
        n_slots=int(admitted.size),
        promised_session_slots=int(promised_total),
        delivered_session_slots=int(delivered_total),
        n_outages=len(runs),
        mtbf_seconds=mtbf,
        mttr_seconds=mttr,
        longest_outage_seconds=(max(runs) * seconds) if runs else 0.0,
        degraded_slot_fraction=float(degraded.mean()),
    )


@dataclass(frozen=True)
class CostReport:
    """What the policy cost, split into the two ways it can lose money."""

    sla_credit_cost: float          # refunds owed for missing the target
    foregone_revenue: float         # capacity that existed and was not sold
    total_cost: float
    cost_per_hour: float
    credit_fraction: float
    achieved_availability: float
    unsold_session_hours: float
    served_session_hours: float
    hours: float

    def to_dict(self) -> dict:
        return {
            "sla_credit_cost": round(self.sla_credit_cost, 4),
            "foregone_revenue": round(self.foregone_revenue, 4),
            "total_cost": round(self.total_cost, 4),
            "cost_per_hour": round(self.cost_per_hour, 6),
            "credit_fraction": self.credit_fraction,
            "achieved_availability": round(self.achieved_availability, 6),
            "unsold_session_hours": round(self.unsold_session_hours, 3),
            "served_session_hours": round(self.served_session_hours, 3),
            "hours": round(self.hours, 3),
        }


def cost_report(admitted: np.ndarray, actual_mbps: np.ndarray,
                sla: ServiceLevelAgreement,
                availability: AvailabilityReport | None = None) -> CostReport:
    """Operational cost of an allocation policy, in currency per hour.

    Two terms, because a policy can lose money in two directions:

    **SLA credits.** Missing the committed availability triggers a refund of a
    fraction of the fee, stepping up as the miss widens. This is what
    over-allocation costs.

    **Foregone revenue.** Capacity the link carried and the allocator declined
    to sell earns nothing. This is what over-conservatism costs, and including
    it is what stops "admit nothing" from being optimal.

    The sum is the quantity to minimise, and it is the only place in this
    project where the two failure directions are on the same scale.
    """
    admitted = np.asarray(admitted, dtype=float)
    actual = np.asarray(actual_mbps, dtype=float)
    report = availability or availability_report(admitted, actual, sla)

    hours = admitted.size * sla.seconds_per_slot / 3600.0
    slot_hours = sla.seconds_per_slot / 3600.0

    served_session_hours = report.delivered_session_slots * slot_hours
    servable = np.floor(np.maximum(actual, 0.0) / sla.bandwidth_per_session_mbps)
    unsold = np.maximum(servable - admitted, 0.0)
    unsold_session_hours = float(unsold.sum()) * slot_hours

    # Credits are owed on the fee for the sessions actually being billed, scaled
    # to the observation window. The monthly fee is prorated by hours observed.
    fraction = sla.credit_fraction(report.availability)
    hours_in_month = 24.0 * 30.0
    billed = served_session_hours / hours_in_month * sla.monthly_fee_per_session
    credit = billed * fraction

    foregone = unsold_session_hours * sla.revenue_per_session_hour
    total = credit + foregone
    return CostReport(
        sla_credit_cost=credit,
        foregone_revenue=foregone,
        total_cost=total,
        cost_per_hour=(total / hours) if hours > 0 else 0.0,
        credit_fraction=fraction,
        achieved_availability=report.availability,
        unsold_session_hours=unsold_session_hours,
        served_session_hours=served_session_hours,
        hours=hours,
    )


@dataclass
class PolicyOutcome:
    """Availability and cost for one allocation policy, ready to tabulate."""

    name: str
    availability: AvailabilityReport
    cost: CostReport
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"policy": self.name, **self.availability.to_dict(),
                **self.cost.to_dict(), **self.extras}


def evaluate_policy(name: str, bound_mbps: np.ndarray, actual_mbps: np.ndarray,
                    sla: ServiceLevelAgreement) -> PolicyOutcome:
    """Run one safe-bound policy all the way through to availability and cost."""
    bound = np.asarray(bound_mbps, dtype=float)
    admitted = np.floor(np.maximum(bound, 0.0) / sla.bandwidth_per_session_mbps)
    report = availability_report(admitted, actual_mbps, sla)
    cost = cost_report(admitted, actual_mbps, sla, report)
    return PolicyOutcome(name=name, availability=report, cost=cost,
                         extras={"mean_admitted": round(float(admitted.mean()), 4)})


def break_even_credit_ratio(conservative: np.ndarray, aggressive: np.ndarray,
                            actual_mbps: np.ndarray,
                            sla: ServiceLevelAgreement) -> dict:
    """How costly must a violation be before the conservative policy wins?

    The two cost terms scale with different prices. Foregone revenue scales with
    `revenue_per_session_hour`; SLA credits scale with the fee and the credit
    ladder. Their ratio therefore decides the ranking, and our defaults are
    guesses.

    Rather than pick a number, solve for the one where the policies tie. Model a
    violation as costing `k` times the revenue on a session-hour, and price each
    policy as

        cost(k) = unsold_session_hours * r + violated_session_hours * r * k

    Both are linear in `k`, so the crossing is closed form. Returned alongside
    the raw quantities so a reader can substitute their own contract.

    A ratio at or below zero means the conservative policy wins outright. A very
    large one means it needs an implausibly expensive violation to be worth it,
    which is itself a finding about the link.
    """
    per = sla.bandwidth_per_session_mbps
    slot_hours = sla.seconds_per_slot / 3600.0
    actual = np.asarray(actual_mbps, dtype=float)
    servable = np.floor(np.maximum(actual, 0.0) / per)

    def terms(bound: np.ndarray) -> tuple[float, float]:
        admitted = np.floor(np.maximum(np.asarray(bound, dtype=float), 0.0) / per)
        unsold = float(np.maximum(servable - admitted, 0.0).sum()) * slot_hours
        violated = float(np.maximum(admitted - servable, 0.0).sum()) * slot_hours
        return unsold, violated

    unsold_c, violated_c = terms(conservative)
    unsold_a, violated_a = terms(aggressive)

    # cost_c(k) = unsold_c + violated_c*k ; cost_a(k) = unsold_a + violated_a*k
    # Tie when (unsold_c - unsold_a) = k * (violated_a - violated_c).
    numerator = unsold_c - unsold_a
    denominator = violated_a - violated_c
    if abs(denominator) < 1e-12:
        ratio = float("inf") if numerator > 0 else float("-inf")
    else:
        ratio = numerator / denominator

    return {
        "break_even_credit_ratio": round(float(ratio), 3),
        "conservative_unsold_session_hours": round(unsold_c, 3),
        "conservative_violated_session_hours": round(violated_c, 3),
        "aggressive_unsold_session_hours": round(unsold_a, 3),
        "aggressive_violated_session_hours": round(violated_a, 3),
        "interpretation": (
            "a violated session-hour must cost at least this many times a sold "
            "session-hour before the conservative policy is cheaper"
        ),
    }
