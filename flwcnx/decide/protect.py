"""Splitting a calibrated capacity bound across flows without halting the
ones that matter.

`admission.py` answers "how many sessions fit" and stops there. This answers
"which flow gets which rate", which is the question a congestion episode
actually poses. It is the consumer of `decide/flows.py` and, through that
module's criticality scores, the only place in the repository where the
calibrated bound turns into a per-flow decision.

## Why the bound and not the forecast

The allocator divides `B_t`, the calibrated lower bound, rather than the point
forecast. That choice is what makes a protected floor mean anything. Reserving
0.5 Mbps for a patient monitor out of a forecast the link then fails to deliver
reserves nothing: the shortfall lands on whichever flow the transport starves
first, and the reservation was a statement about a number rather than about the
link. Dividing a bound whose overestimation rate is held at a stated budget
makes the reservation an assertion that can be checked, and
`scripts/run_protection.py` checks it.

## The allocation rule

Criticality-weighted max-min fairness with protected floors. For a weight
`omega_f`, a floor `m_f` and a demand `d_f`, each flow receives

    r_f(tau) = clip(omega_f * tau, m_f, d_f)

and `tau` is raised until the rates sum to the capacity. This is progressive
filling (Bertsekas and Gallager, section 6.5.2) with two modifications: the
per-flow weights make the fill rates unequal, and the floors mean a protected
flow starts filled rather than empty.

The sum is continuous, non-decreasing and piecewise linear in `tau`, with
breakpoints where a flow leaves its floor and where it reaches its demand, so
`tau` is solved exactly by sorting the breakpoints and walking the segments.
No bisection and no iteration to a tolerance: the result is the exact
weighted max-min fair point, in `O(n log n)`.

Three properties follow from the form and are pinned by tests:

- **Floors are honoured whenever they are jointly feasible.** If the protected
  floors sum to no more than the capacity, every protected flow gets at least
  its floor. This is the guarantee the layer exists to provide.
- **Work conserving.** The allocation sums to `min(capacity, total demand)`, so
  capacity a critical flow does not want is not held idle for it.
- **Nothing is halted outright.** `weight_floor` keeps every weight strictly
  positive, so even a flow scored at zero criticality receives a share rather
  than a stop. Killing a TCP connection does not save the capacity it was
  using; it defers the same bytes into a retry, usually into the same episode.

## When the floors do not fit

If the protected floors exceed the capacity, no allocation satisfies them and
the honest thing is to say so. The layer degrades by strict criticality order,
granting floors from the most critical down until the capacity is exhausted,
and returns the list of flows whose floor it could not meet in
`AllocationResult.breached`. That field is an alarm, not a diagnostic: it is
the case where the link physically cannot carry what has been declared
critical, and the answer is an operator decision rather than an allocation.

A layer that silently shaded every floor down proportionally would look better
in aggregate and would leave every protected flow below the rate at which it is
useful, which is the worst available outcome.

## What is reproduced and what is not

Weighted max-min fairness, progressive filling and class-based priority are
long-published (Bertsekas and Gallager; Demers, Keshav and Shenker, SIGCOMM
1989; Parekh and Gallager, ToN 1993; RFC 2474). The baselines in this module
are reimplementations of the standard policies, present so the protection layer
is measured against what a real shaper does rather than against nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flwcnx.decide.flows import (
    CLASS_PRIOR_LOG_ODDS,
    CriticalityScorer,
    FlowClass,
    FlowObservation,
    FlowSpec,
    ScorerConfig,
)

__all__ = [
    "AllocationResult",
    "FlowDemand",
    "POLICIES",
    "ProtectionConfig",
    "ProtectionController",
    "allocate_by_class",
    "allocate_equal_share",
    "allocate_protected",
    "allocate_shed_largest",
    "weighted_max_min",
]


#: Relative slack when comparing a sum of floors against a capacity. Purely
#: numerical: the same floors summed in two different orders do not give the
#: same double.
_FLOOR_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ProtectionConfig:
    """Knobs for the allocator."""

    #: Smallest share weight any flow can have. Strictly positive so that no
    #: flow is ever allocated zero while capacity remains, which is the
    #: difference between throttling a background sync and killing it.
    weight_floor: float = 0.05
    #: Sharpens the preference for critical flows. At 1.0 the weight is linear
    #: in criticality; above 1.0 the mid-range is pushed down, which widens the
    #: gap between a confidently critical flow and an uncertain one.
    weight_exponent: float = 1.0
    #: Honour the floors of protected flows. Turning this off reduces the layer
    #: to weighted fair queueing and is how the ablation measures what the
    #: floors themselves are worth.
    honour_floors: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.weight_floor <= 1.0:
            raise ValueError("weight_floor must lie in (0, 1]")
        if self.weight_exponent <= 0:
            raise ValueError("weight_exponent must be positive")

    def weight(self, criticality: float) -> float:
        c = min(max(criticality, 0.0), 1.0)
        return self.weight_floor + (1.0 - self.weight_floor) * c ** self.weight_exponent


@dataclass(frozen=True)
class FlowDemand:
    """One flow's ask for one slot, as the allocator sees it.

    Deliberately the whole input: every policy in this module is a pure
    function of the capacity and a list of these, so two policies compared on
    the same slot are compared on identical information. The only thing that
    differs between the protection policy and the oracle is what was written
    into `criticality` and `protected`.
    """

    flow_id: str
    demand_mbps: float
    floor_mbps: float = 0.0
    criticality: float = 0.5
    protected: bool = False
    declared: FlowClass = FlowClass.STANDARD

    def __post_init__(self) -> None:
        if self.demand_mbps < 0 or self.floor_mbps < 0:
            raise ValueError("demand and floor must be non-negative")
        if self.floor_mbps > self.demand_mbps:
            raise ValueError(f"flow {self.flow_id}: floor exceeds demand")


@dataclass(frozen=True)
class AllocationResult:
    """What every flow got, and what the allocator could not do."""

    rates: dict[str, float]
    capacity_mbps: float
    #: Protected flows whose floor could not be met. Non-empty means the link
    #: cannot carry what has been declared critical, which is an operator
    #: decision rather than an allocation.
    breached: tuple[str, ...] = ()
    #: Flows allocated below their own floor, protected or not. A superset of
    #: `breached`: an unprotected video call below its floor is the expected
    #: cost of congestion, not a failure of the layer.
    below_floor: tuple[str, ...] = ()

    @property
    def allocated_mbps(self) -> float:
        return sum(self.rates.values())

    @property
    def protection_breached(self) -> bool:
        return bool(self.breached)

    def utilisation(self) -> float:
        return self.allocated_mbps / self.capacity_mbps if self.capacity_mbps > 0 else 0.0

    def wasted_mbps(self, flows: list[FlowDemand]) -> float:
        """Capacity spent on flows that got too little to be useful.

        A video call at 0.3 Mbps against a 1.5 Mbps floor is not a degraded
        call, it is a dropped one, and the 0.3 bought nothing. Reported rather
        than reclaimed: reclaiming it means deciding to kill the call, which is
        a policy this layer deliberately leaves to the operator.
        """
        return sum(self.rates.get(f.flow_id, 0.0) for f in flows
                   if 0.0 < self.rates.get(f.flow_id, 0.0) < f.floor_mbps)


# -- the core solver ---------------------------------------------------------


def weighted_max_min(capacity_mbps: float, weights: dict[str, float],
                     floors: dict[str, float], demands: dict[str, float]
                     ) -> dict[str, float]:
    """Exact weighted max-min fair rates under per-flow floors and caps.

    Solves `sum_f clip(omega_f * tau, m_f, d_f) = capacity` for `tau` by walking
    the breakpoints of that piecewise-linear sum. Assumes the floors fit;
    callers handle infeasibility, because what to do about it is a policy
    question and not a numerical one.
    """
    ids = list(demands)
    if not ids:
        return {}
    total_demand = sum(demands.values())
    if capacity_mbps >= total_demand:
        return dict(demands)

    total_floor = sum(floors.get(i, 0.0) for i in ids)
    # Equality is feasible and lands at tau = 0, which the sweep below handles;
    # only a strict shortfall is the caller's problem. The tolerance is not a
    # softening of that rule: the caller resolves infeasibility by summing the
    # floors and then hands them back to be summed again in a different order,
    # and binary floating point makes those two sums differ by an ulp or two on
    # a link carrying hundreds of megabits. Trimming inside the tolerance keeps
    # the sweep well posed; anything larger is a real infeasibility and raises.
    tolerance = _FLOOR_TOLERANCE * max(1.0, total_floor)
    if capacity_mbps < total_floor - tolerance:
        raise ValueError(
            f"floors sum to {total_floor:.6f} Mbps against a capacity of "
            f"{capacity_mbps:.6f}; resolve the infeasibility before filling"
        )
    if total_floor > capacity_mbps:
        scale = capacity_mbps / total_floor if total_floor > 0 else 0.0
        floors = {i: floors.get(i, 0.0) * scale for i in ids}
        total_floor = capacity_mbps

    # Two events per flow: leaving the floor at tau = m/omega, reaching the
    # demand cap at tau = d/omega. Between events the total is A + B * tau.
    events: list[tuple[float, float, float]] = []
    for i in ids:
        omega = weights[i]
        if omega <= 0:
            raise ValueError(f"flow {i}: weight must be positive")
        events.append((floors.get(i, 0.0) / omega, -floors.get(i, 0.0), omega))
        events.append((demands[i] / omega, demands[i], -omega))
    events.sort(key=lambda e: e[0])

    a, b, tau = total_floor, 0.0, 0.0
    k = 0
    while k < len(events):
        tau_next = events[k][0]
        if a + b * tau_next >= capacity_mbps:
            break
        tau = tau_next
        while k < len(events) and events[k][0] == tau_next:
            a += events[k][1]
            b += events[k][2]
            k += 1

    # b > 0 here: the total is continuous and rises from total_floor (below the
    # capacity) to total_demand (above it), so the crossing segment has slope.
    tau_star = (capacity_mbps - a) / b if b > 0 else tau
    return {i: min(max(weights[i] * tau_star, floors.get(i, 0.0)), demands[i])
            for i in ids}


# -- the protection policy ---------------------------------------------------


def allocate_protected(capacity_mbps: float, flows: list[FlowDemand],
                       config: ProtectionConfig | None = None) -> AllocationResult:
    """Criticality-weighted max-min fair allocation with protected floors."""
    config = config or ProtectionConfig()
    if not flows:
        return AllocationResult({}, capacity_mbps)
    capacity_mbps = max(capacity_mbps, 0.0)

    weights = {f.flow_id: config.weight(f.criticality) for f in flows}
    demands = {f.flow_id: f.demand_mbps for f in flows}
    floors = {f.flow_id: (f.floor_mbps if (f.protected and config.honour_floors) else 0.0)
              for f in flows}

    breached: tuple[str, ...] = ()
    if sum(floors.values()) > capacity_mbps * (1.0 + _FLOOR_TOLERANCE):
        floors, breached = _degrade_floors(capacity_mbps, flows, floors)

    rates = weighted_max_min(capacity_mbps, weights, floors, demands)
    return AllocationResult(rates=rates, capacity_mbps=capacity_mbps, breached=breached,
                            below_floor=_below_floor(rates, flows))


def _degrade_floors(capacity_mbps: float, flows: list[FlowDemand],
                    floors: dict[str, float]) -> tuple[dict[str, float], tuple[str, ...]]:
    """Grant floors by strict criticality order until the capacity runs out.

    The alternative, shading every floor down proportionally, keeps the arithmetic
    tidy and leaves every protected flow below the rate at which it does anything.
    Ties break toward the smaller floor, which serves more flows with the same
    capacity.
    """
    order = sorted((f for f in flows if floors.get(f.flow_id, 0.0) > 0.0),
                   key=lambda f: (-f.criticality, f.floor_mbps, f.flow_id))
    granted: dict[str, float] = dict.fromkeys(floors, 0.0)
    breached: list[str] = []
    remaining = capacity_mbps
    for flow in order:
        need = floors[flow.flow_id]
        if need <= remaining:
            granted[flow.flow_id] = need
            remaining -= need
        else:
            breached.append(flow.flow_id)
    return granted, tuple(breached)


def _below_floor(rates: dict[str, float], flows: list[FlowDemand]) -> tuple[str, ...]:
    # A tolerance, because the exact tau lands on a floor by construction and
    # binary floating point will sit a few ulps under it.
    return tuple(f.flow_id for f in flows
                 if f.floor_mbps > 0.0 and rates.get(f.flow_id, 0.0) < f.floor_mbps - 1e-9)


# -- baselines ---------------------------------------------------------------


def allocate_equal_share(capacity_mbps: float, flows: list[FlowDemand],
                         config: ProtectionConfig | None = None) -> AllocationResult:
    """Unweighted max-min fairness. What a fair queue does, and the reference
    point for how much the weighting is worth."""
    if not flows:
        return AllocationResult({}, capacity_mbps)
    capacity_mbps = max(capacity_mbps, 0.0)
    rates = weighted_max_min(
        capacity_mbps,
        weights=dict.fromkeys((f.flow_id for f in flows), 1.0),
        floors=dict.fromkeys((f.flow_id for f in flows), 0.0),
        demands={f.flow_id: f.demand_mbps for f in flows},
    )
    return AllocationResult(rates, capacity_mbps, below_floor=_below_floor(rates, flows))


def allocate_shed_largest(capacity_mbps: float, flows: list[FlowDemand],
                          config: ProtectionConfig | None = None) -> AllocationResult:
    """Throttle the heaviest flows first, which is the policy this layer exists
    to argue against.

    Present because it is what a rate-based shaper actually does, and because
    the cost of it is not obvious until it is measured: the flows it sheds first
    are the large ones, and a radiology push, an instrument upload and an
    operating system update are all large.
    """
    if not flows:
        return AllocationResult({}, capacity_mbps)
    capacity_mbps = max(capacity_mbps, 0.0)
    rates = {f.flow_id: 0.0 for f in flows}
    remaining = capacity_mbps
    # Smallest first: the small flows survive, the large ones absorb the cut.
    for flow in sorted(flows, key=lambda f: (f.demand_mbps, f.flow_id)):
        grant = min(flow.demand_mbps, remaining)
        rates[flow.flow_id] = grant
        remaining -= grant
        if remaining <= 0:
            break
    return AllocationResult(rates, capacity_mbps, below_floor=_below_floor(rates, flows))


def allocate_by_class(capacity_mbps: float, flows: list[FlowDemand],
                      config: ProtectionConfig | None = None) -> AllocationResult:
    """Strict priority by declared class, equal share within a class.

    The DiffServ policy (RFC 2474), and the baseline that matters most, because
    it is the one a competent operator would deploy today. Comparing against it
    isolates what the *dynamic* evidence buys over a static declaration: it gets
    every correctly declared flow right, and is wrong on exactly the two cases
    the evidence exists for, the undeclared critical flow and the misdeclared
    bulk one.
    """
    if not flows:
        return AllocationResult({}, capacity_mbps)
    capacity_mbps = max(capacity_mbps, 0.0)
    rates = {f.flow_id: 0.0 for f in flows}
    remaining = capacity_mbps

    by_class: dict[FlowClass, list[FlowDemand]] = {}
    for flow in flows:
        by_class.setdefault(flow.declared, []).append(flow)

    for klass in sorted(by_class, key=lambda c: -CLASS_PRIOR_LOG_ODDS.get(c, 0.0)):
        tier = by_class[klass]
        if remaining <= 0:
            break
        share = weighted_max_min(
            remaining,
            weights=dict.fromkeys((f.flow_id for f in tier), 1.0),
            floors=dict.fromkeys((f.flow_id for f in tier), 0.0),
            demands={f.flow_id: f.demand_mbps for f in tier},
        )
        rates.update(share)
        remaining -= sum(share.values())
    return AllocationResult(rates, capacity_mbps, below_floor=_below_floor(rates, flows))


#: Every policy the evaluation runs, keyed by the name used in result files and
#: figures. The oracle is not here: it is `allocate_protected` fed the ground
#: truth as criticality, which is what makes scorer error separable from
#: allocator error.
POLICIES = {
    "protected": allocate_protected,
    "by_class": allocate_by_class,
    "equal_share": allocate_equal_share,
    "shed_largest": allocate_shed_largest,
}


# -- wiring ------------------------------------------------------------------


@dataclass
class ProtectionController:
    """Scorer plus allocator, stepped one slot at a time.

    The object a deployment would hold. `step` takes the calibrated bound for
    the coming horizon and what the monitor saw during the last one, and
    returns rates. Everything it uses to decide slot `t` existed before slot
    `t` began, which is the same causality the rest of the pipeline keeps.
    """

    scorer: CriticalityScorer = field(default_factory=CriticalityScorer)
    config: ProtectionConfig = field(default_factory=ProtectionConfig)
    _slot: int = field(default=0, repr=False)

    @classmethod
    def build(cls, scorer_config: ScorerConfig | None = None,
              protection_config: ProtectionConfig | None = None) -> ProtectionController:
        return cls(scorer=CriticalityScorer(scorer_config),
                   config=protection_config or ProtectionConfig())

    def step(self, bound_mbps: float,
             observations: dict[str, tuple[FlowSpec, FlowObservation]]
             ) -> AllocationResult:
        slot = self._slot
        self._slot += 1
        criticality = self.scorer.update(slot, observations)
        flows = [
            FlowDemand(
                flow_id=flow_id,
                # Demand is what the flow asked for this slot; the spec's figure
                # is the fallback for a flow observed before it has sent.
                demand_mbps=max(obs.demand_mbps, 0.0) or spec.demand_mbps,
                floor_mbps=min(spec.floor_mbps,
                               max(obs.demand_mbps, 0.0) or spec.demand_mbps),
                criticality=criticality[flow_id],
                protected=self.scorer.is_protected(flow_id),
                declared=spec.declared,
            )
            for flow_id, (spec, obs) in observations.items()
        ]
        return allocate_protected(bound_mbps, flows, self.config)
