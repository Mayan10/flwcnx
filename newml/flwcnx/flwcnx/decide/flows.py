"""Online criticality scoring for individual flows.

The rest of `decide/` reasons about the link as a single pipe: the calibrated
bound says how many megabits the next horizon can be trusted to carry, and
`admission.py` turns that into a session count. That is the right abstraction
for measuring a forecaster and the wrong one for shedding load, because it
treats every megabit as interchangeable. When the bound falls and something has
to give, "which megabit" is the entire question.

The naive answer, and the one most shapers implement, is to throttle whatever
is consuming the most. That rule is wrong in exactly the case that matters. A
radiology study being pushed to a regional archive against a reporting deadline
is a large, sustained, single-direction transfer, and so is an operating system
update. Bandwidth consumed carries almost no information about whether halting
the flow is acceptable.

This module infers that instead, from evidence the host can actually observe,
and it does so online because criticality is not a static property. A nightly
backup with a six hour window is the most sheddable thing on the link at 02:00
and the least sheddable at 07:55, without anything about the flow changing
except the clock.

## The model

Each flow carries a running log-odds that it is critical:

    L(t) = clip( rho * L(t-1) + (1 - rho) * ( prior + sum_k w_k z_k(t) ),
                 -L_max, +L_max )

    criticality c(t) = sigmoid( L(t) )

This is a first-order recursive Bayesian update with a forgetting factor, the
standard form for sequential evidence accumulation under non-stationarity
(Vovk, Gammerman and Shafer, chapter 2, for the log-odds pooling; the
forgetting factor is the usual exponential discount). Three properties earn it
its place here:

- **Absent evidence is zero, not negative.** A channel that cannot see anything
  this slot returns 0.0 and moves the posterior by exactly the decay term. In a
  linear score, a missing signal has to be imputed; in log-odds it is free.
- **The prior is the initial condition**, so a flow declared life-safety is
  protected on its first packet rather than after an evidence window. A layer
  that needs five seconds of observation before it will protect a patient
  monitor has not solved the problem.
- **Clipping is load-bearing.** Without it an accumulator that has seen sixty
  consecutive seconds of bulk transfer needs another sixty to change its mind.
  `L_max` bounds how much history any flow can bank, which bounds how long a
  wrong classification can persist after the evidence turns.

## The channels

Seven, each returning a bounded score in [-1, 1] that is weighted into the
update. They are deliberately weak and deliberately redundant, because no single
one of them is safe on its own.

`attention`        the owning process has focus, and the user is typing in it
`interactivity`    small bidirectional exchanges with idle gaps
`elasticity`       measured: did offered load fall when we last throttled it
`deadline`         slack between the deadline and the ETA at the current rate
`irreversibility`  progress already spent that a restart would throw away
`volume`           sustained high-rate single-direction transfer
`recurrence`       how routine this peer and process pair is

Several are asymmetric on purpose, and the asymmetries are the design. Presence
of user attention is strong evidence; absence is weak, because a clinical
transfer nobody is watching is still clinical. `volume` is the channel that
could reconstruct the very failure this layer exists to prevent, so it carries
the smallest weight of the seven.

`elasticity` is the only closed-loop channel. When the allocator throttles a
flow it creates an experiment: an elastic flow's offered load falls to meet the
new rate, an inelastic one keeps demanding. The system learns from its own
actions rather than only from observation.

## What this is not

Nothing here is a novel mechanism. Log-odds pooling of weak evidence is
textbook, hysteresis on a binary decision is textbook, and prioritising traffic
by class is DiffServ (RFC 2474). Deadline-driven scheduling of network flows is
D3 (Wilson et al., SIGCOMM 2011) and PDQ (Hong et al., SIGCOMM 2012). What is
specific to this project is the coupling: the scores here are consumed by
`decide/protect.py` against a *calibrated* capacity bound, so the protection
budget is one the link is known to be able to honour at a stated risk, rather
than a point forecast the shaper hopes is right.

**The weights below are hand-set, not learned.** There is no labelled flow
corpus for this link, and fitting them on the synthetic workload in
`decide/workload.py` and then evaluating on the same generator would be
circular. `scripts/run_protection.py --sensitivity` perturbs them to show what
the conclusions do and do not survive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "CHANNEL_WEIGHTS",
    "CLASS_PRIOR_LOG_ODDS",
    "CriticalityScorer",
    "FlowClass",
    "FlowObservation",
    "FlowSpec",
    "FlowState",
    "ScorerConfig",
    "channel_scores",
]


class FlowClass(str, Enum):
    """What a flow says it is.

    A declaration is a prior, not a verdict. Trusting it absolutely gives every
    process on the host a one-bit override of the shaper, which is why DSCP
    marks are routinely rewritten at administrative boundaries. Treating it as
    evidence to be confirmed or contradicted keeps the useful part.
    """

    LIFE_SAFETY = "life_safety"      # patient telemetry, alarms, telesurgery
    CLINICAL = "clinical"            # imaging transfer, teleconsultation
    RESEARCH = "research"            # instrument capture, long compute upload
    INTERACTIVE = "interactive"      # shells, remote desktops, editors
    STANDARD = "standard"            # the default, and what most flows declare
    BULK = "bulk"                    # declared batch transfer
    BACKGROUND = "background"        # sync, backup, telemetry upload


#: Prior log-odds of criticality per declared class. `STANDARD` is deliberately
#: 0.0: the default declaration carries no information in either direction, and
#: most of the flows this layer has to get right arrive wearing it.
CLASS_PRIOR_LOG_ODDS: dict[FlowClass, float] = {
    FlowClass.LIFE_SAFETY: 4.0,
    FlowClass.CLINICAL: 2.5,
    FlowClass.RESEARCH: 1.0,
    FlowClass.INTERACTIVE: 0.5,
    FlowClass.STANDARD: 0.0,
    FlowClass.BULK: -1.5,
    FlowClass.BACKGROUND: -2.5,
}

#: Weight of each evidence channel in log-odds units. `volume` is the smallest
#: because it is the channel that most resembles the failure mode this layer
#: exists to prevent, and `deadline` the largest because it is the only one that
#: changes a flow's answer over its own lifetime.
CHANNEL_WEIGHTS: dict[str, float] = {
    "attention": 1.6,
    "interactivity": 1.0,
    "elasticity": 1.4,
    "deadline": 1.8,
    "irreversibility": 1.2,
    "volume": 0.5,
    "recurrence": 0.8,
}


@dataclass(frozen=True)
class FlowSpec:
    """What is known about a flow when it starts, and does not change.

    Every field is something a host-side monitor can fill from `/proc`, a
    connection tracker and the owning process, with the exception of
    `bytes_total`, which is known for an HTTP transfer with a content length and
    unknown for a stream. Unknown is represented as 0.0 and disables the
    deadline and irreversibility channels for that flow rather than guessing.
    """

    flow_id: str
    declared: FlowClass = FlowClass.STANDARD
    #: Rate below which the flow is useless rather than slow. A video call at
    #: 0.2 Mbps is not a degraded video call, it is a dropped one, and capacity
    #: spent holding it there is wasted. This is what the allocator protects.
    floor_mbps: float = 0.0
    #: Offered load if nothing constrained it.
    demand_mbps: float = 1.0
    #: Total transfer size in megabits, 0.0 when unbounded or unknown.
    bytes_total_mbit: float = 0.0
    #: Seconds from flow start to its deadline, inf when there is none.
    deadline_s: float = math.inf
    #: Whether an interrupted transfer resumes from where it stopped. A
    #: non-resumable transfer at 95% is the most expensive thing on the link
    #: to halt and looks identical to a fresh one to a rate-based shaper.
    resumable: bool = True
    #: How routine this process and peer pair is, in [0, 1]. A nightly backup
    #: scores near 1; a transfer the user started by hand scores near 0.
    recurrence: float = 0.0
    #: Ground truth, for evaluation only. The scorer never reads this.
    truly_critical: bool | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.floor_mbps < 0 or self.demand_mbps < 0:
            raise ValueError("floor and demand must be non-negative")
        if self.floor_mbps > self.demand_mbps:
            raise ValueError(
                f"flow {self.flow_id}: floor {self.floor_mbps} exceeds demand "
                f"{self.demand_mbps}, which is not a flow the allocator can serve"
            )
        if not 0.0 <= self.recurrence <= 1.0:
            raise ValueError("recurrence must lie in [0, 1]")


@dataclass(frozen=True)
class FlowObservation:
    """What the monitor saw of one flow during one decision slot."""

    #: Payload megabits carried in each direction this slot.
    mbit_down: float = 0.0
    mbit_up: float = 0.0
    packets_down: int = 0
    packets_up: int = 0
    #: Fraction of the slot with no packet in either direction, in [0, 1].
    idle_fraction: float = 0.0
    #: The owning process has user focus.
    foreground: bool = False
    #: Keystrokes and clicks attributed to the owner this slot.
    user_events: int = 0
    #: Offered load this slot, which is what the flow asked for rather than
    #: what it got. The difference between this and the delivered rate is how
    #: the elasticity channel gets its experiment.
    demand_mbps: float = 0.0
    #: Rate the allocator granted last slot, and whether that was a throttle.
    granted_last_mbps: float = 0.0
    throttled_last: bool = False
    #: Megabits still to transfer, 0.0 when unbounded or unknown.
    remaining_mbit: float = 0.0
    #: Seconds until the deadline. Negative means already late.
    deadline_remaining_s: float = math.inf
    #: Fraction of the transfer already completed, in [0, 1].
    progress: float = 0.0
    age_s: float = 0.0


@dataclass(frozen=True)
class ScorerConfig:
    """Knobs for the accumulator and the hysteresis.

    `rho` sets the memory: at 0.8 a channel that flips sign moves the posterior
    most of the way within about five slots, which at a 5 s horizon is 25 s.
    Slower than that and the deadline channel cannot act in time; faster and the
    score chases per-slot noise in the packet counters.
    """

    rho: float = 0.8
    max_log_odds: float = 8.0
    weights: dict[str, float] = field(default_factory=lambda: dict(CHANNEL_WEIGHTS))
    priors: dict[FlowClass, float] = field(
        default_factory=lambda: dict(CLASS_PRIOR_LOG_ODDS)
    )
    #: Hysteresis band. A flow becomes protected at `protect_threshold` and only
    #: loses protection below `release_threshold`. A single threshold makes the
    #: protected set flap across the boundary, and every flap is a rate change
    #: the transport below has to absorb.
    protect_threshold: float = 0.60
    release_threshold: float = 0.45
    #: Slots a flow must stay protected before it can be demoted, on top of the
    #: hysteresis band. Guards the case where the evidence itself is noisy
    #: enough to cross the whole band.
    min_protected_slots: int = 3
    #: Slots without an observation before a flow's state is discarded, so the
    #: scorer's memory is bounded by the number of *active* flows.
    idle_eviction_slots: int = 60
    #: Reference scales for the channels. Exposed because the right values
    #: depend on the link: `volume_scale_mbps` should sit near the rate at which
    #: a transfer stops looking interactive on this access link.
    volume_scale_mbps: float = 20.0
    small_packet_bytes: float = 400.0
    deadline_scale_s: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        if self.release_threshold > self.protect_threshold:
            raise ValueError("release threshold must not exceed the protect threshold")
        if self.max_log_odds <= 0:
            raise ValueError("max_log_odds must be positive")


@dataclass
class FlowState:
    """The scorer's running state for one flow."""

    log_odds: float
    protected: bool = False
    protected_slots: int = 0
    slots_seen: int = 0
    last_seen_slot: int = -1
    #: Slot at which this flow first crossed the protect threshold, so the
    #: evaluation can report how long detection took.
    first_protected_slot: int | None = None
    flips: int = 0

    @property
    def criticality(self) -> float:
        return _sigmoid(self.log_odds)


def _sigmoid(x: float) -> float:
    # Branch to avoid overflow in exp for large negative x. The clip in the
    # update makes this unreachable at the default L_max, but the scorer is
    # usable with a wider clip and should not depend on that.
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# -- evidence channels -------------------------------------------------------
#
# Each returns a score in [-1, 1]. Zero means "this channel saw nothing this
# slot", which in log-odds is the correct representation of absent evidence.


def _attention(obs: FlowObservation, cfg: ScorerConfig) -> float:
    """Is the user in front of this flow right now.

    Asymmetric by design. Focus plus typing is the strongest single indication
    that a transfer is one the user is waiting on. Its absence is only weak
    counter-evidence, because the background clinical cases are precisely the
    ones nobody is watching, and a symmetric channel would bury them.
    """
    raw = 0.65 * float(obs.foreground) + 0.35 * math.tanh(obs.user_events / 2.0)
    return -0.25 + 1.25 * raw


def _interactivity(obs: FlowObservation, cfg: ScorerConfig) -> float:
    """Small bidirectional exchanges with gaps, versus a saturating stream."""
    packets = obs.packets_down + obs.packets_up
    if packets == 0:
        return 0.0
    mean_payload_bytes = (obs.mbit_down + obs.mbit_up) * 125_000.0 / packets
    # 1 when packets are far below the small-packet reference, 0 at and above it.
    small = 1.0 - math.tanh(mean_payload_bytes / cfg.small_packet_bytes)
    hi = max(obs.packets_down, obs.packets_up)
    bidirectional = min(obs.packets_down, obs.packets_up) / hi if hi else 0.0
    score = 0.5 * small + 0.3 * bidirectional + 0.2 * _clamp(obs.idle_fraction, 0.0, 1.0)
    return 2.0 * score - 1.0


def _elasticity(obs: FlowObservation, cfg: ScorerConfig) -> float:
    """Did offered load fall when we last throttled this flow.

    The closed-loop channel. Throttling is an intervention, and an intervention
    is an experiment: an elastic transfer's offered load collapses to whatever
    it was given, an inelastic one keeps asking. Reading this off the response
    to our own action is far more reliable than inferring elasticity from the
    traffic shape, which is why this channel outweighs `interactivity`.

    Returns 0.0 when no throttle has been applied, because in that case there
    is no experiment and therefore no evidence.
    """
    if not obs.throttled_last or obs.granted_last_mbps <= 0.0:
        return 0.0
    # Demand persisting at or above what it was granted means the flow did not
    # back off. Ratios above 1 are clipped: a flow asking for twice what it got
    # is no more inelastic than one asking for exactly what it got.
    persistence = _clamp(obs.demand_mbps / obs.granted_last_mbps, 0.0, 1.0)
    return 2.0 * persistence - 1.0


def _deadline(obs: FlowObservation, cfg: ScorerConfig) -> float:
    """Slack between the deadline and the ETA at the rate currently granted.

    The channel that makes this layer dynamic rather than a classifier. Nothing
    about a backup changes between 02:00 and 07:55; the slack does, and so does
    the right decision.
    """
    if not math.isfinite(obs.deadline_remaining_s) or obs.remaining_mbit <= 0.0:
        return 0.0
    rate = max(obs.granted_last_mbps, obs.demand_mbps, 1e-6)
    eta_s = obs.remaining_mbit / rate
    slack_s = obs.deadline_remaining_s - eta_s
    # Negative slack means the flow misses its deadline at the current rate.
    return -math.tanh(slack_s / cfg.deadline_scale_s)


def _irreversibility(obs: FlowObservation, spec: FlowSpec, cfg: ScorerConfig) -> float:
    """Work already spent that halting would discard.

    A resumable transfer loses only the time; a non-resumable one at 95% loses
    everything. To a rate-based shaper the two are indistinguishable, which is
    how a long instrument upload gets killed at the last minute and restarted
    into the same congestion that killed it.
    """
    if spec.resumable:
        # Still mildly protective near completion: finishing frees the capacity.
        return -0.2 + 0.4 * _clamp(obs.progress, 0.0, 1.0)
    return -0.2 + 1.2 * _clamp(obs.progress, 0.0, 1.0)


def _volume(obs: FlowObservation, cfg: ScorerConfig) -> float:
    """Sustained high-rate single-direction transfer.

    The one channel that points the same way as the policy this layer replaces,
    and the one carrying the least weight. It is included because bulk transfer
    really is weak evidence of low criticality, and excluded from carrying a
    decision because acting on it alone is the failure being measured.
    """
    if obs.demand_mbps <= 0.0:
        return 0.0
    rate_score = math.tanh(obs.demand_mbps / cfg.volume_scale_mbps)
    total = obs.mbit_down + obs.mbit_up
    asymmetry = abs(obs.mbit_down - obs.mbit_up) / total if total > 0 else 0.0
    return -rate_score * (0.5 + 0.5 * asymmetry)


def _recurrence(spec: FlowSpec, cfg: ScorerConfig) -> float:
    """How routine this flow is. Static, but cheap and genuinely informative."""
    return -(2.0 * _clamp(spec.recurrence, 0.0, 1.0) - 1.0)


def channel_scores(spec: FlowSpec, obs: FlowObservation,
                   cfg: ScorerConfig | None = None) -> dict[str, float]:
    """Every channel's score for one flow in one slot, for audit and figures.

    Exposed because a criticality score that cannot be explained is not usable
    by an operator. Every protection decision this layer makes can be traced to
    the channels that drove it.
    """
    cfg = cfg or ScorerConfig()
    return {
        "attention": _attention(obs, cfg),
        "interactivity": _interactivity(obs, cfg),
        "elasticity": _elasticity(obs, cfg),
        "deadline": _deadline(obs, cfg),
        "irreversibility": _irreversibility(obs, spec, cfg),
        "volume": _volume(obs, cfg),
        "recurrence": _recurrence(spec, cfg),
    }


# -- the scorer --------------------------------------------------------------


class CriticalityScorer:
    """Tracks the criticality of every active flow, one slot at a time.

    Causal by construction: `update` for slot t is a function of the state after
    slot t-1 and the observations from slot t, and nothing else. There is no
    path by which a later slot can change an earlier score, which is the
    property `tests/test_flows.py` pins by truncating a trace.
    """

    def __init__(self, config: ScorerConfig | None = None) -> None:
        self.config = config or ScorerConfig()
        self.states: dict[str, FlowState] = {}
        self._slot = -1

    # -- state ---------------------------------------------------------------

    def state_for(self, spec: FlowSpec) -> FlowState:
        """The state for a flow, created at its class prior if new.

        Seeding at the prior rather than at zero is what lets a declared
        life-safety flow be protected on its first slot. A flow that had to earn
        its protection from neutral would spend its first seconds sheddable,
        and the first seconds are when a congestion episode is already underway.
        """
        state = self.states.get(spec.flow_id)
        if state is None:
            prior = self.config.priors.get(spec.declared, 0.0)
            state = FlowState(log_odds=_clamp(prior, -self.config.max_log_odds,
                                              self.config.max_log_odds))
            state.protected = _sigmoid(state.log_odds) >= self.config.protect_threshold
            if state.protected:
                state.first_protected_slot = max(self._slot, 0)
            self.states[spec.flow_id] = state
        return state

    def forget(self, flow_id: str) -> None:
        self.states.pop(flow_id, None)

    def _evict_idle(self, slot: int) -> None:
        cutoff = slot - self.config.idle_eviction_slots
        stale = [k for k, v in self.states.items() if v.last_seen_slot < cutoff]
        for key in stale:
            del self.states[key]

    # -- the update ----------------------------------------------------------

    def update(self, slot: int,
               observations: dict[str, tuple[FlowSpec, FlowObservation]]
               ) -> dict[str, float]:
        """Advance every observed flow by one slot, returning criticalities.

        `observations` maps flow id to its spec and this slot's observation.
        Flows absent from the mapping are left untouched rather than decayed:
        an unobserved flow is not evidence of anything, and a flow that goes
        quiet for a few slots should not lose its protection for it. Flows
        unobserved for `idle_eviction_slots` are dropped entirely.
        """
        cfg = self.config
        self._slot = slot
        out: dict[str, float] = {}

        for flow_id, (spec, obs) in observations.items():
            if spec.flow_id != flow_id:
                raise ValueError(f"key {flow_id!r} does not match spec {spec.flow_id!r}")
            state = self.state_for(spec)

            evidence = cfg.priors.get(spec.declared, 0.0)
            scores = channel_scores(spec, obs, cfg)
            for name, score in scores.items():
                evidence += cfg.weights.get(name, 0.0) * score

            state.log_odds = _clamp(cfg.rho * state.log_odds + (1.0 - cfg.rho) * evidence,
                                    -cfg.max_log_odds, cfg.max_log_odds)
            state.slots_seen += 1
            state.last_seen_slot = slot

            criticality = _sigmoid(state.log_odds)
            self._apply_hysteresis(state, criticality, slot)
            out[flow_id] = criticality

        self._evict_idle(slot)
        return out

    def _apply_hysteresis(self, state: FlowState, criticality: float, slot: int) -> None:
        cfg = self.config
        if state.protected:
            state.protected_slots += 1
            demotable = state.protected_slots > cfg.min_protected_slots
            if criticality < cfg.release_threshold and demotable:
                state.protected = False
                state.protected_slots = 0
                state.flips += 1
        elif criticality >= cfg.protect_threshold:
            state.protected = True
            state.protected_slots = 1
            state.flips += 1
            if state.first_protected_slot is None:
                state.first_protected_slot = slot

    # -- readouts ------------------------------------------------------------

    def criticality(self, flow_id: str) -> float:
        state = self.states.get(flow_id)
        return state.criticality if state else 0.5

    def is_protected(self, flow_id: str) -> bool:
        state = self.states.get(flow_id)
        return bool(state and state.protected)

    def protected_ids(self) -> frozenset[str]:
        return frozenset(k for k, v in self.states.items() if v.protected)

    def flap_rate(self) -> float:
        """Protection changes per flow-slot observed.

        The number to watch when tuning the hysteresis band. Every flip is a
        rate change the transport underneath has to absorb, so a scorer that
        classifies well and flaps constantly has not helped.
        """
        slots = sum(s.slots_seen for s in self.states.values())
        flips = sum(max(s.flips - 1, 0) for s in self.states.values())
        return flips / slots if slots else 0.0
