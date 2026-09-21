"""A labelled flow workload for evaluating the protection layer.

**Nothing generated here is a result, and the module lives under `eval/` rather
than `ingest/` so that boundary is structural rather than a convention.** The
capacity it is driven with is measured; the flows are a model.

## Why a model at all

The protection layer needs per-flow evidence and per-flow ground truth, and
neither the StarNet traces nor WetLinks carry either: both are link-level
throughput series with no flow table, no process attribution and no user
attention signal. No public LEO dataset has them, because collecting them means
instrumenting the host rather than the link.

The standard way out, and the one every adaptive-streaming paper takes, is to
drive a simulated workload with a real capacity trace. That is what happens
here: `scripts/run_protection.py` takes the calibrated bound and the realised
throughput from a trained pipeline over the real traces, and this module
supplies the flows that contend for them. **Every claim about the protection
layer is therefore a claim about this workload**, and the limitation is stated
wherever one of those claims is.

## What the model has to get right to be worth running

A generator whose critical flows are the small ones and whose bulk flows are
the large ones measures nothing, because the trivial policy already wins. The
archetypes are chosen so the two policies this layer competes with each fail on
a case they cannot see:

- **`pacs_image_push` and `research_upload`** are critical, high rate, sustained
  and single direction. They are what a rate-based shaper throttles first, and
  they are indistinguishable from a system update by rate alone. Neither
  declares itself, because the applications that generate them mostly do not.
- **`greedy_downloader`** declares itself clinical and is a download. Any policy
  that trusts declarations hands it the link.
- **`video_stream`** is interactive, in the foreground, inelastic and not
  critical. It is there so the evidence channels cannot win by equating
  "the user is watching" with "must not be shed".

Two archetypes are deliberately easy, because a real link is mostly easy and a
workload made only of hard cases overstates how often the question is difficult.

## The shortfall model, and why it is the interesting part

Rates are allocated against the *bound*, and the link delivers the *realised*
throughput. When the bound overestimates, the shortfall is shared in proportion
to the allocated rates, which is what an unmanaged bottleneck does. A protected
floor is therefore honoured by the allocator on every slot it is feasible, and
*delivered* only on the slots where the bound was safe.

That is the whole coupling between this layer and the rest of the repository.
The allocator's guarantee is algorithmic and unconditional; the guarantee the
user experiences is only as good as the bound's risk control, which is the
quantity `calibrate/` exists to hold at a budget. The evaluation reports both
and the gap between them.

## What the model deliberately does not claim

The elasticity of each archetype, the packet-size profiles, the arrival rates
and the mix are ours. So is `truly_critical`, which is a stipulation and not a
measurement: it is defined here as "throttling this flow below its floor causes
harm the operator would not accept". A scorer's measured accuracy against this
workload is accuracy against our model of the traffic. It bounds nothing about
real traffic, and the README says so beside every number that comes from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from thalweg.decide.flows import FlowClass, FlowObservation, FlowSpec

__all__ = [
    "ARCHETYPES",
    "ActiveFlow",
    "Archetype",
    "FlowWorkload",
    "WorkloadSpec",
    "archetype_table",
]


@dataclass(frozen=True)
class Archetype:
    """One kind of flow, and how it behaves when it is throttled."""

    name: str
    declared: FlowClass
    #: The stipulated label. Throttling this below its floor causes harm the
    #: operator would not accept.
    truly_critical: bool
    nominal_mbps: float
    floor_fraction: float          # floor as a fraction of nominal demand
    duration_s: tuple[float, float]
    #: Elastic flows reduce offered load to whatever they were granted, the way
    #: a congestion-controlled bulk transfer does. Inelastic ones keep asking,
    #: which is what a real-time stream does until it fails.
    elastic: bool
    mean_packet_bytes: float
    #: Uplink bytes as a fraction of downlink, which is what separates a
    #: conversation from a download in the packet counters.
    uplink_ratio: float
    idle_fraction: float
    foreground_probability: float
    user_events_per_slot: float
    resumable: bool
    recurrence: float
    #: Seconds of deadline budget as a multiple of the unthrottled duration.
    #: inf for a flow with no deadline. A value near 1 is tight.
    deadline_slack_multiple: float
    #: Expected flows of this kind alive at once.
    concurrency: float

    @property
    def floor_mbps(self) -> float:
        return self.nominal_mbps * self.floor_fraction


#: The mix. Three critical archetypes and five not, which is roughly the shape
#: of a clinical site's access link: the traffic that must not be shed is a
#: minority of the bytes and almost all of the consequence.
ARCHETYPES: tuple[Archetype, ...] = (
    # -- critical -----------------------------------------------------------
    Archetype(
        # Patient telemetry. Tiny, constant, inelastic, nobody watching it, and
        # correctly declared. The easy case, and the one a rate-based shaper
        # also gets right, because it is too small to be worth throttling.
        name="patient_monitor", declared=FlowClass.LIFE_SAFETY, truly_critical=True,
        nominal_mbps=0.5, floor_fraction=0.8, duration_s=(600.0, 3600.0),
        elastic=False, mean_packet_bytes=180.0, uplink_ratio=0.9, idle_fraction=0.15,
        foreground_probability=0.0, user_events_per_slot=0.0, resumable=False,
        recurrence=0.6, deadline_slack_multiple=math.inf, concurrency=2.0,
    ),
    Archetype(
        # A radiology study pushed to a regional archive against a reporting
        # deadline. High rate, sustained, single direction, non-resumable, and
        # it does not declare itself, because the modality vendor's uploader
        # does not set DSCP. This is the flow the whole layer is about.
        name="pacs_image_push", declared=FlowClass.STANDARD, truly_critical=True,
        nominal_mbps=28.0, floor_fraction=0.55, duration_s=(90.0, 300.0),
        elastic=False, mean_packet_bytes=1400.0, uplink_ratio=0.03, idle_fraction=0.0,
        foreground_probability=0.1, user_events_per_slot=0.05, resumable=False,
        recurrence=0.15, deadline_slack_multiple=1.6, concurrency=0.8,
    ),
    Archetype(
        # A teleconsultation. Inelastic, interactive, in the foreground, and
        # correctly declared. Below its floor it is a dropped call rather than
        # a degraded one, which is what the floor is for.
        name="teleconsult_video", declared=FlowClass.CLINICAL, truly_critical=True,
        nominal_mbps=3.0, floor_fraction=0.55, duration_s=(180.0, 1200.0),
        elastic=False, mean_packet_bytes=900.0, uplink_ratio=0.75, idle_fraction=0.02,
        foreground_probability=0.95, user_events_per_slot=1.5, resumable=False,
        recurrence=0.25, deadline_slack_multiple=math.inf, concurrency=1.2,
    ),
    Archetype(
        # The personally critical case from the brief: an instrument capture or
        # a thesis dataset the user is waiting on, with a submission deadline
        # and no resume. Undeclared and large, so it looks exactly like a
        # download to everything except the deadline and the user.
        name="research_upload", declared=FlowClass.STANDARD, truly_critical=True,
        nominal_mbps=16.0, floor_fraction=0.45, duration_s=(240.0, 900.0),
        elastic=False, mean_packet_bytes=1300.0, uplink_ratio=12.0, idle_fraction=0.0,
        foreground_probability=0.55, user_events_per_slot=0.6, resumable=False,
        recurrence=0.05, deadline_slack_multiple=1.35, concurrency=0.6,
    ),
    # -- not critical -------------------------------------------------------
    Archetype(
        # The hard negative. Interactive, foreground, inelastic, and nobody is
        # harmed by it buffering. Without this the attention and interactivity
        # channels would be free wins.
        name="video_stream", declared=FlowClass.STANDARD, truly_critical=False,
        nominal_mbps=9.0, floor_fraction=0.3, duration_s=(300.0, 2400.0),
        elastic=False, mean_packet_bytes=1100.0, uplink_ratio=0.02, idle_fraction=0.05,
        foreground_probability=0.85, user_events_per_slot=0.4, resumable=True,
        recurrence=0.55, deadline_slack_multiple=math.inf, concurrency=1.5,
    ),
    Archetype(
        # The gaming case. Claims to be clinical, is a download manager.
        name="greedy_downloader", declared=FlowClass.CLINICAL, truly_critical=False,
        nominal_mbps=45.0, floor_fraction=0.0, duration_s=(120.0, 600.0),
        elastic=True, mean_packet_bytes=1460.0, uplink_ratio=0.01, idle_fraction=0.0,
        foreground_probability=0.2, user_events_per_slot=0.05, resumable=True,
        recurrence=0.35, deadline_slack_multiple=math.inf, concurrency=0.7,
    ),
    Archetype(
        name="os_update", declared=FlowClass.STANDARD, truly_critical=False,
        nominal_mbps=32.0, floor_fraction=0.0, duration_s=(180.0, 1200.0),
        elastic=True, mean_packet_bytes=1460.0, uplink_ratio=0.01, idle_fraction=0.02,
        foreground_probability=0.05, user_events_per_slot=0.0, resumable=True,
        recurrence=0.8, deadline_slack_multiple=math.inf, concurrency=0.9,
    ),
    Archetype(
        # Correctly declared background, huge deadline slack, fully resumable.
        # The easy negative, and the flow the layer should shed first.
        name="cloud_backup", declared=FlowClass.BACKGROUND, truly_critical=False,
        nominal_mbps=22.0, floor_fraction=0.0, duration_s=(600.0, 3600.0),
        elastic=True, mean_packet_bytes=1460.0, uplink_ratio=40.0, idle_fraction=0.03,
        foreground_probability=0.0, user_events_per_slot=0.0, resumable=True,
        recurrence=0.95, deadline_slack_multiple=12.0, concurrency=1.0,
    ),
)


def archetype_table() -> list[dict]:
    """The mix as a table, for the workload figure and the committed summary."""
    return [
        {
            "archetype": a.name,
            "declared": a.declared.value,
            "truly_critical": a.truly_critical,
            "nominal_mbps": a.nominal_mbps,
            "floor_mbps": round(a.floor_mbps, 2),
            "elastic": a.elastic,
            "resumable": a.resumable,
            "deadline": "none" if math.isinf(a.deadline_slack_multiple)
                        else f"{a.deadline_slack_multiple:g}x",
            "concurrency": a.concurrency,
        }
        for a in ARCHETYPES
    ]


@dataclass(frozen=True)
class WorkloadSpec:
    """How much of the workload to generate, and how hard to make it."""

    seed: int = 1337
    seconds_per_slot: float = 5.0
    #: Scales every archetype's concurrency together. Above 1 the link is
    #: oversubscribed more often, which is the regime the layer is for.
    load_multiplier: float = 1.0
    #: Fraction of flows whose declaration is dropped to STANDARD, on top of
    #: the archetypes that are undeclared by construction. Models the host
    #: where nothing sets DSCP.
    declaration_loss: float = 0.0
    #: Consecutive slots below a useful rate before a flow gives up. Without
    #: it the generator has no steady state above capacity: arrivals continue
    #: while starved transfers take proportionally longer, so the active set
    #: grows for the whole run and the load level stops meaning what it says.
    #: Real transfers do not wait forever either. 60 slots is five minutes at
    #: the default horizon, and abandonment is reported rather than hidden,
    #: because a critical transfer that gave up is the worst outcome the layer
    #: has and counting it as "not violating its floor any more" would be the
    #: wrong bookkeeping.
    abandon_after_slots: int = 60
    #: The rate below which a slot counts toward giving up, for a flow whose
    #: own floor is zero. An elastic transfer squeezed to nothing for five
    #: minutes gives up too.
    abandon_floor_mbps: float = 0.5


@dataclass
class ActiveFlow:
    """One flow in flight, and everything the simulation needs to advance it."""

    spec: FlowSpec
    archetype: Archetype
    started_slot: int
    total_mbit: float
    remaining_mbit: float
    #: Megabits still outstanding when this flow entered the simulation. Warm
    #: started flows enter part way through, so the work they actually did here
    #: is less than their total and the completion reference has to use this.
    work_at_entry_mbit: float
    deadline_remaining_s: float
    foreground: bool
    granted_last_mbps: float = 0.0
    delivered_last_mbps: float = 0.0
    throttled_last: bool = False
    demand_mbps: float = 0.0
    demand_peak_mbps: float = 0.0
    slots_below_floor: int = 0
    slots_alive: int = 0
    delivered_mbit: float = 0.0
    starved_run: int = 0
    abandoned: bool = False

    @property
    def progress(self) -> float:
        return 0.0 if self.total_mbit <= 0 else min(
            self.delivered_mbit / self.total_mbit, 1.0)


class FlowWorkload:
    """Generates flows, observes them, and advances them under an allocation.

    The loop is three calls per slot and the order matters:

        observations = workload.observe(t)          # what the monitor saw
        allocation   = controller.step(bound, obs)  # the decision
        workload.apply(t, allocation.rates, realised_mbps)

    `apply` is where the link, rather than the allocator, has the last word:
    the realised throughput is shared out in proportion to the allocated rates,
    so an allocation written against a bound the link does not meet is not
    delivered. Protection is measured on both sides of that.
    """

    def __init__(self, spec: WorkloadSpec | None = None) -> None:
        self.spec = spec or WorkloadSpec()
        # Two independent streams. `rng_structure` draws arrivals, durations,
        # sizes and declarations; `rng_behaviour` draws per-slot foreground
        # switches and user events. Splitting them is what makes two policies
        # comparable: the arrival process and every flow's spec are then
        # identical across runs at the same seed, so the only thing that
        # differs between policies is what they did about the same flows.
        self.rng = np.random.default_rng(self.spec.seed)
        self.rng_behaviour = np.random.default_rng(self.spec.seed + 1_000_003)
        self.active: dict[str, ActiveFlow] = {}
        self.completed: list[ActiveFlow] = []
        self._next_id = 0
        self._slot = -1

    # -- arrivals ------------------------------------------------------------

    def _spawn(self, archetype: Archetype, slot: int) -> ActiveFlow:
        self._next_id += 1
        flow_id = f"{archetype.name}#{self._next_id}"

        duration_s = float(self.rng.uniform(*archetype.duration_s))
        total_mbit = archetype.nominal_mbps * duration_s
        deadline_s = (math.inf if math.isinf(archetype.deadline_slack_multiple)
                      else duration_s * archetype.deadline_slack_multiple)

        declared = archetype.declared
        if (declared is not FlowClass.STANDARD
                and self.rng.random() < self.spec.declaration_loss):
            declared = FlowClass.STANDARD

        spec = FlowSpec(
            flow_id=flow_id, declared=declared,
            floor_mbps=archetype.floor_mbps, demand_mbps=archetype.nominal_mbps,
            bytes_total_mbit=total_mbit, deadline_s=deadline_s,
            resumable=archetype.resumable, recurrence=archetype.recurrence,
            truly_critical=archetype.truly_critical,
        )
        return ActiveFlow(
            spec=spec, archetype=archetype, started_slot=slot, total_mbit=total_mbit,
            remaining_mbit=total_mbit, work_at_entry_mbit=total_mbit,
            deadline_remaining_s=deadline_s,
            foreground=bool(self.rng.random() < archetype.foreground_probability),
            demand_mbps=archetype.nominal_mbps,
            demand_peak_mbps=archetype.nominal_mbps,
        )

    def _arrivals(self, slot: int) -> None:
        """Poisson arrivals per archetype, tuned so the expected number alive
        matches `concurrency`: rate = concurrency / mean duration."""
        for archetype in ARCHETYPES:
            mean_duration = float(np.mean(archetype.duration_s))
            target = archetype.concurrency * self.spec.load_multiplier
            per_slot = target / mean_duration * self.spec.seconds_per_slot
            for _ in range(int(self.rng.poisson(per_slot))):
                flow = self._spawn(archetype, slot)
                self.active[flow.spec.flow_id] = flow

    def warm_start(self, slot: int = 0) -> None:
        """Seed the link with flows already part way through.

        Running the arrival process from an empty link does not work here: the
        long-lived archetypes have lifetimes in the thousands of seconds, so
        reaching steady state that way costs more slots than most traces have,
        and the first thousand decisions would measure a filling transient
        rather than the layer. This places `concurrency` flows per archetype
        directly, each at a uniformly random point in its own lifetime.
        """
        for archetype in ARCHETYPES:
            target = archetype.concurrency * self.spec.load_multiplier
            # Probabilistic rounding, so a concurrency of 0.6 means 0.6 flows on
            # average rather than 0 or 1 every time.
            count = int(target) + int(self.rng.random() < (target - int(target)))
            for _ in range(count):
                flow = self._spawn(archetype, slot)
                consumed = float(self.rng.uniform(0.0, 0.9)) * flow.total_mbit
                flow.delivered_mbit = consumed
                flow.remaining_mbit = flow.total_mbit - consumed
                flow.work_at_entry_mbit = flow.remaining_mbit
                if math.isfinite(flow.deadline_remaining_s):
                    flow.deadline_remaining_s *= 1.0 - consumed / flow.total_mbit
                # Seed the counters so the first observation is not an empty one.
                flow.delivered_last_mbps = archetype.nominal_mbps
                flow.granted_last_mbps = archetype.nominal_mbps
                self.active[flow.spec.flow_id] = flow

    # -- observation ---------------------------------------------------------

    def observe(self, slot: int) -> dict[str, tuple[FlowSpec, FlowObservation]]:
        """What a host-side monitor saw of each active flow during the last slot.

        Derived from what was *delivered*, not from what was allocated, because
        a monitor counts packets and the packets are what the link carried.
        """
        if slot != self._slot:
            self._arrivals(slot)
            self._slot = slot

        dt = self.spec.seconds_per_slot
        out: dict[str, tuple[FlowSpec, FlowObservation]] = {}
        for flow_id, flow in self.active.items():
            arch = flow.archetype
            delivered_mbit = flow.delivered_last_mbps * dt
            up_share = arch.uplink_ratio / (1.0 + arch.uplink_ratio)
            mbit_up = delivered_mbit * up_share
            mbit_down = delivered_mbit - mbit_up
            packets = delivered_mbit * 125_000.0 / arch.mean_packet_bytes

            events = (int(self.rng_behaviour.poisson(arch.user_events_per_slot))
                      if flow.foreground else 0)
            out[flow_id] = (
                flow.spec,
                FlowObservation(
                    mbit_down=mbit_down, mbit_up=mbit_up,
                    packets_down=int(packets * (1.0 - up_share)),
                    packets_up=int(packets * up_share),
                    idle_fraction=arch.idle_fraction,
                    foreground=flow.foreground, user_events=events,
                    demand_mbps=flow.demand_mbps,
                    demand_peak_mbps=flow.demand_peak_mbps,
                    granted_last_mbps=flow.granted_last_mbps,
                    throttled_last=flow.throttled_last,
                    remaining_mbit=flow.remaining_mbit,
                    deadline_remaining_s=flow.deadline_remaining_s,
                    progress=flow.progress,
                    age_s=(slot - flow.started_slot) * dt,
                ),
            )
        return out

    # -- advance -------------------------------------------------------------

    def apply(self, slot: int, rates: dict[str, float], realised_mbps: float
              ) -> dict[str, float]:
        """Advance every flow by one slot under what the link actually carried.

        The allocator divided a bound. If the bound overestimated, the link
        cannot carry the sum of the rates and the shortfall is shared in
        proportion to them, which is what an unmanaged bottleneck does. This is
        the only place the difference between a bound and a forecast becomes
        something a user experiences.
        """
        dt = self.spec.seconds_per_slot
        allocated = sum(rates.get(f, 0.0) for f in self.active)
        scale = 1.0 if allocated <= realised_mbps or allocated <= 0 else (
            realised_mbps / allocated)

        delivered: dict[str, float] = {}
        finished: list[str] = []
        useful = self.spec.abandon_floor_mbps
        for flow_id, flow in self.active.items():
            granted = max(rates.get(flow_id, 0.0), 0.0)
            got = granted * scale
            delivered[flow_id] = got

            flow.throttled_last = granted < flow.demand_mbps - 1e-9
            flow.granted_last_mbps = granted
            flow.delivered_last_mbps = got
            flow.delivered_mbit += got * dt
            flow.remaining_mbit = max(flow.remaining_mbit - got * dt, 0.0)
            flow.deadline_remaining_s -= dt
            flow.slots_alive += 1
            if got < flow.spec.floor_mbps - 1e-9:
                flow.slots_below_floor += 1

            # Offered load for the next slot. An elastic flow's congestion
            # control collapses its demand toward what it received; an inelastic
            # one keeps asking for its nominal rate until it fails. This is what
            # the elasticity channel is measuring, and it is generated here
            # rather than read off the label.
            if flow.archetype.elastic:
                flow.demand_mbps = min(flow.archetype.nominal_mbps,
                                       0.4 * flow.demand_mbps + 0.6 * max(got, 0.05))
            else:
                flow.demand_mbps = flow.archetype.nominal_mbps
            flow.demand_peak_mbps = max(flow.demand_peak_mbps, flow.demand_mbps)

            # Foreground is sticky: a user does not switch window every slot.
            if self.rng_behaviour.random() < 0.05:
                flow.foreground = bool(
                    self.rng_behaviour.random() < flow.archetype.foreground_probability)

            # Giving up. A flow held below a useful rate for long enough stops
            # waiting, which is both what real transfers do and what keeps the
            # active set bounded when the link is oversubscribed.
            if got < max(flow.spec.floor_mbps, useful) - 1e-9:
                flow.starved_run += 1
            else:
                flow.starved_run = 0

            if flow.remaining_mbit <= 1e-9:
                finished.append(flow_id)
            elif flow.starved_run >= self.spec.abandon_after_slots:
                flow.abandoned = True
                finished.append(flow_id)

        for flow_id in finished:
            self.completed.append(self.active.pop(flow_id))
        return delivered

    # -- readouts ------------------------------------------------------------

    def snapshot(self) -> list[ActiveFlow]:
        return list(self.active.values())

    def all_flows(self) -> list[ActiveFlow]:
        return self.completed + list(self.active.values())

    def offered_mbps(self) -> float:
        return sum(f.demand_mbps for f in self.active.values())


def oracle_view(observations: dict[str, tuple[FlowSpec, FlowObservation]]
                ) -> dict[str, float]:
    """Ground-truth criticality, for the oracle policy.

    Kept here rather than in `decide/` so that no module the system actually
    runs has a path to the label. The oracle is `allocate_protected` fed this,
    which is what keeps scorer error separable from allocator error.
    """
    return {flow_id: (1.0 if spec.truly_critical else 0.0)
            for flow_id, (spec, _) in observations.items()}
