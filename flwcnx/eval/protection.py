"""Running and scoring the protection layer over a measured capacity series.

The harness for `decide/flows.py` and `decide/protect.py`, in the same shape as
`eval/metrics.py` is for the calibration layer: the script is thin and every
quantity it reports is defined here, once.

## The two sides of the guarantee, and why both are reported

`allocate_protected` honours a protected floor on every slot the floors are
jointly feasible. That is an algorithmic property and it holds unconditionally.
It is also not what a user experiences, because the allocator divides a *bound*
and the link delivers whatever it delivers. Every metric with a `_allocated`
suffix scores the allocator; every `_delivered` one scores what arrived after
the link had its say.

The gap between the two is the price of the bound being wrong, which makes it a
direct read on the calibration layer rather than on this one. Running the same
workload against the point forecast and against the calibrated bound separates
the two cleanly: the point forecast lets the allocator write larger floors and
the link fails more of them.

## Scoring the scorer separately from the allocator

The oracle policy is `allocate_protected` fed the ground-truth labels as
criticality. Its violation rate is therefore the floor imposed by physics and
by the allocation rule, with classification error removed. Any gap between the
protection policy and the oracle is scorer error and nothing else, and any
violation the oracle also commits is a slot on which the link could not carry
what was declared critical no matter who decided.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flwcnx.decide.flows import CriticalityScorer, ScorerConfig, channel_scores
from flwcnx.decide.protect import (
    POLICIES,
    AllocationResult,
    FlowDemand,
    ProtectionConfig,
    allocate_protected,
)
from flwcnx.eval.workload import FlowWorkload, WorkloadSpec, oracle_view

__all__ = [
    "ORACLE",
    "PolicyRun",
    "average_precision",
    "jain_index",
    "run_policy",
    "sweep_load",
]

#: The oracle is not in `POLICIES` because it is not a policy anything could
#: deploy. It is the protection allocator with classification error removed.
ORACLE = "oracle"


def jain_index(values: Sequence[float]) -> float:
    """Jain's fairness index. 1.0 is a perfectly equal split, 1/n the worst.

    Reported over the non-critical flows only. Protecting critical traffic is
    supposed to make the overall split unequal, so a fairness index over
    everything would measure the layer working and call it a fault. Within the
    class the layer claims to treat alike, unfairness is a real defect.
    """
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if array.size == 0 or not np.any(array > 0):
        return float("nan")
    return float(array.sum() ** 2 / (array.size * np.sum(array ** 2)))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, by the step-wise definition.

    Written out rather than imported so the definition is fixed here: AP is
    `sum_k (R_k - R_{k-1}) * P_k` over thresholds in descending score order,
    which is the interpolation-free version and the one that does not flatter
    a classifier on an imbalanced problem. The workload is imbalanced by
    construction, so this matters.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    hits = labels[order].astype(float)
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, hits.size + 1)
    # Only the ranks where a positive is retrieved move recall, so only those
    # contribute a rectangle.
    return float(np.sum(precision * hits) / positives)


@dataclass
class PolicyRun:
    """One policy over one capacity series, with everything needed to score it."""

    policy: str
    records: pd.DataFrame           # one row per flow-slot
    slots: pd.DataFrame             # one row per decision slot
    flows: pd.DataFrame             # one row per flow that ever existed
    scorer_flaps: float = float("nan")
    config: dict = field(default_factory=dict)

    def summary(self) -> dict:
        rows, slots, flows = self.records, self.slots, self.flows
        critical = rows[rows["truly_critical"]]
        ordinary = rows[~rows["truly_critical"]]

        def rate(frame: pd.DataFrame, column: str) -> float:
            return float(frame[column].mean()) if len(frame) else float("nan")

        out = {
            "policy": self.policy,
            "n_slots": int(len(slots)),
            "n_flow_slots": int(len(rows)),
            "n_flows": int(len(flows)),
            "mean_offered_mbps": float(slots["offered_mbps"].mean()),
            "mean_capacity_mbps": float(slots["capacity_mbps"].mean()),
            "mean_realised_mbps": float(slots["realised_mbps"].mean()),
            "oversubscription": float(slots["offered_mbps"].mean()
                                      / max(slots["realised_mbps"].mean(), 1e-9)),

            # -- the headline. A critical flow held below the rate at which it
            #    is useful, scored on both sides of the link.
            "critical_violation_allocated": rate(critical, "below_floor_allocated"),
            "critical_violation_delivered": rate(critical, "below_floor_delivered"),
            "ordinary_violation_delivered": rate(ordinary, "below_floor_delivered"),

            # -- the cost side. A layer that protects by wasting capacity has
            #    not protected anything.
            "critical_goodput_ratio": _goodput(critical),
            "ordinary_goodput_ratio": _goodput(ordinary),
            "utilisation": float(slots["delivered_mbps"].sum()
                                 / max(slots["realised_mbps"].sum(), 1e-9)),
            "wasted_fraction": float(slots["wasted_mbps"].sum()
                                     / max(slots["delivered_mbps"].sum(), 1e-9)),
            # Jain over delivered-over-demanded, not over raw rates. The
            # non-critical archetypes ask for anything from 9 to 45 Mbps, so an
            # index over rates would score a correct proportional split as
            # unfair. The share of its own ask is what a flow experiences.
            #
            # Columns are selected before the apply rather than dropped inside
            # it with `include_groups`, whose meaning has changed twice across
            # pandas 2 and 3. Selecting is equivalent and version independent.
            "ordinary_jain": jain_index(
                ordinary.groupby("flow_id")[GOODPUT_COLUMNS]
                .apply(_goodput).tolist()),

            # -- the alarm, which is the case where no allocation would do.
            "breach_slot_rate": float(slots["breached"].mean()),

            # -- the worst outcome the layer has. A transfer held below a
            #    useful rate long enough stops waiting, and a critical one that
            #    gave up is not the same as one that was merely slow.
            "critical_abandon_rate": _abandon_rate(flows, critical=True),
            "ordinary_abandon_rate": _abandon_rate(flows, critical=False),
        }

        if "criticality" in rows and rows["criticality"].notna().any():
            scored = rows.dropna(subset=["criticality"])
            out |= {
                "scorer_ap": average_precision(scored["criticality"].to_numpy(),
                                               scored["truly_critical"].to_numpy()),
                "scorer_precision": _precision(scored),
                "scorer_recall": _recall(scored),
                "scorer_flap_rate": self.scorer_flaps,
                "median_detect_slots": _median_detect(flows),
            }

        done = flows[flows["completed"]]
        out["completion_inflation_ordinary"] = (
            float(done.loc[~done["truly_critical"], "completion_inflation"].median())
            if (~done["truly_critical"]).any() else float("nan"))
        out["completion_inflation_critical"] = (
            float(done.loc[done["truly_critical"], "completion_inflation"].median())
            if done["truly_critical"].any() else float("nan"))
        return out

    def by_archetype(self) -> pd.DataFrame:
        grouped = self.records.groupby("archetype")
        return pd.DataFrame({
            "policy": self.policy,
            "truly_critical": grouped["truly_critical"].first(),
            "flow_slots": grouped.size(),
            "violation_allocated": grouped["below_floor_allocated"].mean(),
            "violation_delivered": grouped["below_floor_delivered"].mean(),
            "goodput_ratio": (self.records.groupby("archetype")[GOODPUT_COLUMNS]
                              .apply(_goodput)),
            "mean_criticality": grouped["criticality"].mean(),
        }).reset_index()


#: The two columns `_goodput` reads, selected before any groupby-apply.
GOODPUT_COLUMNS = ["delivered_mbps", "demand_mbps"]


def _goodput(frame: pd.DataFrame) -> float:
    demanded = float(frame["demand_mbps"].sum())
    return float(frame["delivered_mbps"].sum() / demanded) if demanded > 0 else float("nan")


def _precision(frame: pd.DataFrame) -> float:
    flagged = frame[frame["protected"]]
    return float(flagged["truly_critical"].mean()) if len(flagged) else float("nan")


def _recall(frame: pd.DataFrame) -> float:
    positives = frame[frame["truly_critical"]]
    return float(positives["protected"].mean()) if len(positives) else float("nan")


def _abandon_rate(flows: pd.DataFrame, critical: bool) -> float:
    subset = flows[flows["truly_critical"] == critical]
    return float(subset["abandoned"].mean()) if len(subset) else float("nan")


def _median_detect(flows: pd.DataFrame) -> float:
    """Slots from a critical flow's first observation to its first protection.

    NaN for a flow never protected, which keeps the median a median over
    detections and leaves the misses to recall. Reporting a miss as a large
    latency would hide it inside a number that otherwise looks fine.
    """
    detected = flows[flows["truly_critical"] & flows["detect_slots"].notna()]
    return float(detected["detect_slots"].median()) if len(detected) else float("nan")


def run_policy(policy: str, capacity_mbps: np.ndarray, realised_mbps: np.ndarray,
               workload_spec: WorkloadSpec | None = None,
               scorer_config: ScorerConfig | None = None,
               protection_config: ProtectionConfig | None = None,
               record_channels: bool = False) -> PolicyRun:
    """Replay one policy slot by slot and record everything it did.

    `capacity_mbps` is what the policy divides, which is the calibrated bound on
    the main path and the point forecast in the comparison arm.
    `realised_mbps` is what the link actually carried and is revealed only after
    the allocation for that slot has been written, the same ordering the rest of
    the pipeline keeps.

    `record_channels` adds one column per evidence channel to the flow-slot
    table. Off by default because it roughly doubles the table, and on for the
    reference run, because a protection decision an operator cannot trace back
    to the evidence that drove it is not one they can act on.
    """
    capacity_mbps = np.asarray(capacity_mbps, dtype=float).ravel()
    realised_mbps = np.asarray(realised_mbps, dtype=float).ravel()
    if capacity_mbps.shape != realised_mbps.shape:
        raise ValueError("capacity and realised series must be the same length")
    if policy != ORACLE and policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}, expected one of "
                         f"{sorted(POLICIES) + [ORACLE]}")

    workload = FlowWorkload(workload_spec or WorkloadSpec())
    workload.warm_start(0)
    scorer = CriticalityScorer(scorer_config)
    allocate: Callable[..., AllocationResult] = (
        allocate_protected if policy == ORACLE else POLICIES[policy])

    rows: list[dict] = []
    slot_rows: list[dict] = []
    first_protected: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    for t in range(capacity_mbps.size):
        observations = workload.observe(t)
        if not observations:
            continue

        if policy == ORACLE:
            criticality = oracle_view(observations)
            protected = {k: bool(v >= 0.5) for k, v in criticality.items()}
        else:
            criticality = scorer.update(t, observations)
            protected = {k: scorer.is_protected(k) for k in observations}

        flows: list[FlowDemand] = []
        for flow_id, (spec, obs) in observations.items():
            first_seen.setdefault(flow_id, t)
            if protected[flow_id] and flow_id not in first_protected:
                first_protected[flow_id] = t
            demand = max(obs.demand_mbps, 0.0) or spec.demand_mbps
            flows.append(FlowDemand(
                flow_id=flow_id, demand_mbps=demand,
                floor_mbps=min(spec.floor_mbps, demand),
                criticality=criticality[flow_id], protected=protected[flow_id],
                declared=spec.declared,
            ))

        result = allocate(capacity_mbps[t], flows, protection_config)
        delivered = workload.apply(t, result.rates, realised_mbps[t])

        for flow in flows:
            spec, obs = observations[flow.flow_id]
            got = delivered.get(flow.flow_id, 0.0)
            granted = result.rates.get(flow.flow_id, 0.0)
            row = {
                "slot": t, "flow_id": flow.flow_id,
                "archetype": flow.flow_id.split("#")[0],
                "truly_critical": bool(spec.truly_critical),
                "declared": spec.declared.value,
                "criticality": flow.criticality, "protected": flow.protected,
                "demand_mbps": flow.demand_mbps, "floor_mbps": flow.floor_mbps,
                "granted_mbps": granted, "delivered_mbps": got,
                "below_floor_allocated": bool(granted < flow.floor_mbps - 1e-9),
                "below_floor_delivered": bool(got < flow.floor_mbps - 1e-9),
                "progress": obs.progress, "foreground": obs.foreground,
                "deadline_remaining_s": obs.deadline_remaining_s,
            }
            if record_channels:
                scores = channel_scores(spec, obs, scorer.config)
                row |= {f"z_{name}": value for name, value in scores.items()}
                row |= {f"w_{name}": scorer.config.weights.get(name, 0.0) * value
                        for name, value in scores.items()}
            rows.append(row)

        slot_rows.append({
            "slot": t, "capacity_mbps": float(capacity_mbps[t]),
            "realised_mbps": float(realised_mbps[t]),
            "offered_mbps": float(sum(f.demand_mbps for f in flows)),
            "allocated_mbps": result.allocated_mbps,
            "delivered_mbps": float(sum(delivered.values())),
            "wasted_mbps": result.wasted_mbps(flows),
            "n_flows": len(flows), "breached": bool(result.breached),
            "n_protected": int(sum(f.protected for f in flows)),
        })

    records = pd.DataFrame(rows)
    return PolicyRun(
        policy=policy, records=records, slots=pd.DataFrame(slot_rows),
        flows=_flow_table(workload, first_seen, first_protected),
        scorer_flaps=scorer.flap_rate() if policy != ORACLE else float("nan"),
        config={
            "policy": policy,
            "workload": vars(workload.spec),
            "scorer": _scorer_snapshot(scorer_config),
            "protection": vars(protection_config or ProtectionConfig()),
        },
    )


def _scorer_snapshot(config: ScorerConfig | None) -> dict:
    config = config or ScorerConfig()
    snapshot = dict(vars(config))
    snapshot["priors"] = {k.value: v for k, v in config.priors.items()}
    return snapshot


def _flow_table(workload: FlowWorkload, first_seen: dict[str, int],
                first_protected: dict[str, int]) -> pd.DataFrame:
    rows = []
    for flow in workload.all_flows():
        flow_id = flow.spec.flow_id
        detect = (first_protected[flow_id] - first_seen[flow_id]
                  if flow_id in first_protected and flow_id in first_seen else None)
        unthrottled_slots = (flow.work_at_entry_mbit / flow.archetype.nominal_mbps
                             / workload.spec.seconds_per_slot)
        rows.append({
            "flow_id": flow_id, "archetype": flow.archetype.name,
            "truly_critical": flow.archetype.truly_critical,
            "declared": flow.spec.declared.value,
            "slots_alive": flow.slots_alive,
            "slots_below_floor": flow.slots_below_floor,
            "delivered_mbit": flow.delivered_mbit, "total_mbit": flow.total_mbit,
            "completed": flow.remaining_mbit <= 1e-9,
            "abandoned": bool(flow.abandoned),
            # How much longer the flow took than it would have on an idle link.
            # The cost the elastic traffic pays for the protection, and the
            # number that stops "protect everything" from looking free.
            "completion_inflation": (flow.slots_alive / unthrottled_slots
                                     if unthrottled_slots > 0 else float("nan")),
            "detect_slots": detect,
        })
    return pd.DataFrame(rows)


def sweep_load(policies: Sequence[str], capacity_mbps: np.ndarray,
               realised_mbps: np.ndarray, load_multipliers: Sequence[float],
               base_spec: WorkloadSpec | None = None, **kwargs) -> pd.DataFrame:
    """One row per policy and offered-load level.

    The sweep is the honest way to report this layer. At light load nothing
    separates the policies because nothing has to be shed, and at heavy enough
    load nothing separates them either because the floors stop fitting and the
    oracle fails too. Reporting a single operating point would let either of
    those be mistaken for the general case.
    """
    base = base_spec or WorkloadSpec()
    rows = []
    for load in load_multipliers:
        spec = WorkloadSpec(seed=base.seed, seconds_per_slot=base.seconds_per_slot,
                            load_multiplier=load,
                            declaration_loss=base.declaration_loss)
        for policy in policies:
            run = run_policy(policy, capacity_mbps, realised_mbps,
                             workload_spec=spec, **kwargs)
            rows.append({"load_multiplier": load} | run.summary())
    return pd.DataFrame(rows)
