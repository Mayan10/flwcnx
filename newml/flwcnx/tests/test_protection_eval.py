"""The flow workload and the protection evaluation harness."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flwcnx.decide.flows import FlowClass
from flwcnx.eval.protection import (
    ORACLE,
    average_precision,
    jain_index,
    run_policy,
    sweep_load,
)
from flwcnx.eval.workload import ARCHETYPES, FlowWorkload, WorkloadSpec, oracle_view

CAPACITY = np.full(120, 60.0)
REALISED = np.full(120, 55.0)


# -- the workload ------------------------------------------------------------

def test_the_mix_contains_hard_cases_in_both_directions():
    """A workload whose critical flows are the small ones measures nothing,
    because the trivial policy already wins it."""
    critical = [a for a in ARCHETYPES if a.truly_critical]
    ordinary = [a for a in ARCHETYPES if not a.truly_critical]
    assert critical and ordinary

    # At least one critical archetype is high rate and does not declare itself,
    # which is what a rate-based shaper throttles first.
    assert any(a.nominal_mbps > 10 and a.declared is FlowClass.STANDARD for a in critical)
    # At least one non-critical archetype declares a high priority class, which
    # is what fools a policy that trusts declarations.
    assert any(a.declared in {FlowClass.CLINICAL, FlowClass.LIFE_SAFETY}
               for a in ordinary)
    # At least one non-critical archetype is interactive and in the foreground,
    # so the attention channel cannot win by itself.
    assert any(a.foreground_probability > 0.5 and not a.elastic for a in ordinary)


def test_every_archetype_has_a_floor_it_can_reach():
    for archetype in ARCHETYPES:
        assert 0.0 <= archetype.floor_mbps <= archetype.nominal_mbps


def test_warm_start_places_flows_part_way_through():
    """Running arrivals from an empty link would spend most of a trace filling,
    because the long archetypes live for thousands of seconds."""
    workload = FlowWorkload(WorkloadSpec(seed=3))
    workload.warm_start(0)
    assert len(workload.active) >= 4
    assert any(f.progress > 0.0 for f in workload.active.values())


def test_the_structural_stream_is_independent_of_the_behavioural_one():
    """What makes two policies comparable: the same seed has to produce the
    same arrivals and the same flow specs whatever the allocator does."""
    def specs(rates: float) -> list[tuple[str, float]]:
        workload = FlowWorkload(WorkloadSpec(seed=11))
        workload.warm_start(0)
        for t in range(40):
            observations = workload.observe(t)
            workload.apply(t, {k: rates for k in observations}, 40.0)
        return sorted((f.spec.flow_id, round(f.total_mbit, 6))
                      for f in workload.all_flows())

    assert specs(0.5) == specs(50.0)


def test_an_elastic_flow_backs_off_and_an_inelastic_one_does_not():
    """The generator has to produce the elasticity signal from behaviour, not
    from the label, or the channel measuring it is reading its own answer."""
    workload = FlowWorkload(WorkloadSpec(seed=5))
    workload.warm_start(0)
    for t in range(30):
        observations = workload.observe(t)
        workload.apply(t, {k: 0.05 for k in observations}, 100.0)

    elastic = [f for f in workload.active.values() if f.archetype.elastic]
    inelastic = [f for f in workload.active.values() if not f.archetype.elastic]
    assert all(f.demand_mbps < f.archetype.nominal_mbps for f in elastic)
    assert all(f.demand_mbps == f.archetype.nominal_mbps for f in inelastic)


def test_the_link_has_the_last_word_over_the_allocator():
    """An allocation written against a bound the link does not meet is not
    delivered. This is the only place the difference between a bound and a
    forecast becomes something a user experiences."""
    workload = FlowWorkload(WorkloadSpec(seed=2))
    workload.warm_start(0)
    observations = workload.observe(0)
    generous = dict.fromkeys(observations, 40.0)
    delivered = workload.apply(0, generous, realised_mbps=10.0)
    assert sum(delivered.values()) == pytest.approx(10.0)
    assert all(v < 40.0 for v in delivered.values())


def test_the_label_is_reachable_only_through_the_oracle_view():
    workload = FlowWorkload(WorkloadSpec(seed=4))
    workload.warm_start(0)
    observations = workload.observe(0)
    assert set(oracle_view(observations)) == set(observations)
    assert set(oracle_view(observations).values()) <= {0.0, 1.0}


# -- the metrics -------------------------------------------------------------

def test_average_precision_is_one_for_a_perfect_ranking():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert average_precision(scores, np.array([1, 1, 0, 0])) == pytest.approx(1.0)


def test_average_precision_matches_the_hand_computed_value():
    # Ranked 1, 0, 1, 0: precision at the two hits is 1.0 and 2/3.
    value = average_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0]))
    assert value == pytest.approx((1.0 + 2.0 / 3.0) / 2)


def test_average_precision_is_undefined_without_both_classes():
    assert np.isnan(average_precision(np.array([0.5, 0.4]), np.array([1, 1])))


def test_jain_index_is_one_for_an_equal_split():
    assert jain_index([2.0, 2.0, 2.0]) == pytest.approx(1.0)
    assert jain_index([4.0, 0.0, 0.0]) == pytest.approx(1 / 3)


# -- the harness -------------------------------------------------------------

def test_an_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown policy"):
        run_policy("wishful", CAPACITY, REALISED)


def test_mismatched_series_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        run_policy("protected", CAPACITY, REALISED[:10])


def test_a_run_records_one_row_per_flow_slot_and_one_per_slot():
    run = run_policy("protected", CAPACITY, REALISED,
                     WorkloadSpec(seed=7, load_multiplier=2.0))
    assert len(run.slots) <= len(CAPACITY)
    assert len(run.records) == run.records.groupby(["slot", "flow_id"]).ngroups
    summary = run.summary()
    assert 0.0 <= summary["critical_violation_allocated"] <= 1.0
    assert summary["critical_violation_delivered"] >= \
        summary["critical_violation_allocated"] - 1e-9


def test_the_oracle_is_the_floor_the_allocator_cannot_beat():
    """Any gap between the protection policy and the oracle is classification
    error and nothing else, so the oracle has to be at least as good."""
    kwargs = {"workload_spec": WorkloadSpec(seed=7, load_multiplier=2.5)}
    oracle = run_policy(ORACLE, CAPACITY, REALISED, **kwargs).summary()
    ours = run_policy("protected", CAPACITY, REALISED, **kwargs).summary()
    assert oracle["critical_violation_allocated"] <= ours["critical_violation_allocated"]


def test_the_protection_policy_beats_every_deployable_baseline_under_load():
    kwargs = {"workload_spec": WorkloadSpec(seed=7, load_multiplier=2.5)}
    ours = run_policy("protected", CAPACITY, REALISED, **kwargs).summary()
    for baseline in ("by_class", "equal_share", "shed_largest"):
        other = run_policy(baseline, CAPACITY, REALISED, **kwargs).summary()
        assert ours["critical_violation_allocated"] < other["critical_violation_allocated"]
        assert ours["critical_goodput_ratio"] > other["critical_goodput_ratio"]


def test_a_run_is_reproducible_at_a_fixed_seed():
    kwargs = {"workload_spec": WorkloadSpec(seed=21, load_multiplier=2.0)}
    def fingerprint(summary: dict) -> dict:
        # NaN is a legitimate value here (a median over an empty set) and is
        # never equal to itself, so compare through a sentinel.
        return {k: ("nan" if isinstance(v, float) and math.isnan(v) else v)
                for k, v in summary.items()}

    first = run_policy("protected", CAPACITY, REALISED, **kwargs).summary()
    second = run_policy("protected", CAPACITY, REALISED, **kwargs).summary()
    assert fingerprint(first) == fingerprint(second)


def test_the_config_snapshot_records_what_produced_the_run():
    run = run_policy("protected", CAPACITY, REALISED, WorkloadSpec(seed=9))
    assert run.config["workload"]["seed"] == 9
    assert "rho" in run.config["scorer"]
    assert "weight_floor" in run.config["protection"]


def test_the_load_sweep_covers_every_policy_at_every_level():
    frame = sweep_load(["protected", "equal_share"], CAPACITY[:60], REALISED[:60],
                       [1.0, 2.0], base_spec=WorkloadSpec(seed=13))
    assert len(frame) == 4
    assert set(frame["load_multiplier"]) == {1.0, 2.0}


def test_the_archetype_breakdown_names_where_the_violations_land():
    run = run_policy("protected", CAPACITY, REALISED,
                     WorkloadSpec(seed=7, load_multiplier=2.5))
    table = run.by_archetype()
    assert set(table["archetype"]) <= {a.name for a in ARCHETYPES}
    assert table["violation_allocated"].between(0.0, 1.0).all()


def test_a_flow_held_below_a_useful_rate_gives_up():
    """Without this the generator has no steady state above capacity: arrivals
    continue while starved transfers take proportionally longer, so the active
    set grows for the whole run."""
    workload = FlowWorkload(WorkloadSpec(seed=8, abandon_after_slots=10))
    workload.warm_start(0)
    started = len(workload.active)
    for t in range(40):
        observations = workload.observe(t)
        workload.apply(t, dict.fromkeys(observations, 0.0), realised_mbps=0.0)
    assert len(workload.active) < started
    assert any(f.abandoned for f in workload.completed)


def test_a_served_flow_never_gives_up():
    workload = FlowWorkload(WorkloadSpec(seed=8, abandon_after_slots=5))
    workload.warm_start(0)
    for t in range(40):
        observations = workload.observe(t)
        workload.apply(t, {k: 100.0 for k in observations}, realised_mbps=1e6)
    assert not any(f.abandoned for f in workload.all_flows())


def test_abandonment_keeps_the_active_set_bounded_under_heavy_load():
    workload = FlowWorkload(WorkloadSpec(seed=8, load_multiplier=4.0))
    workload.warm_start(0)
    sizes = []
    for t in range(400):
        observations = workload.observe(t)
        # A link carrying far less than is offered.
        workload.apply(t, dict.fromkeys(observations, 0.3), realised_mbps=20.0)
        sizes.append(len(workload.active))
    # The second half is not systematically larger than the first, which is
    # what "bounded" means here.
    assert np.mean(sizes[200:]) < np.mean(sizes[:200]) * 1.6


def test_abandonment_is_reported_rather_than_hidden():
    """A critical transfer that gave up is the worst outcome the layer has.
    Counting it as no longer violating its floor would be the wrong
    bookkeeping."""
    run = run_policy("protected", np.full(200, 3.0), np.full(200, 3.0),
                     WorkloadSpec(seed=8, load_multiplier=3.0,
                                  abandon_after_slots=20))
    summary = run.summary()
    assert summary["critical_abandon_rate"] > 0.0
    assert "abandoned" in run.flows
