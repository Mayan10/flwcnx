"""The protective allocator: weighted max-min fairness with protected floors."""

from __future__ import annotations

import random

import pytest

from flwcnx.decide.flows import FlowClass, FlowObservation, FlowSpec
from flwcnx.decide.protect import (
    POLICIES,
    AllocationResult,
    FlowDemand,
    ProtectionConfig,
    ProtectionController,
    allocate_by_class,
    allocate_equal_share,
    allocate_protected,
    allocate_shed_largest,
    weighted_max_min,
)


def demand(flow_id: str, d: float, floor: float = 0.0, c: float = 0.5,
           protected: bool = False, declared: FlowClass = FlowClass.STANDARD) -> FlowDemand:
    return FlowDemand(flow_id, demand_mbps=d, floor_mbps=floor, criticality=c,
                      protected=protected, declared=declared)


def brute_force_tau(capacity: float, flows: list[FlowDemand],
                    cfg: ProtectionConfig) -> dict[str, float]:
    """Independent bisection on the same equation, for cross-checking the
    exact sweep. Slow and obviously correct, which is the point."""
    weights = {f.flow_id: cfg.weight(f.criticality) for f in flows}
    floors = {f.flow_id: (f.floor_mbps if f.protected else 0.0) for f in flows}

    def total(tau: float) -> float:
        return sum(min(max(weights[f.flow_id] * tau, floors[f.flow_id]), f.demand_mbps)
                   for f in flows)

    lo, hi = 0.0, max(f.demand_mbps / weights[f.flow_id] for f in flows) + 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if total(mid) < capacity:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2
    return {f.flow_id: min(max(weights[f.flow_id] * tau, floors[f.flow_id]), f.demand_mbps)
            for f in flows}


# -- the solver --------------------------------------------------------------

def test_abundant_capacity_gives_everyone_their_demand():
    flows = [demand("a", 10.0), demand("b", 5.0)]
    result = allocate_protected(1_000.0, flows)
    assert result.rates == {"a": 10.0, "b": 5.0}
    assert not result.protection_breached


def test_the_allocation_is_work_conserving():
    """Sums to min(capacity, total demand). Capacity a critical flow does not
    want must not be held idle for it."""
    flows = [demand("a", 10.0, c=0.9), demand("b", 40.0, c=0.1), demand("c", 5.0, c=0.5)]
    for capacity in (1.0, 12.0, 30.0, 54.9, 55.0, 90.0):
        result = allocate_protected(capacity, flows)
        assert result.allocated_mbps == pytest.approx(min(capacity, 55.0))


def test_the_exact_sweep_agrees_with_bisection_on_random_instances():
    rng = random.Random(1337)
    cfg = ProtectionConfig()
    for _ in range(300):
        n = rng.randint(1, 12)
        flows = []
        for i in range(n):
            d = rng.uniform(0.1, 50.0)
            floor = rng.uniform(0.0, d) if rng.random() < 0.5 else 0.0
            flows.append(demand(f"f{i}", d, floor, c=rng.random(),
                                protected=rng.random() < 0.4))
        total_floor = sum(f.floor_mbps for f in flows if f.protected)
        capacity = rng.uniform(total_floor + 1e-3, sum(f.demand_mbps for f in flows))
        exact = allocate_protected(capacity, flows, cfg).rates
        approx = brute_force_tau(capacity, flows, cfg)
        for key in exact:
            assert exact[key] == pytest.approx(approx[key], abs=1e-6)


def test_zero_capacity_allocates_nothing_without_raising():
    result = allocate_protected(0.0, [demand("a", 10.0, 2.0, protected=True)])
    assert result.allocated_mbps == pytest.approx(0.0)
    assert result.breached == ("a",)


def test_an_empty_flow_set_is_not_an_error():
    for policy in POLICIES.values():
        assert policy(50.0, []).rates == {}


def test_weighted_max_min_refuses_an_infeasible_fill():
    with pytest.raises(ValueError, match="resolve the infeasibility"):
        weighted_max_min(1.0, {"a": 1.0}, {"a": 5.0}, {"a": 10.0})


def test_a_non_positive_weight_is_rejected():
    with pytest.raises(ValueError, match="weight must be positive"):
        weighted_max_min(5.0, {"a": 0.0}, {"a": 0.0}, {"a": 10.0})


def test_a_floor_above_demand_is_rejected():
    with pytest.raises(ValueError, match="floor exceeds demand"):
        FlowDemand("a", demand_mbps=1.0, floor_mbps=5.0)


def test_weight_floor_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="weight_floor"):
        ProtectionConfig(weight_floor=0.0)


# -- the guarantee -----------------------------------------------------------

def test_protected_floors_are_honoured_whenever_they_jointly_fit():
    """The guarantee the layer exists to provide, on random instances."""
    rng = random.Random(99)
    for _ in range(400):
        flows = [demand(f"f{i}", d := rng.uniform(1.0, 40.0),
                        floor=rng.uniform(0.1, d), c=rng.random(),
                        protected=rng.random() < 0.5)
                 for i in range(rng.randint(2, 10))]
        need = sum(f.floor_mbps for f in flows if f.protected)
        capacity = rng.uniform(need, need + 60.0)      # feasible by construction
        result = allocate_protected(capacity, flows)
        assert result.breached == ()
        for flow in flows:
            if flow.protected:
                assert result.rates[flow.flow_id] >= flow.floor_mbps - 1e-9


def test_nothing_is_halted_while_capacity_remains():
    """A flow scored at zero criticality is throttled, not stopped. Killing a
    connection does not save its bytes, it defers them into a retry."""
    flows = [demand("critical", 50.0, 10.0, c=1.0, protected=True),
             demand("background", 50.0, c=0.0)]
    rates = allocate_protected(30.0, flows).rates
    assert rates["background"] > 0.0
    assert rates["critical"] > rates["background"]


def test_more_critical_flows_are_never_allocated_less_at_equal_demand():
    flows = [demand(f"f{i}", 20.0, c=i / 9.0) for i in range(10)]
    rates = allocate_protected(60.0, flows).rates
    ordered = [rates[f"f{i}"] for i in range(10)]
    assert ordered == sorted(ordered)


def test_the_floors_ablation_changes_only_the_floors():
    flows = [demand("a", 20.0, 8.0, c=0.95, protected=True), demand("b", 40.0, c=0.2)]
    with_floors = allocate_protected(10.0, flows, ProtectionConfig())
    without = allocate_protected(10.0, flows, ProtectionConfig(honour_floors=False))
    assert with_floors.rates["a"] >= 8.0
    assert without.rates["a"] < 8.0
    assert without.allocated_mbps == pytest.approx(with_floors.allocated_mbps)


# -- infeasibility -----------------------------------------------------------

def test_infeasible_floors_degrade_by_criticality_and_are_reported():
    """The alarm case. Strict order rather than shading every floor down,
    because a proportional shade leaves every protected flow below the rate at
    which it does anything."""
    flows = [demand("monitor", 1.0, 1.0, c=0.99, protected=True),
             demand("call", 3.0, 3.0, c=0.80, protected=True),
             demand("imaging", 25.0, 20.0, c=0.70, protected=True)]
    result = allocate_protected(5.0, flows)
    assert result.breached == ("imaging",)
    assert result.protection_breached
    assert result.rates["monitor"] >= 1.0
    assert result.rates["call"] >= 3.0


def test_ties_in_criticality_break_toward_serving_more_flows():
    flows = [demand("big", 30.0, 20.0, c=0.8, protected=True),
             demand("small_a", 3.0, 3.0, c=0.8, protected=True),
             demand("small_b", 3.0, 3.0, c=0.8, protected=True)]
    result = allocate_protected(7.0, flows)
    assert result.breached == ("big",)
    assert result.rates["small_a"] >= 3.0 and result.rates["small_b"] >= 3.0


def test_below_floor_is_a_superset_of_breached():
    flows = [demand("protected_flow", 10.0, 9.0, c=0.9, protected=True),
             demand("unprotected_call", 4.0, 3.0, c=0.2)]
    result = allocate_protected(6.0, flows)
    assert "unprotected_call" in result.below_floor
    assert set(result.breached) <= set(result.below_floor)


def test_wasted_capacity_is_reported_not_reclaimed():
    flows = [demand("call", 4.0, 3.0, c=0.2), demand("bulk", 100.0, c=0.2)]
    result = allocate_protected(3.0, flows)
    assert 0.0 < result.wasted_mbps(flows) == result.rates["call"]


# -- the baselines, and the case for the layer -------------------------------

def test_shedding_the_largest_flow_halts_the_critical_transfer():
    """The motivating failure, in one assertion. The imaging push and the
    operating system update are the same shape to a rate-based shaper, and the
    imaging push is larger."""
    flows = [demand("imaging", 25.0, 15.0, c=0.95, protected=True,
                    declared=FlowClass.STANDARD),
             demand("os_update", 18.0, c=0.05),
             demand("chat", 0.3, c=0.4)]
    naive = allocate_shed_largest(20.0, flows)
    ours = allocate_protected(20.0, flows)
    assert naive.rates["imaging"] < 2.0          # first against the wall
    assert ours.rates["imaging"] >= 15.0         # floor held
    assert ours.rates["os_update"] > 0.0         # and the update is not killed


def test_class_priority_is_fooled_by_a_declaration_and_the_scorer_is_not():
    """What the dynamic evidence buys over a static policy. The static policy
    is wrong on exactly the two cases the evidence exists for."""
    flows = [
        # Undeclared but genuinely critical: scored high, declared STANDARD.
        demand("undeclared_critical", 10.0, 6.0, c=0.93, protected=True,
               declared=FlowClass.STANDARD),
        # Declared clinical and actually a download: scored low.
        demand("misdeclared_bulk", 40.0, c=0.08, declared=FlowClass.CLINICAL),
    ]
    static = allocate_by_class(12.0, flows)
    ours = allocate_protected(12.0, flows)
    assert static.rates["misdeclared_bulk"] > static.rates["undeclared_critical"]
    assert ours.rates["undeclared_critical"] >= 6.0
    assert ours.rates["undeclared_critical"] > ours.rates["misdeclared_bulk"]


def test_equal_share_is_max_min_fair():
    flows = [demand("a", 1.0), demand("b", 50.0), demand("c", 50.0)]
    rates = allocate_equal_share(31.0, flows).rates
    assert rates["a"] == pytest.approx(1.0)
    assert rates["b"] == pytest.approx(15.0)
    assert rates["c"] == pytest.approx(15.0)


def test_every_policy_is_work_conserving_and_never_over_allocates():
    rng = random.Random(7)
    for _ in range(200):
        flows = [demand(f"f{i}", rng.uniform(0.1, 30.0), c=rng.random())
                 for i in range(rng.randint(1, 8))]
        capacity = rng.uniform(0.0, 120.0)
        for name, policy in POLICIES.items():
            result = policy(capacity, flows)
            expected = min(capacity, sum(f.demand_mbps for f in flows))
            assert result.allocated_mbps == pytest.approx(expected, abs=1e-6), name
            assert all(v >= -1e-12 for v in result.rates.values()), name


# -- the controller ----------------------------------------------------------

def test_the_controller_protects_a_declared_monitor_from_its_first_slot():
    controller = ProtectionController.build()
    monitor = FlowSpec("monitor", declared=FlowClass.LIFE_SAFETY,
                       floor_mbps=0.4, demand_mbps=0.5)
    bulk = FlowSpec("bulk", declared=FlowClass.BACKGROUND, demand_mbps=80.0)
    observations = {
        "monitor": (monitor, FlowObservation(demand_mbps=0.5, packets_down=60,
                                             packets_up=60, mbit_down=0.3, mbit_up=0.2)),
        "bulk": (bulk, FlowObservation(demand_mbps=80.0, packets_down=30_000,
                                       mbit_down=400.0)),
    }
    first = controller.step(5.0, observations)
    assert first.rates["monitor"] >= 0.4
    assert first.rates["bulk"] > 0.0
    assert not first.protection_breached


def test_the_controller_result_is_an_allocation_result():
    controller = ProtectionController.build()
    spec = FlowSpec("a", demand_mbps=4.0)
    out = controller.step(10.0, {"a": (spec, FlowObservation(demand_mbps=4.0))})
    assert isinstance(out, AllocationResult)
    assert out.utilisation() == pytest.approx(0.4)
