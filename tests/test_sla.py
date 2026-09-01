"""Tests for the availability and cost model.

The model turns rates into money, so the tests that matter are the ones that
pin the arithmetic and the ones that prove the trade-off is genuinely two-sided.
"""

from __future__ import annotations

import numpy as np
import pytest

from flwcnx.decide.sla import (
    DEFAULT_CREDIT_TIERS,
    ServiceLevelAgreement,
    availability_report,
    cost_report,
    evaluate_policy,
)

SLA = ServiceLevelAgreement()


# -- availability ------------------------------------------------------------


def test_perfect_delivery_is_full_availability():
    admitted = np.full(100, 5.0)
    actual = np.full(100, 200.0)          # 20 sessions servable, 5 admitted
    r = availability_report(admitted, actual, SLA)
    assert r.availability == 1.0
    assert r.n_outages == 0
    assert r.mttr_seconds == 0.0
    assert np.isinf(r.mtbf_seconds)
    assert r.meets_target


def test_availability_counts_session_slots_not_whole_slots():
    """Admitting 20 and dropping 1 is not an outage, it is 95% availability.

    Counting whole slots would make every policy look identical, because almost
    every slot has at least one dropped session somewhere.
    """
    admitted = np.full(10, 20.0)
    actual = np.full(10, 190.0)           # 19 servable of 20 promised
    r = availability_report(admitted, actual, SLA)
    assert r.availability == pytest.approx(0.95)
    assert r.degraded_slot_fraction == 1.0     # every slot is degraded ...
    assert r.availability > 0.9                # ... and availability is still high


def test_nines_are_computed_the_standard_way():
    admitted, actual = np.full(1000, 10.0), np.full(1000, 100.0)
    r = availability_report(admitted, actual, SLA)
    assert np.isinf(r.nines)

    # 99% delivered is two nines.
    admitted = np.full(100, 100.0)
    actual = np.full(100, 990.0)
    r = availability_report(admitted, actual, SLA)
    assert r.availability == pytest.approx(0.99)
    assert r.nines == pytest.approx(2.0, abs=1e-6)


def test_mtbf_and_mttr_separate_one_long_outage_from_many_short_ones():
    """A single availability number cannot tell these apart. Operators care."""
    sla = ServiceLevelAgreement(seconds_per_slot=1.0)
    good, bad = 200.0, 50.0               # 20 servable vs 5, against 10 admitted

    one_long = np.array([good] * 40 + [bad] * 20 + [good] * 40)
    many_short = np.array(([good] * 8 + [bad] * 2) * 10)
    admitted = np.full(100, 10.0)

    a = availability_report(admitted, one_long, sla)
    b = availability_report(admitted, many_short, sla)

    assert a.availability == pytest.approx(b.availability, abs=0.01)
    assert a.n_outages == 1
    assert b.n_outages == 10
    assert a.mttr_seconds > b.mttr_seconds
    assert a.longest_outage_seconds == 20.0
    assert b.longest_outage_seconds == 2.0


def test_empty_input_is_available_rather_than_undefined():
    r = availability_report(np.array([]), np.array([]), SLA)
    assert r.availability == 1.0 and r.n_slots == 0


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        availability_report(np.zeros(5), np.zeros(4), SLA)


# -- credits -----------------------------------------------------------------


@pytest.mark.parametrize(("availability", "expected"), [
    (0.9999, 0.00), (0.999, 0.00), (0.9989, 0.10),
    (0.99, 0.10), (0.9899, 0.25), (0.96, 0.25), (0.94, 0.50),
])
def test_credit_ladder_steps_up_as_the_miss_widens(availability, expected):
    assert SLA.credit_fraction(availability) == expected


def test_credit_tiers_are_ordered_worst_first():
    """The scan returns the first match, so a mis-ordered ladder under-credits."""
    floors = [f for f, _ in DEFAULT_CREDIT_TIERS]
    assert floors == sorted(floors)


# -- cost --------------------------------------------------------------------


def test_over_conservative_policy_pays_in_foregone_revenue():
    """The term that stops 'admit nothing' from being optimal."""
    actual = np.full(1000, 200.0)         # 20 sessions available every slot
    timid = evaluate_policy("timid", np.full(1000, 10.0), actual, SLA)

    assert timid.availability.availability == 1.0     # never breaks a promise
    assert timid.cost.sla_credit_cost == 0.0          # never owes a credit
    assert timid.cost.foregone_revenue > 0.0          # and still loses money
    assert timid.cost.total_cost > 0.0


def test_over_aggressive_policy_pays_in_sla_credits():
    actual = np.full(1000, 100.0)         # only 10 sessions are servable
    greedy = evaluate_policy("greedy", np.full(1000, 400.0), actual, SLA)

    assert greedy.availability.availability < 0.5
    assert greedy.cost.sla_credit_cost > 0.0
    assert greedy.cost.foregone_revenue == 0.0


def test_a_matched_policy_beats_both_extremes_on_total_cost():
    """The trade-off has an interior optimum, which is the point of the model."""
    rng = np.random.default_rng(0)
    actual = rng.normal(200.0, 20.0, 4000).clip(min=0)
    timid = evaluate_policy("timid", np.full(4000, 60.0), actual, SLA)
    greedy = evaluate_policy("greedy", np.full(4000, 400.0), actual, SLA)
    matched = evaluate_policy("matched", np.full(4000, 165.0), actual, SLA)

    assert matched.cost.total_cost < timid.cost.total_cost
    assert matched.cost.total_cost < greedy.cost.total_cost


def test_cost_scales_with_observation_length_but_cost_per_hour_does_not():
    rng = np.random.default_rng(1)
    actual = rng.normal(200.0, 20.0, 2000).clip(min=0)
    bound = np.full(2000, 150.0)
    short = evaluate_policy("s", bound[:1000], actual[:1000], SLA)
    long = evaluate_policy("l", bound, actual, SLA)

    assert long.cost.hours == pytest.approx(2 * short.cost.hours)
    assert long.cost.cost_per_hour == pytest.approx(short.cost.cost_per_hour, rel=0.25)


def test_report_dicts_are_json_safe():
    import json
    actual = np.full(50, 150.0)
    outcome = evaluate_policy("p", np.full(50, 120.0), actual, SLA)
    json.dumps(outcome.to_dict())         # raises if a numpy type leaked through


def test_seconds_per_slot_drives_the_time_axis():
    admitted, actual = np.full(720, 10.0), np.full(720, 50.0)
    fast = cost_report(admitted, actual, ServiceLevelAgreement(seconds_per_slot=1.0))
    slow = cost_report(admitted, actual, ServiceLevelAgreement(seconds_per_slot=10.0))
    assert slow.hours == pytest.approx(10 * fast.hours)
