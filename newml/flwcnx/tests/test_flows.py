"""The online criticality scorer."""

from __future__ import annotations

import math

import pytest

from flwcnx.decide.flows import (
    CLASS_PRIOR_LOG_ODDS,
    CriticalityScorer,
    FlowClass,
    FlowObservation,
    FlowSpec,
    ScorerConfig,
    channel_scores,
)


def bulk_spec(flow_id: str = "bulk", **kw) -> FlowSpec:
    base = {"declared": FlowClass.BACKGROUND, "floor_mbps": 0.0, "demand_mbps": 30.0,
            "recurrence": 0.9, "resumable": True}
    return FlowSpec(flow_id=flow_id, **(base | kw))


def bulk_obs(**kw) -> FlowObservation:
    base = {"mbit_down": 150.0, "mbit_up": 2.0, "packets_down": 12_000,
            "packets_up": 400, "idle_fraction": 0.0, "foreground": False,
            "demand_mbps": 30.0, "granted_last_mbps": 30.0, "age_s": 60.0}
    return FlowObservation(**(base | kw))


def call_spec(flow_id: str = "call", **kw) -> FlowSpec:
    base = {"declared": FlowClass.CLINICAL, "floor_mbps": 1.5, "demand_mbps": 2.5}
    return FlowSpec(flow_id=flow_id, **(base | kw))


def call_obs(**kw) -> FlowObservation:
    base = {"mbit_down": 12.0, "mbit_up": 10.0, "packets_down": 3_000,
            "packets_up": 2_800, "idle_fraction": 0.05, "foreground": True,
            "user_events": 4, "demand_mbps": 2.5, "granted_last_mbps": 2.5}
    return FlowObservation(**(base | kw))


# -- the spec contract -------------------------------------------------------

def test_a_floor_above_demand_is_rejected():
    with pytest.raises(ValueError, match="exceeds demand"):
        FlowSpec(flow_id="f", floor_mbps=10.0, demand_mbps=2.0)


def test_recurrence_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="recurrence"):
        FlowSpec(flow_id="f", recurrence=1.4)


def test_release_threshold_above_protect_threshold_is_rejected():
    with pytest.raises(ValueError, match="release threshold"):
        ScorerConfig(protect_threshold=0.5, release_threshold=0.8)


# -- the prior is the initial condition --------------------------------------

def test_a_declared_life_safety_flow_is_protected_on_its_first_slot():
    """The whole point of seeding at the prior. A patient monitor that has to
    earn protection over an evidence window is unprotected exactly when a
    congestion episode is already running."""
    scorer = CriticalityScorer()
    spec = FlowSpec(flow_id="monitor", declared=FlowClass.LIFE_SAFETY,
                    floor_mbps=0.4, demand_mbps=0.5)
    assert scorer.state_for(spec).protected


def test_a_background_flow_is_not_protected_on_its_first_slot():
    scorer = CriticalityScorer()
    assert not scorer.state_for(bulk_spec()).protected


def test_priors_are_ordered_by_class():
    ordered = [FlowClass.BACKGROUND, FlowClass.BULK, FlowClass.STANDARD,
               FlowClass.INTERACTIVE, FlowClass.RESEARCH, FlowClass.CLINICAL,
               FlowClass.LIFE_SAFETY]
    values = [CLASS_PRIOR_LOG_ODDS[c] for c in ordered]
    assert values == sorted(values)


# -- the channels ------------------------------------------------------------

def test_absent_evidence_scores_exactly_zero():
    """Zero is the log-odds representation of "this channel saw nothing", and
    several channels rely on being able to say it."""
    scores = channel_scores(FlowSpec(flow_id="f", recurrence=0.5), FlowObservation())
    assert scores["elasticity"] == 0.0     # never throttled, so no experiment
    assert scores["deadline"] == 0.0       # no deadline
    assert scores["interactivity"] == 0.0  # no packets
    assert scores["volume"] == 0.0         # no demand
    assert scores["recurrence"] == 0.0     # exactly routine-neutral


def test_every_channel_stays_inside_the_unit_band():
    extremes = [
        (bulk_spec(), bulk_obs()),
        (call_spec(), call_obs()),
        (bulk_spec(resumable=False), bulk_obs(progress=1.0, user_events=10_000,
                                              foreground=True, idle_fraction=1.0)),
        (call_spec(recurrence=1.0), call_obs(demand_mbps=1e6, granted_last_mbps=1e-9,
                                             throttled_last=True, remaining_mbit=1e9,
                                             deadline_remaining_s=-1e6)),
    ]
    for spec, obs in extremes:
        for name, value in channel_scores(spec, obs).items():
            assert -1.0 <= value <= 1.0, f"{name} escaped the band at {value}"


def test_elasticity_separates_backing_off_from_holding_firm():
    cfg = ScorerConfig()
    elastic = channel_scores(bulk_spec(), bulk_obs(throttled_last=True,
                                                   granted_last_mbps=5.0,
                                                   demand_mbps=5.0), cfg)
    inelastic = channel_scores(call_spec(), call_obs(throttled_last=True,
                                                     granted_last_mbps=0.5,
                                                     demand_mbps=2.5), cfg)
    # An elastic flow's offered load collapses to what it was granted, so the
    # ratio pins at 1 either way. The separation comes from demand *below* the
    # grant, which is what a flow that has finished backing off looks like.
    backed_off = channel_scores(bulk_spec(), bulk_obs(throttled_last=True,
                                                      granted_last_mbps=5.0,
                                                      demand_mbps=0.5), cfg)
    assert backed_off["elasticity"] < 0.0
    assert inelastic["elasticity"] == pytest.approx(1.0)
    assert elastic["elasticity"] == pytest.approx(1.0)


def test_deadline_pressure_flips_sign_as_slack_runs_out():
    """The dynamic claim, in one assertion. Nothing about the flow changes
    except how much time is left."""
    spec = bulk_spec(bytes_total_mbit=6_000.0, deadline_s=3_600.0)
    relaxed = channel_scores(spec, bulk_obs(remaining_mbit=3_000.0,
                                            deadline_remaining_s=3_000.0))
    urgent = channel_scores(spec, bulk_obs(remaining_mbit=3_000.0,
                                           deadline_remaining_s=60.0))
    assert relaxed["deadline"] < -0.5
    assert urgent["deadline"] > 0.5


def test_irreversibility_distinguishes_a_resumable_transfer_at_the_same_progress():
    late = FlowObservation(progress=0.95)
    assert (channel_scores(bulk_spec(resumable=False), late)["irreversibility"]
            > channel_scores(bulk_spec(resumable=True), late)["irreversibility"])


def test_attention_is_asymmetric():
    """Presence of a user is strong evidence, absence is weak. A background
    clinical transfer must not be buried by nobody watching it."""
    watched = channel_scores(call_spec(), call_obs(foreground=True, user_events=6))
    unwatched = channel_scores(call_spec(), call_obs(foreground=False, user_events=0))
    assert watched["attention"] > 0.8
    assert -0.3 < unwatched["attention"] < 0.0


def test_volume_never_outvotes_the_rest():
    """The channel that could reconstruct the failure this layer prevents is
    capped below any two of the others."""
    from flwcnx.decide.flows import CHANNEL_WEIGHTS

    others = sorted(v for k, v in CHANNEL_WEIGHTS.items() if k != "volume")
    assert CHANNEL_WEIGHTS["volume"] < others[0] + others[1]
    assert CHANNEL_WEIGHTS["volume"] == min(CHANNEL_WEIGHTS.values())


# -- the accumulator ---------------------------------------------------------

def test_an_undeclared_interactive_flow_earns_protection_from_evidence_alone():
    """The case a static class policy cannot reach: declared STANDARD, so its
    prior is 0, and it is protected only because of what it does."""
    scorer = CriticalityScorer()
    spec = call_spec("undeclared", declared=FlowClass.STANDARD)
    for t in range(20):
        scorer.update(t, {"undeclared": (spec, call_obs())})
    assert scorer.is_protected("undeclared")
    assert scorer.criticality("undeclared") > 0.6


def test_a_misdeclared_bulk_flow_loses_the_protection_its_declaration_bought():
    """The gaming case. Anything on the host can claim to be clinical; the
    evidence has to be able to contradict it."""
    scorer = CriticalityScorer()
    spec = bulk_spec("liar", declared=FlowClass.CLINICAL)
    assert scorer.state_for(spec).protected      # the declaration is believed at first
    for t in range(60):
        scorer.update(t, {"liar": (spec, bulk_obs())})
    assert not scorer.is_protected("liar")
    assert scorer.criticality("liar") < 0.45


def test_the_accumulator_is_clipped_at_the_configured_bound():
    """Without a clip a flow banks unbounded evidence, and unbounded evidence
    takes unbounded time to overturn."""
    cfg = ScorerConfig(max_log_odds=1.5)
    scorer = CriticalityScorer(cfg)
    for t in range(200):
        scorer.update(t, {"b": (bulk_spec("b"), bulk_obs())})
    assert scorer.states["b"].log_odds == pytest.approx(-cfg.max_log_odds)


def test_recovery_time_does_not_grow_with_the_history_behind_it():
    """The property the clip exists for. A flow that has looked like bulk for
    an hour must react to new evidence as fast as one that has looked like bulk
    for a minute, or the layer cannot serve a transfer that turns urgent."""
    urgent = bulk_obs(foreground=True, user_events=8, remaining_mbit=3_000.0,
                      deadline_remaining_s=10.0, throttled_last=True,
                      granted_last_mbps=1.0, demand_mbps=30.0)

    def recover(history_slots: int) -> list[float]:
        scorer = CriticalityScorer()
        for t in range(history_slots):
            scorer.update(t, {"b": (bulk_spec("b"), bulk_obs())})
        return [scorer.update(history_slots + t,
                              {"b": (bulk_spec("b", resumable=False), urgent)})["b"]
                for t in range(30)]

    short, long = recover(150), recover(3_000)
    assert short == pytest.approx(long, abs=1e-9)
    assert long[-1] > long[0]


def test_mixed_evidence_lands_near_undecided_rather_than_confidently_wrong():
    """A recurrent bulk transfer that has turned deadline-urgent is genuinely
    ambiguous: the deadline and the user say protect, the traffic shape and the
    declaration say shed. The scorer is required to sit near 0.5 and say so,
    not to resolve it by fiat."""
    scorer = CriticalityScorer()
    urgent = bulk_obs(foreground=True, user_events=8, remaining_mbit=3_000.0,
                      deadline_remaining_s=10.0, throttled_last=True,
                      granted_last_mbps=1.0, demand_mbps=30.0)
    for t in range(60):
        value = scorer.update(t, {"b": (bulk_spec("b", resumable=False), urgent)})["b"]
    assert 0.35 < value < 0.65


def test_hysteresis_stops_a_borderline_flow_from_flapping():
    scorer = CriticalityScorer(ScorerConfig(min_protected_slots=3))
    spec = call_spec("edge", declared=FlowClass.STANDARD)
    # Alternate the single strongest channel on and off every slot.
    for t in range(60):
        scorer.update(t, {"edge": (spec, call_obs(foreground=bool(t % 2),
                                                  user_events=4 * (t % 2)))})
    assert scorer.flap_rate() < 0.1


def test_scoring_is_causal():
    """Slot t's score depends on slots up to t and nothing after. Truncating
    the trace must not move any earlier decision."""
    spec = call_spec("c", declared=FlowClass.STANDARD)
    observations = [call_obs(foreground=bool(t % 3), user_events=t % 5) for t in range(40)]

    full, short = CriticalityScorer(), CriticalityScorer()
    long_run = [full.update(t, {"c": (spec, o)})["c"] for t, o in enumerate(observations)]
    short_run = [short.update(t, {"c": (spec, o)})["c"]
                 for t, o in enumerate(observations[:25])]
    assert long_run[:25] == short_run


def test_an_unobserved_flow_is_held_rather_than_decayed():
    """Going quiet is not evidence of being unimportant. A paused transfer
    that loses its protection resumes into the shedding it was protected from."""
    scorer = CriticalityScorer()
    spec = FlowSpec(flow_id="m", declared=FlowClass.LIFE_SAFETY, floor_mbps=0.4,
                    demand_mbps=0.5)
    scorer.update(0, {"m": (spec, FlowObservation(demand_mbps=0.5))})
    before = scorer.criticality("m")
    for t in range(1, 10):
        scorer.update(t, {})
    assert scorer.criticality("m") == before
    assert scorer.is_protected("m")


def test_idle_flows_are_evicted_so_memory_tracks_active_flows():
    scorer = CriticalityScorer(ScorerConfig(idle_eviction_slots=5))
    scorer.update(0, {"gone": (bulk_spec("gone"), bulk_obs())})
    assert "gone" in scorer.states
    scorer.update(20, {})
    assert "gone" not in scorer.states


def test_a_key_that_disagrees_with_its_spec_is_rejected():
    scorer = CriticalityScorer()
    with pytest.raises(ValueError, match="does not match"):
        scorer.update(0, {"a": (bulk_spec("b"), bulk_obs())})


def test_criticality_stays_a_probability():
    scorer = CriticalityScorer()
    for t in range(100):
        for spec, obs in [(bulk_spec(), bulk_obs()), (call_spec(), call_obs())]:
            value = scorer.update(t, {spec.flow_id: (spec, obs)})[spec.flow_id]
            assert 0.0 < value < 1.0 and math.isfinite(value)
