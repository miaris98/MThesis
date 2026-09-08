"""Tests for the closed-loop driving score.

The scoring rule is the part of closed-loop evaluation that cannot be checked by looking
at a rollout: a wrong penalty product or a missing collision debounce produces numbers
that are plausible, ordered, and wrong. These pin the behaviour that a 30-minute
simulator run would not reveal.
"""
import pytest

from src.eval.driving_metrics import (
    DrivingMetrics,
    TerminationReason,
    aggregate,
    classify_collision,
    PENALTY_COLLISION_VEHICLE,
    PENALTY_COLLISION_PEDESTRIAN,
    PENALTY_RED_LIGHT,
)


def _drive(m: DrivingMetrics, seconds: float, speed_mps: float, final_idx: int,
           dt: float = 0.05, off_road: bool = False, t0: float = 0.0) -> float:
    """Ticks a metrics object through a constant-speed stretch, advancing the route
    index linearly, and returns the simulation time it ended at."""
    steps = max(1, int(seconds / dt))
    t = t0
    for i in range(1, steps + 1):
        t = t0 + i * dt
        idx = int(round(final_idx * i / steps))
        m.tick(t_s=t, speed_mps=speed_mps, route_idx=idx, off_road=off_road)
    return t


def test_a_clean_completed_route_scores_one():
    m = DrivingMetrics(route_id="r0", total_route_points=101)
    _drive(m, seconds=20.0, speed_mps=10.0, final_idx=100)

    assert m.route_completion == pytest.approx(1.0)
    assert m.infraction_penalty == pytest.approx(1.0)
    assert m.driving_score == pytest.approx(1.0)
    assert m.termination_reason == TerminationReason.COMPLETED


def test_sustained_contact_counts_once_not_once_per_frame():
    """The failure this whole module exists to prevent. A collision sensor re-fires every
    frame of contact; without the per-actor debounce a two-second scrape at 20 FPS logs
    40 infractions and the penalty product underflows to ~1e-9, making a scrape
    indistinguishable from destroying the car."""
    m = DrivingMetrics(route_id="r0", total_route_points=101)
    _drive(m, seconds=10.0, speed_mps=10.0, final_idx=100)

    counted = sum(
        m.add_collision(t_s=1.0 + i * 0.05, actor_type_id="vehicle.tesla.model3", actor_id=7)
        for i in range(40)  # 2 s of contact at 20 FPS
    )

    assert counted == 1
    assert m.to_dict()["collisions_vehicle"] == 1
    assert m.infraction_penalty == pytest.approx(PENALTY_COLLISION_VEHICLE)


def test_distinct_actors_each_count_even_in_the_same_instant():
    """The debounce is keyed on actor id, not on time alone - hitting two different cars
    in a pileup is two infractions, and keying on time would swallow the second."""
    m = DrivingMetrics(route_id="r0", total_route_points=101)
    _drive(m, seconds=10.0, speed_mps=10.0, final_idx=100)

    assert m.add_collision(1.0, "vehicle.tesla.model3", actor_id=1) is True
    assert m.add_collision(1.0, "vehicle.audi.tt", actor_id=2) is True

    assert m.infraction_penalty == pytest.approx(PENALTY_COLLISION_VEHICLE ** 2)


def test_a_long_continuous_contact_is_one_collision_however_long_it_lasts():
    """Regression test for a real artifact. With a fixed 5 s lockout, a car wedged
    against a stationary van logged a fresh collision every 5.0 s for the whole wedge -
    the first live rollout produced hits at t=34.8, 39.8, 44.8, 49.8, 54.8, all the same
    actor, and a route driven to 96% completion scored 0.026. Contact that never breaks
    is one collision no matter how many frames it spans."""
    m = DrivingMetrics(route_id="r0", total_route_points=101, contact_gap_s=1.0)
    _drive(m, seconds=60.0, speed_mps=1.0, final_idx=100)

    counted = sum(
        m.add_collision(t_s=10.0 + i * 0.05, actor_type_id="vehicle.volkswagen.t2", actor_id=7)
        for i in range(600)  # 30 seconds of unbroken contact at 20 FPS
    )

    assert counted == 1
    assert m.to_dict()["collisions_vehicle"] == 1


def test_the_same_actor_counts_again_only_after_contact_actually_breaks():
    m = DrivingMetrics(route_id="r0", total_route_points=101, contact_gap_s=1.0)
    _drive(m, seconds=30.0, speed_mps=10.0, final_idx=100)

    assert m.add_collision(1.00, "vehicle.tesla.model3", actor_id=7) is True
    assert m.add_collision(1.05, "vehicle.tesla.model3", actor_id=7) is False  # same contact
    assert m.add_collision(1.10, "vehicle.tesla.model3", actor_id=7) is False
    assert m.add_collision(6.50, "vehicle.tesla.model3", actor_id=7) is True   # separated, re-hit

    assert m.to_dict()["collisions_vehicle"] == 2


def test_penalties_compound_across_kinds():
    m = DrivingMetrics(route_id="r0", total_route_points=101)
    _drive(m, seconds=10.0, speed_mps=10.0, final_idx=100)

    m.add_collision(1.0, "walker.pedestrian.0001", actor_id=1)
    m.add_collision(2.0, "vehicle.tesla.model3", actor_id=2)
    m.add_red_light_violation(3.0, light_id=5)

    expected = PENALTY_COLLISION_PEDESTRIAN * PENALTY_COLLISION_VEHICLE * PENALTY_RED_LIGHT
    assert m.infraction_penalty == pytest.approx(expected)
    assert m.driving_score == pytest.approx(expected)  # route completion is 1.0


def test_classify_collision_buckets_props_and_layout_as_static():
    assert classify_collision("walker.pedestrian.0023") == "pedestrian"
    assert classify_collision("vehicle.audi.tt") == "vehicle"
    assert classify_collision("static.prop.streetbarrier") == "static"
    assert classify_collision("traffic.speed_limit.30") == "static"
    assert classify_collision("") == "static"


def test_route_completion_keeps_the_furthest_point_not_the_last_one():
    """A car that overshoots a junction and reverses, or oscillates, must be scored on
    how far it got. Reading the live index would score it on where it stopped."""
    m = DrivingMetrics(route_id="r0", total_route_points=101)
    m.tick(t_s=1.0, speed_mps=10.0, route_idx=80)
    m.tick(t_s=2.0, speed_mps=10.0, route_idx=40)

    assert m.raw_route_completion == pytest.approx(0.8)


def test_offroad_share_is_distance_weighted_not_tick_weighted():
    """A vehicle parked off the drivable lane accumulates ticks but no distance. Counting
    ticks would let standing still off-road dominate a route that was otherwise driven
    correctly, which reads as a much worse policy than it is."""
    m = DrivingMetrics(route_id="r0", total_route_points=101)
    t = _drive(m, seconds=10.0, speed_mps=10.0, final_idx=100, off_road=False)
    _drive(m, seconds=60.0, speed_mps=0.0, final_idx=100, off_road=True, t0=t)

    assert m.offroad_fraction == pytest.approx(0.0, abs=1e-9)
    assert m.route_completion == pytest.approx(1.0)


def test_offroad_distance_reduces_route_completion():
    m = DrivingMetrics(route_id="r0", total_route_points=101)
    t = _drive(m, seconds=10.0, speed_mps=10.0, final_idx=50, off_road=False)
    _drive(m, seconds=10.0, speed_mps=10.0, final_idx=100, off_road=True, t0=t)

    assert m.offroad_fraction == pytest.approx(0.5, abs=1e-2)
    assert m.raw_route_completion == pytest.approx(1.0)
    assert m.route_completion == pytest.approx(0.5, abs=1e-2)


def test_blocked_requires_a_continuous_stop():
    """Any motion resets the clock. A car crawling in stop-and-go traffic is not blocked,
    and a detector that summed stopped time would eventually declare it so."""
    m = DrivingMetrics(route_id="r0", total_route_points=101, blocked_timeout_s=10.0)

    t = _drive(m, seconds=8.0, speed_mps=0.0, final_idx=0)
    assert not m.is_blocked()

    t = _drive(m, seconds=1.0, speed_mps=5.0, final_idx=10, t0=t)   # moved: clock resets
    _drive(m, seconds=8.0, speed_mps=0.0, final_idx=10, t0=t)
    assert not m.is_blocked()


def test_blocked_trips_after_the_timeout_and_is_reported_as_the_reason():
    m = DrivingMetrics(route_id="r0", total_route_points=101, blocked_timeout_s=10.0)
    _drive(m, seconds=12.0, speed_mps=0.0, final_idx=5)

    assert m.is_blocked()
    assert m.termination_reason == TerminationReason.BLOCKED
    assert m.driving_score < 0.1


def test_a_stalled_route_scores_near_zero_without_any_infraction():
    """Route completion is the other half of the metric: a policy that never leaves the
    spawn point commits no infractions at all and must still score ~0. A collision count
    alone - which is what the pre-existing eval script reported - cannot express this."""
    m = DrivingMetrics(route_id="r0", total_route_points=101)
    _drive(m, seconds=30.0, speed_mps=0.0, final_idx=0)

    assert m.infraction_penalty == pytest.approx(1.0)
    assert m.driving_score == pytest.approx(0.0)


def test_aggregate_reports_means_and_per_km_rates():
    records = [
        DrivingMetrics("a", 101).to_dict(),
        DrivingMetrics("b", 101).to_dict(),
    ]
    records[0].update({"driving_score": 1.0, "route_completion": 1.0,
                       "distance_driven_m": 1000.0, "collisions_vehicle": 1})
    records[1].update({"driving_score": 0.0, "route_completion": 0.0,
                       "distance_driven_m": 0.0, "collisions_vehicle": 0})

    agg = aggregate(records)

    assert agg["num_routes"] == 2
    assert agg["driving_score"] == pytest.approx(0.5)
    assert agg["total_km"] == pytest.approx(1.0)
    assert agg["collisions_vehicle_per_km"] == pytest.approx(1.0)


def test_aggregate_reports_no_rate_rather_than_a_perfect_one_when_nothing_moved():
    """Per-km rates divide by distance. A fleet that never moved has an undefined rate,
    not a flawless one, and reporting 0.0 would rank a policy that cannot drive at all
    above one that drives and occasionally scrapes."""
    records = [DrivingMetrics("a", 101).to_dict()]
    agg = aggregate(records)

    assert agg["total_km"] == pytest.approx(0.0)
    assert agg["collisions_vehicle_per_km"] is None


def test_aggregate_of_nothing_is_not_an_exception():
    assert aggregate([])["num_routes"] == 0
