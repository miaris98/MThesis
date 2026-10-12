"""Direct unit tests for `PIDController` in isolation.

Zero tests existed for this class before today (confirmed via the code-review-graph's
`tests_for` query) despite it being the exact component whose `target_speed` wiring bug
was found and fixed this session (see test_longitudinal_control.py) - that fix was only
verified end-to-end through `policy.act()`, never against the controller's own PID math.
This file pins the controller's contract directly: sign conventions, integral clamping,
state reset, and the throttle/brake mutual-exclusivity invariant every downstream caller
relies on.
"""
import numpy as np
import pytest

from src.models.world_on_rails.pid_controller import PIDController
from tests._sanity import soft_check


def test_zero_error_produces_zero_correction():
    """Aim point dead ahead, at exactly the target speed: nothing to correct."""
    pid = PIDController(target_speed=20.0)
    steer, throttle, brake = pid.control_from_waypoints(
        waypoints=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        current_speed_kmh=20.0)
    assert steer == pytest.approx(0.0, abs=1e-6)
    assert throttle == pytest.approx(0.0, abs=1e-6)
    assert brake == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("lateral_sign", [1.0, -1.0])
def test_steer_sign_follows_lateral_offset(lateral_sign):
    """(x_forward, y_lateral): a positive lateral offset must steer the same sign,
    otherwise the car corrects away from its aim point instead of toward it."""
    pid = PIDController()
    aim_y = 2.0 * lateral_sign
    steer, _, _ = pid.control_from_waypoints(
        waypoints=np.array([[0.0, 0.0], [0.0, 0.0], [5.0, aim_y]]),
        current_speed_kmh=20.0)
    assert steer * lateral_sign > 0, (steer, lateral_sign)
    # A 2m lateral offset at 5m ahead (~22 deg) is a moderate correction, not an emergency swerve.
    # It technically satisfies the sign check even if it's clipped to full lock - but a
    # controller that saturates on a routine offset is a tuning smell (kp_steer too high, or the
    # clip is masking a units mistake), worth a look even though nothing here is "wrong" per se.
    soft_check(abs(steer) < 0.99,
               f"steer saturated to {steer:+.3f} for a moderate ~22deg aim-point offset - "
               f"check kp_steer/ki_steer/kd_steer tuning rather than assuming this is intended")


def test_steer_integral_is_clamped():
    """A sustained lateral error must not let the integral term wind up unboundedly -
    it is clamped to [-1, 1] in the implementation."""
    pid = PIDController(ki_steer=0.05)
    for _ in range(200):
        pid.control_from_waypoints(
            waypoints=np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 10.0]]),
            current_speed_kmh=20.0)
    assert -1.0 <= pid.steer_error_integral <= 1.0


def test_speed_integral_is_clamped():
    """Same guarantee for the longitudinal integral, clamped to [-10, 10]."""
    pid = PIDController(ki_speed=0.05)
    for _ in range(500):
        pid.control_from_waypoints(
            waypoints=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            current_speed_kmh=0.0, target_speed_kmh=100.0)
    assert -10.0 <= pid.speed_error_integral <= 10.0


def test_reset_clears_all_accumulated_state():
    pid = PIDController()
    for _ in range(10):
        pid.control_from_waypoints(
            waypoints=np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]),
            current_speed_kmh=5.0, target_speed_kmh=25.0)
    assert pid.steer_error_integral != 0.0
    assert pid.speed_error_integral != 0.0

    pid.reset()
    assert pid.steer_error_integral == 0.0
    assert pid.steer_error_prev == 0.0
    assert pid.speed_error_integral == 0.0
    assert pid.speed_error_prev == 0.0


def test_explicit_target_speed_overrides_the_stored_default():
    """`target_speed_kmh` passed per-call must win over `self.target_speed` - this is
    the exact mechanism act() uses to feed the target-speed head's prediction in."""
    pid = PIDController(target_speed=20.0)
    # At the stored default (20 km/h) a car doing 20 km/h needs no correction...
    _, throttle_at_default, brake_at_default = pid.control_from_waypoints(
        waypoints=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        current_speed_kmh=20.0)
    assert throttle_at_default == pytest.approx(0.0, abs=1e-6)
    assert brake_at_default == pytest.approx(0.0, abs=1e-6)

    pid.reset()
    # ...but an explicit override to 5 km/h at the same current speed must brake.
    _, throttle_override, brake_override = pid.control_from_waypoints(
        waypoints=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        current_speed_kmh=20.0, target_speed_kmh=5.0)
    assert throttle_override == 0.0
    assert brake_override > 0.0


def test_none_target_speed_falls_back_to_stored_default():
    """This is the exact fallback path that produced the constant-20km/h bug before the
    fix: act() must pass a real value, but the controller's own contract for `None` has
    to keep working so a policy built with use_target_speed=False is unaffected."""
    pid = PIDController(target_speed=42.0)
    pid_ref = PIDController(target_speed=42.0)
    a = pid.control_from_waypoints(
        waypoints=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        current_speed_kmh=10.0, target_speed_kmh=None)
    b = pid_ref.control_from_waypoints(
        waypoints=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        current_speed_kmh=10.0, target_speed_kmh=42.0)
    assert a == b


@pytest.mark.parametrize("current,target", [(0.0, 30.0), (30.0, 0.0), (20.0, 20.0), (5.0, 4.9)])
def test_throttle_and_brake_are_never_both_nonzero(current, target):
    pid = PIDController()
    _, throttle, brake = pid.control_from_waypoints(
        waypoints=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        current_speed_kmh=current, target_speed_kmh=target)
    assert not (throttle > 0.0 and brake > 0.0), (throttle, brake)


@pytest.mark.parametrize("n_points", [1, 2])
def test_short_waypoint_list_uses_the_last_point_not_a_crash(n_points):
    """`waypoints[min(2, len - 1)]` is the guard against an IndexError when a policy
    predicts fewer than 3 waypoints - it must resolve to the last available point."""
    pid = PIDController()
    wps = np.array([[1.0, 0.0], [2.0, 3.0]][:n_points])
    steer, _, _ = pid.control_from_waypoints(waypoints=wps, current_speed_kmh=10.0,
                                             target_speed_kmh=10.0)
    expected_angle_sign = 1 if wps[-1][1] > 0 else (-1 if wps[-1][1] < 0 else 0)
    assert (steer > 0) == (expected_angle_sign > 0) or steer == pytest.approx(0.0, abs=1e-6)


def test_outputs_are_clipped_to_valid_control_ranges():
    """Even under an extreme error, steer must stay in [-1, 1] and throttle/brake in
    [0, 1] - CARLA's control API rejects values outside these ranges."""
    pid = PIDController()
    steer, throttle, brake = pid.control_from_waypoints(
        waypoints=np.array([[0.01, 0.0], [0.01, 1000.0], [0.01, 1000.0]]),
        current_speed_kmh=0.0, target_speed_kmh=500.0)
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0
