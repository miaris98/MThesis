"""Waypoint-to-control conversion for World on Rails policies.

Split out of wor_policy.py: this is a classical controller, not a network, and both
the CNN-headed and Qwen-headed policies hand it the same (N, 2) ego-frame waypoints.
Kept importable from wor_policy for backwards compatibility.
"""
import math
import os
from typing import Optional, Tuple

import numpy as np


def tfpp_brake_rule(throttle: float, brake: float, speed_mps: float,
                    target_speed_kmh: Optional[float]) -> Tuple[float, float]:
    """TF++'s longitudinal brake rule (carla_garage team_code/model.py:control_pid_direct): full brake when the target
    speed is ~0 or the car is more than brake_ratio (1.1) above it; otherwise the inputs pass through."""
    if target_speed_kmh is None:
        return throttle, brake
    tgt_mps = target_speed_kmh / 3.6
    if tgt_mps < 0.01 or speed_mps / max(tgt_mps, 1e-6) > 1.1:
        return 0.0, 1.0
    return throttle, brake


def tfpp_brake_override(throttle: float, brake: float, speed_mps: float,
                        target_speed_kmh: Optional[float]) -> Tuple[float, float]:
    """tfpp_brake_rule, off unless WOR_TFPP_BRAKE_RULE=1 (older switch; WOR_CONTROLLER=brake is the same rule).

    Lives here, shared by both policies' act(): it used to be written into the qwen policy only, so the same environment
    flag was silently ignored by the cnn arm (arm H) and would have confounded any head-vs-head comparison run
    with it.
    """
    if os.environ.get("WOR_TFPP_BRAKE_RULE") != "1":
        return throttle, brake
    return tfpp_brake_rule(throttle, brake, speed_mps, target_speed_kmh)


# 2026-11-20, controller study: WOR_CONTROLLER picks how (waypoints, target speed) become (steer, throttle, brake). The network
# and its outputs are unchanged; only this conversion differs.
#   pid       the original controller below (default; bit-identical to before)
#   brake     pid + TF++'s brake rule (tfpp_brake_rule)
#   pdm_long  PDM-Lite's longitudinal controller (the expert that recorded our data; TF++ uses it with its target-speed head:
#             carla_garage team_code/nav_planner.py:get_throttle + model.py:control_pid_direct): full brake at target ~0 or
#             speed > 1.1 x target, else a fitted linear-regression throttle law; no integral, so no windup. Steering = pid.
#   pdm_lat   PDM-Lite's lateral PID (lateral_controller.py / nav_planner.py:LateralPIDController): heading error to the point
#             that lies clip(0.9755 * v_kmh + 1.9153, 24, 105) / 10 m along the predicted waypoints (2.4-10.5 m, the expert's
#             own lookahead), gains kp 3.118 / ki 0.641 / kd 1.378, integral = mean of the last 6 errors. Speed = pid.
#   pdm       pdm_lat + pdm_long
#   brake_lat pdm_lat steering + pid speed with TF++'s brake rule (added 2026-11-20 after the screen: brake and pdm_lat helped
#             separately, pdm_long hurt; this is the combination of the two helpful parts)
CONTROLLERS = ("pid", "brake", "pdm_long", "pdm_lat", "pdm", "brake_lat")
PDM_LAT = dict(kp=3.118357247806046, kd=1.3782508892109167, ki=0.6406067986034124, speed_scale=0.9755321901954155,
               speed_offset=1.9152884533402488, min_points=24.0, max_points=105.0, points_per_meter=10.0, window=6)
PDM_LONG_PARAMS = (1.1990342347353184, -0.8057602384167799, 1.710818710950062, 0.921890257450335, 1.556497522998393,
                   -0.7013479734904027, 1.031266635497984)
PDM_LONG_MAX_ACCEL_KMH = 1.89      # speed error (km/h) above which PDM-Lite gives full throttle
PDM_BRAKE_RATIO = 1.1              # TF++ control_pid_direct's brake ratio, applied before get_throttle's own 1.031 ratio


def controller_mode(mode: Optional[str] = None) -> str:
    m = (mode if mode is not None else os.environ.get("WOR_CONTROLLER", "pid")).strip().lower() or "pid"
    if m not in CONTROLLERS:
        raise ValueError(f"WOR_CONTROLLER={m!r}: expected one of {', '.join(CONTROLLERS)}")
    return m


def pdm_lookahead_m(speed_kmh: float) -> float:
    """PDM-Lite's lookahead distance in metres (route points 10 cm apart: 24..105 points = 2.4..10.5 m)."""
    p = PDM_LAT
    return float(np.clip(p["speed_scale"] * speed_kmh + p["speed_offset"], p["min_points"], p["max_points"]) / p["points_per_meter"])


def point_along(waypoints: np.ndarray, dist_m: float) -> np.ndarray:
    """The point dist_m metres along the polyline origin -> waypoints[0] -> ... (linear interpolation); the last waypoint
    when the polyline is shorter than dist_m."""
    pts = np.vstack([np.zeros((1, 2)), np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < dist_m:
        return pts[-1]
    k = int(np.searchsorted(cum, dist_m, side="left"))
    k = min(max(k, 1), len(pts) - 1)
    s = seg[k - 1]
    t = 0.0 if s <= 1e-9 else (dist_m - cum[k - 1]) / s
    return pts[k - 1] + t * (pts[k] - pts[k - 1])


def pdm_throttle_brake(target_mps: float, speed_mps: float) -> Tuple[float, float]:
    """TF++ direct-mode longitudinal control (model.py:control_pid_direct + nav_planner.get_throttle), brake as 0/1."""
    if target_mps < 0.01 or speed_mps / max(target_mps, 1e-6) > PDM_BRAKE_RATIO:
        return 0.0, 1.0
    if target_mps < 1.0 / 3.6:            # avoid very small target speeds
        target_mps = 1.0 / 3.6
    v, tgt = speed_mps * 3.6, target_mps * 3.6
    err = tgt - v
    if err > PDM_LONG_MAX_ACCEL_KMH:
        return 1.0, 0.0
    if v / tgt > PDM_LONG_PARAMS[-1]:
        return 0.0, 1.0
    e = max(err, 0.0) / 100.0
    v /= 100.0
    f = np.array([v, v ** 2, 100 * e, e ** 2, v * e, v ** 2 * e])
    return float(np.clip(f @ np.array(PDM_LONG_PARAMS[:-1]), 0.0, 1.0)), 0.0


class PIDController:
    """
    Translates selected waypoint trajectory / Q-value paths into smooth CARLA vehicle controls.
    """
    def __init__(
        self,
        kp_steer: float = 0.75,
        ki_steer: float = 0.05,
        kd_steer: float = 0.15,
        kp_speed: float = 1.0,
        ki_speed: float = 0.05,
        kd_speed: float = 0.10,
        target_speed: float = 20.0,  # km/h
        mode: Optional[str] = None   # None: WOR_CONTROLLER (default "pid"), see CONTROLLERS above
    ):
        self.mode = controller_mode(mode)
        self._lat_window = []        # pdm_lat / pdm: the last PDM_LAT["window"] normalised heading errors
        self._mode_logged = False
        self.kp_steer = kp_steer
        self.ki_steer = ki_steer
        self.kd_steer = kd_steer

        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kd_speed = kd_speed
        self.target_speed = target_speed

        self.steer_error_integral = 0.0
        self.steer_error_prev = 0.0

        self.speed_error_integral = 0.0
        self.speed_error_prev = 0.0

    def reset(self):
        self.steer_error_integral = 0.0
        self.steer_error_prev = 0.0
        self.speed_error_integral = 0.0
        self.speed_error_prev = 0.0
        self._lat_window = []

    def _pdm_steer(self, waypoints: np.ndarray, current_speed_kmh: float) -> float:
        aim = point_along(waypoints, pdm_lookahead_m(current_speed_kmh))
        # a stopped car's waypoints collapse onto the origin, where the heading is noise: hold the wheel straight
        err = 0.0 if float(np.hypot(aim[0], aim[1])) < 0.5 else math.atan2(float(aim[1]), float(aim[0])) * 180.0 / math.pi / 90.0
        self._lat_window.append(err)
        self._lat_window = self._lat_window[-PDM_LAT["window"]:]
        deriv = 0.0 if len(self._lat_window) == 1 else self._lat_window[-1] - self._lat_window[-2]
        integ = float(np.mean(self._lat_window))
        return float(np.clip(PDM_LAT["kp"] * err + PDM_LAT["kd"] * deriv + PDM_LAT["ki"] * integ, -1.0, 1.0))

    def control_from_waypoints(
        self,
        waypoints: np.ndarray,
        current_speed_kmh: float,
        target_speed_kmh: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Computes (steer, throttle, brake) from predicted ego-frame waypoints.
        Waypoints shape: (N, 2) where (x_forward, y_lateral).
        """
        if target_speed_kmh is None:
            target_speed_kmh = self.target_speed
        if not self._mode_logged:   # one line per process: proves which controller a lane ran (grep "[controller]")
            print(f"[controller] WOR_CONTROLLER={self.mode}", flush=True)
            self._mode_logged = True

        steer, throttle, brake = self._pid_control(waypoints, current_speed_kmh, target_speed_kmh)
        if self.mode in ("pdm_lat", "pdm", "brake_lat"):
            steer = self._pdm_steer(waypoints, current_speed_kmh)
        if self.mode in ("pdm_long", "pdm"):
            throttle, brake = pdm_throttle_brake(target_speed_kmh / 3.6, current_speed_kmh / 3.6)
        elif self.mode in ("brake", "brake_lat"):
            throttle, brake = tfpp_brake_rule(throttle, brake, current_speed_kmh / 3.6, target_speed_kmh)
        return steer, throttle, brake

    def _pid_control(self, waypoints: np.ndarray, current_speed_kmh: float,
                     target_speed_kmh: float) -> Tuple[float, float, float]:
        """The original controller (WOR_CONTROLLER=pid). Its integrators also run under the other modes, unused, so
        switching modes never changes this path's state."""
        # 1. Lateral Control (Pure Pursuit / PID on aim waypoint)
        aim_point = waypoints[min(2, len(waypoints) - 1)]
        dx = float(aim_point[0])
        dy = float(aim_point[1])

        # Heading angle error to target waypoint
        desired_angle = math.atan2(dy, max(dx, 0.5))
        steer_error = desired_angle

        self.steer_error_integral = np.clip(self.steer_error_integral + steer_error, -1.0, 1.0)
        steer_deriv = steer_error - self.steer_error_prev
        self.steer_error_prev = steer_error

        steer = (
            self.kp_steer * steer_error +
            self.ki_steer * self.steer_error_integral +
            self.kd_steer * steer_deriv
        )
        steer = float(np.clip(steer, -1.0, 1.0))

        # 2. Longitudinal Control (Speed PID)
        speed_error = (target_speed_kmh - current_speed_kmh) / 3.6  # convert to m/s
        self.speed_error_integral = np.clip(self.speed_error_integral + speed_error, -10.0, 10.0)
        speed_deriv = speed_error - self.speed_error_prev
        self.speed_error_prev = speed_error

        accel = (
            self.kp_speed * speed_error +
            self.ki_speed * self.speed_error_integral +
            self.kd_speed * speed_deriv
        )

        if accel >= 0:
            throttle = float(np.clip(accel, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-accel, 0.0, 1.0))

        return steer, throttle, brake
