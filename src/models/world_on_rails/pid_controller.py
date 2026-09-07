"""Waypoint-to-control conversion for World on Rails policies.

Split out of wor_policy.py: this is a classical controller, not a network, and both
the CNN-headed and Qwen-headed policies hand it the same (N, 2) ego-frame waypoints.
Kept importable from wor_policy for backwards compatibility.
"""
import math
from typing import Optional, Tuple

import numpy as np


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
        target_speed: float = 20.0  # km/h
    ):
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
