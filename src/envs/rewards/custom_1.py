"""Custom_1: progress-and-violations reward - dense forward progress plus event penalties."""
from typing import Any, Dict, Tuple

from src.envs.rewards.base import BaseReward


class Custom1Reward(BaseReward):
    """
    This project's own formulation: one positive signal (verified forward progress) and a
    set of penalties, with no term that pays out while stationary.

    Scale rationale, derived from the 10k-step telemetry run:
    at ~20 km/h a step earns roughly +0.23, so a 100-step clean episode is worth about +23
    while a single off-road costs -8. The failure is recoverable in ~35 steps of good
    driving, which keeps a usable gradient toward "drive further without violating".

    The earlier calibration (progress 0.2, off-road -20) made this unwinnable: breaking even
    on one off-road needed ~217 consecutive clean steps against a median episode of 10, so
    every episode was dominated by the terminal penalty and the advantage signal was noise.
    """
    NAME = "custom_1"
    SOURCE = "this project"

    PROGRESS_PER_METER = 1.0

    # Lane discipline. At the typical observed deviation (~0.55 m) this costs about 20% of
    # the progress reward; at the lane edge it exceeds it, so drifting stops being profitable
    # well before the off-road cliff rather than only at it.
    LANE_CENTER_PENALTY = -0.5
    MIN_HALF_LANE = 1.0

    COLLISION_PENALTY = -10.0
    OFF_ROAD_PENALTY = -8.0

    STALL_GRACE_STEPS = 40
    STALL_TIMEOUT_STEPS = 80
    STALL_SPEED_KMH = 2.0
    STALL_PENALTY = -5.0

    RED_LIGHT_VIOLATION_PENALTY = -3.0

    OBSTACLE_DANGER_DIST = 10.0
    OBSTACLE_VIOLATION_PENALTY = -3.0

    TTC_DANGER_SECONDS = 2.0
    TTC_VIOLATION_PENALTY = -2.0

    def compute_reward(
        self, state: Dict[str, Any], curriculum_factor: float = 1.0, dt: float = 0.05
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate total step reward (progress minus violations) and its decomposition."""
        speed_kmh = float(state.get("speed_kmh", 0.0))
        heading_cos = float(state.get("heading_cos", 1.0))
        lateral_dist = abs(float(state.get("lateral_dist", 0.0)))
        lane_width = float(state.get("lane_width", 3.5))
        throttle = float(state.get("throttle", 0.0))
        brake = float(state.get("brake", 0.0))
        is_at_red_light = bool(state.get("is_at_red_light", False))
        is_junction = bool(state.get("is_junction", False))
        min_obs_dist = float(state.get("min_obs_dist", 99.0))
        is_pedestrian = bool(state.get("is_pedestrian", False))
        ttc_seconds = float(state.get("ttc_seconds", 99.0))
        is_collision = bool(state.get("is_collision", False))
        is_off_road = bool(state.get("is_off_road", False))
        time_step = int(state.get("time_step", 0))

        # 1. Dense forward progress: metres advanced along the lane tangent this step.
        forward_speed_ms = (speed_kmh / 3.6) * max(0.0, heading_cos)
        r_progress = self.PROGRESS_PER_METER * forward_speed_ms * dt

        # 2. Lane discipline: quadratic in normalized lateral deviation. Junctions are exempt
        #    because turns legitimately carry the ego far from any single lane centreline.
        r_lane = 0.0
        if not is_junction:
            half_lane = max(self.MIN_HALF_LANE, lane_width / 2.0)
            r_lane = self.LANE_CENTER_PENALTY * ((lateral_dist / half_lane) ** 2)

        # 3. Traffic light: a legal stop is neutral, running the light is a violation.
        r_light = 0.0
        if is_at_red_light and speed_kmh >= self.STALL_SPEED_KMH and brake <= 0.2:
            r_light = self.RED_LIGHT_VIOLATION_PENALTY
            r_progress = 0.0

        # 4. Obstacle proximity: only when accelerating into danger, not while braking.
        r_obstacle = 0.0
        near_obstacle = min_obs_dist < self.OBSTACLE_DANGER_DIST
        if near_obstacle and throttle > 0.2 and brake <= 0.2:
            danger = 1.0 - (min_obs_dist / self.OBSTACLE_DANGER_DIST)
            r_obstacle = self.OBSTACLE_VIOLATION_PENALTY * (danger ** 2) * (2.0 if is_pedestrian else 1.0)

        r_ttc = 0.0
        if ttc_seconds < self.TTC_DANGER_SECONDS:
            danger = (self.TTC_DANGER_SECONDS - ttc_seconds) / self.TTC_DANGER_SECONDS
            r_ttc = self.TTC_VIOLATION_PENALTY * (danger ** 2)

        # 5. Stall detection, exempt while legitimately held up.
        is_stalled = self._track_stall(
            speed_kmh, time_step, exempt=(is_at_red_light or near_obstacle),
            grace_steps=self.STALL_GRACE_STEPS, timeout_steps=self.STALL_TIMEOUT_STEPS,
            stall_speed_kmh=self.STALL_SPEED_KMH
        )

        # 6. Terminal violations, softened during curriculum warmup.
        r_terminal = 0.0
        if is_collision:
            r_terminal = self.COLLISION_PENALTY
        elif is_off_road:
            r_terminal = self.OFF_ROAD_PENALTY
        elif is_stalled:
            r_terminal = self.STALL_PENALTY

        alpha = curriculum_factor
        total = r_progress + (r_lane + r_light + r_obstacle + r_ttc + r_terminal) * alpha

        return total, self._blank_info(
            r_progress=r_progress, r_lane=r_lane, r_light=r_light,
            r_obstacle=r_obstacle, r_ttc=r_ttc, r_terminal=r_terminal, is_stalled=is_stalled
        )
