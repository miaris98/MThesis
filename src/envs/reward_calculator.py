"""Progress-and-violations driving reward: dense forward progress plus event-triggered penalties."""
from typing import Dict, Tuple, Any


class RewardCalculator:
    """
    Minimal two-signal reward for autonomous driving in CARLA.

    1. A small dense reward for verified forward progress along the lane (speed projected onto
       the lane tangent, integrated over time). Standing still earns exactly zero - never
       negative, never positive - so idling is reward-neutral rather than a safe local optimum.
    2. One-time/event penalties for violations: collision, off-road (including wrong-way and
       lane departure, flagged upstream by DrivingStateExtractor), stalling, running a red
       light, and unsafe proximity to obstacles/pedestrians. Legal, non-violating behavior
       (e.g. correctly stopping at a red light) is reward-neutral, not reward-positive.

    This intentionally drops the previous dense shaping terms (lane-centering well, heading
    alignment, steering/throttle smoothness) that let a stationary vehicle accumulate reward
    without moving.
    """
    PROGRESS_PER_METER = 1.0

    COLLISION_PENALTY = -25.0
    OFF_ROAD_PENALTY = -20.0

    STALL_GRACE_STEPS = 40
    STALL_TIMEOUT_STEPS = 30
    STALL_SPEED_KMH = 2.0
    STALL_PENALTY = -15.0

    RED_LIGHT_VIOLATION_PENALTY = -3.0

    OBSTACLE_DANGER_DIST = 10.0
    OBSTACLE_VIOLATION_PENALTY = -3.0

    TTC_DANGER_SECONDS = 2.0
    TTC_VIOLATION_PENALTY = -2.0

    def __init__(self, desired_speed: float = 25.0):
        # Retained for API/CLI compatibility; pace is no longer part of the reward signal.
        self.desired_speed = desired_speed
        self.stalled_steps = 0

    def reset_episode_tracking(self) -> None:
        """Reset stall tracking for a fresh episode."""
        self.stalled_steps = 0

    def compute_reward(self, state: Dict[str, Any], curriculum_factor: float = 1.0, dt: float = 0.05) -> Tuple[float, Dict[str, Any]]:
        """Calculate total step reward (progress - violations) and its decomposition."""
        speed_kmh = float(state.get("speed_kmh", 0.0))
        heading_cos = float(state.get("heading_cos", 1.0))
        throttle = float(state.get("throttle", 0.0))
        brake = float(state.get("brake", 0.0))
        is_at_red_light = bool(state.get("is_at_red_light", False))
        min_obs_dist = float(state.get("min_obs_dist", 99.0))
        is_pedestrian = bool(state.get("is_pedestrian", False))
        ttc_seconds = float(state.get("ttc_seconds", 99.0))
        is_collision = bool(state.get("is_collision", False))
        is_off_road = bool(state.get("is_off_road", False))
        time_step = int(state.get("time_step", 0))

        # 1. Dense forward-progress reward: distance advanced along the lane tangent this step.
        forward_speed_ms = (speed_kmh / 3.6) * max(0.0, heading_cos)
        r_progress = self.PROGRESS_PER_METER * forward_speed_ms * dt

        # 2. Traffic light compliance: a legal stop is neutral; running the light is a violation.
        r_light = 0.0
        if is_at_red_light:
            if speed_kmh < self.STALL_SPEED_KMH or brake > 0.2:
                self.stalled_steps = 0
            else:
                r_light = self.RED_LIGHT_VIOLATION_PENALTY
                r_progress = 0.0

        # 3. Obstacle proximity violation: only when accelerating into danger, not while braking.
        r_obstacle = 0.0
        if min_obs_dist < self.OBSTACLE_DANGER_DIST and throttle > 0.2 and brake <= 0.2:
            danger_scale = 1.0 - (min_obs_dist / self.OBSTACLE_DANGER_DIST)
            multiplier = 2.0 if is_pedestrian else 1.0
            r_obstacle = self.OBSTACLE_VIOLATION_PENALTY * (danger_scale ** 2) * multiplier

        r_ttc = 0.0
        if ttc_seconds < self.TTC_DANGER_SECONDS:
            danger_scale = (self.TTC_DANGER_SECONDS - ttc_seconds) / self.TTC_DANGER_SECONDS
            r_ttc = self.TTC_VIOLATION_PENALTY * (danger_scale ** 2)

        # 4. Stall detection: no forward progress for STALL_TIMEOUT_STEPS after a startup grace period.
        if time_step > self.STALL_GRACE_STEPS and not is_at_red_light and min_obs_dist >= self.OBSTACLE_DANGER_DIST:
            if speed_kmh < self.STALL_SPEED_KMH:
                self.stalled_steps += 1
            else:
                self.stalled_steps = 0
        is_stalled = bool(self.stalled_steps >= self.STALL_TIMEOUT_STEPS)

        # 5. Terminal violations, softened during curriculum warmup.
        r_terminal = 0.0
        if is_collision:
            r_terminal = self.COLLISION_PENALTY
        elif is_off_road:
            r_terminal = self.OFF_ROAD_PENALTY
        elif is_stalled:
            r_terminal = self.STALL_PENALTY

        alpha = curriculum_factor
        r_light_s = r_light * alpha
        r_obstacle_s = r_obstacle * alpha
        r_ttc_s = r_ttc * alpha
        r_terminal_s = r_terminal * alpha

        total_reward = r_progress + r_light_s + r_obstacle_s + r_ttc_s + r_terminal_s

        sub_info = {
            "r_progress": r_progress,
            "r_light": r_light,
            "r_obstacle": r_obstacle,
            "r_ttc": r_ttc,
            "r_terminal": r_terminal,
            "is_stalled": is_stalled
        }
        return total_reward, sub_info
