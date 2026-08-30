"""Progress-and-violations driving reward: dense forward progress plus event-triggered penalties."""
from typing import Dict, Tuple, Any


class RewardCalculator:
    """
    Minimal driving reward built from one positive signal and a set of penalties.

    1. A small dense reward for verified forward progress along the lane (speed projected onto
       the lane tangent, integrated over time). Driving sideways or backwards earns exactly
       zero, so no amount of spinning or wall-riding pays out.
    2. A continuous lane-discipline penalty proportional to squared lateral deviation, which
       is what stops the "full-throttle hard-left lock" trap: holding a constant steering
       lock walks the ego off the lane centre and bleeds reward long before it crashes.
    3. Event penalties for violations: collision, off-road (including wrong-way and boundary
       crossing, flagged upstream by DrivingStateExtractor), stalling, running a red light,
       and unsafe proximity to obstacles/pedestrians.

    No term is ever positive except progress, so standing still cannot be farmed for reward.
    The old dense shaping terms (heading alignment, steering/throttle smoothness, comfort)
    stay removed - they were what let a stationary vehicle accumulate reward without moving.
    """
    # Progress is deliberately small: ~0.1/step at 36 km/h, so a single collision (-25)
    # outweighs roughly 250 steps of perfect driving.
    PROGRESS_PER_METER = 0.2

    # Lane discipline. Scaled so that hugging the lane edge (|d_lat| ~= half lane width)
    # roughly cancels the progress reward, while small deviations stay near-free.
    LANE_CENTER_PENALTY = -0.15
    MIN_HALF_LANE = 1.0

    COLLISION_PENALTY = -25.0
    OFF_ROAD_PENALTY = -20.0

    STALL_GRACE_STEPS = 40
    STALL_TIMEOUT_STEPS = 80
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

        # 1. Dense forward-progress reward: distance advanced along the lane tangent this step.
        forward_speed_ms = (speed_kmh / 3.6) * max(0.0, heading_cos)
        r_progress = self.PROGRESS_PER_METER * forward_speed_ms * dt

        # 2. Lane discipline: quadratic penalty on normalized lateral deviation from lane centre.
        #    Junctions are exempt - lane geometry there is ambiguous and turns legitimately
        #    carry the ego far from any single lane's centreline.
        r_lane = 0.0
        if not is_junction:
            half_lane = max(self.MIN_HALF_LANE, lane_width / 2.0)
            normalized_offset = lateral_dist / half_lane
            r_lane = self.LANE_CENTER_PENALTY * (normalized_offset ** 2)

        # 3. Traffic light compliance: a legal stop is neutral; running the light is a violation.
        r_light = 0.0
        if is_at_red_light:
            if speed_kmh < self.STALL_SPEED_KMH or brake > 0.2:
                self.stalled_steps = 0
            else:
                r_light = self.RED_LIGHT_VIOLATION_PENALTY
                r_progress = 0.0

        # 4. Obstacle proximity violation: only when accelerating into danger, not while braking.
        r_obstacle = 0.0
        if min_obs_dist < self.OBSTACLE_DANGER_DIST and throttle > 0.2 and brake <= 0.2:
            danger_scale = 1.0 - (min_obs_dist / self.OBSTACLE_DANGER_DIST)
            multiplier = 2.0 if is_pedestrian else 1.0
            r_obstacle = self.OBSTACLE_VIOLATION_PENALTY * (danger_scale ** 2) * multiplier

        r_ttc = 0.0
        if ttc_seconds < self.TTC_DANGER_SECONDS:
            danger_scale = (self.TTC_DANGER_SECONDS - ttc_seconds) / self.TTC_DANGER_SECONDS
            r_ttc = self.TTC_VIOLATION_PENALTY * (danger_scale ** 2)

        # 5. Stall detection: no forward progress for STALL_TIMEOUT_STEPS after a startup grace period.
        if time_step > self.STALL_GRACE_STEPS and not is_at_red_light and min_obs_dist >= self.OBSTACLE_DANGER_DIST:
            if speed_kmh < self.STALL_SPEED_KMH:
                self.stalled_steps += 1
            else:
                self.stalled_steps = 0
        is_stalled = bool(self.stalled_steps >= self.STALL_TIMEOUT_STEPS)

        # 6. Terminal violations, softened during curriculum warmup.
        r_terminal = 0.0
        if is_collision:
            r_terminal = self.COLLISION_PENALTY
        elif is_off_road:
            r_terminal = self.OFF_ROAD_PENALTY
        elif is_stalled:
            r_terminal = self.STALL_PENALTY

        alpha = curriculum_factor
        total_reward = (
            r_progress
            + (r_lane + r_light + r_obstacle + r_ttc + r_terminal) * alpha
        )

        sub_info = {
            "r_progress": r_progress,
            "r_lane": r_lane,
            "r_light": r_light,
            "r_obstacle": r_obstacle,
            "r_ttc": r_ttc,
            "r_terminal": r_terminal,
            "is_stalled": is_stalled
        }
        return total_reward, sub_info
