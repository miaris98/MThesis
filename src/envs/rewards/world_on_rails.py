"""World on Rails (WoR) reward function from Chen et al., ICCV 2021."""
from typing import Any, Dict, Tuple
from src.envs.rewards.base import BaseReward


class WorldOnRailsReward(BaseReward):
    """
    Official World on Rails (WoR) Reward & Cost Formulation (Chen et al., ICCV 2021).
    
    Combines:
    1. Forward Progress along target rails / route: v_ego * dt * cos(heading_error)
    2. Severe Collision Penalty: -25.0 (immediate episode termination)
    3. Traffic Light Violation Penalty: -20.0 (charged upon entering junction on red)
    4. Off-Road / Drivable Lane Penalty: -20.0 (triggered when lateral deviation exceeds lane boundary)
    5. Lane-Centering Cost: -0.20 * (|d_lateral| / (lane_width / 2))^2
    6. Anti-Stall Guard: -15.0 after 80 steps below 2 km/h
    """
    NAME = "wor"
    SOURCE = "Learning to drive from a world on rails (Chen et al., ICCV 2021)"

    PROGRESS_WEIGHT = 0.50          # Progress reward scaling per meter
    COLLISION_PENALTY = -25.0       # Terminal collision cost
    RED_LIGHT_PENALTY = -20.0       # Red light infraction cost
    OFF_ROAD_PENALTY = -20.0        # Off-road boundary termination cost
    LANE_CENTERING_WEIGHT = -0.20   # Centering quadratic penalty
    STALL_TERMINAL_PENALTY = -15.0  # Terminal stall cost

    STALL_TIMEOUT_STEPS = 80
    STALL_SPEED_KMH = 2.0

    def __init__(self, **kwargs: Any):
        super().__init__()
        self._light_latched = False

    def reset_episode_tracking(self) -> None:
        """Reset event latches for a new episode."""
        super().reset_episode_tracking()
        self._light_latched = False

    def compute_reward(
        self,
        state: Dict[str, Any],
        curriculum_factor: float = 1.0,
        dt: float = 0.05
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates the step reward matching the World on Rails Markov Decision Process.
        """
        speed_kmh = float(state.get("speed_kmh", 0.0))
        heading_cos = float(state.get("heading_cos", 1.0))
        lateral_dist = float(state.get("lateral_dist", 0.0))
        lane_width = float(state.get("lane_width", 3.5))
        is_at_red_light = bool(state.get("is_at_red_light", False))
        is_collision = bool(state.get("is_collision", False))
        is_off_road = bool(state.get("is_off_road", False))
        is_junction = bool(state.get("is_junction", False))
        brake = float(state.get("brake", 0.0))
        time_step = int(state.get("time_step", 0))

        # 1. Forward Progress along Road Rails
        forward_speed_ms = (speed_kmh / 3.6) * max(0.0, heading_cos)
        r_progress = self.PROGRESS_WEIGHT * forward_speed_ms * dt

        # Zero out progress reward if actively running a red light
        running_red = is_at_red_light and speed_kmh >= self.STALL_SPEED_KMH and brake <= 0.2
        if running_red:
            r_progress = 0.0

        # 2. Red Light Violation
        r_light = 0.0
        if running_red and not self._light_latched:
            self._light_latched = True
            r_light = self.RED_LIGHT_PENALTY

        # 3. Continuous Lane-Centering Penalty (Exempt inside junctions)
        r_lane = 0.0
        if not is_junction and not is_off_road:
            half_width = max(1.0, lane_width / 2.0)
            norm_dist = min(1.5, abs(lateral_dist) / half_width)
            r_lane = self.LANE_CENTERING_WEIGHT * (norm_dist ** 2)

        # 4. Terminal Events (Collision, Off-Road, Stall)
        r_terminal = 0.0
        is_stalled = False
        if is_collision:
            r_terminal = self.COLLISION_PENALTY
        elif is_off_road:
            r_terminal = self.OFF_ROAD_PENALTY
        else:
            # Anti-stall check
            is_stalled = self._track_stall(
                speed_kmh=speed_kmh,
                time_step=time_step,
                exempt=(is_at_red_light or float(state.get("min_obs_dist", 99.0)) < 10.0),
                timeout_steps=self.STALL_TIMEOUT_STEPS,
                stall_speed_kmh=self.STALL_SPEED_KMH
            )
            if is_stalled:
                r_terminal = self.STALL_TERMINAL_PENALTY

        # Total composite reward
        raw_reward = r_progress + r_lane + r_light + r_terminal

        return raw_reward, self._blank_info(
            r_progress=r_progress,
            r_lane=r_lane,
            r_light=r_light,
            r_obstacle=0.0,
            r_ttc=0.0,
            r_terminal=r_terminal,
            is_stalled=is_stalled,
        )
