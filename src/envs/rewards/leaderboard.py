"""CARLA Leaderboard driving-score reward: route completion scaled by multiplicative infractions."""
from typing import Any, Dict, Tuple

from src.envs.rewards.base import BaseReward


class LeaderboardReward(BaseReward):
    """
    Dense adaptation of the official CARLA Leaderboard driving score.

    The offline metric is `driving_score = route_completion x prod(infraction_penalties)`,
    where each infraction multiplies a running penalty coefficient. Constants are taken
    verbatim from the vendored leaderboard implementation
    (papers_and_code/plant/leaderboard/leaderboard/utils/statistics_manager.py).

    Turning an episode-level product into a per-step reward: the agent is paid for each
    metre of route progress, scaled by the penalty coefficient accumulated so far. An early
    infraction therefore devalues the entire remainder of the episode, which reproduces the
    metric's defining property - infractions are multiplicative, not additive, so they are
    punished in proportion to how much driving they contaminate.
    """
    NAME = "leaderboard"
    SOURCE = "CARLA Leaderboard driving score (vendored in papers_and_code/plant, also LEAD)"

    PENALTY_COLLISION_PEDESTRIAN = 0.50
    PENALTY_COLLISION_VEHICLE = 0.60
    PENALTY_COLLISION_STATIC = 0.65
    PENALTY_TRAFFIC_LIGHT = 0.70
    PENALTY_STOP = 0.80

    # Route completion is measured in metres travelled; this converts it to reward units.
    PROGRESS_PER_METER = 1.0
    # The offline metric simply ends the route on failure. For RL we still need a terminal
    # signal, kept small because the multiplicative coefficient carries most of the cost.
    TERMINAL_PENALTY = -5.0
    OFF_ROAD_LANE_PENALTY = 0.65

    STALL_TIMEOUT_STEPS = 80
    STALL_SPEED_KMH = 2.0

    def reset_episode_tracking(self) -> None:
        """Reset the running infraction coefficient and event latches for a new route."""
        super().reset_episode_tracking()
        self.score_penalty = 1.0
        self._collision_latched = False
        self._light_latched = False
        self._off_road_latched = False

    def compute_reward(
        self, state: Dict[str, Any], curriculum_factor: float = 1.0, dt: float = 0.05
    ) -> Tuple[float, Dict[str, Any]]:
        """Pay for route progress scaled by the infraction coefficient accumulated so far."""
        speed_kmh = float(state.get("speed_kmh", 0.0))
        heading_cos = float(state.get("heading_cos", 1.0))
        brake = float(state.get("brake", 0.0))
        is_at_red_light = bool(state.get("is_at_red_light", False))
        is_pedestrian = bool(state.get("is_pedestrian", False))
        min_obs_dist = float(state.get("min_obs_dist", 99.0))
        is_collision = bool(state.get("is_collision", False))
        is_off_road = bool(state.get("is_off_road", False))
        time_step = int(state.get("time_step", 0))

        # Infractions latch once per episode so a multi-frame contact is charged a single time.
        r_terminal = 0.0
        if is_collision and not self._collision_latched:
            self._collision_latched = True
            if is_pedestrian:
                coeff = self.PENALTY_COLLISION_PEDESTRIAN
            elif min_obs_dist < 10.0:
                coeff = self.PENALTY_COLLISION_VEHICLE
            else:
                coeff = self.PENALTY_COLLISION_STATIC
            self.score_penalty *= coeff
            r_terminal = self.TERMINAL_PENALTY

        r_light = 0.0
        running_red = is_at_red_light and speed_kmh >= self.STALL_SPEED_KMH and brake <= 0.2
        if running_red and not self._light_latched:
            self._light_latched = True
            self.score_penalty *= self.PENALTY_TRAFFIC_LIGHT
            r_light = self.TERMINAL_PENALTY * 0.2

        # OUTSIDE_ROUTE_LANES in the official metric; charged once on leaving the lane.
        r_lane = 0.0
        if is_off_road and not self._off_road_latched:
            self._off_road_latched = True
            self.score_penalty *= self.OFF_ROAD_LANE_PENALTY
            r_lane = self.TERMINAL_PENALTY * 0.5
            if r_terminal == 0.0:
                r_terminal = self.TERMINAL_PENALTY

        # Route completion for this step, devalued by every infraction so far.
        forward_speed_ms = (speed_kmh / 3.6) * max(0.0, heading_cos)
        r_progress = self.PROGRESS_PER_METER * forward_speed_ms * dt * self.score_penalty

        is_stalled = self._track_stall(
            speed_kmh, time_step, exempt=(is_at_red_light or min_obs_dist < 10.0),
            timeout_steps=self.STALL_TIMEOUT_STEPS, stall_speed_kmh=self.STALL_SPEED_KMH
        )
        if is_stalled and r_terminal == 0.0:
            # VEHICLE_BLOCKED: the offline metric fails the route outright.
            r_terminal = self.TERMINAL_PENALTY

        alpha = curriculum_factor
        total = r_progress + (r_lane + r_light + r_terminal) * alpha

        info = self._blank_info(
            r_progress=r_progress, r_lane=r_lane, r_light=r_light,
            r_terminal=r_terminal, is_stalled=is_stalled
        )
        info["score_penalty"] = self.score_penalty
        return total, info
