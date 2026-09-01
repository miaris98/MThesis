"""gym-carla reward (Chen et al.), as consumed by papers_and_code/interp-e2e-driving."""
from typing import Any, Dict, Tuple

from src.envs.rewards.base import BaseReward


class InterpE2EReward(BaseReward):
    """
    The gym-carla reward used by "Interpretable End-to-End Urban Autonomous Driving with
    Latent Deep Reinforcement Learning" (Chen, Li, Tomizuka et al.).

    Original weighted sum:
        r = 200*r_collision + v_lon + 10*r_fast + r_out + 5*r_steer + 0.2*r_lat - 0.1
    with
        r_collision = -1 on contact
        v_lon       = longitudinal speed in m/s (the dense driver)
        r_fast      = -1 when above the desired speed
        r_out       = -1 when outside the lane
        r_steer     = -steer^2
        r_lat       = -|steer| * v_lon
        -0.1        = constant per-step cost discouraging idling

    NOTE ON PROVENANCE: papers_and_code/interp-e2e-driving vendors only the training script
    and agents; the reward itself lives in the external `gym_carla` package, which is NOT in
    this repository. This is reconstructed from the published formulation rather than copied
    from vendored source, so treat the constants as the paper's rather than verified-in-repo.
    """
    NAME = "interp_e2e"
    SOURCE = "Chen et al., gym-carla (reconstructed from paper; gym_carla not vendored)"

    W_COLLISION = 200.0
    W_FAST = 10.0
    W_OUT = 1.0
    W_STEER = 5.0
    W_LAT = 0.2
    STEP_COST = -0.1

    STALL_TIMEOUT_STEPS = 80
    STALL_SPEED_KMH = 2.0

    def compute_reward(
        self, state: Dict[str, Any], curriculum_factor: float = 1.0, dt: float = 0.05
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate the weighted-sum reward; components are mapped onto the shared telemetry slots."""
        speed_kmh = float(state.get("speed_kmh", 0.0))
        heading_cos = float(state.get("heading_cos", 1.0))
        steer = float(state.get("steer", 0.0))
        is_at_red_light = bool(state.get("is_at_red_light", False))
        min_obs_dist = float(state.get("min_obs_dist", 99.0))
        is_collision = bool(state.get("is_collision", False))
        is_off_road = bool(state.get("is_off_road", False))
        time_step = int(state.get("time_step", 0))

        # Longitudinal speed along the lane, in m/s - the paper's dense driving signal.
        v_lon = (speed_kmh / 3.6) * max(0.0, heading_cos)

        # Desired speed is held in km/h by this project; the paper caps speed rather than
        # tracking it, penalizing only the overshoot.
        r_fast = -1.0 if speed_kmh > self.desired_speed else 0.0
        r_collision = -1.0 if is_collision else 0.0
        r_out = -1.0 if is_off_road else 0.0
        r_steer = -(steer ** 2)
        r_lat = -abs(steer) * v_lon

        is_stalled = self._track_stall(
            speed_kmh, time_step, exempt=(is_at_red_light or min_obs_dist < 10.0),
            timeout_steps=self.STALL_TIMEOUT_STEPS, stall_speed_kmh=self.STALL_SPEED_KMH
        )

        r_terminal = self.W_COLLISION * r_collision
        r_progress = v_lon + self.STEP_COST
        r_lane = self.W_OUT * r_out + self.W_STEER * r_steer + self.W_LAT * r_lat
        r_obstacle = self.W_FAST * r_fast

        alpha = curriculum_factor
        total = r_progress + (r_lane + r_obstacle + r_terminal) * alpha

        return total, self._blank_info(
            r_progress=r_progress, r_lane=r_lane, r_obstacle=r_obstacle,
            r_terminal=r_terminal, is_stalled=is_stalled
        )
