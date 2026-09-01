"""Roach reward (Zhang et al., ICCV 2021) - the RL coach behind TCP and ThinkTwice."""
from typing import Any, Dict, Tuple

from src.envs.rewards.base import BaseReward


class RoachReward(BaseReward):
    """
    Reward from "End-to-End Urban Driving by Imitating a Reinforcement Learning Coach"
    (Zhang, Liniger, Dai, Yu, Van Gool - ICCV 2021), whose agent supervises both TCP and
    ThinkTwice in papers_and_code.

    Formulation: r = r_speed + r_position + r_rotation + r_action + r_terminal
        r_speed    : 1 - |v - v_desired| / v_max, where v_desired collapses to 0 when a
                     hazard (red light, obstacle) is present - so stopping for a hazard is
                     rewarded exactly as much as cruising is when the road is clear.
        r_position : -0.5 * |lateral deviation from lane centre|
        r_rotation : -|heading error| in radians
        r_action   : -0.1 when steering exceeds the comfort envelope
        r_terminal : -1 on collision / blocked / route deviation

    NOTE ON PROVENANCE: papers_and_code/TCP/roach and .../ThinkTwice/roach vendor only the
    PPO buffer and the birdview wrapper, which delegate to `carla_gym` - not present in this
    repository. Reconstructed from the paper, not copied from vendored source.

    The distinguishing idea versus Custom_1 is the hazard-aware speed target: rather than
    penalizing violations, Roach rewrites what "correct speed" means, so a correct stop and
    correct cruising are equally valuable and the agent is never torn between them.
    """
    NAME = "roach"
    SOURCE = "Zhang et al. ICCV 2021 (reconstructed from paper; carla_gym not vendored)"

    MAX_SPEED_KMH = 40.0
    HAZARD_DISTANCE = 10.0
    W_POSITION = -0.5
    W_ROTATION = -1.0
    STEER_COMFORT_LIMIT = 0.7
    ACTION_PENALTY = -0.1
    TERMINAL_PENALTY = -1.0

    STALL_TIMEOUT_STEPS = 80
    STALL_SPEED_KMH = 2.0

    def compute_reward(
        self, state: Dict[str, Any], curriculum_factor: float = 1.0, dt: float = 0.05
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate the Roach reward with a hazard-dependent desired speed."""
        speed_kmh = float(state.get("speed_kmh", 0.0))
        heading_cos = float(state.get("heading_cos", 1.0))
        lateral_dist = abs(float(state.get("lateral_dist", 0.0)))
        steer = float(state.get("steer", 0.0))
        is_at_red_light = bool(state.get("is_at_red_light", False))
        min_obs_dist = float(state.get("min_obs_dist", 99.0))
        ttc_seconds = float(state.get("ttc_seconds", 99.0))
        is_collision = bool(state.get("is_collision", False))
        is_off_road = bool(state.get("is_off_road", False))
        time_step = int(state.get("time_step", 0))

        # A hazard rewrites the target speed to zero: stopping then scores as highly as
        # cruising does on clear road, instead of being merely un-penalized.
        hazard = is_at_red_light or min_obs_dist < self.HAZARD_DISTANCE or ttc_seconds < 2.0
        v_desired = 0.0 if hazard else self.desired_speed

        r_speed = 1.0 - abs(speed_kmh - v_desired) / self.MAX_SPEED_KMH
        r_speed = max(-1.0, min(1.0, r_speed))

        r_position = self.W_POSITION * lateral_dist

        # heading_cos = cos(heading error); recover the absolute angle in radians.
        clamped = max(-1.0, min(1.0, heading_cos))
        heading_error = (1.0 - clamped) * 1.5708  # 0 rad when aligned, pi/2 when perpendicular
        r_rotation = self.W_ROTATION * heading_error

        r_action = self.ACTION_PENALTY if abs(steer) > self.STEER_COMFORT_LIMIT else 0.0

        is_stalled = self._track_stall(
            speed_kmh, time_step, exempt=hazard,
            timeout_steps=self.STALL_TIMEOUT_STEPS, stall_speed_kmh=self.STALL_SPEED_KMH
        )

        r_terminal = 0.0
        if is_collision or is_off_road or is_stalled:
            r_terminal = self.TERMINAL_PENALTY

        alpha = curriculum_factor
        total = r_speed + (r_position + r_rotation + r_action + r_terminal) * alpha

        info = self._blank_info(
            r_progress=r_speed, r_lane=r_position + r_action,
            r_light=r_rotation, r_terminal=r_terminal, is_stalled=is_stalled
        )
        info["v_desired"] = v_desired
        return total, info
