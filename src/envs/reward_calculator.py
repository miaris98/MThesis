"""Multi-objective driving reward function with Gaussian potential wells and curriculum gating."""
import math
from typing import Dict, Tuple, Any


class RewardCalculator:
    """
    Computes dense multi-objective reinforcement learning reward for autonomous driving in CARLA.
    Balances lane centering, heading alignment, target speed, comfort, obstacle avoidance, and traffic laws.
    """
    def __init__(self, desired_speed: float = 25.0):
        self.desired_speed = desired_speed
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.stalled_steps = 0

    def reset_episode_tracking(self) -> None:
        """Reset step-differential trackers for a fresh episode."""
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.stalled_steps = 0

    def compute_reward(self, state: Dict[str, Any], curriculum_factor: float = 1.0) -> Tuple[float, Dict[str, Any]]:
        """Calculate total step reward and sub-reward decomposition dictionary."""
        speed_kmh = float(state.get("speed_kmh", 0.0))
        heading_cos = float(state.get("heading_cos", 1.0))
        heading_cos_far = float(state.get("heading_cos_far", 1.0))
        lateral_dist = float(state.get("lateral_dist", 0.0))
        curve_factor = float(state.get("curve_factor", 1.0))
        is_junction = bool(state.get("is_junction", False))
        
        steer = float(state.get("steer", 0.0))
        throttle = float(state.get("throttle", 0.0))
        brake = float(state.get("brake", 0.0))
        is_at_red_light = bool(state.get("is_at_red_light", False))
        
        min_obs_dist = float(state.get("min_obs_dist", 99.0))
        is_pedestrian = bool(state.get("is_pedestrian", False))
        ttc_seconds = float(state.get("ttc_seconds", 99.0))
        
        is_collision = bool(state.get("is_collision", False))
        is_off_road = bool(state.get("is_off_road", False))
        time_step = int(state.get("time_step", 0))

        # 1. Dual-Horizon Heading alignment [-0.5, +0.5]
        r_heading = 0.35 * max(-1.0, min(1.0, heading_cos)) + 0.15 * max(-1.0, min(1.0, heading_cos_far))
        
        # 2. Gaussian Lane Potential Well [-2.0, +1.0]
        lat_norm = min(3.0, max(0.0, lateral_dist))
        r_lateral = 1.0 * math.exp(-(lat_norm ** 2) / (2.0 * (0.45 ** 2))) - 0.5 * min(4.0, lat_norm ** 2)
        r_lateral = max(-2.0, min(1.0, r_lateral))
        r_boundary = -1.5 * min(4.0, (max(0.0, lat_norm - 0.9) ** 2))

        # 3. Speed & Velocity Progress along Lane Tangent [0.0, +1.5]
        adaptive_target_speed = self.desired_speed * curve_factor
        if is_junction:
            adaptive_target_speed = min(adaptive_target_speed, 15.0)
            
        v_proj = speed_kmh * max(0.0, heading_cos)
        speed_diff = abs(v_proj - adaptive_target_speed)
        lane_centering_gate = max(0.0, 1.0 - (lat_norm / 1.5)) * max(0.0, heading_cos)
        
        if not is_at_red_light:
            if speed_diff <= 3.0:
                raw_speed_r = 1.5
            else:
                raw_speed_r = 1.5 * max(0.0, 1.0 - (speed_diff - 3.0) / max(10.0, adaptive_target_speed))
            r_speed = raw_speed_r * lane_centering_gate
        else:
            r_speed = 0.0

        # 4. Steering Smoothness, Rate & Envelope Regularization [-1.0, 0.0]
        steer_diff = abs(steer - self.prev_steer)
        self.prev_steer = steer
        r_steer_rate = -0.3 * min(1.0, steer_diff)
        r_steer_mag = -0.35 * min(1.0, steer ** 2)
        steer_max_allowed = max(0.20, min(0.60, 15.0 / (speed_kmh + 10.0)))
        r_steer_envelope = -1.5 * min(2.0, (max(0.0, abs(steer) - steer_max_allowed) ** 2))
        r_steer = max(-1.0, r_steer_rate + r_steer_mag + r_steer_envelope)

        # 5. Comfort & Throttle-Brake Jitter Penalty [-0.5, 0.0]
        throttle_diff = abs(throttle - self.prev_throttle)
        self.prev_throttle = throttle
        r_comfort = -0.3 * (throttle * brake) - 0.2 * min(1.0, throttle_diff)

        # 6. Wrong-Way / Reverse Driving Penalty [-3.0, 0.0]
        r_wrong_way = -3.0 * max(0.0, -heading_cos) * min(speed_kmh / 5.0, 1.0)

        # 7. Traffic Light Compliance [-3.0, +1.5]
        if is_at_red_light:
            if speed_kmh < 2.0 or brake > 0.2:
                self.stalled_steps = 0
                r_light = 1.5
            else:
                r_light = -3.0
        else:
            r_light = 0.0

        # 8. Obstacle Proximity Barrier & TTC [-3.0, +1.5]
        r_obstacle = 0.0
        if min_obs_dist < 10.0:
            barrier_scale = 1.0 - (min_obs_dist / 10.0)
            multiplier = 2.0 if is_pedestrian else 1.0
            if brake > 0.2 or speed_kmh < 2.0:
                r_obstacle = 1.5 * barrier_scale * multiplier
            elif throttle > 0.2:
                r_obstacle = -3.0 * (barrier_scale ** 2) * multiplier

        r_ttc = -2.0 * min(2.0, (max(0.0, (2.0 - ttc_seconds) / 2.0) ** 2)) if ttc_seconds < 2.0 else 0.0

        # 9. Idle & Stall Penalties (with 40-step acceleration grace period)
        if time_step > 40 and not is_at_red_light and min_obs_dist >= 10.0:
            if speed_kmh < 2.0:
                self.stalled_steps += 1
                r_idle = -0.5
            else:
                self.stalled_steps = 0
                r_idle = 0.0
        else:
            r_idle = 0.0

        is_stalled = bool(self.stalled_steps >= 120)

        # 10. Terminal Penalties
        r_terminal = 0.0
        if is_collision:
            r_terminal = -25.0
        elif is_off_road:
            r_terminal = -20.0
        elif is_stalled:
            r_terminal = -15.0

        alpha = curriculum_factor
        r_boundary_s = r_boundary * alpha
        r_wrong_way_s = r_wrong_way * alpha
        r_light_s = r_light if r_light > 0 else (r_light * alpha)
        r_obstacle_s = r_obstacle if r_obstacle > 0 else (r_obstacle * alpha)
        r_ttc_s = r_ttc * alpha
        r_terminal_s = r_terminal * alpha

        total_reward = (
            r_speed + r_heading + r_lateral + r_boundary_s + r_steer + 
            r_comfort + r_wrong_way_s + r_light_s + r_obstacle_s + r_ttc_s + r_idle + r_terminal_s
        )

        sub_info = {
            "r_speed": r_speed,
            "r_heading": r_heading,
            "r_lateral": r_lateral,
            "r_boundary": r_boundary,
            "r_steer": r_steer,
            "r_comfort": r_comfort,
            "r_wrong_way": r_wrong_way,
            "r_light": r_light,
            "r_obstacle": r_obstacle,
            "r_ttc": r_ttc,
            "r_idle": r_idle,
            "r_terminal": r_terminal,
            "is_stalled": is_stalled
        }
        return total_reward, sub_info
