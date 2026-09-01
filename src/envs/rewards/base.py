"""Base contract for swappable driving reward functions."""
from typing import Any, Dict, Tuple

# Every reward function reports into this fixed set of telemetry slots, so the CSV schema and
# MLflow metrics stay identical no matter which one is active and runs remain comparable.
# Rewards that have no analogue for a slot simply leave it at 0.0.
REWARD_COMPONENTS = ("r_progress", "r_lane", "r_light", "r_obstacle", "r_ttc", "r_terminal")


class BaseReward:
    """
    Interface shared by every reward function.

    Implementations receive the full per-step state dict assembled by DrivingStateExtractor
    and return (total_reward, sub_info). `sub_info` must carry every key in REWARD_COMPONENTS
    plus `is_stalled`, which the environment consults for stall termination.
    """
    #: Human-readable name shown in logs and MLflow params.
    NAME = "base"
    #: Where this formulation comes from, for the thesis writeup.
    SOURCE = "abstract base class"

    def __init__(self, desired_speed: float = 25.0, **kwargs: Any):
        self.desired_speed = desired_speed
        self.reset_episode_tracking()

    def reset_episode_tracking(self) -> None:
        """Clear any per-episode accumulators. Called on every env reset."""
        self.stalled_steps = 0

    def compute_reward(
        self, state: Dict[str, Any], curriculum_factor: float = 1.0, dt: float = 0.05
    ) -> Tuple[float, Dict[str, Any]]:
        """Return (total_reward, sub_info) for one simulation step."""
        raise NotImplementedError

    @staticmethod
    def _blank_info(**overrides: Any) -> Dict[str, Any]:
        """Build a sub_info dict with every telemetry slot present and zeroed."""
        info: Dict[str, Any] = {k: 0.0 for k in REWARD_COMPONENTS}
        info["is_stalled"] = False
        info.update(overrides)
        return info

    def _track_stall(
        self, speed_kmh: float, time_step: int, exempt: bool,
        grace_steps: int = 40, timeout_steps: int = 80, stall_speed_kmh: float = 2.0
    ) -> bool:
        """
        Shared stall detector so every reward function terminates idling the same way.

        `exempt` covers legitimate reasons to be stopped (red light, obstacle ahead) and both
        freezes and resets the counter, so a legal stop can never accumulate toward a stall.
        """
        if exempt:
            self.stalled_steps = 0
            return False
        if time_step > grace_steps:
            if speed_kmh < stall_speed_kmh:
                self.stalled_steps += 1
            else:
                self.stalled_steps = 0
        return bool(self.stalled_steps >= timeout_steps)
