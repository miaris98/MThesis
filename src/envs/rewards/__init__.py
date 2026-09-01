"""Swappable driving reward functions, selectable at training time via --reward-fn."""
from typing import Any, Dict, List, Type

from src.envs.rewards.base import BaseReward, REWARD_COMPONENTS
from src.envs.rewards.custom_1 import Custom1Reward
from src.envs.rewards.leaderboard import LeaderboardReward
from src.envs.rewards.interp_e2e import InterpE2EReward
from src.envs.rewards.roach import RoachReward

REWARD_REGISTRY: Dict[str, Type[BaseReward]] = {
    Custom1Reward.NAME: Custom1Reward,
    LeaderboardReward.NAME: LeaderboardReward,
    InterpE2EReward.NAME: InterpE2EReward,
    RoachReward.NAME: RoachReward,
}


def available_rewards() -> List[str]:
    """Names accepted by --reward-fn, for CLI choices and error messages."""
    return sorted(REWARD_REGISTRY.keys())


def make_reward(name: str = "custom_1", **kwargs: Any) -> BaseReward:
    """Instantiate a reward function by registry name."""
    key = str(name).lower().strip()
    if key not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward function '{name}'. Available: {available_rewards()}")
    return REWARD_REGISTRY[key](**kwargs)


__all__ = [
    "BaseReward", "REWARD_COMPONENTS", "REWARD_REGISTRY",
    "Custom1Reward", "LeaderboardReward", "InterpE2EReward", "RoachReward",
    "available_rewards", "make_reward",
]
