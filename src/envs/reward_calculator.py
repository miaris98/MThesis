"""Backward-compatible alias for the default reward function.

The reward is now a swappable component; see src/envs/rewards/ for the registry and
`--reward-fn` for selection. `RewardCalculator` remains bound to Custom_1, this project's
own formulation, so older imports and checkpoints keep resolving.
"""
from src.envs.rewards import Custom1Reward, make_reward, available_rewards

RewardCalculator = Custom1Reward

__all__ = ["RewardCalculator", "Custom1Reward", "make_reward", "available_rewards"]
