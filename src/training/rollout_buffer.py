"""Rollout buffer storing trajectory transitions and computing GAE advantages."""
from typing import List, Tuple, Optional
import torch
import numpy as np


class RolloutBuffer:
    """
    Trajectory transition storage buffer for on-policy PPO reinforcement learning.
    Computes Generalized Advantage Estimation (GAE) and prepares batched GPU tensors.
    """
    def __init__(self, buffer_size: int, gamma: float = 0.99, gae_lambda: float = 0.95, device: Optional[torch.device] = None):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cpu")
        self.reset()

    def reset(self) -> None:
        """Clear all stored transition lists."""
        self.obs_images: List[torch.Tensor] = []
        self.obs_visual_features: List[torch.Tensor] = []
        self.obs_speeds: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        speed: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        obs_img: Optional[torch.Tensor] = None,
        obs_vis: Optional[torch.Tensor] = None
    ) -> None:
        """Store a single transition step in the buffer."""
        if obs_img is not None:
            self.obs_images.append(obs_img.squeeze(0))
        if obs_vis is not None:
            self.obs_visual_features.append(obs_vis.squeeze(0))

        self.obs_speeds.append(speed.squeeze(0))
        self.actions.append(action.squeeze(0))
        self.log_probs.append(log_prob.squeeze(0))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(value.squeeze(0))

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        next_done: bool
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - float(next_done)
                nxt_val = next_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                nxt_val = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * nxt_val * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        b_vis = torch.stack(self.obs_visual_features) if self.obs_visual_features else None
        b_spd = torch.stack(self.obs_speeds)
        b_act = torch.stack(self.actions)
        b_logp = torch.stack(self.log_probs)
        b_adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        b_ret = torch.tensor(returns, dtype=torch.float32, device=self.device)
        b_val = torch.stack(self.values)

        # Standardize Advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        return b_vis, b_spd, b_act, b_logp, b_adv, b_ret, b_val
