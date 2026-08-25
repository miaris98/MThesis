"""Rollout buffer storing trajectory transitions and computing GAE advantages across parallel environments."""
from typing import List, Tuple, Optional, Union
import torch
import numpy as np


class RolloutBuffer:
    """
    Trajectory transition storage buffer for on-policy PPO reinforcement learning.
    Supports single-environment and multi-environment vectorized rollouts (T, num_envs, ...).
    Computes Generalized Advantage Estimation (GAE) independently per worker to prevent
    cross-environment trajectory leakage and flattens batches for PPO optimization.
    """
    def __init__(
        self, 
        buffer_size: int, 
        gamma: float = 0.99, 
        gae_lambda: float = 0.95, 
        device: Optional[torch.device] = None,
        num_envs: int = 1
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cpu")
        self.num_envs = num_envs
        self.reset()

    def reset(self) -> None:
        """Clear all stored transition lists."""
        self.obs_images: List[torch.Tensor] = []
        self.obs_visual_features: List[torch.Tensor] = []
        self.obs_speeds: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def total_transitions(self) -> int:
        """Total number of individual transition steps across all environments (T * num_envs)."""
        return len(self.rewards) * self.num_envs

    def add(
        self,
        speed: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: Union[float, np.ndarray, torch.Tensor],
        done: Union[bool, np.ndarray, torch.Tensor],
        value: torch.Tensor,
        obs_img: Optional[torch.Tensor] = None,
        obs_vis: Optional[torch.Tensor] = None
    ) -> None:
        """Store a single timestep across all parallel environments in the buffer."""
        # Ensure tensors are shaped (num_envs, ...)
        spd_t = speed.view(self.num_envs, -1)
        act_t = action.view(self.num_envs, -1)
        logp_t = log_prob.view(self.num_envs)
        val_t = value.view(self.num_envs)

        rew_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(self.num_envs)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(self.num_envs)

        if obs_img is not None:
            self.obs_images.append(obs_img.view(self.num_envs, *obs_img.shape[1:]))
        if obs_vis is not None:
            self.obs_visual_features.append(obs_vis.view(self.num_envs, -1))

        self.obs_speeds.append(spd_t)
        self.actions.append(act_t)
        self.log_probs.append(logp_t)
        self.rewards.append(rew_t)
        self.dones.append(done_t)
        self.values.append(val_t)

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        next_done: Union[bool, np.ndarray, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) and discounted returns independently
        along the time dimension for each environment worker.
        Returns flattened tensors of shape (T * num_envs, ...) for PPO minibatch optimization.
        """
        T = len(self.rewards)
        N = self.num_envs

        t_rewards = torch.stack(self.rewards)  # (T, N)
        t_dones = torch.stack(self.dones)      # (T, N)
        t_values = torch.stack(self.values)    # (T, N)

        nxt_val = next_value.view(N)
        nxt_done = torch.as_tensor(next_done, dtype=torch.float32, device=self.device).view(N)

        advantages = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        last_gae_lam = torch.zeros(N, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - nxt_done
                next_vals = nxt_val
            else:
                next_non_terminal = 1.0 - t_dones[t + 1]
                next_vals = t_values[t + 1]

            delta = t_rewards[t] + self.gamma * next_vals * next_non_terminal - t_values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + t_values

        # Flatten (T, N, ...) into (T * N, ...)
        b_vis = torch.stack(self.obs_visual_features).view(T * N, -1) if self.obs_visual_features else None
        b_spd = torch.stack(self.obs_speeds).view(T * N, -1)
        b_act = torch.stack(self.actions).view(T * N, -1)
        b_logp = torch.stack(self.log_probs).view(T * N)
        b_adv = advantages.view(T * N)
        b_ret = returns.view(T * N)
        b_val = t_values.view(T * N)

        # Standardize Advantages across all rollout samples
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        return b_vis, b_spd, b_act, b_logp, b_adv, b_ret, b_val
