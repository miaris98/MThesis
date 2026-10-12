"""EfficientZero v2-style Multi-step Latent Dynamics and Branch Predictor.

Features:
    1. Recurrent Latent Transition / Dynamics Model: z_{k+1} = Dynamics(z_k, a_k)
    2. Reward Prediction Head: r_hat_k = RewardHead(z_k, a_k)
    3. Future Value Prediction Head: V_hat_{k+1} = ValueHead(z_{k+1})
    4. Self-Supervised Latent Consistency (SimSiam-style cosine similarity):
       aligns rollout latent states with real target future observation representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional


class LatentDynamicsBlock(nn.Module):
    """Residual transition dynamics block mapping (z_k, a_k) -> z_{k+1}."""
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim), action: (B,)
        a_emb = self.action_embed(action)
        inp = torch.cat([z, a_emb], dim=-1)
        dz = self.net(inp)
        # Residual step with normalization to keep latent scale bounded
        z_next = self.norm(z + dz)
        return z_next


class EfficientZeroV2Predictor(nn.Module):
    """
    Multi-Step Branch Predictor inspired by EfficientZero v2.
    Unrolls forward latent dynamics for K steps and predicts rewards, values, and representations.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 4,
        unroll_steps: int = 5,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.unroll_steps = unroll_steps

        # 1. Latent Dynamics
        self.dynamics = LatentDynamicsBlock(latent_dim, action_dim, hidden_dim)

        # 2. Reward Head: (z_k, a_k) -> scalar predicted reward
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 3. Value Head: z_{k+1} -> scalar predicted value
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 4. Consistency Projector / Predictor (SimSiam architecture to avoid representation collapse)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def predict_step(self, z: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict 1 step ahead: returns (z_next, reward_hat, value_next_hat)."""
        a_emb = self.dynamics.action_embed(action)
        reward_inp = torch.cat([z, a_emb], dim=-1)
        reward_hat = self.reward_head(reward_inp).squeeze(-1)
        
        z_next = self.dynamics(z, action)
        val_next_hat = self.value_head(z_next).squeeze(-1)
        return z_next, reward_hat, val_next_hat

    def unroll_trajectory(
        self,
        z_0: torch.Tensor,
        actions_seq: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Unroll multi-step branch predictions across sequence of actions.
        Args:
            z_0: initial latent state (B, latent_dim)
            actions_seq: (B, K) sequence of discrete action indices
        Returns:
            Dict containing lists of length K for latent_states, rewards, and values.
        """
        B, K = actions_seq.shape
        latent_states = []
        pred_rewards = []
        pred_values = []
        pred_projections = []

        curr_z = z_0
        for k in range(K):
            act_k = actions_seq[:, k]
            next_z, r_hat, v_hat = self.predict_step(curr_z, act_k)
            proj = self.predictor(self.projector(next_z))
            
            latent_states.append(next_z)
            pred_rewards.append(r_hat)
            pred_values.append(v_hat)
            pred_projections.append(proj)
            curr_z = next_z

        return {
            "latent_states": latent_states,
            "rewards": pred_rewards,
            "values": pred_values,
            "projections": pred_projections
        }

    @staticmethod
    def compute_consistency_loss(pred_proj: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:
        """Negative cosine similarity with stop-gradient on target (SimSiam formula)."""
        p = F.normalize(pred_proj, dim=-1)
        z = F.normalize(target_proj.detach(), dim=-1)
        return -(p * z).sum(dim=-1).mean()
