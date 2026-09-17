"""Unified IMPALA-CNN + GTrXL + EfficientZero v2 Agent Architecture.

Combines:
    1. IMPALA-CNN visual feature tokenizer.
    2. GTrXL (Gated Transformer-XL) deep policy trunk with trainable GRU skip gates.
    3. Actor (Policy) and Critic (Value) heads.
    4. EfficientZero v2 multi-step branch dynamics predictor for self-supervised forward planning.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from atari_qwen.models.visual_encoders import ImpalaCNNEncoder
from atari_qwen.models.gtrxl_layers import GTrXLBlock
from atari_qwen.models.efficientzero_v2_predictor import EfficientZeroV2Predictor


class ImpalaGTrXLAgent(nn.Module):
    """
    IMPALA-CNN + GTrXL + EfficientZero v2 Latent Branch Predictor.
    """
    def __init__(
        self,
        action_dim: int = 4,
        in_channels: int = 4,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        unroll_steps: int = 5,
        bg_init: float = 2.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.unroll_steps = unroll_steps

        # 1. IMPALA-CNN Visual Tokenizer (84x84 -> 121 spatial tokens of dim embed_dim)
        self.visual_encoder = ImpalaCNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            return_spatial_tokens=True
        )

        # 2. Learnable [STATE] and [POLICY] query tokens
        self.state_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.policy_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 + 128, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. GTrXL Transformer Trunk with Trainable GRU Skip Gating
        self.blocks = nn.ModuleList([
            GTrXLBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                bg_init=bg_init
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Heads
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        # 5. EfficientZero v2 Multi-Step Branch Predictor
        self.predictor = EfficientZeroV2Predictor(
            latent_dim=embed_dim,
            action_dim=action_dim,
            unroll_steps=unroll_steps,
            hidden_dim=ffn_dim // 2
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.state_token, std=0.02)
        nn.init.trunc_normal_(self.policy_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.actor_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.critic_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes raw observation through IMPALA-CNN + GTrXL trunk.
        Returns:
            latent_z: (B, embed_dim) compressed state representation
            policy_repr: (B, embed_dim) decision representation for actor head
        """
        B = obs.shape[0]
        vis_tokens = self.visual_encoder(obs)  # (B, 121, embed_dim)
        N_v = vis_tokens.shape[1]

        state_tok = self.state_token.expand(B, -1, -1)
        pol_tok = self.policy_token.expand(B, -1, -1)
        seq = torch.cat([state_tok, pol_tok, vis_tokens], dim=1)  # (B, 2 + N_v, embed_dim)

        seq = seq + self.pos_embed[:, : seq.shape[1], :]
        seq = self.pos_drop(seq)

        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)

        latent_z = seq[:, 0]     # [STATE] query token
        policy_repr = seq[:, 1]  # [POLICY] query token
        return latent_z, policy_repr

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard forward pass for RL execution.
        Returns:
            logits: (B, action_dim) policy action logits
            value: (B, 1) state value estimate V(s)
            latent_z: (B, embed_dim) latent state for branch rollouts
        """
        latent_z, policy_repr = self.encode_observation(obs)
        logits = self.actor_head(policy_repr)
        value = self.critic_head(latent_z)
        return logits, value, latent_z

    def unroll_branches(self, latent_z: torch.Tensor, actions_seq: torch.Tensor) -> Dict[str, list]:
        """Unrolls future branches in latent space using EfficientZero v2 dynamics."""
        return self.predictor.unroll_trajectory(latent_z, actions_seq)

    def evaluate_actions_lookahead(
        self,
        obs: torch.Tensor,
        gamma: float = 0.99
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1-step latent tree lookahead for all candidate actions in parallel:
        Q(s, a) = r(s, a) + gamma * V(s')
        Returns:
            q_values: (B, action_dim) estimated Q-values across all discrete actions
            latent_z: (B, embed_dim) root latent state
        """
        latent_z, _ = self.encode_observation(obs)
        B = obs.shape[0]
        q_vals = []
        for a in range(self.action_dim):
            act_tensor = torch.full((B,), a, dtype=torch.long, device=obs.device)
            _, r_hat, v_hat = self.predictor.predict_step(latent_z, act_tensor)
            q_a = r_hat + gamma * v_hat
            q_vals.append(q_a.unsqueeze(-1))
        q_values = torch.cat(q_vals, dim=-1)  # (B, action_dim)
        return q_values, latent_z
