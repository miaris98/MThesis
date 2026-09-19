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
        # 2.0 (~88% skip per gate, the "Stabilizing Transformers for RL" default) was measured
        # in struggle-solutions S-035 to still be frozen at init after 15k-60k training steps --
        # 3-5 orders of magnitude short of the huge step budgets that default was designed for.
        # 0.0 (~50% skip) lets visual information reach the policy/value heads much sooner.
        bg_init: float = 0.0,
        # Isolation test per struggle-solutions S-036/S-037: False replaces every block's GRU
        # gating with plain residual addition, to test whether the gating mechanism itself
        # (rather than raw step budget) is why the actor's output stays input-invariant.
        use_gru_gating: bool = True,
        # True reproduces the pre-S-038 actor init (gain=0.01 on EVERY actor_head layer) for A/B.
        legacy_actor_init: bool = False,
        # E41: orthogonal init on the visual encoder's Conv2d/Linear layers instead of the
        # default PyTorch Kaiming-uniform (ImpalaCNNEncoder has no explicit init at all).
        cnn_orthogonal_init: bool = False,
        # S-043: DECISIVE FIX confirmed on the incremental-integration testbed (I1 vs I1b) --
        # ImpalaCNNEncoder's missing weight init alone reproduced the whole-investigation 0/11
        # collapse pattern in an otherwise-proven pipeline, and NatureCNNEncoder's own
        # kaiming_normal_ scheme (not E41's orthogonal init) fully recovered a clean 0->17 climb.
        # Testing whether this alone fixes ImpalaGTrXLAgent itself.
        cnn_kaiming_init: bool = False,
        # E47: replace actor_head's Sequential(Linear,GELU,Linear) with a single Linear, so
        # logits are directly proportional to policy_repr with no internal GELU that can die.
        single_layer_actor_head: bool = False,
        # E26: skip the final trunk LayerNorm for policy_repr specifically (still applied to
        # latent_z), testing whether normalizing across the feature dim crushes the variance of
        # low-magnitude spatial activations right before the actor head reads them.
        norm_policy_repr: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.unroll_steps = unroll_steps
        self.legacy_actor_init = legacy_actor_init
        self.single_layer_actor_head = single_layer_actor_head
        self.norm_policy_repr = norm_policy_repr

        # 1. IMPALA-CNN Visual Tokenizer (84x84 -> 121 spatial tokens of dim embed_dim)
        self.visual_encoder = ImpalaCNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            return_spatial_tokens=True,
            orthogonal_init=cnn_orthogonal_init,
            kaiming_init=cnn_kaiming_init,
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
                bg_init=bg_init,
                use_gru_gating=use_gru_gating
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Heads
        if single_layer_actor_head:
            self.actor_head = nn.Sequential(nn.Linear(embed_dim, action_dim))
        else:
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
        self._actor_linears = [m for m in self.actor_head if isinstance(m, nn.Linear)]

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

        # The small 0.01 gain belongs on the OUTPUT layer only (near-uniform initial policy);
        # applying it to hidden layers too compounds to ~1e-4 attenuation, which measurably
        # crushed the actor's gradient into the shared trunk to ~65x below the critic's and
        # left the logits input-invariant (struggle-solutions S-038, experiment E1/E2).
        if self.legacy_actor_init:
            actor_gains = [0.01] * len(self._actor_linears)
        else:
            actor_gains = [2.0 ** 0.5] * (len(self._actor_linears) - 1) + [0.01]
        for m, gain in zip(self._actor_linears, actor_gains):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        critic_linears = [m for m in self.critic_head.modules() if isinstance(m, nn.Linear)]
        critic_gains = [2.0 ** 0.5] * (len(critic_linears) - 1) + [1.0]
        for m, gain in zip(critic_linears, critic_gains):
            nn.init.orthogonal_(m.weight, gain=gain)
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
        normed = self.norm(seq)

        latent_z = normed[:, 0]  # [STATE] query token
        # E26: optionally skip the final LayerNorm for policy_repr specifically, on the
        # hypothesis that normalizing across the feature dim crushes low-magnitude spatial
        # variance right before the actor head reads it. latent_z (critic + EZ2) is unaffected.
        policy_repr = normed[:, 1] if self.norm_policy_repr else seq[:, 1]
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
