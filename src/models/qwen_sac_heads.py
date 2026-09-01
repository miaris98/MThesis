"""Qwen transformer actor and Q-critic heads for Soft Actor-Critic continuous control."""
from typing import Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint

from src.models.transformer.layers import RMSNorm, QwenTransformerBlock

QWEN_PRESETS = {
    "100m": dict(depth=12, embed_dim=768, num_heads=12, ffn_dim=2816),
    "500m": dict(depth=28, embed_dim=1024, num_heads=16, ffn_dim=4096),
    "900m": dict(depth=24, embed_dim=1536, num_heads=24, ffn_dim=6144),
}


def resolve_preset(model_size: str) -> dict:
    """Map a policy-arch string such as 'qwen100m' onto its transformer dimensions."""
    key = str(model_size).lower()
    for size in ("900m", "500m", "100m"):
        if size in key:
            return QWEN_PRESETS[size]
    return QWEN_PRESETS["100m"]


class _QwenTrunk(nn.Module):
    """Shared stack of Qwen blocks with gradient checkpointing during training."""

    def __init__(self, depth: int, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            QwenTransformerBlock(dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            # Checkpointing trades ~30% extra compute for a large activation-memory saving,
            # which is what makes three concurrent Qwen trunks fit alongside CARLA on one GPU.
            if self.training and tokens.requires_grad:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return self.final_norm(tokens)


class QwenSACActor(nn.Module):
    """
    Qwen transformer policy producing a state-dependent tanh-squashed Gaussian.

    Unlike the PPO decision transformer - whose log_std is a single global parameter - this
    head emits log_std per observation, so the policy can be confident on straight road and
    uncertain at a junction. SAC's temperature tuning depends on that being state-aware.
    """

    def __init__(self, in_features: int, action_dim: int = 3, model_size: str = "100m"):
        super().__init__()
        cfg = resolve_preset(model_size)
        embed_dim = cfg["embed_dim"]
        self.action_dim = action_dim

        self.vision_proj = nn.Linear(in_features, embed_dim)
        self.speed_proj = nn.Linear(1, embed_dim)
        self.actor_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trunk = _QwenTrunk(**cfg)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 2 * action_dim)
        )
        nn.init.trunc_normal_(self.actor_token, std=0.02)
        with torch.no_grad():
            # Match the env's action mapping: throttle = (a0 + 1) / 2, steer = a1, brake = a2.
            self.head[2].bias.data[0] = 0.5
            self.head[2].bias.data[1] = 0.0
            self.head[2].bias.data[2] = -0.5

    def forward(self, visual_features: torch.Tensor, speed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) for the pre-squash Gaussian over actions."""
        B = visual_features.shape[0]
        v_tok = self.vision_proj(visual_features.float()).unsqueeze(1)
        s_tok = self.speed_proj(speed.float().view(B, 1) / 50.0).unsqueeze(1)
        a_tok = self.actor_token.expand(B, -1, -1)

        tokens = torch.cat([a_tok, v_tok, s_tok], dim=1)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=visual_features.is_cuda):
            tokens = self.trunk(tokens)
            out = self.head(tokens[:, 0]).float()
        return out.chunk(2, dim=-1)


class QwenQCritic(nn.Module):
    """
    Qwen transformer Q-function over (visual features, speed, action).

    The action enters as its own token rather than being concatenated to a flat vector, so
    attention can relate the proposed control directly to the visual context. Each critic
    owns an independent trunk - sharing one would correlate the twin estimates and defeat
    the min(Q1, Q2) overestimation correction that SAC relies on.
    """

    def __init__(self, in_features: int, action_dim: int = 3, model_size: str = "100m"):
        super().__init__()
        cfg = resolve_preset(model_size)
        embed_dim = cfg["embed_dim"]

        self.vision_proj = nn.Linear(in_features, embed_dim)
        self.speed_proj = nn.Linear(1, embed_dim)
        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.q_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trunk = _QwenTrunk(**cfg)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 1)
        )
        nn.init.trunc_normal_(self.q_token, std=0.02)

    def forward(self, visual_features: torch.Tensor, speed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return the scalar Q estimate for each state-action pair in the batch."""
        B = visual_features.shape[0]
        v_tok = self.vision_proj(visual_features.float()).unsqueeze(1)
        s_tok = self.speed_proj(speed.float().view(B, 1) / 50.0).unsqueeze(1)
        a_tok = self.action_proj(action.float()).unsqueeze(1)
        q_tok = self.q_token.expand(B, -1, -1)

        tokens = torch.cat([q_tok, v_tok, s_tok, a_tok], dim=1)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=visual_features.is_cuda):
            tokens = self.trunk(tokens)
            q = self.head(tokens[:, 0]).squeeze(-1).float()
        return q
