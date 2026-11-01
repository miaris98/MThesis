"""TODO A73 (2026-11-01): latent world model on the frozen pooled regnety features (Design B: RL in imagination, offline).

State = the frozen encoder's (C, gh, gw) pooled map of a frame (the A59 pooled cache: C = 1512, 4 x 4), as gh * gw tokens projected to
`embed_dim` and passed through `repr_depth` blocks. Dynamics: the state tokens plus one action token (A71's 40-action grid: 8 speed bins x 5
lateral offsets) through a `dyn_depth` trunk, then `n_ensemble` linear heads, each with its own action embedding, that each predict the next
state tokens as a residual (LOMPO-style ensemble on a shared trunk; their spread is the conservatism penalty, and the per-member action
embeddings are what make it rise on actions the data does not contain). Reward and value from the mean-pooled tokens; a policy head over the 40
actions. No pixel reconstruction (MILE weights images at 0): the latent target is the next frame's representation by an EMA encoder (LAW).

trunk = "qwen": the project's QwenTransformerBlock at a _MODEL_SIZES size (A73-Q; "30m" = 8 blocks x 512, the CARLA/Atari network), or
explicit small dims for tests; "mlp": per-token MLP blocks with a token-mean mixer (A73-W, the conv-head stand-in).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.transformer.layers import QwenTransformerBlock, RMSNorm

N_ACTIONS = 40                                          # = len(counterfactual_q.TF_SPEEDS) x len(lat_targets)


@dataclass(frozen=True)
class WorldModelConfig:
    feat_channels: int = 1512
    grid: Tuple[int, int] = (4, 4)
    n_actions: int = N_ACTIONS
    trunk: str = "qwen"                                 # "qwen" | "mlp"
    size: Optional[str] = None                          # a qwen_wor_policy._MODEL_SIZES key; overrides the dims below
    embed_dim: int = 512
    repr_depth: int = 2
    dyn_depth: int = 8
    num_heads: int = 8
    ffn_dim: int = 1536
    n_ensemble: int = 5
    qk_norm: bool = True                                # B67: QK-norm for transformer RL heads

    def resolved(self) -> "WorldModelConfig":
        if self.size is None:
            return self
        from src.models.world_on_rails.qwen_wor_policy import _MODEL_SIZES
        s = _MODEL_SIZES[str(self.size).lower()]
        return replace(self, embed_dim=s["embed_dim"], dyn_depth=s["depth"], num_heads=s["num_heads"], ffn_dim=s["ffn_dim"])


class _MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mix = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.ffn(h + self.mix(h.mean(1, keepdim=True)))


def _blocks(cfg: WorldModelConfig, depth: int) -> nn.Sequential:
    if cfg.trunk == "qwen":
        return nn.Sequential(*[QwenTransformerBlock(cfg.embed_dim, num_heads=cfg.num_heads, ffn_dim=cfg.ffn_dim, qk_norm=cfg.qk_norm)
                               for _ in range(depth)])
    if cfg.trunk == "mlp":
        return nn.Sequential(*[_MLPBlock(cfg.embed_dim, cfg.ffn_dim) for _ in range(depth)])
    raise ValueError(f"trunk must be 'qwen' or 'mlp', got {cfg.trunk!r}")


class LatentWorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig = WorldModelConfig()):
        super().__init__()
        cfg = cfg.resolved()
        self.cfg = cfg
        D, T = cfg.embed_dim, cfg.grid[0] * cfg.grid[1]
        self.n_tokens = T
        self.proj = nn.Linear(cfg.feat_channels, D)
        self.pos = nn.Parameter(torch.zeros(1, T + 1, D))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.repr = _blocks(cfg, cfg.repr_depth)
        self.repr_norm = RMSNorm(D)
        self.action_embed = nn.Embedding(cfg.n_actions, D)
        self.dyn = _blocks(cfg, cfg.dyn_depth)
        self.dyn_norm = RMSNorm(D)
        self.ens = nn.ModuleList([nn.Linear(D, D) for _ in range(cfg.n_ensemble)])
        # each member also sees the action through its own embedding: on actions the data never shows these stay independent random vectors,
        # so the members disagree there; with the action only in the shared trunk the spread was lower on unseen actions than on seen ones
        self.ens_action = nn.ModuleList([nn.Embedding(cfg.n_actions, D) for _ in range(cfg.n_ensemble)])
        self.reward_head = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, 1))
        self.value_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        self.policy_head = nn.Linear(D, cfg.n_actions)

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, C, gh, gw) frozen pooled map -> (B, gh * gw, D) state tokens."""
        x = feats.float().flatten(2).transpose(1, 2)
        x = self.proj(x) + self.pos[:, :self.n_tokens]
        return self.repr_norm(self.repr(x))

    def step(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(next state = ensemble mean (B, T, D), every head's next state (E, B, T, D), reward (B,)) for actions a (B,) long."""
        a_tok = self.action_embed(a.long()).unsqueeze(1) + self.pos[:, self.n_tokens:]
        h = self.dyn_norm(self.dyn(torch.cat([s, a_tok], 1)))
        hs, ha = h[:, :self.n_tokens], h[:, self.n_tokens]
        s_all = torch.stack([s + head(hs + emb(a.long()).unsqueeze(1)) for head, emb in zip(self.ens, self.ens_action)])
        r = self.reward_head(torch.cat([hs.mean(1), ha], -1)).squeeze(-1)
        return s_all.mean(0), s_all, r

    def value(self, s: torch.Tensor) -> torch.Tensor:
        return self.value_head(s.mean(1)).squeeze(-1)

    def policy_logits(self, s: torch.Tensor) -> torch.Tensor:
        return self.policy_head(s.mean(1))
