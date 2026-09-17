"""GTrXL (Gated Transformer-XL) layers with trainable GRU-style skip gating.

References:
    Parisotto et al. "Stabilizing Transformers for Reinforcement Learning" (DeepMind, ICML 2020).
    Replaces static identity addition x + f(x) with GRUGating(x, f(x)) to provide smooth,
    stable gradient highways for deep RL training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GRUGating(nn.Module):
    """
    GRU-style gating mechanism for Transformer skip connections:
        r = sigmoid(W_r * x + U_r * y)
        z = sigmoid(W_z * x + U_z * y - b_g)
        h_tilde = tanh(W_g * x + U_g * (r * y))
        out = (1 - z) * x + z * h_tilde

    With gate bias b_g initialized to +2.0, z starts close to 0, making the layer
    act approximately as an identity map initially, allowing deep gradient flow.
    """
    def __init__(self, dim: int, bg_init: float = 2.0):
        super().__init__()
        self.w_r = nn.Linear(dim, dim, bias=False)
        self.u_r = nn.Linear(dim, dim, bias=False)
        self.w_z = nn.Linear(dim, dim, bias=False)
        self.u_z = nn.Linear(dim, dim, bias=False)
        self.w_g = nn.Linear(dim, dim, bias=False)
        self.u_g = nn.Linear(dim, dim, bias=False)
        
        # Trainable gate bias, initialized to positive value (default 2.0)
        self.bg = nn.Parameter(torch.full((dim,), bg_init))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.w_r(x) + self.u_r(y))
        z = torch.sigmoid(self.w_z(x) + self.u_z(y) - self.bg)
        h_tilde = torch.tanh(self.w_g(x) + self.u_g(r * y))
        out = (1.0 - z) * x + z * h_tilde
        return out


class GTrXLBlock(nn.Module):
    """
    A single GTrXL (Gated Transformer-XL) block:
        1. LayerNorm(x) -> Multi-Head Self-Attention -> GRUGating(x, attn_out)
        2. LayerNorm(h) -> SwiGLU / MLP Feed-Forward -> GRUGating(h, ffn_out)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        bg_init: float = 2.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Attention sub-layer
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate1 = GRUGating(dim, bg_init=bg_init)

        # Feed-forward sub-layer (SwiGLU)
        self.norm2 = nn.LayerNorm(dim)
        self.w_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, dim, bias=False)
        self.gate2 = GRUGating(dim, bg_init=bg_init)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout.p > 0:
                attn_weights = self.dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(attn_out)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + Attention + GRU Gating
        attn_out = self._attention(self.norm1(x))
        x = self.gate1(x, attn_out)

        # Pre-LN + FFN + GRU Gating
        ffn_out = self._ffn(self.norm2(x))
        x = self.gate2(x, ffn_out)
        return x
