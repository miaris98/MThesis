"""CNN-Like Lightweight Transformer Trunk for Sample-Efficient Discrete RL & MCTS.

Combines the translation equivariance and local inductive bias of CNNs with the
expressive dynamic routing and capacity of modern transformers:
1. 2D Relative Positional Bias:
   Replaces learned absolute coordinate tables with translation-equivariant 2D relative offsets
   (Delta_r, Delta_c), preserving spatial invariance across grid positions.
2. Local Neighborhood Attention:
   Restricts spatial-to-spatial attention to a local spatial window (e.g. 3x3), eliminating
   global feature perturbation that destabilizes representation consistency loss, while granting
   the prepended [POLICY] token a global receptive field over the full board.
3. Depthwise Separable 2D Convolution in the Residual Stream:
   ConvNeXt/MetaFormer-style local spatial mixing via 3x3 depthwise convolution (groups=dim).
4. QK-Norm & Trainable Alpha Residual Gating:
   Per-head RMSNorm on queries and keys to prevent logit explosion under AdamW, plus trainable
   skip scalars (alpha_conv, alpha_attn, alpha_ffn) starting at 0.1.
5. Lightweight Parameterization:
   Presets sized between 0.2M and 4.2M parameters (vs. 25.2M in Qwen-30m), slashing MCTS
   simulation latency and allowing 32-64 simulation scaling within a fixed GPU budget.
"""
from typing import Optional, Tuple, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer.layers import RMSNorm, SwiGLU


CONV_TRANSFORMER_SIZES: Dict[str, Dict[str, int]] = {
    "nano":  {"dim": 64,  "depth": 2, "heads": 2, "ffn_dim": 128},   # ~0.2M params
    "micro": {"dim": 128, "depth": 2, "heads": 4, "ffn_dim": 256},   # ~0.7M params
    "mini":  {"dim": 128, "depth": 4, "heads": 4, "ffn_dim": 384},   # ~1.4M params
    "small": {"dim": 256, "depth": 4, "heads": 4, "ffn_dim": 768},   # ~4.1M params (target 3M-5M)
}


def normalized_entropy(p: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of each attention row over the keys it may see, divided by log(number of those keys): 1 is uniform, 0 is one key.
    `allowed` is 1 where a key is visible (broadcast against p's last two dims)."""
    k = allowed.sum(-1).clamp(min=2).to(p.dtype)
    ent = -(p * torch.log(p.clamp(min=1e-12))).masked_fill(allowed == 0, 0.0).sum(-1)
    return ent / torch.log(k)


class RelativePositionBias2D(nn.Module):
    """2D Relative Positional Bias table for an (H, W) spatial token grid.
    
    For spatial tokens (i, j), the bias depends strictly on the spatial offset:
        Delta_r = r_i - r_j in [-(H-1), H-1]
        Delta_c = c_i - c_j in [-(W-1), W-1]
    
    When has_policy_token is True, token 0 is a global policy readout token:
    - Policy -> Spatial: learned 2D spatial bias map per head of shape (H, W)
    - Spatial -> Policy: learned 2D spatial bias map per head of shape (H, W)
    - Policy -> Policy: learned scalar bias per head
    """
    def __init__(self, heads: int, h: int = 6, w: int = 6, has_policy_token: bool = True):
        super().__init__()
        self.heads = heads
        self.h = h
        self.w = w
        self.hw = h * w
        self.has_policy_token = has_policy_token
        
        # Spatial-to-spatial relative table: (2*H - 1) * (2*W - 1) offsets
        num_offsets = (2 * h - 1) * (2 * w - 1)
        self.spatial_rel_table = nn.Parameter(torch.zeros(heads, num_offsets))
        
        # Initialize with locality bias: -0.5 * (|Delta_r| + |Delta_c|)
        rows = torch.arange(-(h - 1), h).view(-1, 1).repeat(1, 2 * w - 1)
        cols = torch.arange(-(w - 1), w).view(1, -1).repeat(2 * h - 1, 1)
        locality_init = -0.5 * (rows.abs().float() + cols.abs().float()).reshape(-1)
        self.spatial_rel_table.data.copy_(locality_init.unsqueeze(0).repeat(heads, 1))
        
        # Precompute 2D relative index buffer for spatial tokens (hw, hw)
        grid_r = torch.arange(h).view(-1, 1).repeat(1, w).reshape(-1)
        grid_c = torch.arange(w).view(1, -1).repeat(h, 1).reshape(-1)
        dr = grid_r[:, None] - grid_r[None, :] + (h - 1)  # 0 to 2*h - 2
        dc = grid_c[:, None] - grid_c[None, :] + (w - 1)  # 0 to 2*w - 2
        rel_idx = dr * (2 * w - 1) + dc
        self.register_buffer("spatial_rel_idx", rel_idx.long(), persistent=False)
        
        if has_policy_token:
            # Policy <-> Spatial biases
            self.pol_to_spatial_bias = nn.Parameter(torch.zeros(heads, self.hw))
            self.spatial_to_pol_bias = nn.Parameter(torch.zeros(heads, self.hw))
            self.pol_to_pol_bias = nn.Parameter(torch.zeros(heads, 1))
            # Initialize with small normal noise
            nn.init.trunc_normal_(self.pol_to_spatial_bias, std=0.02)
            nn.init.trunc_normal_(self.spatial_to_pol_bias, std=0.02)
            nn.init.trunc_normal_(self.pol_to_pol_bias, std=0.02)

    def forward(self) -> torch.Tensor:
        """Returns attention bias of shape (heads, N, N) where N = hw or 1 + hw."""
        # Spatial-to-spatial bias: (heads, hw, hw)
        spatial_bias = self.spatial_rel_table[:, self.spatial_rel_idx]
        
        if not self.has_policy_token:
            return spatial_bias
            
        # Compose full (heads, 1 + hw, 1 + hw) bias tensor
        heads = self.heads
        N = 1 + self.hw
        full_bias = spatial_bias.new_empty(heads, N, N)
        
        # Policy -> Policy (top-left 1x1)
        full_bias[:, 0:1, 0:1] = self.pol_to_pol_bias.unsqueeze(-1)
        # Policy -> Spatial (top row, shape (heads, 1, hw))
        full_bias[:, 0:1, 1:] = self.pol_to_spatial_bias.unsqueeze(1)
        # Spatial -> Policy (left column, shape (heads, hw, 1))
        full_bias[:, 1:, 0:1] = self.spatial_to_pol_bias.unsqueeze(-1)
        # Spatial -> Spatial (bottom-right hw x hw)
        full_bias[:, 1:, 1:] = spatial_bias
        
        return full_bias


class LocalNeighborhoodAttention2D(nn.Module):
    """Multi-Head Attention with 2D Relative Positional Bias, Local Spatial Window Masking,
    and per-head QK-Norm.
    
    - Spatial-to-spatial pairs outside window_size (e.g. 3x3) are masked to zero attention weight.
    - Policy token (index 0) is UNMASKED to maintain full global visibility over all spatial cells.
    """
    def __init__(self, dim: int, num_heads: int = 4, h: int = 6, w: int = 6,
                 window_size: int = 3, dropout: float = 0.0, qk_norm: bool = True,
                 has_policy_token: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.h = h
        self.w = w
        self.hw = h * w
        self.has_policy_token = has_policy_token
        self.window_size = window_size
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        
        self.rel_pos = RelativePositionBias2D(heads=num_heads, h=h, w=w, has_policy_token=has_policy_token)
        
        # Build local window mask
        self._build_window_mask(h, w, window_size, has_policy_token)

    def _build_window_mask(self, h: int, w: int, window_size: int, has_policy_token: bool):
        radius = window_size // 2
        grid_r = torch.arange(h).view(-1, 1).repeat(1, w).reshape(-1)
        grid_c = torch.arange(w).view(1, -1).repeat(h, 1).reshape(-1)
        dr = (grid_r[:, None] - grid_r[None, :]).abs()
        dc = (grid_c[:, None] - grid_c[None, :]).abs()
        spatial_in_window = (dr <= radius) & (dc <= radius)
        
        N = (1 + self.hw) if has_policy_token else self.hw
        mask = torch.zeros(N, N, dtype=torch.bool)
        
        if has_policy_token:
            # Policy token (row 0) can attend to ALL spatial cells (global readout)
            mask[0, :] = True
            # Spatial cells can attend to Policy token (optional, enable for bidirectional)
            mask[:, 0] = True
            # Spatial cells can attend to other spatial cells only within window
            mask[1:, 1:] = spatial_in_window
        else:
            mask = spatial_in_window
            
        # Store as additive float mask: 0.0 for in-window, -10000.0 for masked
        float_mask = torch.where(mask, torch.zeros(1), torch.full((1,), -10000.0))
        self.register_buffer("window_mask", float_mask.unsqueeze(0), persistent=False)  # (1, N, N)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.q_norm is not None:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
            
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        
        # Add 2D relative position bias: (heads, N, N) -> (1, H, N, N)
        rel_bias = self.rel_pos()
        scores = scores + rel_bias.unsqueeze(0)
        
        # Add local neighborhood mask
        scores = scores + self.window_mask.unsqueeze(1)
        
        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout.p > 0:
            attn_weights = self.dropout(attn_weights)
            
        attn_out = torch.matmul(attn_weights, v)  # (B, H, N, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(attn_out)
        
        if return_weights:
            return out, attn_weights
        return out, None


class ConvTransformerBlock(nn.Module):
    """Hybrid Transformer Block with:
    1. 2D Depthwise Convolutional residual branch for local inductive spatial mixing.
    2. Local Neighborhood Attention with 2D Relative Positional Bias & QK-Norm.
    3. SwiGLU Feed-Forward Network.
    4. Trainable Alpha Residual Scaling on all three branches.
    """
    def __init__(self, dim: int, num_heads: int = 4, h: int = 6, w: int = 6,
                 ffn_dim: int = 512, window_size: int = 3, dropout: float = 0.0,
                 qk_norm: bool = True, has_policy_token: bool = True):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.hw = h * w
        self.has_policy_token = has_policy_token
        
        # 1. Depthwise Convolution branch (ConvNeXt / MetaFormer style)
        self.norm_conv = RMSNorm(dim)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw_proj = nn.Linear(dim, dim)
        if has_policy_token:
            self.pol_conv_proj = nn.Linear(dim, dim)
        self.alpha_conv = nn.Parameter(0.1 * torch.ones(dim))
        
        # 2. Local Neighborhood Attention branch
        self.norm_attn = RMSNorm(dim)
        self.attn = LocalNeighborhoodAttention2D(
            dim=dim, num_heads=num_heads, h=h, w=w, window_size=window_size,
            dropout=dropout, qk_norm=qk_norm, has_policy_token=has_policy_token
        )
        self.alpha_attn = nn.Parameter(0.1 * torch.ones(dim))
        
        # 3. SwiGLU FFN branch
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim=ffn_dim)
        self.alpha_ffn = nn.Parameter(0.1 * torch.ones(dim))

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 3x3 depthwise convolution over spatial tokens and projects policy token."""
        if self.has_policy_token:
            pol_tok = x[:, 0:1]  # (B, 1, C)
            spatial = x[:, 1:]   # (B, HW, C)
            
            # Reshape spatial to (B, C, H, W) for depthwise conv
            B, HW, C = spatial.shape
            spatial_2d = spatial.transpose(1, 2).view(B, C, self.h, self.w)
            conv_out_2d = self.dw_conv(spatial_2d)
            conv_out = conv_out_2d.flatten(2).transpose(1, 2)  # (B, HW, C)
            conv_out = self.dw_proj(conv_out)
            
            pol_out = self.pol_conv_proj(pol_tok)
            return torch.cat([pol_out, conv_out], dim=1)
        else:
            B, HW, C = x.shape
            x_2d = x.transpose(1, 2).view(B, C, self.h, self.w)
            conv_out_2d = self.dw_conv(x_2d)
            conv_out = conv_out_2d.flatten(2).transpose(1, 2)
            return self.dw_proj(conv_out)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # Branch 1: Depthwise Convolution
        x = x + self.alpha_conv * self._forward_conv(self.norm_conv(x))
        
        # Branch 2: Local Neighborhood Attention (weights only on request: the diagnostics read them)
        attn_out, w = self.attn(self.norm_attn(x), return_weights=return_attn)
        x = x + self.alpha_attn * attn_out
        
        # Branch 3: SwiGLU FFN
        x = x + self.alpha_ffn * self.ffn(self.norm_ffn(x))
        return (x, w) if return_attn else x


class ConvTransformerTrunk(nn.Module):
    """Prediction Trunk built with CNN-Like Lightweight Transformer blocks.
    
    Replaces PolicyTokenTrunk and conventional conv prediction heads:
    - Input: (B, C, H, W) latent state
    - Projects state into tokens, prepends a learned [POLICY] token
    - Passes sequence through depthwise conv + local relative attention + SwiGLU blocks
    - Value and policy logits are read out from the policy token via Linear-GELU-Linear heads
    - Heads' last layers are zero-initialized (uniform prior, zero initial value)
    """
    def __init__(self, channels: int, hw: int, dim: int, depth: int, heads: int, ffn_dim: int,
                 num_actions: int, support_size: int, window_size: int = 3, qk_norm: bool = True):
        super().__init__()
        self.channels = channels
        self.hw = hw
        self.h = int(round(math.sqrt(hw)))
        self.w = self.h
        assert self.h * self.w == hw, f"hw={hw} must be a perfect square"
        self.dim = dim
        self.depth = depth
        
        # Project conv channels into transformer embedding dim
        self.inp = nn.Linear(channels, dim)
        
        # Learned global policy query token (excluded from weight decay)
        self.policy_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.policy_token, std=0.02)
        
        # Stack of hybrid ConvTransformer blocks
        self.blocks = nn.ModuleList([
            ConvTransformerBlock(
                dim=dim, num_heads=heads, h=self.h, w=self.w, ffn_dim=ffn_dim,
                window_size=window_size, qk_norm=qk_norm, has_policy_token=True
            ) for _ in range(depth)
        ])
        self.norm_out = RMSNorm(dim)
        
        # Value and Policy prediction heads reading from the policy token
        self.value_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, support_size)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, num_actions)
        )
        
        # Zero-initialize the final prediction layers (EZ-V2 init_zero)
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.zeros_(self.policy_head[-1].weight)
        nn.init.zeros_(self.policy_head[-1].bias)

    def tokens(self, s: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) state -> (B, 1 + HW, dim) sequence with policy token at index 0."""
        B, C, H, W = s.shape
        t = self.inp(s.flatten(2).transpose(1, 2))  # (B, HW, dim) - NO absolute positional table!
        return torch.cat([self.policy_token.expand(B, -1, -1), t], dim=1)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.tokens(s)
        for blk in self.blocks:
            h = blk(h)
        policy_repr = self.norm_out(h)[:, 0]  # Read from [POLICY] token
        return self.value_head(policy_repr), self.policy_head(policy_repr)

    @torch.no_grad()
    def attention_entropy(self, s: torch.Tensor) -> dict:
        """Diagnostic (Gate 1, 2026-11-03): normalised attention entropy per layer, averaged over batch, heads and queries (1 = uniform over
        the keys a query may see, 0 = one key), plus the same for the [POLICY] query alone. Eval mode, no gradients; the training flag is
        restored."""
        was_training = self.training
        self.eval()
        h = self.tokens(s)
        per_layer, policy_row = [], []
        for blk in self.blocks:
            h, w = blk(h, return_attn=True)
            allowed = (blk.attn.window_mask[0] == 0).to(w.dtype)
            H = normalized_entropy(w, allowed)                       # (B, heads, N)
            per_layer.append(float(H.mean()))
            policy_row.append(float(H[:, :, 0].mean()))
        self.train(was_training)
        return {"per_layer": per_layer, "policy_row": policy_row}

