"""Fundamental transformer building blocks: RMSNorm, SwiGLU, and Attention with Trainable Skips."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) used in modern LLMs/VLMs."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit (SwiGLU) Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class QwenAttentionWithSkip(nn.Module):
    """Multi-Head Self-Attention with Trainable Residual Skip Vector (Alpha)."""
    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
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


class QwenTransformerBlock(nn.Module):
    """Qwen-style Transformer Block with Trainable Attention Skip Connection (Qwen Alpha Gating)."""
    def __init__(self, dim: int, num_heads: int = 16, ffn_dim: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = QwenAttentionWithSkip(dim, num_heads=num_heads, dropout=dropout)
        self.alpha_attn = nn.Parameter(0.1 * torch.ones(dim))

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim=ffn_dim)
        self.alpha_ffn = nn.Parameter(0.1 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.alpha_attn * self.attn(self.norm1(x))
        x = x + self.alpha_ffn * self.ffn(self.norm2(x))
        return x
