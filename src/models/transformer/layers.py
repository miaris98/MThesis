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
        # The variance is computed in fp32 and only the normalized result is cast back,
        # matching the reference LLaMA/Qwen implementations. Under fp16 autocast the
        # naive version squares activations in half precision: the sum over `dim`
        # elements reaches fp16's 65504 ceiling at activation magnitudes well inside
        # the range an alpha-gated residual stream reaches after a dozen blocks, and
        # the resulting inf/NaN silently costs an optimizer step (the AMP GradScaler
        # skips it) rather than raising.
        dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return normed.to(dtype) * self.weight


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
    """Multi-Head Self-Attention with Trainable Residual Skip Vector (Alpha).

    `qk_norm` (TODO B67, 2026-10-26) adds Qwen3's per-head RMSNorm on the queries and keys before the dot product. Without it, AdamW at
    3e-4 grows the q/k weights until the attention logits reach ~9,000 and every token attends to a single token (Atari B66, update 8,004;
    attention-logit growth, Wortsman et al. 2023). With it the logits are bounded by the norm gains, whatever the size of the q/k weights.
    Off by default: no extra parameters, and the forward pass is the original one, so every existing checkpoint (E15 included) loads
    and computes exactly as before."""
    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:  # RMSNorm's fp32 gain promotes a bf16 input to fp32; SDPA needs q, k and v in one dtype
            q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

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
    def __init__(self, dim: int, num_heads: int = 16, ffn_dim: int = 4096, dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = QwenAttentionWithSkip(dim, num_heads=num_heads, dropout=dropout, qk_norm=qk_norm)
        self.alpha_attn = nn.Parameter(0.1 * torch.ones(dim))

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim=ffn_dim)
        self.alpha_ffn = nn.Parameter(0.1 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.alpha_attn * self.attn(self.norm1(x))
        x = x + self.alpha_ffn * self.ffn(self.norm2(x))
        return x
