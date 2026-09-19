"""Qwen-style Transformer Policy and Value Network for Atari Discrete Control."""
import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.distributions.categorical import Categorical

from src.models.transformer.layers import RMSNorm, QwenTransformerBlock
from atari_qwen.models.visual_encoders import NatureCNNEncoder, ImpalaCNNEncoder, PatchTokenizer
from atari_qwen.models.gtrxl_layers import GTrXLBlock
from atari_qwen.config.atari_config import QWEN_ATARI_PRESETS


class QwenAtariActorCritic(nn.Module):
    """
    Qwen-style Actor-Critic policy for discrete Atari action spaces.
    
    Architecture:
      1. Visual Tokenizer (Nature CNN, Impala CNN, or Patch Tokenizer) converts (B, 4, 84, 84) -> (B, N_v, embed_dim)
      2. Learnable [ACTOR] and [CRITIC] tokens prepended -> sequence length 2 + N_v
      3. Stack of QwenTransformerBlock (RMSNorm, Multi-Head Self-Attention with Trainable Alpha Skips, SwiGLU FFN)
      4. Final RMSNorm
      5. Actor Head: Linear -> GELU -> Categorical logits over action_dim
      6. Critic Head: Linear -> GELU -> Scalar state value V(s)
    """
    def __init__(
        self,
        action_dim: int,
        in_channels: int = 4,
        preset: str = "tiny",
        encoder_type: str = "nature_cnn",
        patch_size: int = 14,
        depth: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
        # Incremental-integration plan (struggle-solutions S-041): start from this proven
        # pipeline (train_ppo.py genuinely climbed 0.00->17.00 with the default "qwen" blocks)
        # and swap in ImpalaGTrXLAgent's components one at a time to isolate which one breaks
        # learning, instead of varying hyperparameters on a stack that already differs from the
        # proven baseline in five places at once. "gtrxl" swaps only the transformer block type
        # (GTrXLBlock with GRU gating off, matching ImpalaGTrXLAgent's use_gru_gating=False
        # isolation config) -- everything else (heads, init, RMSNorm, PPO trainer) stays as-is.
        block_type: str = "qwen",  # "qwen" (proven default) or "gtrxl"
        # S-043: ImpalaCNNEncoder has no explicit weight init by default (unlike
        # NatureCNNEncoder). True applies NatureCNNEncoder's own kaiming_normal_ scheme to it --
        # only relevant when encoder_type="impala_cnn".
        impala_kaiming_init: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.encoder_type = encoder_type.lower()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.block_type = block_type

        # Resolve preset dimensions if not explicitly provided
        cfg = QWEN_ATARI_PRESETS.get(preset.lower(), QWEN_ATARI_PRESETS["tiny"])
        self.depth = depth or cfg["depth"]
        self.embed_dim = embed_dim or cfg["embed_dim"]
        self.num_heads = num_heads or cfg["num_heads"]
        self.ffn_dim = ffn_dim or cfg["ffn_dim"]

        # 1. Visual Tokenizer
        if "impala" in self.encoder_type:
            self.visual_encoder = ImpalaCNNEncoder(
                in_channels=in_channels,
                embed_dim=self.embed_dim,
                return_spatial_tokens=True,
                kaiming_init=impala_kaiming_init,
            )
            # 11x11 grid = 121 visual tokens
            max_visual_tokens = 128
        elif "patch" in self.encoder_type:
            self.visual_encoder = PatchTokenizer(
                in_channels=in_channels,
                embed_dim=self.embed_dim,
                patch_size=patch_size
            )
            # 84/14 = 6x6 = 36 tokens
            max_visual_tokens = 64
        else:  # Default: Nature CNN
            self.visual_encoder = NatureCNNEncoder(
                in_channels=in_channels,
                embed_dim=self.embed_dim,
                return_spatial_tokens=True
            )
            # 7x7 grid = 49 visual tokens
            max_visual_tokens = 64

        # 2. Dual Query Tokens
        self.actor_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.critic_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # 3. Positional Embeddings (covering 2 query tokens + max visual tokens)
        total_seq_len = 2 + max_visual_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, total_seq_len, self.embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 4. Transformer Blocks: QwenTransformerBlock (RMSNorm, SwiGLU, Trainable Alpha Gating --
        # the proven default) or GTrXLBlock (GRU-gated, gating disabled to isolate just the
        # block's attention/FFN structure) per block_type.
        if self.block_type == "gtrxl":
            self.blocks = nn.ModuleList([
                GTrXLBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=dropout,
                    bg_init=0.0,
                    use_gru_gating=False,
                )
                for _ in range(self.depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                QwenTransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=dropout
                )
                for _ in range(self.depth)
            ])
        self.final_norm = RMSNorm(self.embed_dim)

        # 5. Actor & Critic Heads
        self.actor_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal / truncated normal initialization for stable RL dynamics."""
        nn.init.trunc_normal_(self.actor_token, std=0.02)
        nn.init.trunc_normal_(self.critic_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Policy output layer orthogonal init with small gain (standard PPO practice)
        for m in self.actor_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Critic output layer orthogonal init with gain 1.0
        for m in self.critic_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_transformer(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through visual encoder and Qwen transformer trunk."""
        B = x.shape[0]
        
        # Extract visual tokens: (B, N_v, embed_dim)
        vis_tokens = self.visual_encoder(x)
        N_v = vis_tokens.shape[1]

        # Expand query tokens: (B, 1, embed_dim)
        a_tok = self.actor_token.expand(B, -1, -1)
        c_tok = self.critic_token.expand(B, -1, -1)

        # Concatenate: [ACTOR, CRITIC, v_1, v_2, ..., v_Nv] -> (B, 2 + N_v, embed_dim)
        tokens = torch.cat([a_tok, c_tok, vis_tokens], dim=1)
        seq_len = tokens.shape[1]

        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :seq_len, :]
        tokens = self.pos_drop(tokens)

        # Run through Qwen Transformer blocks with alpha gating
        for block in self.blocks:
            if self.training and self.use_gradient_checkpointing and tokens.requires_grad:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)

        tokens = self.final_norm(tokens)
        
        actor_repr = tokens[:, 0]   # Contextualized [ACTOR] token
        critic_repr = tokens[:, 1]  # Contextualized [CRITIC] token
        return actor_repr, critic_repr

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        _, critic_repr = self._forward_transformer(x)
        return self.critic_head(critic_repr).squeeze(-1)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute action distribution and state value.
        
        Returns:
            action: (B,) selected discrete action indices
            log_prob: (B,) log probability of selected actions
            entropy: (B,) entropy of the action distribution
            value: (B,) scalar state value V(s)
        """
        actor_repr, critic_repr = self._forward_transformer(x)
        
        logits = self.actor_head(actor_repr)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic_head(critic_repr).squeeze(-1)
        
        return action, log_prob, entropy, value
