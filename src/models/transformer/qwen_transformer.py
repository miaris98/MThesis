"""Qwen Decision Transformer and 500M Vision Transformer for CARLA End-to-End Driving."""
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint

from src.models.transformer.layers import RMSNorm, QwenTransformerBlock


class Qwen500MVisionTransformer(nn.Module):
    """
    ~500M Parameter Qwen-Style Vision Transformer for CARLA End-to-End Driving Policy.
    - 28 Layers, 1024 Hidden Dimension, 16 Attention Heads, 4096 SwiGLU FFN.
    - Multi-Camera Panorama Patch Tokenizer (16x16) + Speed Token + [DRIVE] Token.
    """
    def __init__(
        self, 
        features_dim: int = 512, 
        depth: int = 28, 
        embed_dim: int = 1024, 
        num_heads: int = 16, 
        ffn_dim: int = 4096,
        patch_size: int = 16,
        freeze_backbone: bool = False,
        weights_path: Optional[str] = None
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.drive_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.speed_proj = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList([
            QwenTransformerBlock(dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(embed_dim)
        
        self.head_proj = nn.Sequential(
            nn.Linear(embed_dim, features_dim),
            nn.GELU(),
            nn.Linear(features_dim, features_dim)
        )

        self._init_weights()

        if weights_path is not None and os.path.exists(weights_path):
            print(f"--> Loading Qwen-500M checkpoint from: {weights_path}")
            try:
                ckpt = torch.load(weights_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
                self.load_state_dict(state_dict, strict=False)
                print("✓ Successfully loaded Qwen-500M weights!")
            except Exception as e:
                print(f"--> Notice: Could not load weights ({e}). Initialized from scratch.")

        if self.freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.head_proj.parameters():
                param.requires_grad = True

    def _init_weights(self):
        nn.init.trunc_normal_(self.drive_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract [DRIVE] token visual representation from multi-camera panorama."""
        img_x = image.float() / 255.0
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=img_x.is_cuda):
            if img_x.ndim == 4 and img_x.shape[-1] == 3:
                img_perm = img_x.permute(0, 3, 1, 2)
            else:
                img_perm = img_x

            patches = self.patch_embed(img_perm).flatten(2).transpose(1, 2)
            B = patches.shape[0]

            drive_tokens = self.drive_token.expand(B, -1, -1)
            tokens = torch.cat([drive_tokens, patches], dim=1)
            
            seq_len = tokens.shape[1]
            tokens = tokens + self.pos_embed[:, :seq_len, :]
            tokens = self.pos_drop(tokens)

            for block in self.blocks:
                tokens = block(tokens)

            tokens = self.final_norm(tokens)
            drive_repr = tokens[:, 0]
            
        return drive_repr.float()

    def forward_with_visual_features(self, visual_features: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        speed_norm = speed.float().view(-1, 1) / 50.0
        speed_emb = self.speed_proj(speed_norm)
        fused = visual_features + speed_emb
        return self.head_proj(fused)

    def forward(self, image: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        visual_features = self.extract_visual_features(image)
        return self.forward_with_visual_features(visual_features, speed)


class QwenDecisionTransformer(nn.Module):
    """
    900M / 500M Parameter Qwen-Style Decision Transformer for PPO Actor-Critic.
    - 28 Layers, 1024 Hidden Dim (500M) / 24 Layers, 1536 Hidden Dim (900M).
    - Trainable Attention Skip Connections (Qwen Alpha Gating).
    """
    def __init__(self, in_features: int = 1536, action_dim: int = 3, model_size: str = "500m"):
        super().__init__()
        if model_size == "900m":
            depth = 24
            embed_dim = 1536
            num_heads = 24
            ffn_dim = 6144
        else:
            depth = 28
            embed_dim = 1024
            num_heads = 16
            ffn_dim = 4096

        self.embed_dim = embed_dim
        
        self.vision_proj = nn.Linear(in_features, embed_dim)
        self.speed_proj = nn.Linear(1, embed_dim)
        
        self.actor_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.critic_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            QwenTransformerBlock(dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(embed_dim)
        
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.actor_token, std=0.02)
        nn.init.trunc_normal_(self.critic_token, std=0.02)
        with torch.no_grad():
            self.actor_head[2].bias.data[0] = 0.5
            self.actor_head[2].bias.data[1] = 0.0
            self.actor_head[2].bias.data[2] = -0.5

    def forward(self, visual_features: torch.Tensor, speed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = visual_features.shape[0]
        v_tok = self.vision_proj(visual_features).unsqueeze(1)
        
        speed_norm = speed.float().view(-1, 1) / 50.0
        s_tok = self.speed_proj(speed_norm).unsqueeze(1)
        
        a_tok = self.actor_token.expand(B, -1, -1)
        c_tok = self.critic_token.expand(B, -1, -1)
        
        tokens = torch.cat([a_tok, c_tok, v_tok, s_tok], dim=1)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=visual_features.is_cuda):
            for block in self.blocks:
                if self.training and tokens.requires_grad:
                    tokens = torch.utils.checkpoint.checkpoint(block, tokens, use_reentrant=False)
                else:
                    tokens = block(tokens)
            tokens = self.final_norm(tokens)
            
            actor_repr = tokens[:, 0]
            critic_repr = tokens[:, 1]
            
            action_mean = self.actor_head(actor_repr).float()
            value = self.critic_head(critic_repr).squeeze(-1).float()
            
        return action_mean, self.actor_log_std, value
