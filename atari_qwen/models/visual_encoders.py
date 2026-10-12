"""Visual tokenizers and feature extractors for Atari frame stacks."""
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureCNNEncoder(nn.Module):
    """
    DeepMind Nature CNN adapted as a visual tokenizer for Transformer architectures.
    Produces either a sequence of spatial visual tokens (B, 49, embed_dim) or pooled token.
    """
    def __init__(self, in_channels: int = 4, embed_dim: int = 256, return_spatial_tokens: bool = True):
        super().__init__()
        self.return_spatial_tokens = return_spatial_tokens
        self.embed_dim = embed_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Feature map size for 84x84 input: 64 channels x 7 x 7 spatial grid = 49 tokens
        self.proj = nn.Linear(64, embed_dim) if return_spatial_tokens else nn.Linear(64 * 7 * 7, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 4, 84, 84) in range [0, 255] or [0.0, 1.0]
        Output:
          If return_spatial_tokens: (B, 49, embed_dim)
          Else: (B, 1, embed_dim)
        """
        # Normalize if uint8 or [0, 255]
        if x.dtype != torch.float32 and x.dtype != torch.bfloat16 and x.dtype != torch.float16:
            x = x.float()
        if x.max() > 1.0:
            x = x / 255.0

        feat = self.conv(x)  # (B, 64, 7, 7)
        B, C, H, W = feat.shape
        
        if self.return_spatial_tokens:
            # Reshape to (B, H*W, C) -> (B, 49, 64)
            feat = feat.flatten(2).transpose(1, 2)
            tokens = self.proj(feat)  # (B, 49, embed_dim)
        else:
            feat = feat.reshape(B, -1)  # (B, 3136)
            tokens = self.proj(feat).unsqueeze(1)  # (B, 1, embed_dim)
            
        return self.layer_norm(tokens)


class ImpalaResidualBlock(nn.Module):
    """Residual block used in Impala CNN."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        out = self.conv0(out)
        out = F.relu(out)
        out = self.conv1(out)
        return x + out


class ImpalaConvSequence(nn.Module):
    """Single stage of Impala CNN: Conv -> MaxPool -> 2x ResBlocks."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res0 = ImpalaResidualBlock(out_channels)
        self.res1 = ImpalaResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res0(x)
        x = self.res1(x)
        return x


class ImpalaCNNEncoder(nn.Module):
    """
    Impala CNN visual feature extractor (SOTA representation for RL on arcade games).
    Produces sequence of spatial tokens for transformer reasoning.
    """
    def __init__(self, in_channels: int = 4, embed_dim: int = 256, return_spatial_tokens: bool = True,
                 # E41 (TODO_GTRXL_COLLAPSE_EXPERIMENTS.md): unlike NatureCNNEncoder above, this
                 # encoder has no explicit weight init at all (default PyTorch Kaiming-uniform).
                 # True applies orthogonal_(gain=sqrt(2)) to every Conv2d/Linear here, to test
                 # whether that's washing out input-dependent signal before it reaches the trunk.
                 orthogonal_init: bool = False,
                 # S-043 follow-up: the incremental-integration test (I1) showed swapping ONLY
                 # this encoder into the proven QwenAtariActorCritic+train_ppo.py pipeline at
                 # 300k steps broke a clean 0->14 climb into the same 0/11-oscillation collapse
                 # pattern seen throughout S-024-S-042 -- E41's orthogonal init made things worse
                 # on the (already-broken) GTrXL stack, but was never tried matching
                 # NatureCNNEncoder's own scheme exactly (kaiming_normal_ fan_out relu on convs,
                 # trunc_normal_ on the projection) rather than a generic orthogonal init.
                 kaiming_init: bool = False):
        super().__init__()
        self.return_spatial_tokens = return_spatial_tokens

        # 3 stages: 16, 32, 32 channels. 84x84 -> 42x42 -> 21x21 -> 11x11
        self.stage1 = ImpalaConvSequence(in_channels, 16)
        self.stage2 = ImpalaConvSequence(16, 32)
        self.stage3 = ImpalaConvSequence(32, 32)

        self.proj = nn.Linear(32, embed_dim) if return_spatial_tokens else nn.Linear(32 * 11 * 11, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        if orthogonal_init:
            self._init_weights_orthogonal()
        elif kaiming_init:
            self._init_weights_kaiming()

    def _init_weights_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def _init_weights_orthogonal(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=2.0 ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32 and x.dtype != torch.bfloat16 and x.dtype != torch.float16:
            x = x.float()
        if x.max() > 1.0:
            x = x / 255.0

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)  # (B, 32, 11, 11)
        
        B, C, H, W = x.shape
        if self.return_spatial_tokens:
            x = x.flatten(2).transpose(1, 2)  # (B, 121, 32)
            tokens = self.proj(x)             # (B, 121, embed_dim)
        else:
            x = x.reshape(B, -1)
            tokens = self.proj(x).unsqueeze(1)
            
        return self.layer_norm(tokens)


class PatchTokenizer(nn.Module):
    """
    ViT-style patch projection for Atari frames.
    84x84 divided into 14x14 patches = 6x6 = 36 visual tokens.
    """
    def __init__(self, in_channels: int = 4, embed_dim: int = 256, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32 and x.dtype != torch.bfloat16 and x.dtype != torch.float16:
            x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
            
        patches = self.patch_embed(x)  # (B, embed_dim, H_p, W_p)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return self.layer_norm(tokens)
