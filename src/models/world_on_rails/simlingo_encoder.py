"""Frozen SimLingo vision tower as a drop-in encoder for the WoR heads (TODO A65-A66, 2026-10-14).

SimLingo (Renz et al., CVPR 2025) fine-tunes the InternVL2-1B vision tower (InternViT-300M + the pixel-shuffle MLP `mlp1`) on CARLA driving; the checkpoint
`RenzKa/simlingo` (epoch 13) holds those weights under `vision_model.image_encoder.model.{vision_model,mlp1}.*`. This module keeps only that part, frozen:

    forward(x) with x = the raw 1024x512 camera frame, (B, 3, H, W) or (B, H, W, 3), [0, 255] or [0, 1]
      1. drop the bottom 4.8/16 of the image (the bonnet; SimLingo's `cut_bottom_quarter`),
      2. resize to 896x448 and cut two 448x448 tiles (InternVL's dynamic aspect-ratio rule for a 2.5:1 frame, max 2 tiles),
      3. ImageNet normalisation, InternViT, pixel shuffle, `mlp1`  -> 256 tokens x 896 channels per tile,
      4. tokens laid back on the image: (B, 896, 16, 32), the interface the heads pool (`out_channels` = 896).
    Train, cache and evaluate through this one function (dataset: --img_size 512x1024 --crop_bottom_frac 0 --route_overlay 0) so the pixels cannot drift apart.

The architecture comes from an OpenGVLab/InternVL2-1B snapshot directory (`SIMLINGO_INTERNVL`, default /workspace/internvl2_1b: config + modelling code, weights are not
read); the tuned weights come from `weights_path` (SimLingo's pytorch_model.pt, read with torch.load(weights_only=True)) while training and from the run's
frozen_backbone.pth (keys `encoder.m.*`) while evaluating, exactly like the regnety encoder.
"""
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_SIM_PREFIX = "vision_model.image_encoder.model."


class SimLingoEncoder(nn.Module):
    def __init__(self, weights_path: Optional[str] = None, freeze_backbone: bool = True, internvl_dir: Optional[str] = None, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        import sys
        extra = os.environ.get("SIMLINGO_PKGS", "/workspace/simlingo_pkgs")
        if os.path.isdir(extra) and extra not in sys.path:
            sys.path.insert(0, extra)                      # only transformers + its small deps live there (--no-deps), so torch / timm / numpy of the venv are not shadowed
        from transformers import AutoConfig, AutoModel
        d = internvl_dir or os.environ.get("SIMLINGO_INTERNVL", "/workspace/internvl2_1b")
        cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
        m = AutoModel.from_config(cfg, trust_remote_code=True)
        m.language_model = None                       # only the vision tower + mlp1 are used
        self.m = m.to(dtype)
        self.freeze_backbone = bool(freeze_backbone)
        self.out_channels = int(cfg.llm_config.hidden_size)
        self.backbone_name = "simlingo"
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)
        if weights_path:
            self._load_simlingo(weights_path)
        if self.freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def _load_simlingo(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        for k in ("state_dict", "module", "model"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
        own = {k[len(_SIM_PREFIX):]: v for k, v in sd.items()
               if k.startswith(_SIM_PREFIX) and (k[len(_SIM_PREFIX):].startswith("vision_model.") or k[len(_SIM_PREFIX):].startswith("mlp1."))}
        missing, unexpected = self.m.load_state_dict({k: v.to(torch.bfloat16) for k, v in own.items()}, strict=False)
        miss = [k for k in missing if k.startswith("vision_model.") or k.startswith("mlp1.")]
        assert own and not miss, f"SimLingo checkpoint does not cover the vision tower: {len(own)} tensors, missing {miss[:5]}"
        print(f"--> SimLingo vision tower: {len(own)} tensors from {path}, unexpected {len(unexpected)}", flush=True)

    def train(self, mode: bool = True):
        return super().train(False if self.freeze_backbone else mode)

    @torch.no_grad()
    def _tiles(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = x[:, :, : H - int(H * 4.8 // 16)]
        assert abs(x.shape[-1] / x.shape[-2] - 1024 / 359) < 0.3, f"unexpected aspect {tuple(x.shape)}: the encoder expects the raw 1024x512 frame"
        x = F.interpolate(x.float(), size=(448, 896), mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
        x = (x - self.mean) / self.std
        return torch.cat([x[..., :448], x[..., 448:]], dim=0)             # (2B, 3, 448, 448): left tiles then right tiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        B = x.shape[0]
        px = self._tiles(x).to(self.m.vision_model.embeddings.patch_embedding.weight.dtype)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            tok = self.m.extract_feature(px).float()                       # (2B, 256, C)
        C = tok.shape[-1]
        tok = tok.view(2, B, 16, 16, C)                                    # [left, right] x batch x y x x x channel
        return torch.cat([tok[0], tok[1]], dim=2).permute(0, 3, 1, 2).contiguous()   # (B, C, 16, 32)
