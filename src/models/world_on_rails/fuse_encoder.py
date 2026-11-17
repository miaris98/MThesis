"""Live side of the two-encoder arm (f1_fuse, 2026-11-09): what the agent feeds a policy trained with --fuse_cache.

Training read one flat vector per frame from two caches (src/training/wor_dataset.py, WOR_FUSE_CACHE):
    [ primary encoder map adaptive-avg-pooled to vision_grid (C1 x gh x gw)  |  second encoder's full map projected on the
      cache's PCA basis (K x Hf x Wf) ]
both stored as fp16 (scripts/training/build_pooled_cache.py: pool_and_store and the --pca_full einsum). This module rebuilds that
vector from a live frame with the same functions, so the policy cannot tell the two apart (QwenWorldOnRailsPolicy._tokenize_state
splits it again at C1*gh*gw). Only the SimLingo second encoder is supported: it is the only one any fuse arm was trained with, and
it takes the raw 1024x512 frame (img_size 512x1024, no crop, no overlay), so the agent passes the camera frame as recorded.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSE_IMG_SIZE = (512, 1024)      # (H, W) the SimLingo fuse cache was built at (= the raw PDM-Lite / CARLA frame)


def fuse_cache_backbone(fuse_cache: str) -> str:
    """The second encoder of a fuse arm, from its cache prefix (run_config.json `fuse_cache`). Hard error for anything but the
    SimLingo cache at the raw frame size: any other pairing has never been trained, so a silent fallback could only drive wrongly."""
    name = str(fuse_cache).replace("\\", "/").rsplit("/", 1)[-1]
    if "_simlingo_" not in name or "_512x1024_" not in name:
        raise NotImplementedError(f"fuse_cache {fuse_cache!r}: the live agent supports only the SimLingo cache at 512x1024")
    return "simlingo"


class FuseSecondEncoder(nn.Module):
    """Frozen second encoder + the PCA basis its cache was written with. forward(rgb) -> (B, K, Hf, Wf), rounded through fp16
    like the cache rows."""

    def __init__(self, basis_path: str, weights_path: Optional[str] = None, k: Optional[int] = None,
                 encoder: Optional[nn.Module] = None):
        super().__init__()
        st = torch.load(basis_path, map_location="cpu")
        mean, basis = st["mean"].float(), st["basis"].float()           # (C,), (K, C)
        if k is not None and basis.shape[0] != int(k):
            raise ValueError(f"{basis_path}: basis has {basis.shape[0]} components, the checkpoint was trained on fuse_pca={k}")
        if encoder is None:
            from src.models.world_on_rails.vision_encoder import build_vision_encoder
            if not weights_path:
                raise ValueError("FuseSecondEncoder needs the SimLingo weights (WOR_FUSE_WEIGHTS): the policy checkpoint does not carry them")
            encoder = build_vision_encoder(backbone_name="simlingo", pretrained=True, freeze_backbone=True, weights_path=weights_path)
        if int(getattr(encoder, "out_channels", basis.shape[1])) != basis.shape[1]:
            raise ValueError(f"{basis_path}: basis is for {basis.shape[1]} channels, the encoder gives {encoder.out_channels}")
        self.enc = encoder.eval()
        self.register_buffer("pca_mean", mean, persistent=False)
        self.register_buffer("pca_basis", basis, persistent=False)
        self.explained = float(st.get("explained", float("nan")))

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        f = self.enc(rgb).float()                                         # (B, C, Hf, Wf)
        pm = torch.einsum("bchw,kc->bkhw", f - self.pca_mean.view(1, -1, 1, 1), self.pca_basis)
        return pm.half().float()


def fused_vision_features(primary_map: torch.Tensor, grid_hw: Tuple[int, int], second_pca: torch.Tensor) -> torch.Tensor:
    """The flat training vector: pooled primary map (pool_and_store: adaptive avg pool in fp32, stored fp16) then the PCA map."""
    pooled = F.adaptive_avg_pool2d(primary_map.float(), tuple(grid_hw)).half().float()
    return torch.cat([pooled.flatten(1), second_pca.float().flatten(1)], dim=1)
