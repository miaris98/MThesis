"""TF++'s camera branch run the way TF++ itself runs it, camera only (2026-11-08).

carla_encoder.py runs the regnety_032 image branch of the TF++ release stage after stage. TF++ (carla_garage team_code/transfuser.py,
TransfuserBackbone.forward) adds a fusion residual after each of the 4 stages: a GPT block over pooled image and LiDAR tokens. Without
it, stages 2-4 see inputs they never saw in training. scripts/analysis/tfpp_fusion_probe.py measured the damage on 73 PDM-Lite frames in
TF++'s own 1024x384 geometry: TF++'s depth head read from our no-fusion features is off by 7.2 m on average, against 3.0 m with the fusion
blocks kept and an empty LiDAR, 2.7 m with a pseudo-LiDAR splatted from the camera's own predicted depth, and 1.7 m with the real LiDAR;
semantics barely change. The policy input stays RGB only (no LiDAR is ever read; pseudo-LiDAR from predicted depth is allowed by the
project rules).

Modes:
  stages        the image stages only (= carla_encoder.py; for a same-geometry control)
  fusion_zero   the full TF++ backbone with an all-zero LiDAR BEV
  fusion_pseudo fusion_zero -> TF++'s depth head -> back-projected points -> TF++'s LiDAR histogram -> the full backbone again

`forward(x)` takes the training pipeline's (B, 3, H, W) image in [0, 1] and returns the stride-32 image features (B, 1512, H/32, W/32)
that TF++'s perspective decoders read. `decoder_maps(feats)` returns TF++'s depth (B, H, W, 0..1 = 0..50 m) and semantic probabilities
(B, 7, H, W). Everything is frozen and in eval mode; every tensor comes from the TF++ checkpoint (no ImageNet weights are fetched).
"""
import glob
import math
import os
import sys
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TFPP_TEAM_CODE = os.path.join(_REPO, "Carla-utils", "carla_garage", "team_code")


def _stub_carla_modules():
    """config.py / transfuser_utils.py / nav_planner.py import carla and CARLA's PythonAPI agents for colours, red-light helpers and
    driving control only. Permissive stubs let the perception modules load on a box without CARLA."""
    class _Any:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

    def _mod_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any

    for mod in ("carla", "agents", "agents.navigation", "agents.navigation.global_route_planner", "agents.navigation.local_planner",
                "agents.tools", "agents.tools.misc"):
        if mod in sys.modules:
            continue
        stub = types.ModuleType(mod)
        stub.__getattr__ = _mod_getattr
        stub.__path__ = []
        sys.modules[mod] = stub


def load_tfpp(tfpp_dir: str, device: str = "cpu"):
    """The released TF++ LidarCenterNet and its config, strict load, eval mode."""
    _stub_carla_modules()
    if TFPP_TEAM_CODE not in sys.path:
        sys.path.insert(0, TFPP_TEAM_CODE)
    import jsonpickle
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import timm
    jsonpickle_numpy.register_handlers()
    from config import GlobalConfig  # noqa: E402  (carla_garage)
    from model import LidarCenterNet  # noqa: E402
    cfg = GlobalConfig()
    with open(os.path.join(tfpp_dir, "config.json"), "rt", encoding="utf-8") as f:
        cfg.__dict__.update(jsonpickle.decode(f.read()).__dict__)
    create = timm.create_model
    timm.create_model = lambda *a, **k: create(*a, **{**k, "pretrained": False})   # every tensor comes from the checkpoint below
    try:
        net = LidarCenterNet(cfg)
    finally:
        timm.create_model = create
    ck = sorted(glob.glob(os.path.join(tfpp_dir, "model*_0.pth")) or glob.glob(os.path.join(tfpp_dir, "model*.pth")))[0]
    sd = torch.load(ck, map_location="cpu")
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"TF++ checkpoint {ck} did not load cleanly: {len(missing)} missing, {len(unexpected)} unexpected")
    for p in net.parameters():
        p.requires_grad = False
    return net.to(device).eval(), cfg, ck


class TfppCamera(nn.Module):
    """Frozen TF++ camera features with the fusion blocks kept (see module docstring)."""

    MODES = ("stages", "fusion_zero", "fusion_pseudo")
    PSEUDO_LIDAR_Z_OFFSET = 1.93   # camera-frame height -> logged LiDAR frame (ground at -0.07 m there; camera 2.0 m above ground)

    def __init__(self, tfpp_dir: str, mode: str = "fusion_zero"):
        super().__init__()
        assert mode in self.MODES, f"mode must be one of {self.MODES}"
        self.mode = mode
        net, cfg, ck = load_tfpp(tfpp_dir, "cpu")
        self.net, self.cfg, self.checkpoint = net, cfg, ck
        self.out_channels = int(net.backbone.num_image_features)
        self.backbone_name = f"tfpp_{mode}"
        print(f"--> TF++ camera ({mode}) from {ck}: {self.out_channels} channels, lidar {cfg.lidar_resolution_height}x{cfg.lidar_resolution_width} "
              f"x{cfg.lidar_seq_len * (2 if cfg.use_ground_plane else 1)}", flush=True)

    def train(self, mode: bool = True):
        return super().train(False)

    def _lidar_channels(self) -> int:
        return (2 if self.cfg.use_ground_plane else 1) * int(self.cfg.lidar_seq_len)

    @torch.no_grad()
    def _backbone(self, rgb255: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        _, _, image_feature_grid = self.net.backbone(rgb255, lidar)
        return image_feature_grid

    @torch.no_grad()
    def _stages(self, rgb255: torch.Tensor) -> torch.Tensor:
        import transfuser_utils as t_u
        x = t_u.normalize_imagenet(rgb255)
        for _, m in self.net.backbone.image_encoder.items():
            x = m(x)
        return x

    @torch.no_grad()
    def decoder_maps(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = torch.sigmoid(self.net.depth_decoder(feats.float())).squeeze(1)
        sem = torch.softmax(self.net.semantic_decoder(feats.float()), dim=1)
        return depth, sem

    @torch.no_grad()
    def pseudo_lidar_bev(self, depth01: torch.Tensor, img_hw: Tuple[int, int], crop_bottom_frac: float) -> torch.Tensor:
        """(B, H, W) predicted depth (0..1 = 0..50 m) of an image made from the 1024x512 camera by cropping the bottom crop_bottom_frac
        and resizing to img_hw -> TF++'s LiDAR histogram (B, C, 256, 256) of the back-projected points above the split height."""
        cfg = self.cfg
        B, H, W = depth01.shape
        dev = depth01.device
        f = float(cfg.camera_width / (2.0 * math.tan(math.radians(float(cfg.camera_fov)) / 2.0)))   # focal length in source pixels (half the FOV)
        cx, cy = cfg.camera_width / 2.0, cfg.camera_height / 2.0
        sx = cfg.camera_width / W
        sy = cfg.camera_height * (1.0 - crop_bottom_frac) / H
        vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        u0, v0 = (uu + 0.5) * sx - 0.5, (vv + 0.5) * sy - 0.5
        d = depth01.float() * 50.0
        x = d
        y = (u0 - cx) / f * d
        z = -(v0 - cy) / f * d + self.PSEUDO_LIDAR_Z_OFFSET
        xl = x + cfg.camera_pos[0] - cfg.lidar_pos[0]
        ok = (d > 0.5) & (d < 49.0) & (z > cfg.lidar_split_height) & (z < cfg.max_height_lidar)
        ppm = float(cfg.pixels_per_meter)
        xi = torch.floor((xl - cfg.min_x) * ppm).long()
        yi = torch.floor((y - cfg.min_y) * ppm).long()
        nx = int((cfg.max_x - cfg.min_x) * ppm)
        ny = int((cfg.max_y - cfg.min_y) * ppm)
        ok = ok & (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        hist = torch.zeros(B, ny * nx, device=dev)
        idx = (yi * nx + xi).clamp(0, ny * nx - 1)    # np.histogramdd(x, y) then .T -> row = y bin, column = x bin
        hist.scatter_add_(1, idx.view(B, -1), ok.view(B, -1).float())
        hist = (hist.clamp(max=cfg.hist_max_per_pixel) / cfg.hist_max_per_pixel).view(B, 1, ny, nx)
        if self._lidar_channels() > 1:   # ground-plane channel (below the split) left empty: not used by the released config
            hist = torch.cat([torch.zeros_like(hist).expand(-1, self._lidar_channels() - 1, -1, -1), hist], 1)
        return hist

    @torch.no_grad()
    def forward(self, x01: torch.Tensor, crop_bottom_frac: Optional[float] = None, return_maps: bool = False):
        """x01 (B, 3, H, W) in [0, 1] -> stride-32 image features; with return_maps also (depth, semantic probabilities) of those features."""
        rgb255 = x01.float() * 255.0
        if self.mode == "stages":
            feats = self._stages(rgb255)
        else:
            B = rgb255.shape[0]
            zero = torch.zeros(B, self._lidar_channels(), self.cfg.lidar_resolution_height, self.cfg.lidar_resolution_width, device=rgb255.device)
            feats = self._backbone(rgb255, zero)
            if self.mode == "fusion_pseudo":
                depth, _ = self.decoder_maps(feats)
                crop = 0.25 if crop_bottom_frac is None else crop_bottom_frac
                feats = self._backbone(rgb255, self.pseudo_lidar_bev(depth, tuple(x01.shape[-2:]), crop))
        if return_maps:
            return feats, self.decoder_maps(feats)
        return feats


def decoder_stat_maps(depth01: torch.Tensor, sem: torch.Tensor, crop_bottom_frac: float, cell: int = 32) -> torch.Tensor:
    """(B, 10, H/cell, W/cell) per stride-32 cell: mean depth, min depth, mean (depth / flat-ground depth, clipped to 2; 2 above the horizon)
    and the 7 TF++ semantic class fractions (2026-11-08, idea 10). Depths in the decoder's 0..1 = 0..50 m units."""
    from src.config.camera import PDM_LITE_CAMERA
    B, H, W = depth01.shape
    Wc, Hc = float(PDM_LITE_CAMERA["width"]), float(PDM_LITE_CAMERA["height"])
    f = Wc / (2.0 * math.tan(math.radians(float(PDM_LITE_CAMERA["fov"])) / 2.0))   # focal length in source pixels (half the FOV)
    sy = Hc * (1.0 - crop_bottom_frac) / H
    v0 = (torch.arange(H, device=depth01.device, dtype=torch.float32) + 0.5) * sy - 0.5
    below = (v0 - Hc / 2.0).clamp(min=1e-3)
    ground = (float(PDM_LITE_CAMERA["z"]) * f / below / 50.0).view(1, H, 1)        # planar depth of a flat road at each row, 0..1 units
    ratio = torch.where((v0 - Hc / 2.0).view(1, H, 1) > 1.0, (depth01 / ground).clamp(max=2.0), torch.full_like(depth01, 2.0))
    d = depth01.unsqueeze(1).float()
    stats = torch.cat([F.avg_pool2d(d, cell), -F.max_pool2d(-d, cell), F.avg_pool2d(ratio.unsqueeze(1).float(), cell),
                       F.avg_pool2d(sem.float(), cell)], dim=1)
    return stats
