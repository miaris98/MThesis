"""Per-vision-token camera geometry, derived on the fly from the one camera definition.

WHY THIS EXISTS
---------------
The trunk receives the encoder's feature map as a flat set of tokens. Self-attention is
permutation-invariant, so without a positional signal the transformer cannot tell the token
covering the road ahead from the one covering the sky - it only has `vision_pos`, a *learned*
embedding that has to discover the image's geometry from scratch, from waypoint supervision
alone. The conv head has the same information implicitly, through the spatial structure of its
convolutions.

But since the camera-parity fix, the geometry is no longer unknown: `src/config/camera.py` pins
the data-collection camera exactly (x=-1.5, z=2.0, pitch=0, fov=110 deg, 1024x512) and the same
definition drives the evaluation agent. So for every cell of the feature grid we can *compute*
where that cell is pointing, and hand the trunk metric geometry instead of making it infer one.

This adds no sensor and no new data: it is a fixed function of the camera intrinsics and the
grid shape, computed once and cached. It is also independent of the frozen encoder, so it does
not invalidate the feature cache.

WHAT EACH CHANNEL IS
--------------------
    0,1  sin/cos of azimuth    - bearing of the cell's ray, left-right. Always defined. This is
                                 the channel that matters most: it maps a column index directly
                                 onto a steering direction.
    2,3  sin/cos of elevation  - up-down angle. Always defined.
    4    ground x (forward)    - where the ray meets the road, in metres ahead of the *ego
                                 origin*, under a flat-world assumption.
    5    ground y (lateral)    - same intersection, metres left/right.
    6    ground valid          - 1 where that intersection exists and is in range, else 0.

Angles are given as sin/cos rather than raw radians because a single unbounded scalar makes a
poor input to a linear layer, and because the pair is smooth everywhere.

THE FLAT-GROUND CAVEAT
----------------------
Channels 4-6 assume the road is a plane at the camera's own height below it. On a slope that is
wrong. It is offered as a *prior*, not a measurement: the learned `vision_pos` is kept alongside
it precisely so the trunk can absorb the residual. Note also that with pitch=0 the horizon sits
at the vertical centre of the uncropped image, which after the bottom-quarter crop is 2/3 of the
way down - so on a 6-row feature grid only the bottom ~2 rows have a valid ground intersection
at all. Those are the near-field road cells, which is where it matters; the rest fall back to
the always-defined angular channels and a zeroed, flagged-invalid ground position.
"""
import math
from typing import Dict, Tuple

import torch

from src.config.camera import PDM_LITE_CAMERA

# sin/cos azimuth, sin/cos elevation, ground x, ground y, ground-valid flag.
RAY_GEOMETRY_CHANNELS = 7

# Normalisation ranges for the ground intersection. Distance to the horizon diverges, so the
# forward channel has to be bounded by something; 75 m is comfortably beyond any distance this
# policy plans over (5 waypoints at PDM-Lite's step size), and 25 m of lateral offset covers a
# wide junction. Values past these clamp rather than explode.
_MAX_FORWARD_M = 75.0
_MAX_LATERAL_M = 25.0

_CACHE: Dict[tuple, torch.Tensor] = {}


def vision_grid_geometry(
    grid_hw: Tuple[int, int],
    crop_bottom_frac: float = 0.0,
    camera: Dict[str, float] = None,
) -> torch.Tensor:
    """Geometry for one feature grid, as (RAY_GEOMETRY_CHANNELS, gh, gw) float32 on CPU.

    Depends only on the grid shape, the bottom crop and the camera - *not* on the resize
    resolution, since scaling an image does not change where its pixels point. Cached, because
    it is the same tensor on every forward pass of a run.

    A 1x1 grid (the globally-pooled ablation) has no meaningful per-cell geometry: one token
    covers the whole frame, so every channel is returned as zero with the valid flag clear.
    """
    cam = camera or PDM_LITE_CAMERA
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    key = (gh, gw, round(float(crop_bottom_frac), 6),
           cam["width"], cam["height"], cam["fov"], cam["z"], cam["x"])
    if key in _CACHE:
        return _CACHE[key]

    out = torch.zeros(RAY_GEOMETRY_CHANNELS, max(gh, 1), max(gw, 1), dtype=torch.float32)
    if gh <= 1 and gw <= 1:
        _CACHE[key] = out
        return out

    src_w, src_h = float(cam["width"]), float(cam["height"])
    # CARLA's `fov` is horizontal - see camera.vertical_fov_deg for why that distinction has
    # already caused a bug in this project.
    focal = (src_w / 2.0) / math.tan(math.radians(float(cam["fov"])) / 2.0)
    cx, cy = src_w / 2.0, src_h / 2.0
    kept_h = src_h * (1.0 - float(crop_bottom_frac))
    cam_h, cam_x = float(cam["z"]), float(cam["x"])

    # Cell centres, in the coordinates of the *uncropped* source image. The crop takes rows off
    # the bottom, so a row index in the cropped image is the same row in the source.
    col = (torch.arange(gw, dtype=torch.float32) + 0.5) / gw * src_w
    row = (torch.arange(gh, dtype=torch.float32) + 0.5) / gh * kept_h
    dx = ((col - cx) / focal).view(1, gw).expand(gh, gw)          # +right
    dy = ((row - cy) / focal).view(gh, 1).expand(gh, gw)          # +down

    azimuth = torch.atan(dx)
    elevation = torch.atan(-dy / torch.sqrt(1.0 + dx * dx))

    # Ray from the camera meets the plane `cam_h` below it only where it points downward.
    below = dy > 1e-6
    t = cam_h / dy.clamp_min(1e-6)
    ego_x = cam_x + t                       # camera sits cam_x metres from the ego origin
    ego_y = t * dx
    valid = below & (ego_x > 0) & (ego_x <= _MAX_FORWARD_M) & (ego_y.abs() <= _MAX_LATERAL_M)

    out[0] = torch.sin(azimuth)
    out[1] = torch.cos(azimuth)
    out[2] = torch.sin(elevation)
    out[3] = torch.cos(elevation)
    out[4] = torch.where(valid, ego_x.clamp(0.0, _MAX_FORWARD_M) / _MAX_FORWARD_M,
                         torch.zeros_like(ego_x))
    out[5] = torch.where(valid, ego_y.clamp(-_MAX_LATERAL_M, _MAX_LATERAL_M) / _MAX_LATERAL_M,
                         torch.zeros_like(ego_y))
    out[6] = valid.to(torch.float32)

    _CACHE[key] = out
    return out


def append_ray_geometry(feats: torch.Tensor, crop_bottom_frac: float = 0.0) -> torch.Tensor:
    """Concatenates the geometry for `feats`' own grid onto its channel dimension.

    Takes the grid from the tensor rather than from a stored config so the same call is correct
    whatever --img_size a run uses, and so a mismatch between the two is impossible by
    construction.
    """
    b, _, h, w = feats.shape
    geo = vision_grid_geometry((h, w), crop_bottom_frac).to(device=feats.device, dtype=feats.dtype)
    return torch.cat([feats, geo.unsqueeze(0).expand(b, -1, -1, -1)], dim=1)
