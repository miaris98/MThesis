"""Shared loss/metric computation and the held-out validation pass for WoR training.

The loss below is the single definition used by both the training loop and
validation, so the two cannot drift apart - a held-out number computed by a second,
separately-maintained copy of the objective is worse than no held-out number at all.
"""
from typing import Dict, Optional
import time

import torch
import torch.nn.functional as F


def waypoint_losses(
    out: Dict[str, torch.Tensor],
    target_wp: torch.Tensor,
    target_q: torch.Tensor,
    wp_loss_weight: float = 1.0,
    q_loss_weight: float = 0.0,
    lateral_loss_weight: float = 3.0
) -> Dict[str, torch.Tensor]:
    """Waypoint imitation loss, Q distillation loss, and interpretable trajectory error.

    Waypoints are (x_forward, y_lateral). The axes are kept separate because forward
    displacement is typically several metres per waypoint while lateral offset - the
    ONLY component the PID controller steers from - is often under a metre. A flat L1
    over both lets the large-magnitude x term dominate the gradient, so the network can
    minimise loss by nailing forward distance while barely fitting y, producing a
    policy that accelerates fine but steers close to zero.

    Reporting them separately is equally deliberate: the combined `wp_loss` is
    `longitudinal + lateral_loss_weight * lateral`, so a change in it cannot be
    attributed to an axis without the split.
    """
    pred_wp = out["selected_waypoints"]
    loss_q = F.mse_loss(out["selected_rail_q"], target_q)
    loss_wp_x = F.l1_loss(pred_wp[..., 0], target_wp[..., 0])
    loss_wp_y = F.l1_loss(pred_wp[..., 1], target_wp[..., 1])
    loss_wp = loss_wp_x + lateral_loss_weight * loss_wp_y
    total = q_loss_weight * loss_q + wp_loss_weight * loss_wp

    with torch.no_grad():
        per_point = torch.norm(pred_wp.float() - target_wp.float(), dim=-1)  # (B, 5)

    return {
        "total": total,
        "q": loss_q,
        "wp": loss_wp,
        "lateral": loss_wp_y,
        "longitudinal": loss_wp_x,
        "ade": per_point.mean(),
        "fde": per_point[:, -1].mean()
    }


def to_device_batch(batch: Dict[str, torch.Tensor], device: str):
    """Moves one batch to the device, converting frames to float NCHW in [0, 1] there.

    RGB crosses PCIe as uint8 HWC - a quarter of the traffic of float32 - and the
    permute lands in channels_last without a copy since the source is already HWC.
    """
    rgb = batch["rgb"].to(device, non_blocking=True)
    rgb = rgb.permute(0, 3, 1, 2).float().div_(255.0)
    return (
        rgb,
        batch["speed"].to(device, non_blocking=True),
        batch["command"].to(device, non_blocking=True),
        batch["route"].to(device, non_blocking=True),
        batch["target_q"].to(device, non_blocking=True),
        batch["target_waypoints"].to(device, non_blocking=True)
    )


@torch.no_grad()
def run_validation(
    model,
    val_loader,
    device: str,
    autocast_ctx,
    use_amp: bool,
    wp_loss_weight: float = 1.0,
    q_loss_weight: float = 0.0,
    lateral_loss_weight: float = 3.0
) -> Dict[str, float]:
    """Evaluates the current weights on the held-out set.

    Returns an empty dict when there is no validation loader, which is what leaves the
    val_* telemetry columns blank rather than filling them with a misleading zero.
    """
    if val_loader is None:
        return {}

    was_training = model.training
    model.eval()
    start = time.time()
    acc = {"loss": 0.0, "ade": 0.0, "fde": 0.0, "lat": 0.0, "lon": 0.0}
    n = 0

    for batch in val_loader:
        rgb, speed, command, route, target_q, target_wp = to_device_batch(batch, device)
        with autocast_ctx(enabled=use_amp):
            out = model(rgb, speed, command, route)
            losses = waypoint_losses(out, target_wp, target_q,
                                     wp_loss_weight, q_loss_weight, lateral_loss_weight)
        acc["loss"] += losses["total"].item()
        acc["ade"] += losses["ade"].item()
        acc["fde"] += losses["fde"].item()
        acc["lat"] += losses["lateral"].item()
        acc["lon"] += losses["longitudinal"].item()
        n += 1

    if was_training:
        model.train()

    d = max(1, n)
    return {
        "val_loss": acc["loss"] / d,
        "val_wp_ade_m": acc["ade"] / d,
        "val_wp_fde_m": acc["fde"] / d,
        "val_wp_lateral_error_m": acc["lat"] / d,
        "val_wp_longitudinal_error_m": acc["lon"] / d,
        "val_time_sec": time.time() - start
    }
