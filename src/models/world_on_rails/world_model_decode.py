"""TODO A73 closed loop (2026-11-02): Q from the latent world model, decoded by A71's rule.

With WOR_QCRITIC set (the A71 decode) and WOR_WORLDMODEL=<wm_epoch.pth> (scripts/training/train_world_model.py), act() replaces the policy's Q head
output - or supplies one, E15 has none - with the world model's h-step lookahead over the 40-action grid, computed from the policy's own frozen
encoder map pooled to the model's grid (what the A73 pooled cache holds). gamma, penalty and the horizon come from the checkpoint's run config;
WOR_WM_HORIZON / WOR_WM_PENALTY override them. The policy's waypoints and speed head are unchanged, so A73 runs on E15 without retraining it.
"""
from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn.functional as F

from src.models.world_on_rails.q_critic import parse_qcritic_env


def _load(policy, path: str, channels: int, device):
    from src.models.world_on_rails.latent_world_model import LatentWorldModel, WorldModelConfig
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mc = dict(ck["model_config"])
    mc["grid"] = tuple(mc["grid"])
    cfg = WorldModelConfig(**mc)
    if cfg.feat_channels != channels:
        raise RuntimeError(f"WOR_WORLDMODEL {path}: built on {cfg.feat_channels} feature channels, this policy's encoder gives {channels}")
    wm = LatentWorldModel(cfg)
    wm.load_state_dict(ck["model"])
    rc = ck.get("run_config", {})
    policy._wm = wm.to(device).eval()
    policy._wm_args = (float(rc.get("gamma", 0.98)), float(os.environ.get("WOR_WM_PENALTY", rc.get("penalty", 1.0))),
                       int(os.environ.get("WOR_WM_HORIZON", rc.get("imag_horizon", 5))))
    print(f"[A73] world-model Q from {path}: gamma, penalty, horizon = {policy._wm_args}", flush=True)


@torch.no_grad()
def world_model_q(policy, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """act()'s hook right after forward: out["q_sa"] from the world model when both switches are set; otherwise out unchanged."""
    path = os.environ.get("WOR_WORLDMODEL")
    feats = getattr(policy, "_last_feats", None)
    if not path or feats is None or parse_qcritic_env(os.environ.get("WOR_QCRITIC")) is None:
        return out
    if getattr(policy, "_wm", None) is None:
        _load(policy, path, feats.shape[1], feats.device)
    from src.training.latent_world_model_train import lookahead_q
    wm = policy._wm
    gamma, penalty, horizon = policy._wm_args
    pooled = F.adaptive_avg_pool2d(feats.float(), wm.cfg.grid)
    q, _ = lookahead_q(wm, wm.encode(pooled), gamma, penalty, horizon)
    out["q_sa"] = q.view(q.shape[0], -1, 5)
    return out
