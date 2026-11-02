"""TODO A73 closed loop (2026-11-02): the latent world model drives through the A71 decoder.

The policy (E15) keeps its waypoints and speed head; with WOR_WORLDMODEL=<wm_epoch.pth> the world model's h-step lookahead over A71's 40-action
grid, computed from the policy's own pooled frozen features (what the A73 pooled cache holds), replaces the Q head's output in act(), and
WOR_QCRITIC decodes it exactly as for A71. No Q head and no retraining of the policy are needed.
"""
import dataclasses

import numpy as np
import pytest
import torch

from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
from src.models.world_on_rails.latent_world_model import LatentWorldModel, WorldModelConfig
from src.training.latent_world_model_train import lookahead_q

KW = dict(backbone_name="resnet18", pretrained=False, model_size="10m", vision_grid=4, use_target_speed=True, route_points=4)


def _save_wm(path, channels):
    cfg = WorldModelConfig(feat_channels=channels, grid=(4, 4), trunk="mlp", embed_dim=32, repr_depth=1, dyn_depth=1, num_heads=4, ffn_dim=64)
    torch.manual_seed(1)
    wm = LatentWorldModel(cfg)
    torch.save({"model": wm.state_dict(), "target": wm.state_dict(), "model_config": dataclasses.asdict(cfg),
                "run_config": {"gamma": 0.98, "penalty": 1.0, "imag_horizon": 2}, "val_routes": [], "metrics": {}}, path)
    return wm


def _act(m, monkeypatch, env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    for k in ("_qcritic_ready", "_safety_ready", "_q_lock", "_wm"):
        m.__dict__.pop(k, None)
    m.controller.control_from_waypoints = lambda waypoints, current_speed_kmh, target_speed_kmh=None: (0.0, 0.5, 0.0)
    rgb = np.random.default_rng(1).integers(0, 255, (96, 256, 3), dtype=np.uint8)
    m.act(rgb=rgb, speed=8.0, command=3, device="cpu", route=np.array([[2.5, 0.0], [7.5, 0.0], [12.5, 0.0], [17.5, 0.0]], np.float32))


def test_world_model_q_feeds_the_a71_decoder_without_a_q_head(tmp_path, monkeypatch):
    m = QwenWorldOnRailsPolicy(**KW).eval()                                       # no Q head at all
    wm = _save_wm(tmp_path / "wm.pth", m.encoder.out_channels).eval()
    _act(m, monkeypatch, {"WOR_QCRITIC": "beta=1", "WOR_WORLDMODEL": str(tmp_path / "wm.pth")})
    q_used = m.last_debug["q_sa"]
    with torch.no_grad():
        pooled = torch.nn.functional.adaptive_avg_pool2d(m._last_feats, (4, 4))
        want, _ = lookahead_q(wm, wm.encode(pooled), gamma=0.98, penalty=1.0, horizon=2)
    assert q_used.shape == (8, 5) and np.allclose(q_used, want.view(8, 5).numpy(), atol=1e-5)


def test_without_the_switch_nothing_changes(tmp_path, monkeypatch):
    m = QwenWorldOnRailsPolicy(**KW).eval()
    _save_wm(tmp_path / "wm.pth", m.encoder.out_channels)
    _act(m, monkeypatch, {"WOR_QCRITIC": None, "WOR_WORLDMODEL": str(tmp_path / "wm.pth")})
    assert m.last_debug.get("q_sa") is None                                       # WOR_QCRITIC off: the world model is not even loaded
    assert getattr(m, "_wm", None) is None


def test_feature_width_mismatch_is_an_error(tmp_path, monkeypatch):
    m = QwenWorldOnRailsPolicy(**KW).eval()
    _save_wm(tmp_path / "wm.pth", m.encoder.out_channels + 1)
    with pytest.raises(RuntimeError, match="channels"):
        _act(m, monkeypatch, {"WOR_QCRITIC": "beta=1", "WOR_WORLDMODEL": str(tmp_path / "wm.pth")})
