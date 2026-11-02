"""TODO A71-W / A72-W (2026-11-02): the WoR network (conv SpatialQHead) with the A71 Q-critic and the A72 V head.

The TODOs ask for both networks ("try both wor and qwen"). The WoR policy already has an attribute `q_head` (the SpatialQHead, the rail Q-map), so
the A71 critic is `q_sa_head` on it; it reads the head's pooled fusion vector (the same 128-d state its target-speed head reads), and act() goes
through the same WOR_QCRITIC decode as the Qwen arm.
"""
import math

import numpy as np
import pytest
import torch

from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy
from src.models.world_on_rails.q_critic import LAT_TARGETS
from src.models.world_on_rails.aux_heads import TARGET_SPEEDS

KW = dict(backbone_name="resnet18", pretrained=False, use_target_speed=True, route_points=4)
NV, ND = len(TARGET_SPEEDS), len(LAT_TARGETS)


def _x(B=2):
    return torch.zeros(B, 96, 256, 3, dtype=torch.uint8), torch.zeros(B, 1), torch.zeros(B, dtype=torch.long), torch.zeros(B, 4, 2)


def test_heads_are_opt_in_and_zero_initialised():
    m = WorldOnRailsPolicy(**KW, use_q_critic=True, use_v_head=True).eval()
    out = m(*_x())
    assert out["q_sa"].shape == (2, NV, ND) and torch.all(out["q_sa"] == 0)
    assert out["v_s"].shape == (2,) and torch.all(out["v_s"] == 0) and out["policy_repr"].shape == (2, 128)
    plain = WorldOnRailsPolicy(**KW).eval()(*_x())
    assert "q_sa" not in plain and "v_s" not in plain


def test_act_reads_the_critic_under_the_switch(monkeypatch):
    torch.manual_seed(0)
    m = WorldOnRailsPolicy(**KW, use_q_critic=True).eval()
    seen = {}
    m.controller.control_from_waypoints = lambda waypoints, current_speed_kmh, target_speed_kmh=None: (
        seen.update(wps=np.array(waypoints), tgt=target_speed_kmh), (0.0, 0.5, 0.0))[1]
    route = np.array([[2.5, 0.0], [7.5, 0.0], [12.5, 0.0], [17.5, 0.0]], np.float32)
    rgb = np.random.default_rng(1).integers(0, 255, (96, 256, 3), dtype=np.uint8)
    monkeypatch.delenv("WOR_QCRITIC", raising=False)
    m.act(rgb, 8.0, command=3, device="cpu", route=route)
    off = dict(seen)
    with torch.no_grad():
        bias = torch.zeros(NV, ND)
        bias[:, LAT_TARGETS.index(3.5)] = 10.0
        m.q_sa_head.net[-1].bias.copy_(bias.reshape(-1))
    monkeypatch.setenv("WOR_QCRITIC", "beta=5,confirm_ticks=1")
    for k in ("_qcritic_ready", "_q_lock"):
        m.__dict__.pop(k, None)
    m.act(rgb, 8.0, command=3, device="cpu", route=route)
    assert m.last_debug["q_lat"] == 3.5 and seen["wps"][-1, 1] > off["wps"][-1, 1] + 1.0


def test_switch_off_is_bit_identical(monkeypatch):
    torch.manual_seed(0)
    plain = WorldOnRailsPolicy(**KW).eval()
    torch.manual_seed(0)
    withq = WorldOnRailsPolicy(**KW, use_q_critic=True).eval()
    withq.load_state_dict(plain.state_dict(), strict=False)
    monkeypatch.delenv("WOR_QCRITIC", raising=False)
    rgb = np.random.default_rng(2).integers(0, 255, (96, 256, 3), dtype=np.uint8)
    route = np.array([[2.5, 0.0], [7.5, 0.5], [12.5, 1.0], [17.5, 1.5]], np.float32)
    assert plain.act(rgb, 5.0, 3, "cpu", route) == withq.act(rgb, 5.0, 3, "cpu", route)


def test_resume_whitelists_the_new_heads():
    from src.training.wor_checkpoint import load_trainable_state
    src = WorldOnRailsPolicy(**KW)
    dst = WorldOnRailsPolicy(**KW, use_q_critic=True, use_v_head=True)
    frozen = [k for k in dst.state_dict() if k.startswith("encoder.")]
    changed = load_trainable_state(dst, {k: v for k, v in src.state_dict().items() if not k.startswith("encoder.")}, frozen, "arm-h-like")
    assert changed and all(k.startswith(("q_sa_head.", "v_head.")) for k in changed)


def test_offline_rl_trains_the_wor_critic(tmp_path, monkeypatch):
    from tests.test_offline_rl import _write_pdm_route
    from src.training.offline_rl import OfflineRLConfig
    from src.training.wor_trainer import WorldOnRailsTrainer
    _write_pdm_route(tmp_path, n=40)
    monkeypatch.setenv("WOR_Q_LABELS", "qtr")
    monkeypatch.setenv("WOR_Q_MIN_COVERAGE", "0.2")
    monkeypatch.setenv("WOR_TRANSITIONS", "1")
    torch.manual_seed(0)
    m = WorldOnRailsPolicy(**KW, use_q_critic=True, use_v_head=True)
    tr = WorldOnRailsTrainer(m, data_dir=str(tmp_path), save_dir=str(tmp_path / "ck"), batch_size=4, num_workers=0, device="cpu",
                             img_size=(96, 256), use_mlflow=False, val_split=0.0, q_critic_loss_weight=1.0, target_speed_loss_weight=0.2,
                             offline_rl=OfflineRLConfig(mode="cql"))
    before = [p.detach().clone() for p in tr.q_target.parameters()]
    metrics = tr.train_epoch(1)
    assert metrics["rl_q_loss"] > 0 and math.isfinite(metrics["rl_cql_loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, tr.q_target.parameters()))


def test_loader_rebuilds_the_wor_heads_from_the_run_config(tmp_path):
    import json
    from src.models.world_on_rails.wor_loader import load_wor_model
    m = WorldOnRailsPolicy(**KW, use_q_critic=True, use_v_head=True)
    ck = tmp_path / "model_epoch_001.pth"
    torch.save({"model_state_dict": m.state_dict(), "epoch": 1, "config": {"use_q_critic": True, "use_v_head": True, "route_points": 4,
                "use_target_speed": True, "target_speed_input": "policy"}}, ck)                # the loader reads the checkpoint's own config
    loaded = load_wor_model(checkpoint_path=str(ck), backbone_name="resnet18", pretrained_backbone=False, device="cpu", policy_arch="cnn", route_points=4)
    assert loaded.q_sa_head is not None and loaded.v_head is not None


def test_train_wor_accepts_the_critic_on_the_cnn_arch():
    import importlib.util
    import types
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("train_wor_w", Path(__file__).resolve().parents[1] / "scripts" / "training" / "train_wor.py")
    tw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tw)
    tw.check_q_critic_args(types.SimpleNamespace(q_critic=1, policy_arch="cnn", q_labels="qcf_v4", target_speed_loss_weight=0.2))
    with pytest.raises(SystemExit):
        tw.check_q_critic_args(types.SimpleNamespace(q_critic=1, policy_arch="cnn", q_labels=None, target_speed_loss_weight=0.2))
