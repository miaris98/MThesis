"""2026-11-21: the waypoint experts' lateral-only option and their logged statistics.

1. lateral_only: the experts' correction has no x (along-path) part, ever, and the y part trains;
2. the mask adds no saved tensor, and a lateral-only checkpoint gives the same outputs with or without the mask (x columns stay zero);
3. the loader rebuilds the option from the checkpoint config;
4. last_stats holds the experts' use and correction size in sane ranges, and the trainer writes them to the telemetry CSV
   (blank for a model without experts).
"""
import csv
import math

import torch

from src.models.world_on_rails import QwenWorldOnRailsPolicy
from src.models.world_on_rails.qwen_wor_policy import MoEWaypointDelta


def _moe(lat):
    torch.manual_seed(0)
    m = MoEWaypointDelta(32, 60, num_experts=8, hidden=16, top_k=2, lateral_only=lat)
    with torch.no_grad():
        m.w2.normal_(0, 0.1)
        m.b2.normal_(0, 0.1)
    return m


def test_lateral_only_has_no_x_part_and_the_y_part_trains():
    m = _moe(True)
    out = m(torch.randn(16, 32))
    xy = out.reshape(16, -1, 2)
    assert torch.equal(xy[..., 0], torch.zeros_like(xy[..., 0]))
    assert xy[..., 1].abs().sum() > 0
    out.pow(2).sum().backward()
    g = m.w2.grad.reshape(8, 16, -1, 2)
    assert torch.equal(g[..., 0], torch.zeros_like(g[..., 0])) and g[..., 1].abs().sum() > 0


def test_mask_off_changes_nothing():
    a, b = _moe(False), _moe(False)
    x = torch.randn(4, 32)
    assert torch.equal(a(x), b(x))
    assert "out_mask" not in a.state_dict()


def _policy(**kw):
    torch.manual_seed(0)
    return QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, route_points=4, model_size="10m", vision_grid=4,
                                  use_target_speed=True, moe_experts=8, moe_hidden=16, moe_top_k=2, **kw)


def test_lateral_only_checkpoint_same_with_or_without_the_mask():
    lat = _policy(moe_lateral_only=True)
    with torch.no_grad():
        lat.trunk.moe.w2.normal_(0, 0.1)
        lat.trunk.moe.w2.view(8, 16, -1, 2)[..., 0] = 0.0      # what training leaves: x columns never get a gradient
        lat.trunk.moe.b2.normal_(0, 0.1)
        lat.trunk.moe.b2.view(8, -1, 2)[..., 0] = 0.0
    plain = _policy()
    assert set(lat.state_dict()) == set(plain.state_dict())
    plain.load_state_dict(lat.state_dict())
    g = torch.Generator().manual_seed(1)
    inp = (torch.rand(3, 3, 128, 128, generator=g), torch.tensor([1.0, 5.0, 9.0]), torch.tensor([3, 3, 2]), torch.rand(3, 4, 2, generator=g))
    with torch.no_grad():
        assert torch.allclose(lat.eval()(*inp)["waypoints"], plain.eval()(*inp)["waypoints"])


def test_loader_rebuilds_lateral_only(tmp_path):
    from src.models.world_on_rails.wor_loader import load_wor_model
    from src.training.wor_trainer import WorldOnRailsTrainer
    t = WorldOnRailsTrainer(model=_policy(moe_lateral_only=True), data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
                            batch_size=4, num_workers=0, device="cpu", synthetic_samples=8, use_mlflow=False, img_size=(128, 128))
    assert t.model_config["moe_lateral_only"] is True
    t.train(num_epochs=1, save_freq=1)
    m = load_wor_model(str(tmp_path / "ck" / "model_epoch_001.pth"), backbone_name="resnet18", pretrained_backbone=False, device="cpu",
                       policy_arch="qwen10m")
    model = m[0] if isinstance(m, tuple) else m
    assert model.trunk.moe.lateral_only and model.moe_lateral_only


def test_stats_are_sane():
    m = _moe(False)
    m(torch.randn(64, 32))
    s = {k: float(v) for k, v in m.last_stats.items()}
    assert 0.0 <= s["moe_load_entropy"] <= 1.0 + 1e-6 and 0.0 < s["moe_load_max"] <= 1.0
    assert 0.0 < s["moe_gate_conf"] <= 1.0 and s["moe_aux"] >= 1.0 - 1e-6   # E * sum f P >= 1 at the optimum, 1 when balanced
    assert s["moe_delta_x"] > 0 and s["moe_delta_y"] > 0
    lat = _moe(True)
    lat(torch.randn(64, 32))
    assert float(lat.last_stats["moe_delta_x"]) == 0.0


def _row(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))[-1]


def test_trainer_writes_the_expert_columns(tmp_path):
    from src.training.wor_trainer import WorldOnRailsTrainer
    for name, model in (("moe", _policy()), ("plain", QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, route_points=4,
                                                                            model_size="10m", vision_grid=4, use_target_speed=True))):
        t = WorldOnRailsTrainer(model=model, data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / name), batch_size=4, num_workers=0,
                                device="cpu", synthetic_samples=16, use_mlflow=False, img_size=(128, 128))
        t.train(num_epochs=1, save_freq=1)
        row = _row(tmp_path / name / "wor_training_telemetry.csv")
        for k in ("moe_aux", "moe_load_max", "moe_load_entropy", "moe_gate_conf", "moe_delta_x", "moe_delta_y"):
            if name == "moe":
                assert row[k] != "" and math.isfinite(float(row[k])), k
            else:
                assert row[k] == "", k
