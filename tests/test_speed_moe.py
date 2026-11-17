"""TODO A81: speed-head experts (mixture of experts on the target-speed decision).

Checks the four properties the arm depends on:
1. zero start: a model with speed experts gives exactly the same outputs as the same model without them, so a run resumed
   from E15 starts at E15;
2. the experts are trainable from that start (gradient reaches their output layer) and add a load-balance term;
3. the routing-split metric is 0 for identical routing, 1 for disjoint routing, NaN when a group is missing;
4. a checkpoint rebuilds into the same arm through load_wor_model, which is the path the CARLA agent uses.
"""
import math

import torch

from src.models.world_on_rails import QwenWorldOnRailsPolicy
from src.models.world_on_rails.qwen_wor_policy import STANDING_MPS, routing_split, speed_moe_inputs
from src.models.world_on_rails.wor_loader import load_wor_model

IMG = (192, 512)


def _model(**kw):
    torch.manual_seed(0)
    return QwenWorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, route_points=4, model_size="10m",
                                  vision_grid=(6, 16), use_target_speed=True, target_speed_input="policy", **kw)


def _batch(b=4):
    g = torch.Generator().manual_seed(1)
    rgb = torch.rand(b, 3, *IMG, generator=g)
    speed = torch.tensor([0.0, 0.2, 6.0, 12.0])[:b]          # two standing, two moving
    return rgb, speed, torch.tensor([3, 3, 2, 4])[:b], torch.zeros(b, 4, 2)


def test_zero_start_matches_the_model_without_experts():
    base = _model().eval()
    moe = _model(speed_moe_experts=16, speed_moe_hidden=32).eval()
    moe.load_state_dict(base.state_dict(), strict=False)     # same shared weights; speed_moe keeps its own (zero-output) init
    rgb, speed, cmd, route = _batch()
    with torch.no_grad():
        a, b = base(rgb, speed, cmd, route), moe(rgb, speed, cmd, route)
    assert torch.equal(a["target_speed_logits"], b["target_speed_logits"])
    assert torch.equal(a["waypoints"], b["waypoints"])
    assert "moe_aux" in b and "speed_moe_split" in b


def test_experts_train_from_the_zero_start():
    m = _model(speed_moe_experts=8, speed_moe_hidden=16).train()
    rgb, speed, cmd, route = _batch()
    out = m(rgb, speed, cmd, route)
    (out["target_speed_logits"].sum() + 0.01 * out["moe_aux"]).backward()
    assert m.speed_moe.w2.grad is not None and float(m.speed_moe.w2.grad.abs().sum()) > 0
    assert m.speed_moe.gate.weight.grad is not None
    assert float(out["moe_aux"]) > 0


def test_inputs_carry_speed_and_standing_flag():
    x = speed_moe_inputs(torch.zeros(3, 5), torch.tensor([0.0, STANDING_MPS, 10.0]))
    assert x.shape == (3, 7)
    assert x[:, -1].tolist() == [1.0, 0.0, 0.0]
    assert torch.allclose(x[:, -2], torch.tensor([0.0, STANDING_MPS / 10.0, 1.0]))


def test_routing_split_bounds():
    standing = torch.tensor([True, True, False, False])
    same = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    disjoint = torch.tensor([[0, 1], [0, 1], [2, 3], [2, 3]])
    assert float(routing_split(same, standing, 4)) == 0.0
    assert float(routing_split(disjoint, standing, 4)) == 1.0
    assert math.isnan(float(routing_split(same, torch.ones(4, dtype=torch.bool), 4)))


def test_checkpoint_rebuilds_the_arm(tmp_path, capsys):
    m = _model(speed_moe_experts=8, speed_moe_hidden=16).eval()
    with torch.no_grad():
        m.speed_moe.w2.normal_(std=0.1)                       # non-zero, so a dropped expert layer would change the output
    cfg = {"backbone": "resnet34", "route_points": 4, "vision_grid": [6, 16], "use_target_speed": True,
           "target_speed_input": "policy", "use_rail_q": True, "use_ray_geometry": False,
           "crop_bottom_frac": m.crop_bottom_frac, "speed_moe_experts": 8, "speed_moe_hidden": 16, "speed_moe_top_k": 2}
    path = tmp_path / "ckpt.pth"
    torch.save({"epoch": 1, "model": m.state_dict(), "config": cfg}, path)
    capsys.readouterr()
    loaded = load_wor_model(checkpoint_path=str(path), backbone_name="resnet34", pretrained_backbone=False,
                            device="cpu", policy_arch="qwen10m", route_points=4).eval()
    printed = capsys.readouterr().out
    assert "Missing keys: 0" in printed and "Unexpected keys: 0" in printed, printed
    rgb, speed, cmd, route = _batch()
    with torch.no_grad():
        assert torch.allclose(m(rgb, speed, cmd, route)["target_speed_logits"],
                              loaded(rgb, speed, cmd, route)["target_speed_logits"], atol=1e-6)
