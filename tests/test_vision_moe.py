"""TODO A82 / A83: vision-token experts and the scenario-labelled waypoint router.

1. zero start: a model with vision experts gives the same outputs as the same model without them (a resumed E15 starts at E15);
2. the vision experts train from that start and add a load-balance term;
3. the waypoint router's scores are exposed, with one row per sample and one column per expert;
4. scenario labels come from the route folder name; unnamed folders are -1;
5. the route-label loss ignores unlabelled rows and is None when no row is labelled;
6. a checkpoint with vision experts rebuilds through load_wor_model with no missing or unexpected keys.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from src.models.world_on_rails import QwenWorldOnRailsPolicy
from src.models.world_on_rails.qwen_wor_policy import route_label_loss
from src.models.world_on_rails.wor_loader import load_wor_model
from src.training.wor_dataset import SCENARIO_CLASSES, scenario_label

IMG = (192, 512)


def _model(**kw):
    torch.manual_seed(0)
    return QwenWorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, route_points=4, model_size="10m",
                                  vision_grid=(6, 16), use_target_speed=True, target_speed_input="policy", **kw)


def _batch(b=4):
    g = torch.Generator().manual_seed(1)
    rgb = torch.rand(b, 3, *IMG, generator=g)
    speed = torch.tensor([0.0, 0.2, 6.0, 12.0])[:b]
    return rgb, speed, torch.tensor([3, 3, 2, 4])[:b], torch.zeros(b, 4, 2)


def test_vision_experts_zero_start_matches_the_model_without_them():
    base = _model().eval()
    vis = _model(vision_moe_experts=8, vision_moe_hidden=16).eval()
    vis.load_state_dict(base.state_dict(), strict=False)   # shared weights; vis_moe keeps its zero-output init
    rgb, speed, cmd, route = _batch()
    with torch.no_grad():
        a, b = base(rgb, speed, cmd, route), vis(rgb, speed, cmd, route)
    assert torch.equal(a["waypoints"], b["waypoints"])
    assert torch.equal(a["target_speed_logits"], b["target_speed_logits"])
    assert "moe_aux" in b and float(b["moe_aux"]) > 0


def test_vision_experts_train_from_the_zero_start():
    m = _model(vision_moe_experts=8, vision_moe_hidden=16).train()
    rgb, speed, cmd, route = _batch()
    out = m(rgb, speed, cmd, route)
    (out["waypoints"].sum() + 0.01 * out["moe_aux"]).backward()
    assert m.vis_moe.w2.grad is not None and float(m.vis_moe.w2.grad.abs().sum()) > 0
    assert m.vis_moe.gate.weight.grad is not None


def test_router_scores_are_exposed_per_sample():
    m = _model(moe_experts=16, moe_top_k=2).eval()
    rgb, speed, cmd, route = _batch()
    with torch.no_grad():
        out = m(rgb, speed, cmd, route)
    assert out["moe_gate_logits"].shape == (4, 16)


def test_scenario_labels_from_folder_names():
    assert SCENARIO_CLASSES[0] == "ParkedObstacle"
    assert scenario_label("data/Town12/RouteScenario_1773_ParkedObstacle_1_25") == 0
    assert scenario_label("data/Town12/RouteScenario_1852_AccidentTwoWays_1") == 1
    assert scenario_label("data/Town12/RouteScenario_2513_ConstructionObstacle_1") == 2     # not ParkedObstacle
    assert scenario_label(r"D:\data\Town07\HazardAtSideLaneTwoWays_1") == 3                   # Windows separators, case-insensitive
    assert scenario_label("data/Town01/route_42") == -1


def test_route_label_loss_ignores_unlabelled_rows():
    logits = torch.randn(4, 5, requires_grad=True)
    labels = torch.tensor([2, -1, 0, -1])
    loss = route_label_loss(logits, labels)
    expected = F.cross_entropy(logits[[0, 2]], torch.tensor([2, 0]))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None and float(logits.grad[1].abs().sum()) == 0.0


def test_route_label_loss_is_none_without_labels():
    assert route_label_loss(torch.randn(3, 5), torch.tensor([-1, -1, -1])) is None


def test_checkpoint_with_vision_experts_rebuilds_the_arm(tmp_path, capsys):
    m = _model(vision_moe_experts=8, vision_moe_hidden=16).eval()
    with torch.no_grad():
        m.vis_moe.w2.normal_(std=0.1)                          # non-zero, so a dropped expert layer would change the output
    cfg = {"backbone": "resnet34", "route_points": 4, "vision_grid": [6, 16], "use_target_speed": True,
           "target_speed_input": "policy", "use_rail_q": True, "use_ray_geometry": False,
           "crop_bottom_frac": m.crop_bottom_frac, "vision_moe_experts": 8, "vision_moe_hidden": 16, "vision_moe_top_k": 2}
    path = tmp_path / "ckpt.pth"
    torch.save({"epoch": 1, "model": m.state_dict(), "config": cfg}, path)
    capsys.readouterr()
    loaded = load_wor_model(checkpoint_path=str(path), backbone_name="resnet34", pretrained_backbone=False,
                            device="cpu", policy_arch="qwen10m", route_points=4).eval()
    printed = capsys.readouterr().out
    assert "Missing keys: 0" in printed and "Unexpected keys: 0" in printed, printed
    rgb, speed, cmd, route = _batch()
    with torch.no_grad():
        assert torch.allclose(m(rgb, speed, cmd, route)["waypoints"], loaded(rgb, speed, cmd, route)["waypoints"], atol=1e-6)


def test_top_k_must_fit_the_experts():
    from src.models.world_on_rails.qwen_wor_policy import MoEWaypointDelta
    with pytest.raises(ValueError):
        MoEWaypointDelta(4, 4, num_experts=2, hidden=8, top_k=4)


def test_token_mask_is_refused_with_attn_pool():
    with pytest.raises(ValueError, match="token_mask"):
        _model(attn_pool=4, token_mask=0.2)


def test_scenario_label_ignores_the_data_root():
    assert scenario_label("D:/Accident_data/Town01/route_42") == -1
    assert scenario_label("D:/Accident_data/Town12/RouteScenario_7_ParkedObstacle_1") == 0


def test_vision_experts_do_not_shift_other_initialisations():
    base = _model()
    vis = _model(vision_moe_experts=8, vision_moe_hidden=16)
    for (name, a), (_, b) in zip(base.target_speed_head.state_dict().items(), vis.target_speed_head.state_dict().items()):
        assert torch.equal(a, b), name


def test_resume_refuses_to_drop_trained_experts():
    from src.training.wor_checkpoint import load_trainable_state
    src = _model(vision_moe_experts=8, vision_moe_hidden=16)
    dst = _model()                                          # built without vision experts
    with pytest.raises(RuntimeError, match="vis_moe"):
        load_trainable_state(dst, src.state_dict(), frozen_keys=[], source="test checkpoint")


def test_scenario_label_reads_the_scenario_folder_above_the_route():
    # the archives unpack as <Town>/data/<Scenario>/<Route>_<Rep>/ (checked on the box, 2026-11-13)
    assert scenario_label("/workspace/dataset/wor_trajectories/Town12/data/Accident/Route4_Rep1") == 1
    assert scenario_label("/workspace/dataset/wor_trajectories/Town12/data/ParkedObstacleTwoWays/Route0_Rep0") == 0
    assert scenario_label("/workspace/dataset/wor_trajectories/Town12/data/VehicleOpensDoorTwoWays/Route3_Rep0") == -1
    assert scenario_label("D:/Accident_data/Town01/data/route_42") == -1
