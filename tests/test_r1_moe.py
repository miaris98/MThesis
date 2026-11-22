"""2026-11-20: R1 (detector object tokens) combined with w4's waypoint experts (16 experts, top-4, hidden 32 on the box; small here).
The two additions must coexist: a resumed RGB checkpoint starts with both at zero output, and a checkpoint holding both loads back
with no missing or unexpected tensors and the same plan."""
import torch

from src.models.world_on_rails.wor_loader import load_wor_model
from src.training.wor_checkpoint import load_trainable_state
from tests.test_r1_object_tokens import ARGS, DET, ENC, M, _files, _policy, _stamp

MOE = dict(moe_experts=4, moe_hidden=8, moe_top_k=2)


def _r1_moe(tmp_path):
    net = _policy(object_tokens=M, object_det_cfg=DET, object_max_range=1e6, object_threshold=0.0, **MOE)
    dp, bp = _files(tmp_path)
    net.load_object_detector(dp, bp)
    return net


def test_resumed_rgb_checkpoint_starts_with_silent_experts(tmp_path, monkeypatch):
    monkeypatch.setenv("WOR_RESUME_WIDEN", "1")
    rgb = _policy()
    net = _r1_moe(tmp_path)
    frozen = [k for k in net.state_dict() if k.startswith("encoder.")]
    changed = load_trainable_state(net, {k: v for k, v in rgb.state_dict().items() if not k.startswith("encoder.")}, frozen, "e15.pth")
    assert any(k.startswith("trunk.moe.") for k in changed) and any(k.startswith("obj_proj.") for k in changed)
    assert float(net.obj_ready) == 1.0
    net.eval()
    fmap = torch.randn(2, ENC, 3, 8)
    with torch.no_grad():
        with_experts = net(None, *ARGS, vision_features=fmap)["waypoints"]
        moe, net.trunk.moe = net.trunk.moe, None
        without = net(None, *ARGS, vision_features=fmap)["waypoints"]
        net.trunk.moe = moe
    assert torch.equal(with_experts, without)          # every expert's last layer is zero: the experts add nothing at the start


def test_r1_with_experts_round_trips_through_a_checkpoint(tmp_path, capsys):
    net = _r1_moe(tmp_path)
    torch.nn.init.normal_(net.obj_proj.weight, std=0.5)
    torch.nn.init.normal_(net.trunk.moe.w2, std=0.5)     # trained-looking experts, so the check sees them
    net.eval()
    path = tmp_path / "r1w4.pth"
    torch.save({"epoch": 30, "model": net.state_dict(), "config": {**_stamp(net), **MOE}}, path)
    capsys.readouterr()
    loaded = load_wor_model(checkpoint_path=str(path), backbone_name="resnet18", pretrained_backbone=False, device="cpu",
                            policy_arch="qwen10m", route_points=4).eval()
    out = capsys.readouterr().out
    assert "Missing keys: 0" in out and "Unexpected keys: 0" in out, out
    assert loaded.trunk.moe is not None and loaded.trunk.moe.num_experts == MOE["moe_experts"]
    fmap = torch.randn(2, ENC, 3, 8)
    with torch.no_grad():
        a = net(None, *ARGS, vision_features=fmap)["waypoints"]
        b = loaded(None, *ARGS, vision_features=fmap)["waypoints"]
    assert torch.allclose(a, b, atol=1e-6)
    moe, net.trunk.moe = net.trunk.moe, None
    with torch.no_grad():
        assert not torch.allclose(a, net(None, *ARGS, vision_features=fmap)["waypoints"], atol=1e-4)   # the experts do act
    net.trunk.moe = moe
