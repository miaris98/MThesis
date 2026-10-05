"""TODO A15: the path head, the aligned cascade, the path labels and the eval-time path steering."""
import numpy as np
import torch

from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
from src.training.wor_dataset import WorldOnRailsDataset

KW = dict(backbone_name="resnet18", pretrained=False, model_size="10m", vision_grid=4, use_target_speed=True, route_points=4)


def _inputs(b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(rgb=torch.rand(b, 3, 96, 256, generator=g), speed=torch.rand(b, 1, generator=g) * 10, command=torch.zeros(b, dtype=torch.long),
                route=torch.randn(b, 4, 2, generator=g))


def test_path_head_is_opt_in_and_zero_initialised():
    plain = QwenWorldOnRailsPolicy(**KW).eval()
    assert "path" not in plain(**_inputs())
    m = QwenWorldOnRailsPolicy(**KW, use_path_head=True).eval()
    out = m(**_inputs())
    nominal = torch.tensor([[3.5 + 2.0 * k, 0.0] for k in range(10)])
    assert out["path"].shape == (3, 10, 2) and torch.allclose(out["path"].detach(), nominal.expand(3, 10, 2))   # starts as 'drive straight'


def test_cascade_starts_as_the_plain_head_and_then_couples():
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(**KW, use_path_head=True, path_cascade=True).eval()
    with torch.no_grad():
        m.path_head[-1].weight.normal_(0, 0.1)          # a non-trivial predicted path
    x = _inputs()
    with torch.no_grad():
        casc = m(**x)["target_speed_logits"]
        m.path_cascade = False
        plain = m(**x)["target_speed_logits"]
        m.path_cascade = True
        assert torch.allclose(casc, plain), "zero-initialised path_embed must leave the speed head unchanged at init"
        m.path_embed.weight.normal_(0, 0.1)
        assert not torch.allclose(m(**x)["target_speed_logits"], plain), "after learning the coupling the speed logits depend on the path"


def test_path_label_sample_and_validity(tmp_path):
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(64, 64))
    ds.is_synthetic = False
    path = [[3.5 + 2 * i, 0.1 * i] for i in range(10)]
    ds.samples = [{"format": "pdm_lite", "rgb_path": str(tmp_path / "a.jpg"), "speed": 5.0, "command": 3, "route": [[0.0, 0.0]] * 4, "target_speed": 5.0,
                   "waypoints": [[1.0, 0.0]] * 5, "path": path},
                  {"format": "pdm_lite", "rgb_path": str(tmp_path / "b.jpg"), "speed": 5.0, "command": 3, "route": [[0.0, 0.0]] * 4, "target_speed": 5.0,
                   "waypoints": [[1.0, 0.0]] * 5, "path": None}]
    a, b = ds[0], ds[1]
    assert a["target_path"].shape == (10, 2) and float(a["path_valid"]) == 1.0 and abs(float(a["target_path"][3, 1]) - 0.3) < 1e-6
    assert float(b["path_valid"]) == 0.0 and float(b["target_path"].abs().sum()) == 0.0


def test_path_steering_replaces_the_lateral_coordinate_only(monkeypatch):
    """WOR_PATH_STEER=1: waypoint y = the path interpolated at the waypoint's x; x (the speed side) is untouched."""
    monkeypatch.setenv("WOR_PATH_STEER", "1")
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(**KW, use_path_head=True).eval()
    with torch.no_grad():
        m.path_head[-1].bias.copy_(torch.tensor([[x, 1.0] for x in np.linspace(2, 20, 10)], dtype=torch.float32).reshape(-1))   # path: y = 1.0 everywhere
    seen = {}
    orig = m.controller.control_from_waypoints
    m.controller.control_from_waypoints = lambda waypoints, current_speed_kmh, target_speed_kmh=None: (seen.setdefault("wps", np.array(waypoints)), (0.0, 0.0, 0.0))[1]
    m.act(rgb=np.random.randint(0, 255, (96, 256, 3), dtype=np.uint8), speed=5.0, command=3, device="cpu", route=np.zeros((4, 2), np.float32))
    assert np.allclose(seen["wps"][:, 1], 1.0, atol=1e-4)


def test_s2d_tokens_and_masking_a54():
    """TODO A54: space-to-depth tokens from the stride-16 level (576 ch x p^2), 6x16 grid at 96x256... and lateral column masking in training only."""
    import torch.nn.functional as F
    m = QwenWorldOnRailsPolicy(backbone_name="regnety_032", pretrained=False, model_size="10m", vision_grid=(2, 5), use_target_speed=True, route_points=4,
                               token_mode="s2d", s2d_patch=3, token_mask=0.5)
    # 96x240 input: stride 16 -> 6x15, patch 3 -> 2x5 tokens
    x = _inputs()
    x["rgb"] = torch.rand(3, 3, 96, 240)
    assert m.vision_proj.in_features == 576 * 9
    m.eval()
    with torch.no_grad():
        out = m(**x)
    assert out["selected_waypoints"].shape == (3, 5, 2) and m.num_vision_tokens == 10
    # masking is a training-time effect: eval is deterministic, train drops columns (outputs differ between two draws)
    m.train()
    torch.manual_seed(1)
    a = m(**x)["selected_waypoints"]
    torch.manual_seed(2)
    b = m(**x)["selected_waypoints"]
    assert not torch.allclose(a, b)
    # the pooled cache cannot feed an s2d model
    try:
        m(None, x["speed"], x["command"], x["route"], vision_features=torch.zeros(3, 1512, 4, 4))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
