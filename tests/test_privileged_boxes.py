"""2026-11-17 ceiling diagnostic: the closed-loop inputs of the privileged teacher are the tokens it trained on."""
import gzip
import json
import math

import numpy as np
import torch

from scripts.training.build_box_targets import VALID_SLOT, frame_targets
from src.agents.privileged_boxes import normalize_angle, relative_position, tokens_from_boxes
from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy

BOXES = [
    {"class": "ego_car", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "extent": [2.4, 1.0, 0.8], "speed": 5.0},
    {"class": "car", "position": [12.0, -1.0, 0.0], "yaw": 0.1, "extent": [2.2, 0.9, 0.7], "speed": 4.0},
    {"class": "walker", "position": [6.0, 3.0, 0.0], "yaw": -1.2, "extent": [0.3, 0.3, 0.9], "speed": 1.2},
    {"class": "traffic_light", "position": [25.0, 2.0, 0.0], "yaw": 3.1, "extent": [1.5, 1.5, 0.5], "state": "Red"},
    {"class": "stop_sign", "position": [30.0, -4.0, 0.0], "yaw": 1.6, "extent": [1.5, 1.5, 0.5]},
    {"class": "static", "position": [8.0, -0.5, 0.0], "yaw": 0.0, "extent": [0.2, 0.2, 0.5]},
    {"class": "static_car", "position": [12.0, 1.0, 0.0], "yaw": 0.0, "extent": [2.0, 0.9, 0.7]},   # same distance as the car: log order decides
    {"class": "static_trafficwarning", "position": [9.0, 0.0, 0.0], "yaw": 0.0, "extent": [1.0, 2.0, 1.0]},
    {"class": "car", "position": [60.0, 0.0, 0.0], "yaw": 0.0, "extent": [2.2, 0.9, 0.7], "speed": 9.0},   # beyond 50 m
]


def test_tokens_match_the_training_builder(tmp_path):
    route = tmp_path / "route"
    (route / "boxes").mkdir(parents=True)
    (route / "rgb").mkdir()
    with gzip.open(route / "boxes" / "0007.json.gz", "wt") as f:
        json.dump(BOXES, f)
    for m in (3, 20):
        _, valid, tok = frame_targets((str(route / "rgb" / "0007.jpg"), 0, 0, 0.0, m, ""))
        assert valid == 1
        np.testing.assert_array_equal(tokens_from_boxes(BOXES, m), tok)
    # the ego, the traffic warning (its own class, which the teacher never saw) and the car beyond 50 m are dropped
    assert int(tokens_from_boxes(BOXES, 20)[:, VALID_SLOT].sum()) == 6


def test_relative_position_is_the_ego_frame():
    yaw = math.radians(90.0)   # CARLA, left-handed: x forward, y right; yaw 90 deg points the ego along world +y
    ego = np.eye(4)
    ego[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    ego[:3, 3] = [10.0, 5.0, 0.0]
    other = np.eye(4)
    other[:3, 3] = [10.0, 15.0, 0.0]
    np.testing.assert_allclose(relative_position(ego, other), [10.0, 0.0, 0.0], atol=1e-9)
    other[:3, 3] = [8.0, 5.0, 0.0]   # world -x is the ego's right when it faces +y
    np.testing.assert_allclose(relative_position(ego, other), [0.0, 2.0, 0.0], atol=1e-9)


def test_normalize_angle():
    assert abs(normalize_angle(3 * math.pi / 2) + math.pi / 2) < 1e-12
    assert abs(normalize_angle(-0.25) + 0.25) < 1e-12


def test_teacher_policy_drives_from_object_tokens():
    torch.manual_seed(0)
    net = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, model_size="10m", vision_grid=4, use_target_speed=True,
                                 route_points=4, privileged_boxes=20).eval()
    tok = torch.as_tensor(tokens_from_boxes(BOXES, 20)).unsqueeze(0)
    route = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]], np.float32)
    steer, throttle, brake = net.act(np.zeros((96, 256, 3), np.uint8), 5.0, command=3, device="cpu", route=route, vision_features=tok)
    assert np.all(np.isfinite([steer, throttle, brake]))
    # the object tokens really are the input: other objects, another plan
    args = (None, torch.tensor([[5.0]]), torch.tensor([3]), torch.as_tensor(route)[None])
    a = net.forward(*args, vision_features=tok)["selected_waypoints"]
    b = net.forward(*args, vision_features=torch.zeros_like(tok))["selected_waypoints"]
    assert not torch.allclose(a, b)


def test_visible_variants_match_the_training_builder(tmp_path):
    """2026-11-18: the camera-visible teachers (_vis, _visns) get the same object selection live as in training."""
    boxes = BOXES + [
        {"class": "car", "position": [-10.0, 0.5, 0.0], "yaw": 0.0, "extent": [2.2, 0.9, 0.7], "speed": 6.0},   # behind the ego
        {"class": "car", "position": [1.0, 9.0, 0.0], "yaw": 1.5, "extent": [2.2, 0.9, 0.7], "speed": 3.0},     # beside it
    ]
    route = tmp_path / "route"
    (route / "boxes").mkdir(parents=True)
    (route / "rgb").mkdir()
    with gzip.open(route / "boxes" / "0007.json.gz", "wt") as f:
        json.dump(boxes, f)
    for variant in ("_vis", "_visns"):
        for crop in (0.0, 0.25):
            _, valid, tok = frame_targets((str(route / "rgb" / "0007.jpg"), 0, 0, crop, 20, variant))
            assert valid == 1
            np.testing.assert_array_equal(tokens_from_boxes(boxes, 20, variant, crop), tok)
    vis = tokens_from_boxes(boxes, 20, "_vis")
    allt = tokens_from_boxes(boxes, 20)
    assert vis[:, VALID_SLOT].sum() < allt[:, VALID_SLOT].sum()            # the rear car is gone
    assert not np.any(np.isclose(vis[:, 0] * 20.0, -10.0) & (vis[:, VALID_SLOT] > 0))
    assert np.all(tokens_from_boxes(boxes, 20, "_visns")[:, 6] == 0.0)     # no speed


def test_unknown_variant_is_refused():
    import pytest
    with pytest.raises(ValueError):
        tokens_from_boxes(BOXES, 20, "_bogus")
