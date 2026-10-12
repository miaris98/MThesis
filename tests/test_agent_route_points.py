"""A checkpoint's route length must survive from training to evaluation.

Every evaluation entry point (`eval_wor.py`, `eval_wor_closed_loop.py`, `bench2drive_agent.py`)
used to build the ego-frame route with a hardcoded `WOR_ROUTE_POINTS = 4`, regardless of what
`--route_points` the checkpoint was actually trained with. The v2 runs use `--route_points 20`:

  - `WorldOnRailsAgent` adopted img_size/crop/overlay/backbone from run_config.json but not
    route_points, so `load_wor_model` was asked to rebuild a 4-point policy - `route_proj`
    (Linear(8, D)) cannot load a (40, D) checkpoint tensor and load_state_dict raises.
  - Even after adopting the count for *construction*, `_ego_frame_route` subsampled the live
    route to the module's hardcoded default, not the agent's own `route_points` - so the
    tensor shape happened to match only by the same accident that made 4 the default in the
    first place, and a 20-point checkpoint would still receive a 4-point route at inference.

These tests exercise the wiring without CARLA (the agent's `carla` import is lazy, confined to
`run_step`'s VehicleControl conversion) and without the network's real weights (a fresh
resnet34 skeleton loads cleanly and is enough to prove the shapes agree).
"""
import json

import torch

from src.agents.wor_agent import WorldOnRailsAgent
from src.models.world_on_rails import WorldOnRailsPolicy


def _write_checkpoint(tmp_path, route_points):
    m = WorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, route_points=route_points)
    ckpt_path = tmp_path / "best_model.pth"
    torch.save({"epoch": 1, "model": m.state_dict(),
                "config": {"backbone": "resnet34", "route_points": route_points}}, ckpt_path)
    with open(tmp_path / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump({"backbone": "resnet34", "route_points": route_points,
                   "img_size": "192x512", "crop_bottom_frac": 0.25, "route_overlay": 0}, fh)
    return str(ckpt_path)


def test_agent_adopts_route_points_from_run_config(tmp_path):
    ckpt = _write_checkpoint(tmp_path, route_points=20)
    agent = WorldOnRailsAgent(checkpoint_path=ckpt, backbone_name="resnet34",
                              pretrained_backbone=False, device="cpu", policy_arch="cnn")
    assert agent.route_points == 20


def test_the_built_network_actually_matches_the_adopted_count(tmp_path):
    """The regression this project hit: adopting the number is not enough if the model built
    from it disagrees. route_proj's input width is route_points * 2."""
    ckpt = _write_checkpoint(tmp_path, route_points=20)
    agent = WorldOnRailsAgent(checkpoint_path=ckpt, backbone_name="resnet34",
                              pretrained_backbone=False, device="cpu", policy_arch="cnn")
    assert agent.net.route_points == 20 == agent.route_points
    assert agent.net.route_mlp[0].in_features == 40


def test_a_default_4point_checkpoint_still_works(tmp_path):
    """Older checkpoints (route_points=4, this project's long-standing default) must still
    load and run exactly as before."""
    ckpt = _write_checkpoint(tmp_path, route_points=4)
    agent = WorldOnRailsAgent(checkpoint_path=ckpt, backbone_name="resnet34",
                              pretrained_backbone=False, device="cpu", policy_arch="cnn")
    assert agent.route_points == 4 == agent.net.route_points


def test_act_accepts_a_route_of_the_adopted_length(tmp_path):
    """End to end: a route array shaped for the checkpoint's own route_points must run without
    a route_proj size mismatch - the failure this project actually hit at evaluation time."""
    ckpt = _write_checkpoint(tmp_path, route_points=20)
    agent = WorldOnRailsAgent(checkpoint_path=ckpt, backbone_name="resnet34",
                              pretrained_backbone=False, device="cpu", policy_arch="cnn")
    # rgb as a torch tensor, route as a plain list: act() calls np.asarray(route) unconditionally,
    # and np.asarray() on a *torch tensor* input hits this machine's broken numpy/torch
    # array-protocol interop (the same root cause as this repo's baseline test_wor_speed_units.py
    # failures) - unrelated to anything under test here, which is route_points parity.
    rgb = torch.zeros(192, 512, 3)
    route = [[0.0, 0.0]] * agent.route_points
    steer, throttle, brake = agent.net.act(rgb=rgb, speed=5.0, command=3, device="cpu",
                                           route=route)
    assert all(isinstance(v, float) for v in (steer, throttle, brake))


def test_a_checkpoint_with_no_run_config_keeps_the_constructor_default(tmp_path):
    """A checkpoint with no run_config.json beside it (older layouts, or one moved
    without its sibling file) cannot adopt anything - _adopt_training_preprocessing
    returns early with a warning. route_points must fall back to the caller's value
    rather than being left undefined or silently reset to some other default."""
    m = WorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, route_points=8)
    ckpt_path = tmp_path / "orphan_model.pth"
    torch.save({"epoch": 1, "model": m.state_dict(), "config": {}}, ckpt_path)
    # Deliberately no run_config.json written beside it.
    agent = WorldOnRailsAgent(checkpoint_path=str(ckpt_path), backbone_name="resnet34",
                              pretrained_backbone=False, device="cpu", policy_arch="cnn",
                              route_points=8)
    assert agent.route_points == 8 == agent.net.route_points
