"""TODO A87 target-point arm (2026-11-21): route_mode dense_tp.

1. the dataset gives route_points - 2 near route points, then TF++'s target point and next target point (falls back when a frame has none);
2. resuming a 4-point checkpoint into a 6-point model widens route_proj with zero weights on the new inputs: the route token, and so the
   whole model, starts exactly where the checkpoint was, whatever the target points are;
3. the driving-time helper steps carla_garage's RoutePlanner (7.5 m / 50 m) and returns route[1] / route[2] in the ego frame.
"""
import types

import numpy as np
import pytest
import torch

from src.models.world_on_rails import QwenWorldOnRailsPolicy
from tests._tfpp_isolation import isolate_tfpp_imports

_isolate = isolate_tfpp_imports()   # carla / shapely / agents stand-ins and team_code must not leak into later test modules


def _ds(route_points, monkeypatch):
    from src.training.wor_dataset import WorldOnRailsDataset
    monkeypatch.setenv("WOR_ROUTE_MODE", "dense_tp")
    ds = WorldOnRailsDataset.__new__(WorldOnRailsDataset)
    ds.route_points, ds.route_key = route_points, "route_original"
    return ds


def test_dataset_layout(monkeypatch):
    ds = _ds(6, monkeypatch)
    meas = {"route_original": [[float(i), 0.0] for i in range(20)], "target_point": [30.0, 2.0], "target_point_next": [60.0, -1.0]}
    r = ds._route_input(meas)
    assert len(r) == 6 and r[:4] == ds._subsample_route(meas["route_original"], n=4)
    assert r[4] == [30.0, 2.0] and r[5] == [60.0, -1.0]
    no_tp = ds._route_input({"route_original": meas["route_original"]})
    assert no_tp[4] == no_tp[3] and no_tp[5] == no_tp[4]                 # no logged target point: the far end of the near route


def test_constructor_accepts_dense_tp(monkeypatch):
    """(2026-11-21 box failure) the constructor validates WOR_ROUTE_MODE too; the layout test above bypassed it."""
    from src.training.wor_dataset import WorldOnRailsDataset
    monkeypatch.setenv("WOR_ROUTE_MODE", "dense_tp")
    WorldOnRailsDataset(data_dir="does-not-exist", synthetic_samples=4, route_points=6, img_size=(32, 64))


def _model(rp):
    torch.manual_seed(0)
    return QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, route_points=rp, model_size="10m", vision_grid=4,
                                  use_target_speed=True).eval()


def test_widened_route_proj_starts_at_the_checkpoint(monkeypatch):
    from src.training.wor_checkpoint import load_trainable_state
    old, new = _model(4), _model(6)
    monkeypatch.setenv("WOR_ROUTE_MODE", "dense_tp")
    load_trainable_state(new, old.state_dict(), frozen_keys=[], source="test")
    near = torch.randn(3, 4, 2)
    tps = torch.randn(3, 2, 2) * 30
    g = torch.Generator().manual_seed(1)
    inp = (torch.rand(3, 3, 128, 128, generator=g), torch.tensor([1.0, 5.0, 9.0]), torch.tensor([3, 3, 2]))
    with torch.no_grad():
        a = old(*inp, route=near)["waypoints"]
        b = new(*inp, route=torch.cat([near, tps], 1))["waypoints"]
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.equal(new.route_proj.weight[:, 8:], torch.zeros_like(new.route_proj.weight[:, 8:]))


def test_without_dense_tp_the_shape_change_is_not_widened(monkeypatch):
    from src.training.wor_checkpoint import load_trainable_state
    old, new = _model(4), _model(6)
    monkeypatch.setenv("WOR_ROUTE_MODE", "dense")
    with pytest.raises(Exception):
        load_trainable_state(new, old.state_dict(), frozen_keys=[], source="test")


def _tf(x, y):
    loc = types.SimpleNamespace(x=x, y=y, z=0.0)
    return types.SimpleNamespace(location=loc)


def test_target_point_helper_on_a_straight_road():
    from src.agents.tfpp_target_point import TfppTargetPoints
    plan = [(_tf(10.0 * i, 0.0), 4) for i in range(12)]                    # sparse plan, a point every 10 m along +x
    tp = TfppTargetPoints(plan)
    ego_inverse = np.eye(4)                                                 # ego at the origin facing +x: world = ego frame
    pts = tp.step((0.0, 0.0, 0.0), ego_inverse)
    assert pts.shape == (2, 2)
    assert np.allclose(pts[0], [10.0, 0.0]) and np.allclose(pts[1], [20.0, 0.0])
    shifted = np.eye(4)
    shifted[0, 3] = -25.0                                                   # ego at x = 25: the inverse matrix subtracts it
    pts = tp.step((25.0, 0.0, 0.0), shifted)
    assert pts[0][0] > 0 and np.allclose(pts[:, 1], 0.0)                    # the target point stays ahead, on the road
