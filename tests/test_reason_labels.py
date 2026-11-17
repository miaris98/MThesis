"""TODO A84: the expert-reason / danger / bucket labels from PDM-Lite logs (src/training/reason_labels.py), the dataset hook and the sampler."""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.world_on_rails import reason_heads
from src.training import reason_labels as rl
from src.training.bucket_sampler import bucket_weights


def _mat(x, y, yaw=0.0):
    m = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    m[:2, :2] = [[c, -s], [s, c]]
    m[0, 3], m[1, 3] = x, y
    return m.tolist()


STRAIGHT = [[2.5 + k, 0.0] for k in range(20)]


def _scene(n, actors_at, speed=0.0):
    """n frames of an ego standing at the origin facing +x; actors_at(k) -> list of boxes at frame k."""
    meas = [{"speed": speed, "target_speed": speed, "speed_limit": 13.89, "ego_matrix": _mat(0, 0), "route": STRAIGHT, "route_original": STRAIGHT}
            for _ in range(n)]
    boxes = [[{"class": "ego_car", "extent": [2.45, 0.92, 0.75], "position": [0, 0, 0], "id": 0}] + actors_at(k) for k in range(n)]
    return meas, boxes


def _car(cid, x, y, yaw=0.0, speed=0.0):
    return {"class": "car", "id": cid, "extent": [2.3, 1.0, 0.8], "matrix": _mat(x, y, yaw), "speed": speed, "number_of_wheels": 4}


def test_class_lists_match_between_label_and_model_side():
    assert rl.REASON_CLASSES == reason_heads.REASON_CLASSES


def test_cause_class():
    assert rl.cause_class(None) == 0 and rl.cause_class("") == 0
    assert rl.cause_class("walker.pedestrian.0002") == 2
    assert rl.cause_class("vehicle.lincoln.mkz") == 1
    assert rl.cause_class("vehicle.bh.crossbike", wheels=2) == 2            # a cyclist is a vulnerable road user
    assert rl.cause_class("traffic.traffic_light") == 3
    assert rl.cause_class("traffic.stop") == 4
    assert rl.cause_class("static.prop.constructioncone") == 5


def test_binding_needs_a_reduced_target_and_a_close_object():
    m = {"speed": 9.9, "target_speed": 9.9, "speed_limit": 13.89, "speed_reduced_by_obj_type": "traffic.traffic_light",
         "speed_reduced_by_obj_distance": 144.0}
    assert not rl.binding(m, rl.ReasonConfig())                              # cruising at 0.72 x limit, light far away (Town02 sample)
    m.update(target_speed=7.0, speed_reduced_by_obj_distance=20.0)
    assert rl.binding(m, rl.ReasonConfig())
    m.update(speed_reduced_by_obj_distance=144.0)
    assert not rl.binding(m, rl.ReasonConfig())                              # reduced (a ramp), but the light is not why
    m.update(speed_reduced_by_obj_distance=None)
    assert not rl.binding(m, rl.ReasonConfig())


def test_danger_crossing_car_is_positive_far_car_is_not():
    cfg = rl.ReasonConfig()
    # a car crossing the ego's path 5 m ahead, 2 m/s laterally (y: 6 -> 0 over 3 s at 4 Hz)
    meas, boxes = _scene(12, lambda k: [_car(7, 5.0, 6.0 - 0.5 * k, yaw=-math.pi / 2, speed=2.0)])
    ego = [np.asarray(m["ego_matrix"]) for m in meas]
    d, ttc = rl.danger_label(0, ego, [0.0] * 12, [STRAIGHT] * 12, boxes, cfg)
    assert d == 1 and 0 < ttc <= cfg.horizon_s
    meas, boxes = _scene(12, lambda k: [_car(7, 40.0, 6.0 - 0.5 * k, yaw=-math.pi / 2, speed=2.0)])
    d, ttc = rl.danger_label(0, ego, [0.0] * 12, [STRAIGHT] * 12, boxes, cfg)
    assert d == 0 and ttc == math.inf


def test_danger_ignores_followers_and_sees_static_obstacles():
    cfg = rl.ReasonConfig()
    ego = [np.asarray(_mat(0, 0))] * 12
    # a follower behind the ego driving into where the counterfactual ego will be: it would have braked for a real ego, so not a danger
    _, boxes = _scene(12, lambda k: [_car(9, -8.0 + 1.5 * k, 0.0, speed=6.0)])
    assert rl.danger_label(0, ego, [0.0] * 12, [STRAIGHT] * 12, boxes, cfg)[0] == 0
    # a cone (static: no world matrix, no id) in the path 4 m ahead
    cone = {"class": "static", "type_id": "static.prop.constructioncone", "extent": [0.2, 0.2, 0.4], "position": [4.0, 0.0, 0.0], "yaw": 0.0}
    _, boxes = _scene(12, lambda k: [cone])
    assert rl.danger_label(0, ego, [0.0] * 12, [STRAIGHT] * 12, boxes, cfg)[0] == 1
    # too few future frames, or an unreadable future box file: no label
    assert rl.danger_label(5, ego, [0.0] * 12, [STRAIGHT] * 12, boxes, cfg)[0] == -1
    boxes[3] = None
    assert rl.danger_label(0, ego, [0.0] * 12, [STRAIGHT] * 12, boxes, cfg)[0] == -1


def test_label_route_buckets_brake_cause_and_start():
    n = 30
    meas, boxes = _scene(n, lambda k: [{"class": "walker", "id": 5, "extent": [0.2, 0.2, 0.9], "matrix": _mat(30.0, 8.0), "speed": 0.0}])
    for i, m in enumerate(meas):                     # cruise 10 m/s, brake hard at frame 10, stand, start again at frame 20
        v = 10.0 if i < 10 else (0.0 if i < 20 else 4.0)
        m.update(speed=v, target_speed=v, ego_matrix=_mat(0.0, 0.0))
    for i in range(8, 20):
        meas[i].update(speed_reduced_by_obj_type="walker.pedestrian.0001", speed_reduced_by_obj_id=5, speed_reduced_by_obj_distance=6.0,
                       target_speed=min(meas[i]["target_speed"], 5.0))
    lab = rl.label_route(meas, boxes)
    assert lab["cause"][12] == 2 and abs(lab["dist"][12] - 6.0) < 1e-6
    assert lab["closing"][12] == pytest.approx(0.0)                          # standing ego, standing walker
    assert lab["brake"][7] == 1 and lab["brake"][2] == 0                     # the drop from 10 to 0 m/s is within 1 s of frame 7
    assert rl.BUCKETS[lab["bucket"][7]] == "brake"
    assert rl.BUCKETS[lab["bucket"][19]] == "start"
    assert rl.BUCKETS[lab["bucket"][25]] == "cruise"
    assert lab["danger"][-1] == -1


def test_save_and_load_roundtrip_refuses_another_config(tmp_path):
    lab = {k: np.zeros(3, np.int8) for k in ("cause", "brake", "danger", "bucket")}
    lab.update(dist=np.full(3, np.nan, np.float32), closing=np.full(3, np.nan, np.float32), ttc=np.full(3, np.inf, np.float32))
    p = str(tmp_path / "reason_v1.npz")
    rl.save_labels(p, ["0000", "0001", "0002"], lab, rl.ReasonConfig())
    idx, cols = rl.load_labels(p)
    assert idx == {"0000": 0, "0001": 1, "0002": 2} and cols["cause"].shape == (3,)
    with pytest.raises(ValueError):
        rl.load_labels(p, rl.ReasonConfig(horizon_s=3.0))


def test_bucket_weights():
    b = np.array([0] * 900 + [1] * 90 + [2] * 10)
    w = bucket_weights(b, 0.0, min_cruise=0.0)
    assert np.allclose(w, 1 / 1000)                                          # alpha 0 = uniform
    w = bucket_weights(b, 1.0, min_cruise=0.0)
    share = [w[b == k].sum() for k in range(3)]
    assert np.allclose(share, 1 / 3)                                         # alpha 1 = every bucket equally often
    w = bucket_weights(b, 1.0, min_cruise=0.5)
    assert abs(w[b == 0].sum() - 0.5) < 1e-9 and abs(w.sum() - 1) < 1e-9     # the cruise floor holds
    rare_cruise = np.array([0] * 100 + [1] * 450 + [2] * 450)                  # cruise only 10% of the frames (the real A84 data: 10.7%)
    plain = bucket_weights(rare_cruise, 0.5, min_cruise=0.0)[rare_cruise == 0].sum()       # sqrt balancing alone: 10 / (10 + 21.2 + 21.2)
    floored = bucket_weights(rare_cruise, 0.5, min_cruise=0.3)[rare_cruise == 0].sum()
    assert abs(plain - 10 / (10 + 2 * 450 ** 0.5)) < 1e-9 and abs(floored - plain) < 1e-12   # the 0.3 floor does not lift a rare cruise bucket
    w = bucket_weights(np.array([-1, -1, 1]), 1.0, min_cruise=0.0)
    assert abs(w[:2].sum() - 0.5) < 1e-9                                     # unlabelled frames count as cruise


def _pdm_route(root: Path, n=24):
    from PIL import Image
    route = root / "Town01" / "data" / "Accident" / "Route0_Rep0"
    (route / "measurements").mkdir(parents=True)
    (route / "rgb").mkdir()
    (route / "boxes").mkdir()
    for i in range(n):
        meas = {"speed": 5.0, "command": 4, "ego_matrix": _mat(float(i), 0.0), "route": STRAIGHT, "route_original": STRAIGHT,
                "target_speed": 5.0, "speed_limit": 13.89}
        with gzip.open(route / "measurements" / f"{i:04d}.json.gz", "wt") as f:
            json.dump(meas, f)
        with gzip.open(route / "boxes" / f"{i:04d}.json.gz", "wt") as f:
            json.dump([{"class": "ego_car", "extent": [2.45, 0.92, 0.75], "position": [0, 0, 0], "id": 0}], f)
        Image.fromarray(np.full((32, 64, 3), 128, np.uint8)).save(route / "rgb" / f"{i:04d}.jpg")
    return route


def test_dataset_attaches_labels_and_refuses_a_missing_tag(tmp_path, monkeypatch):
    import subprocess
    import sys
    from src.training.wor_dataset import WorldOnRailsDataset
    _pdm_route(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "training" / "build_reason_labels.py"
    r = subprocess.run([sys.executable, str(script), "--data_dir", str(tmp_path), "--tag", "t1", "--workers", "1", "--use_mlflow", "0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "label noise" in r.stdout
    monkeypatch.setenv("WOR_REASON_LABELS", "t1")
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=False)
    assert all(s["reason"] is not None for s in ds.samples)
    item = ds[0]
    for k in ("reason_cause", "reason_dist", "reason_dist_valid", "reason_brake", "reason_danger", "bucket"):
        assert k in item
    assert item["reason_cause"].dtype == torch.long and float(item["reason_dist_valid"]) == 0.0
    monkeypatch.setenv("WOR_REASON_LABELS", "missing_tag")
    with pytest.raises(RuntimeError):
        WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=False)
