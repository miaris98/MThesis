"""TODO A71: scripts/training/build_q_labels.py on a synthetic PDM-Lite route (measurements/ + boxes/, gzip JSON, the carla_garage layout)."""
import gzip
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from src.training.counterfactual_q import QConfig, load_q_labels

_spec = importlib.util.spec_from_file_location("build_q_labels", Path(__file__).resolve().parents[1] / "scripts" / "training" / "build_q_labels.py")
bql = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bql)


def _write_route(root: Path, n: int, obstacle_x=None) -> Path:
    rd = root / "Town12" / "Accident" / "Route7_Rep0"
    (rd / "measurements").mkdir(parents=True)
    (rd / "boxes").mkdir()
    for i in range(n):
        x = 2.0 * i                                               # 8 m/s at 4 Hz
        ego = [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0]]
        meas = {"speed": 8.0, "target_speed": 8.0, "ego_matrix": ego, "route_original": [[2.5 + j, 0.0] for j in range(20)],
                "route": [[2.5 + j, 0.0] for j in range(20)], "command": 4}
        boxes = [{"class": "ego_car", "extent": [2.45, 0.92, 0.75], "position": [0, 0, 0], "yaw": 0.0, "id": 1},
                 {"class": "ego_info", "num_lanes_same_direction": 1, "num_lanes_opposite_direction": 1, "ego_lane_number": 0, "is_in_junction": False}]
        if obstacle_x is not None:
            boxes.append({"class": "static_car", "extent": [2.0, 0.9, 0.8], "position": [obstacle_x - x, 0.0, 0.0], "yaw": 0.0})
        for sub, obj in (("measurements", meas), ("boxes", boxes)):
            with gzip.open(rd / sub / f"{i:04d}.json.gz", "wt") as f:
                json.dump(obj, f)
    return rd


def test_find_routes_and_label_one(tmp_path):
    cfg = QConfig()
    rd = _write_route(tmp_path, cfg.horizon + 8, obstacle_x=60.0)
    assert bql.find_routes(str(tmp_path)) == [str(rd)]
    stats = bql.label_route_dir(str(rd), "qtest", cfg, overwrite=False)
    assert stats["frames"] == cfg.horizon + 8 and stats["labelled"] == 8
    z = np.load(rd / "qtest.npz")
    assert z["q"].shape == (cfg.horizon + 8, 8, 5) and list(z["frame"][:2]) == ["0000", "0001"]
    assert json.loads(str(z["config"]))["horizon"] == cfg.horizon
    labels = load_q_labels(str(rd), "qtest")
    assert len(labels) == 8 and labels["0000"].shape == (40,)
    # the expert's own action (8 m/s, in lane) is the best or nearly the best while the obstacle is still far ahead
    assert 0.0 <= stats["expert_top3_share"] <= 1.0 and stats["expert_top3_share"] > 0.5
    # a second call without --overwrite leaves the file alone
    assert bql.label_route_dir(str(rd), "qtest", cfg, overwrite=False)["skipped"] is True


def test_expert_action_index_uses_target_speed_and_route_shift():
    cfg = QConfig()
    meas = {"target_speed": 10.2, "route": [[2.5 + j, 3.4 if j > 3 else 0.0] for j in range(20)], "route_original": [[2.5 + j, 0.0] for j in range(20)]}
    v, d = bql.expert_action_index(meas, cfg)
    assert cfg.speed_targets[v] == 10.0 and cfg.lat_targets[d] == 3.5
    # When stopped waiting behind an obstacle (target_speed < 1.0), the expert stays at its current offset r[0] - ro[0]
    meas_stopped = {"target_speed": 0.0, "route": [[2.5 + j, -1.75 if j > 3 else 0.0] for j in range(20)], "route_original": [[2.5 + j, 0.0] for j in range(20)]}
    v0, d0 = bql.expert_action_index(meas_stopped, cfg)
    assert cfg.speed_targets[v0] == 0.0 and cfg.lat_targets[d0] == 0.0
