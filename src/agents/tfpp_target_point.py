"""TF++'s target point and next target point at driving time (TODO A87 target-point arm, 2026-11-21).

carla_garage's sensor_agent steps its RoutePlanner(7.5 m, 50 m) over the sparse global plan every tick and conditions on route[1] (the target
point) and route[2] (the next one), falling back to route[1] / route[0] near the end of the route. PDM-Lite logged its `target_point` /
`target_point_next` with the same planner, which is what the dense_tp training mode reads. Here the planner is carla_garage's own class,
stepped with the ego position, and the points are projected like every other route point of this project's agent: inverse ego matrix,
x forward, y lateral.
"""
from typing import Sequence

import numpy as np


class TfppTargetPoints(object):

    def __init__(self, sparse_plan_world: Sequence, min_distance: float = 7.5, max_distance: float = 50.0):
        """`sparse_plan_world`: the leaderboard's downsampled (carla.Transform, RoadOption) plan (AutonomousAgent._global_plan_world_coord)."""
        from src.baselines.tfpp_cam import install_import_shims
        install_import_shims()
        from nav_planner import RoutePlanner   # carla_garage/team_code
        self.planner = RoutePlanner(min_distance, max_distance)
        self.planner.set_route(sparse_plan_world, gps=False)

    def step(self, ego_xyz, inverse_matrix) -> np.ndarray:
        """(2, 2) float32: the target point and the next target point in the ego frame (rows), after advancing the planner to `ego_xyz`."""
        r = self.planner.run_step(np.asarray(ego_xyz, dtype=np.float64))
        if len(r) > 2:
            tp, tpn = r[1][0], r[2][0]
        elif len(r) > 1:
            tp = tpn = r[1][0]
        else:
            tp = tpn = r[0][0]
        inv = np.asarray(inverse_matrix, dtype=np.float64)
        return np.asarray([(inv @ np.array([p[0], p[1], p[2], 1.0]))[:2] for p in (tp, tpn)], dtype=np.float32)
