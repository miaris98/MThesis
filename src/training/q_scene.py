"""TODO A71: the scene of one logged PDM-Lite frame for the counterfactual Q labels (split out of counterfactual_q.py, 2026-11-02).

build_scene turns frame k of a route (measurements + boxes) into a Scene: route_original extended with later frames' routes, the ego's projection
(s0, d0), actor tracks moved into frame k's ego frame (same-lane followers dropped, statics held fixed), the over-speed limit per future frame and
the route-lane anchor of the drivable mask. logged_step_reward is A71's reward of the expert's own logged step (A72).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from src.training.counterfactual_q import (LANE_WIDTH, QConfig, RouteFrame, Scene, _box_mat, _half_extents, _is_obstacle, _mat, _wrap,
                                           drivable_lateral_mask, red_light_ahead, speed_limit_of)


def _route_points(meas: dict) -> np.ndarray:
    pts = np.asarray(meas.get("route_original") or meas.get("route") or [[1.0, 0.0]], dtype=np.float64)
    return pts.reshape(len(pts), -1)[:, :2]


def _extended_route(frames, k: int, cfg: QConfig, inv0: np.ndarray) -> RouteFrame:
    """route_original of frame k, extended past its ~22 m with the route_original of later frames moved into frame k (inv(E_k) @ E_{k+t})."""
    pts = _route_points(frames[k][0])
    for t in range(4, cfg.horizon + 1, 4):
        meas = frames[k + t][0]
        rel = inv0 @ _mat(meas["ego_matrix"])
        later = _route_points(meas) @ rel[:2, :2].T + rel[:2, 3]
        rf = RouteFrame(pts, from_origin=False)
        s, d = rf.project(later[:, 0], later[:, 1])
        ext = later[(s > rf.cum[-1] + 0.5) & (np.abs(d) < 2.0)]
        if len(ext):
            pts = np.vstack([pts, ext[np.argsort(rf.project(ext[:, 0], ext[:, 1])[0])]])
    return RouteFrame(pts, from_origin=False)


def _route_lane_info(frames, k: int, route: RouteFrame, inv0: np.ndarray, d0: float, ego_info: Optional[dict]):
    """(ego_info, mask anchor): off the route lane (mid-bypass) ego_info describes the lane under the ego, whose "opposite" lanes include the
    route lane, so the latest earlier frame in the route lane is used (review 2026-11-01); anchor None = the ego's own lane."""
    if abs(d0) <= LANE_WIDTH / 2:
        return ego_info, None
    for j in range(k - 1, max(k - 80, 0) - 1, -1):
        if frames[j] is None or "ego_matrix" not in frames[j][0]:
            continue
        rel = inv0 @ _mat(frames[j][0]["ego_matrix"])
        dj = float(route.project(np.array([rel[0, 3]]), np.array([rel[1, 3]]))[1][0])
        info_j = next((b for b in frames[j][1] if b.get("class") == "ego_info"), None)
        if abs(dj) <= LANE_WIDTH / 2 and info_j is not None:
            return info_j, dj
    return ego_info, None


def logged_step_reward(frames: List[Tuple[dict, list]], k: int, cfg: QConfig) -> Optional[float]:
    """TODO A72: A71's step reward for the expert's real step k -> k+1 (progress along route_original, over-speed against frame k+1's limit,
    lateral and off-road cost at the pose reached); None without a logged successor. Collisions are not checked: the expert's logged path
    does not collide (<= 0.6% of frames at margin 0, the 2026-11-01 check). The limit uses no standing-vehicle set (a vehicle cause keeps the
    expert's target speed), the conservative side of speed_limit_of."""
    if k + 1 >= len(frames) or frames[k] is None or frames[k + 1] is None or "ego_matrix" not in frames[k + 1][0]:
        return None
    meas0, boxes0 = frames[k]
    inv0 = np.linalg.inv(_mat(meas0["ego_matrix"]))
    route = RouteFrame(_route_points(meas0), from_origin=False)
    (s0,), (d0,) = route.project(np.zeros(1), np.zeros(1))
    rel = inv0 @ _mat(frames[k + 1][0]["ego_matrix"])
    (s1,), (d1,) = route.project(np.array([rel[0, 3]]), np.array([rel[1, 3]]))
    info, anchor = _route_lane_info(frames, k, route, inv0, d0, next((b for b in boxes0 if b.get("class") == "ego_info"), None))
    off = ~drivable_lateral_mask(np.array([d1 - (d0 if anchor is None else anchor)]), info)[0]
    v1 = float(frames[k + 1][0].get("speed", 0.0))
    limit = speed_limit_of(frames[k + 1][0], frames[k + 1][1]) + cfg.overspeed_margin
    return float(cfg.w_progress * (s1 - s0) / (cfg.v_max * cfg.dt) - cfg.w_overspeed * max(0.0, v1 - limit) / cfg.v_max
                 - cfg.w_lateral * abs(d1) / LANE_WIDTH - cfg.w_offroad * float(off))


def build_scene(frames: List[Tuple[dict, list]], k: int, cfg: QConfig) -> Optional[Scene]:
    """The scene of frame k from the logged (measurements, boxes) list of one route; None without `horizon` logged successors."""
    if k + cfg.horizon >= len(frames) or any(frames[j] is None for j in range(k, k + cfg.horizon + 1)):
        return None
    meas0, boxes0 = frames[k]
    inv0 = np.linalg.inv(_mat(meas0["ego_matrix"]))
    route = _extended_route(frames, k, cfg, inv0)
    s0, d0 = (float(x[0]) for x in route.project(np.zeros(1), np.zeros(1)))
    ego = next((b for b in boxes0 if b.get("class") == "ego_car"), None)
    ego_half = (float(ego["extent"][0]), float(ego["extent"][1])) if ego else (2.45, 0.92)
    ego_info = next((b for b in boxes0 if b.get("class") == "ego_info"), None)
    ego_info, anchor = _route_lane_info(frames, k, route, inv0, d0, ego_info)

    tracks, statics, followers = {}, {}, set()
    for t in range(cfg.horizon + 1):
        meas, boxes = frames[k + t]
        rel_t = inv0 @ _mat(meas["ego_matrix"])                 # frame k+t's ego frame -> frame k's ego frame
        for b in boxes:
            if not _is_obstacle(b):
                continue
            m = rel_t @ _box_mat(b)
            row = (m[0, 3], m[1, 3], math.atan2(m[1, 0], m[0, 0])) + _half_extents(b)
            if "id" not in b:                                   # parked cars and props carry no id: world-fixed from the frame they appear
                statics.setdefault((round(row[0] / 0.5), round(row[1] / 0.5)), (t, row))
                continue
            if b["id"] not in tracks and row[0] < -ego_half[0] and abs(row[1]) < LANE_WIDTH / 2 and abs(_wrap(row[2])) < 0.5:
                followers.add(b["id"])                          # a same-lane follower: on rails it would rear-end every braking ego
            tracks.setdefault(b["id"], {})[t] = row
    rows = []
    for aid, tr in tracks.items():
        if aid in followers:
            continue
        r = np.full((cfg.horizon + 1, 5), np.nan)
        for t, v in tr.items():
            r[t] = v
        rows.append(r)
    for t0, row in statics.values():
        r = np.full((cfg.horizon + 1, 5), np.nan)
        r[t0:] = row
        rows.append(r)
    boxes = np.stack(rows, axis=1) if rows else np.full((cfg.horizon + 1, 0, 5), np.nan)
    if boxes.shape[1]:
        with np.errstate(all="ignore"):
            near = np.nanmin(np.hypot(boxes[..., 0], boxes[..., 1]), axis=0) <= cfg.max_actor_range
        boxes = boxes[:, near]
    standing = {aid for aid, tr in tracks.items() if len(tr) > cfg.horizon // 2
                and max(math.hypot(r[0] - q[0], r[1] - q[1]) for r in tr.values() for q in tr.values()) <= 0.5}
    if any(red_light_ahead(*frames[k + t]) for t in range(cfg.horizon + 1)):   # a red light anywhere in the horizon: the expert's speed throughout
        tgt = np.array([float(frames[k + t][0].get("target_speed", frames[k + t][0].get("speed", 0.0))) for t in range(cfg.horizon + 1)])
    else:
        tgt = np.array([speed_limit_of(frames[k + t][0], frames[k + t][1], standing) for t in range(cfg.horizon + 1)])
    return Scene(route, boxes, ego_half, float(meas0.get("speed", 0.0)), tgt, ego_info, s0, d0, anchor)
