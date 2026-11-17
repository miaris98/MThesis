"""TODO A84: labels for WHY the expert slows down, a counterfactual "danger" (crash) label, and sampling buckets, from PDM-Lite's own logs.

Behaviour cloning copies the expert's next poses and target speed, never the cause of a slowdown. PDM-Lite logs the cause per frame
(`speed_reduced_by_obj_type` / `_id` / `_distance`, the hazard flags) and every actor's box with a world matrix, so the reason and a
counterfactual can be labelled offline, without images and without CARLA. Labels only: the policy never sees any of this as input.

Per frame:
  cause    REASON_CLASSES index of the object that BINDS the expert's speed (its target speed is below the free cruise speed and the object is
           close enough to matter), else 0 ("none"). A red light 144 m away is logged as the reducing object on every frame of a straight
           approach although the expert cruises (checked on the Town02 sample), hence the distance gate.
  dist     metres to that object (NaN when cause == 0).
  closing  ego speed minus the object's speed along the ego heading, m/s (NaN when cause == 0; lights, stop signs and props move at 0).
  brake    1 = the expert's measured speed drops >= brake_drop within brake_window_s (a hard brake is coming), 0 = not, -1 = no future frames.
  danger   1 = proceeding along the expert's planned route at v_cf = max(current speed, creep_mps) for horizon_s would bring the ego's footprint
           within margin_m of another road user's RECORDED future footprint or of a static obstacle; 0 = clear; -1 = not enough future frames.
           Open loop (actors do not react to the counterfactual ego) and only actors that are not behind the ego (followers would react).
           This is the "would we crash if we went on" classifier label (TODO A37's question asked as one binary per frame, the user's
           2026-11-16 crash-classifier idea): a yield, a stop for a crossing pedestrian, waiting for the oncoming lane on a two-way bypass.
  ttc      seconds to the first counterfactual contact (inf when clear).
  bucket   BUCKETS index (one per frame, first match in priority order) for the re-balancing sampler (src/training/bucket_sampler.py).

The free cruise speed is PDM-Lite's ratio_target_speed_limit (0.72) times the logged speed limit: 13.89 m/s x 0.72 = 10.0 m/s, the
target the expert holds on the Town02 sample whenever nothing reduces it.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

REASON_CLASSES = ("none", "vehicle", "vulnerable", "light", "stop", "static")
BUCKETS = ("cruise", "brake", "danger", "start", "vulnerable", "vehicle", "static", "light_stop", "lane_change", "junction")
# Boxes that are not obstacles: the ego, scene metadata, and the traffic-control objects (their effect is the cause label, not a footprint)
NON_OBSTACLE_CLASSES = frozenset({"ego_car", "ego_info", "weather", "traffic_light", "traffic_light_vqa", "stop_sign", "stop_sign_vqa"})
EGO_HALF_EXTENT = (2.446, 0.918)    # half length, half width (m) of the PDM-Lite ego box (Lincoln MKZ); used when the frame has no ego box


@dataclass(frozen=True)
class ReasonConfig:
    """Every threshold of the label pass. Stored in each label file; the dataset refuses a file built with a different config."""
    version: int = 1
    fps: float = 4.0                 # PDM-Lite measurement frames per second
    cruise_ratio: float = 0.72       # PDM-Lite ratio_target_speed_limit: free target speed = 0.72 x speed limit
    bind_margin: float = 0.5         # the target speed must be this far below the free cruise speed for an object to bind (m/s)
    bind_min_dist: float = 15.0      # an object closer than this can bind at any speed (m)
    bind_time: float = 3.0           # ... and one within bind_time x speed + bind_extra (m) at speed
    bind_extra: float = 10.0
    horizon_s: float = 2.0           # danger look-ahead
    creep_mps: float = 3.0           # the counterfactual speed of a standing or slow ego
    margin_m: float = 0.25           # footprints closer than this count as contact
    brake_drop: float = 2.0          # brake = speed drop of at least this much ...
    brake_window_s: float = 1.0      # ... within this window
    onset_drop: float = 4.0          # brake bucket also covers an expert target-speed drop of at least this much ...
    onset_window_s: float = 2.0      # ... within this window (TODO A38's brake-onset frames)
    start_speed: float = 0.5         # start-from-stop: below this speed now ...
    start_gain: float = 1.0          # ... and at least this much faster one second later
    swerve_m: float = 0.5            # the expert's route bent this far off the plan = a bypass (swerve.SWERVE_SHIFT_M)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


def cause_class(obj_type: Optional[str], wheels: Optional[int] = None) -> int:
    """REASON_CLASSES index of a `speed_reduced_by_obj_type` string. Two-wheelers count as vulnerable road users."""
    if not obj_type:
        return 0
    t = str(obj_type).lower()
    if t.startswith("walker"):
        return 2
    if t.startswith("vehicle"):
        return 2 if wheels == 2 else 1
    if "traffic_light" in t:
        return 3
    if t.startswith("traffic.stop") or "stop" in t.split(".")[-1]:
        return 4
    if t.startswith("static"):
        return 5
    return 0


def free_cruise(meas: dict, cfg: ReasonConfig) -> Optional[float]:
    lim = meas.get("speed_limit")
    return None if lim is None else cfg.cruise_ratio * float(lim)


def binding(meas: dict, cfg: ReasonConfig) -> bool:
    """True when the logged reducing object actually limits the expert's target speed at this frame."""
    if not meas.get("speed_reduced_by_obj_type"):
        return False
    dist = meas.get("speed_reduced_by_obj_distance")
    if dist is None or not math.isfinite(float(dist)):
        return False
    v = max(float(meas.get("speed", 0.0)), float(meas.get("target_speed", 0.0)))
    if float(dist) > max(cfg.bind_min_dist, cfg.bind_time * v + cfg.bind_extra):
        return False
    cruise = free_cruise(meas, cfg)
    return cruise is None or float(meas.get("target_speed", 0.0)) < cruise - cfg.bind_margin


def _polyline(route) -> np.ndarray:
    """Ego-frame route -> (N, 2) polyline from the ego origin; empty or short routes give a straight line ahead."""
    pts = np.zeros((1, 2))
    if route:
        arr = np.asarray(route, dtype=np.float64)[:, :2]
        if arr.ndim == 2 and len(arr):
            pts = np.vstack([pts, arr])
    if len(pts) < 2:
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    pts = pts[keep]
    if len(pts) < 2:
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    return pts


def path_at(poly: np.ndarray, dists: np.ndarray):
    """Positions and unit headings at arc lengths `dists` along `poly`, extended straight past its end."""
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    idx = np.clip(np.searchsorted(cum, dists, side="right") - 1, 0, len(seg) - 1)
    head = seg[idx] / seg_len[idx, None]
    pos = poly[idx] + head * (dists - cum[idx])[:, None]
    return pos, head


def _discs(center: np.ndarray, heading: np.ndarray, half_len: float, half_wid: float):
    """Three disc centres along the body axis and their common radius (a capsule of the box)."""
    a = max(0.0, half_len - half_wid)
    offs = np.array([-a, 0.0, a])
    return center[..., None, :] + heading[..., None, :] * offs[:, None], half_wid


def _local(inv_ego: np.ndarray, matrix) -> tuple:
    m = np.asarray(matrix, dtype=np.float64)
    p = inv_ego @ m[:, 3]
    h = inv_ego[:3, :3] @ m[:3, 0]
    n = math.hypot(h[0], h[1]) or 1.0
    return p[:2], np.array([h[0] / n, h[1] / n])


def _box_half(box) -> tuple:
    ext = box.get("extent") or [0.5, 0.5, 0.5]
    hl, hw = float(ext[0]), float(ext[1])
    if str(box.get("class", "")).startswith("walker"):
        hl = hw = max(hl, hw)
    return hl, hw


def danger_label(i: int, ego: Sequence[np.ndarray], speeds: Sequence[float], routes: Sequence, boxes: Sequence[Optional[list]],
                 cfg: ReasonConfig, ego_half=EGO_HALF_EXTENT):
    """(danger 1/0/-1, ttc s) for frame i. `ego` holds 4x4 world matrices, `boxes` the per-frame box lists (None = unreadable)."""
    k_max = int(round(cfg.horizon_s * cfg.fps))
    if i + k_max >= len(ego) or boxes[i] is None or any(boxes[i + k] is None for k in range(1, k_max + 1)):
        return -1, math.inf
    inv_i = np.linalg.inv(ego[i])
    v_cf = max(float(speeds[i]), cfg.creep_mps)
    times = np.arange(1, k_max + 1) / cfg.fps
    pos, head = path_at(_polyline(routes[i]), v_cf * times)
    e_c, e_r = _discs(pos, head, ego_half[0], ego_half[1])            # (K, 3, 2)
    e_r += cfg.margin_m
    # Static obstacles (props, parked cars: no world matrix, they do not move): their frame-i position holds at every step.
    stat_c, stat_r = [], []
    for b in boxes[i]:
        cls = str(b.get("class", ""))
        if cls in NON_OBSTACLE_CLASSES or "matrix" in b or b.get("position") is None:
            continue
        hl, hw = _box_half(b)
        yaw = float(b.get("yaw", 0.0))
        c, r = _discs(np.asarray(b["position"][:2], dtype=np.float64), np.array([math.cos(yaw), math.sin(yaw)]), hl, hw)
        stat_c.append(c)
        stat_r.append(np.full(3, r))
    stat_c = np.concatenate(stat_c) if stat_c else np.zeros((0, 2))
    stat_r = np.concatenate(stat_r) if stat_r else np.zeros(0)
    # Dynamic actors: tracked by id through the future frames. An actor behind the ego when first seen (at frame i, or at its first future
    # frame if it appears later) is a follower: it reacted to the expert, so its recorded future says nothing about a counterfactual ego.
    followers, first_seen = set(), set()
    for b in boxes[i]:
        if str(b.get("class", "")) in NON_OBSTACLE_CLASSES or "matrix" not in b:
            continue
        first_seen.add(b.get("id"))
        if _local(inv_i, b["matrix"])[0][0] < -ego_half[0]:
            followers.add(b.get("id"))
    for k in range(1, k_max + 1):
        dyn_c, dyn_r = [], []
        for b in boxes[i + k]:
            cls = str(b.get("class", ""))
            if cls in NON_OBSTACLE_CLASSES or "matrix" not in b:
                continue
            bid = b.get("id")
            p, h = _local(inv_i, b["matrix"])
            if bid not in first_seen:
                first_seen.add(bid)
                if p[0] < -ego_half[0]:
                    followers.add(bid)
            if bid in followers:
                continue
            hl, hw = _box_half(b)
            c, r = _discs(p, h, hl, hw)
            dyn_c.append(c)
            dyn_r.append(np.full(3, r))
        cand_c = np.concatenate([stat_c] + dyn_c) if dyn_c else stat_c
        cand_r = np.concatenate([stat_r] + dyn_r) if dyn_r else stat_r
        if len(cand_c) == 0:
            continue
        d = np.linalg.norm(e_c[k - 1][:, None, :] - cand_c[None, :, :], axis=-1)        # (3, n)
        if (d < e_r + cand_r[None, :]).any():
            return 1, float(times[k - 1])
    return 0, math.inf


def _obj_speed_along(meas: dict, boxes_i: Optional[list], ego_i: np.ndarray) -> Optional[float]:
    """The reducing object's speed along the ego heading (m/s); 0 for lights, stop signs and props; None if the box is not found."""
    oid = meas.get("speed_reduced_by_obj_id")
    cls = cause_class(meas.get("speed_reduced_by_obj_type"))
    if cls in (3, 4, 5):
        return 0.0
    if boxes_i is None or oid is None:
        return None
    for b in boxes_i:
        if b.get("id") == oid and "matrix" in b:
            _, h = _local(np.linalg.inv(ego_i), b["matrix"])
            return float(b.get("speed", 0.0)) * float(h[0])
    return None


def _wheels(meas: dict, boxes_i: Optional[list]) -> Optional[int]:
    oid = meas.get("speed_reduced_by_obj_id")
    if boxes_i is None or oid is None:
        return None
    for b in boxes_i:
        if b.get("id") == oid:
            w = b.get("number_of_wheels")
            return None if w is None else int(w)
    return None


def label_route(meas: Sequence[Optional[dict]], boxes: Sequence[Optional[list]], cfg: ReasonConfig = ReasonConfig()) -> Dict[str, np.ndarray]:
    """All per-frame labels of one route. `meas[i]` / `boxes[i]` are the parsed JSON of frame i (None = unreadable frame)."""
    from src.training.swerve import brake_onset_flags, route_shift
    n = len(meas)
    ego = [np.asarray(m["ego_matrix"], dtype=np.float64) if m else np.eye(4) for m in meas]
    speeds = [float(m.get("speed", 0.0)) if m else 0.0 for m in meas]
    routes = [(m.get("route") if m else None) for m in meas]
    out = {k: np.full(n, -1, dtype=np.int8) for k in ("cause", "brake", "danger", "bucket")}
    out["dist"] = np.full(n, np.nan, dtype=np.float32)
    out["closing"] = np.full(n, np.nan, dtype=np.float32)
    out["ttc"] = np.full(n, np.inf, dtype=np.float32)
    kb = int(round(cfg.brake_window_s * cfg.fps))
    onset = brake_onset_flags([m.get("target_speed") if m else None for m in meas], int(round(cfg.onset_window_s * cfg.fps)), cfg.onset_drop)
    for i in range(n):
        m = meas[i]
        if not m:
            continue
        if binding(m, cfg):
            c = cause_class(m.get("speed_reduced_by_obj_type"), _wheels(m, boxes[i]))
            out["cause"][i] = c
            if c:
                out["dist"][i] = float(m["speed_reduced_by_obj_distance"])
                along = _obj_speed_along(m, boxes[i], ego[i])
                if along is not None:
                    out["closing"][i] = speeds[i] - along
        else:
            out["cause"][i] = 0
        fut = [speeds[j] for j in range(i + 1, min(n, i + kb + 1)) if meas[j]]
        if len(fut) == kb:
            out["brake"][i] = int(min(fut) <= speeds[i] - cfg.brake_drop)
        dg, ttc = danger_label(i, ego, speeds, routes, boxes, cfg)
        out["danger"][i], out["ttc"][i] = dg, ttc
        start = len(fut) == kb and speeds[i] < cfg.start_speed and fut[-1] >= speeds[i] + cfg.start_gain
        swerve = route_shift(m.get("route"), m.get("route_original")) > cfg.swerve_m
        c = int(out["cause"][i])
        if out["brake"][i] == 1 or bool(onset[i]):
            b = "brake"
        elif dg == 1:
            b = "danger"
        elif start:
            b = "start"
        elif c == 2:
            b = "vulnerable"
        elif c == 1:
            b = "vehicle"
        elif c == 5 or swerve:
            b = "static"
        elif c in (3, 4):
            b = "light_stop"
        elif m.get("changed_route"):
            b = "lane_change"
        elif m.get("junction"):
            b = "junction"
        else:
            b = "cruise"
        out["bucket"][i] = BUCKETS.index(b)
    return out


def expert_consistency(meas: Sequence[Optional[dict]], labels: Dict[str, np.ndarray], cfg: ReasonConfig = ReasonConfig()) -> Dict[str, int]:
    """Label-noise counts (TODO A37 step 0): on danger frames the expert should slow down or stay slow.
    went = standing/slow (< creep) and at least creep one second later; kept = moving and not slower one second later."""
    n, k = len(meas), int(round(cfg.fps))
    c = {"danger": 0, "slow_danger": 0, "slow_went": 0, "moving_danger": 0, "moving_kept": 0}
    for i in range(n):
        if labels["danger"][i] != 1 or i + k >= n or not meas[i] or not meas[i + k]:
            continue
        v0, v1 = float(meas[i].get("speed", 0.0)), float(meas[i + k].get("speed", 0.0))
        c["danger"] += 1
        if v0 < cfg.creep_mps:
            c["slow_danger"] += 1
            c["slow_went"] += int(v1 >= cfg.creep_mps)
        else:
            c["moving_danger"] += 1
            c["moving_kept"] += int(v1 >= v0 - 0.25)
    return c


def save_labels(path: str, frame_ids: List[str], labels: Dict[str, np.ndarray], cfg: ReasonConfig) -> None:
    import os
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, frame_ids=np.asarray(frame_ids), config=np.asarray(cfg.to_json()), **labels)
    os.replace(tmp, path)


LABEL_KEYS = ("cause", "dist", "closing", "brake", "danger", "ttc", "bucket")


def label_path(route_dir: str, tag: str) -> str:
    import os
    return os.path.join(route_dir, f"reason_{tag}.npz")


def load_labels(path: str, cfg: ReasonConfig = ReasonConfig()):
    """-> ({frame id: row}, {key: column array}). Refuses a file built with another config (a silent mix of label definitions)."""
    z = np.load(path, allow_pickle=False)
    if str(z["config"]) != cfg.to_json():
        raise ValueError(f"{path} was built with config {str(z['config'])}, this run expects {cfg.to_json()}: rebuild the labels")
    return {str(fid): j for j, fid in enumerate(z["frame_ids"])}, {k: z[k] for k in LABEL_KEYS}
