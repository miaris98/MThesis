"""TODO A71 (2026-11-01, after the Gemini review): counterfactual Q labels on PDM-Lite logs, computed without the simulator.

World-on-Rails idea on our action space. The other actors replay their logged tracks ("on rails"); only the ego is simulated, in the frame of the
leaderboard route (`route_original`, extended with later frames' routes): arc length s along the route, lateral offset d from it (+d = +y =
right, CARLA's ego frame), speed v. The lattice starts where the ego actually is: (s0, d0) is the ego's projection onto the route, so a frame in
the middle of a bypass starts one lane off the route rather than on it. An action is a (target speed, target lateral offset) pair: the 8 TF++
speed bins of the target-speed head x `lat_targets` (offsets from the route, the same frame act() uses at evaluation).

    Q(s0, a) = return of committing to a's lateral target for the whole horizon and holding its speed target for `hold` steps,
               then the Bellman-optimal speed continuation (re-chosen every step) up to `horizon` steps.

The commitment is what makes the label informative: with everything re-chosen every 0.25 s almost any first action can be undone by braking on
the next step, so Q(s0, .) would be flat, and a lateral target released after the hold makes +-1.75 and +-3.5 indistinguishable (review
2026-11-01). One step = one logged frame (4 Hz).

Reward per step (driving one step at v_max in lane earns w_progress):
    progress   +w_progress * (lattice distance travelled) / (v_max * dt)
    over-speed -w_overspeed * max(0, v - (expert target speed at that frame + margin)) / v_max. Red lights, stop signs and curves are only in the
               logs as the expert's target speed dropping, so this term carries them without a stop-line geometry PDM-Lite does not log.
    lateral    -w_lateral * |d| / lane width
    off-road   -w_offroad where d leaves the lanes the frame's `ego_info` reports around the ego's own lane
    collision  collision_penalty, terminal: the ego box (+ margin) overlaps a logged actor box anywhere along the step's swept stations (actors at
               mid-step) or at its end (actors at the step's end); no tunnelling at 5 m per step. Same-lane followers are not obstacles: a logged
               follower never reacts, so it would make every braking action a rear-end collision.

`Scene.rollout_return` evaluates one explicit action sequence with the same transitions; the brute-force test checks the dynamic programme
against it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

TF_SPEEDS = (0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0)   # = aux_heads.TARGET_SPEEDS (kept torch-free here)
LANE_WIDTH = 3.5
_IGNORED = ("ego_car", "ego_info", "weather", "landmark")


@dataclass(frozen=True)
class QConfig:
    dt: float = 0.25                        # one logged frame
    horizon: int = 12                       # 3 s, the label audit's horizon
    hold: int = 4                           # the speed target is held 1 s (matching act()'s hold_ticks=20); the lateral target for the whole horizon
    gamma: float = 0.95
    speed_targets: Tuple[float, ...] = TF_SPEEDS
    lat_targets: Tuple[float, ...] = (-LANE_WIDTH, -LANE_WIDTH / 2, 0.0, LANE_WIDTH / 2, LANE_WIDTH)
    v_max: float = 20.0
    v_res: float = 0.5
    s_res: float = 0.5
    a_acc: float = 2.0                      # the label audit's authorities (PDM-Lite brakes at 3.7-8.7 m/s^2)
    a_dec: float = 7.0
    lat_rate: float = 1.75                  # m/s: a full lane change in 2 s; the lateral lattice step is lat_rate * dt
    margin: float = 0.1                     # added to the ego half-extents; 0.3 made 6.7% of the expert's own logged bypass paths "collide"
                                            # on ParkedObstacleTwoWays (0.5% at 0), measured 2026-11-01 on 775 frames
    w_progress: float = 1.0
    w_overspeed: float = 3.0
    overspeed_margin: float = 0.5
    w_lateral: float = 0.05
    w_offroad: float = 1.0
    collision_penalty: float = -10.0
    max_actor_range: float = 80.0

    def __post_init__(self):
        if not 1 <= self.hold <= self.horizon:
            raise ValueError(f"hold must be in [1, horizon]; got hold={self.hold}, horizon={self.horizon}")


# ---------------------------------------------------------------- geometry

def boxes_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Separating-axis test for oriented rectangles (cx, cy, yaw, half_length, half_width); broadcasts over leading dimensions."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    dx, dy = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    ca, sa, cb, sb = np.cos(a[..., 2]), np.sin(a[..., 2]), np.cos(b[..., 2]), np.sin(b[..., 2])
    sep = np.zeros(dx.shape, dtype=bool)
    for ax, ay in ((ca, sa), (-sa, ca), (cb, sb), (-sb, cb)):
        proj_a = a[..., 3] * np.abs(ax * ca + ay * sa) + a[..., 4] * np.abs(-ax * sa + ay * ca)
        proj_b = b[..., 3] * np.abs(ax * cb + ay * sb) + b[..., 4] * np.abs(-ax * sb + ay * cb)
        sep |= np.abs(dx * ax + dy * ay) > proj_a + proj_b
    return ~sep


class RouteFrame:
    """The route as a polyline; pose(s, d) -> (x, y, yaw) in the ego frame, linear extrapolation past both ends. `from_origin` prepends the ego
    origin when the first point is far from it (a convenience for routes known to start at the ego; labels and act() use from_origin=False)."""

    def __init__(self, points: np.ndarray, from_origin: bool = True):
        pts = np.asarray(points, dtype=np.float64)
        pts = pts.reshape(len(pts), -1)[:, :2] if pts.size else np.zeros((0, 2))
        if len(pts) == 0 or (from_origin and np.hypot(*pts[0]) > 0.5):
            pts = np.vstack([[0.0, 0.0], pts])
        keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-3]
        self.pts = pts[keep]
        if len(self.pts) < 2:
            self.pts = np.vstack([self.pts[:1], self.pts[:1] + [1.0, 0.0]]) if len(self.pts) else np.array([[0.0, 0.0], [1.0, 0.0]])
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.seg_yaw = np.arctan2(seg[:, 1], seg[:, 0])
        self.cum = np.r_[0.0, np.cumsum(self.seg_len)]

    def pose(self, s: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=np.float64)
        i = np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1)
        yaw = self.seg_yaw[i]
        along = s - self.cum[i]                                     # negative before the first point, past the end along the last segment
        x = self.pts[i, 0] + along * np.cos(yaw) - np.asarray(d) * np.sin(yaw)
        y = self.pts[i, 1] + along * np.sin(yaw) + np.asarray(d) * np.cos(yaw)
        return x, y, yaw

    def project(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(s, d) of points: the nearest segment by distance (the end segments extend past their ends), never a sort on x."""
        p = np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], -1).reshape(-1, 1, 2)
        a = self.pts[:-1][None]                                       # (1, n_seg, 2)
        u = np.stack([np.cos(self.seg_yaw), np.sin(self.seg_yaw)], -1)[None]
        t = ((p - a) * u).sum(-1)                                     # (n_pts, n_seg) along each segment
        lo, hi = np.zeros(len(self.seg_len)), self.seg_len.copy()
        lo[0], hi[-1] = -np.inf, np.inf
        tc = np.clip(t, lo, hi)
        foot = a + tc[..., None] * u
        j = np.argmin(np.hypot(*(p - foot).transpose(2, 0, 1)), axis=1)
        k = np.arange(p.shape[0])
        n = np.stack([-u[0, j, 1], u[0, j, 0]], -1)                   # (-sin, cos): +d = +y = right
        return self.cum[j] + tc[k, j], ((p[:, 0] - a[0, j]) * n).sum(-1)


def drivable_lateral_mask(d: np.ndarray, ego_info: Optional[dict]) -> np.ndarray:
    """True where a lateral offset d from the ego's own lane stays on a lane `ego_info` reports (ego_lane_number counts from the leftmost
    same-direction lane, data_agent.py; oncoming lanes lie further left). Without lane information, or in a junction, only the own lane."""
    d = np.asarray(d, dtype=np.float64)
    slack = LANE_WIDTH / 2 + 0.25
    if not ego_info or ego_info.get("is_in_junction"):
        return np.abs(d) <= slack
    same = int(ego_info.get("num_lanes_same_direction") or 1)
    lane = int(ego_info.get("ego_lane_number") or 0)
    left = lane + int(ego_info.get("num_lanes_opposite_direction") or 0)
    right = max(same - 1 - lane, 0)
    return (d >= -(left * LANE_WIDTH + slack)) & (d <= right * LANE_WIDTH + slack)


def _mat(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return np.vstack([m, [0.0, 0.0, 0.0, 1.0]]) if m.shape == (3, 4) else m


def _box_mat(b: dict) -> np.ndarray:
    c, s = math.cos(b.get("yaw", 0.0)), math.sin(b.get("yaw", 0.0))
    p = b.get("position", [0.0, 0.0, 0.0])
    return np.array([[c, -s, 0, p[0]], [s, c, 0, p[1]], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def _is_obstacle(b: dict) -> bool:
    cls = str(b.get("class", ""))
    if cls in _IGNORED or cls.startswith(("traffic_light", "stop_sign")) or "extent" not in b:
        return False
    return cls.startswith("static") or cls in ("car", "walker", "bicycle", "motorcycle", "truck", "bus", "vehicle")


def _half_extents(b: dict) -> Tuple[float, float]:
    """(half length, half width) as logged. data_agent.py already swaps CARLA's buggy static extent ('BUG: static_extent x and y are swapped',
    line 1109) before saving, so no class is swapped here; swapping again turned parked cars sideways on real logs (2026-11-01 check)."""
    e = b["extent"]
    return float(e[0]), float(e[1])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- scene and lattice

@dataclass
class Scene:
    route: RouteFrame
    actor_boxes: np.ndarray                 # (H+1, M, 5) in the frame-0 ego frame; NaN where the actor is not logged at that step
    ego_half: Tuple[float, float]
    v0: float
    target_speed: np.ndarray                # (H+1,) the expert's target speed at each future frame
    ego_info: Optional[dict]
    s0: float = 0.0                         # the ego's projection onto the route
    d0: float = 0.0

    def collision_grids(self, cfg: QConfig, d_grid: np.ndarray, s_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(end, mid), each (H+1, n_d, n_s) bool: the ego at (s0 + s, d) overlaps an actor at step t's end / at the middle of step t-1 -> t."""
        S, D = np.meshgrid(self.s0 + s_grid, d_grid, indexing="ij")
        x, y, yaw = self.route.pose(S.ravel(), D.ravel())
        hl, hw = self.ego_half[0] + cfg.margin, self.ego_half[1] + cfg.margin
        ego = np.stack([x, y, yaw, np.full_like(x, hl), np.full_like(x, hw)], -1)
        r_ego = math.hypot(hl, hw)
        shape = (self.actor_boxes.shape[0], len(d_grid), len(s_grid))
        end, mid = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
        for t in range(1, self.actor_boxes.shape[0]):
            a1, a0 = self.actor_boxes[t], self.actor_boxes[t - 1]
            both = np.isfinite(a0[:, 0]) & np.isfinite(a1[:, 0])
            am = np.where(both[:, None], a1, a1)
            am = np.where(both[:, None], np.c_[(a0[:, :2] + a1[:, :2]) / 2, a0[:, 2] + _wrap(a1[:, 2] - a0[:, 2]) / 2, a1[:, 3:]], am)
            for grid, act in ((end, a1), (mid, am)):
                act = act[np.isfinite(act[:, 0])]
                if not len(act):
                    continue
                near = np.hypot(ego[:, None, 0] - act[None, :, 0], ego[:, None, 1] - act[None, :, 1]) <= r_ego + np.hypot(act[:, 3], act[:, 4])[None]
                ci, ai = np.nonzero(near)                           # bounding-circle pre-check, then the exact test on the candidate pairs
                hit = np.zeros(len(ego), dtype=bool)
                if len(ci):
                    hit[ci[boxes_overlap(ego[ci], act[ai])]] = True
                grid[t] = hit.reshape(len(s_grid), len(d_grid)).T
        return end, mid

    def rollout_return(self, actions: Sequence[Tuple[int, int]], cfg: QConfig) -> Tuple[float, bool]:
        """(discounted return, collided) of one explicit action sequence on the same lattice (the brute-force reference)."""
        lat = _Lattice(self, cfg)
        d, v, s = lat.d0, lat.v0, 0
        ret, disc = 0.0, 1.0
        for t, (av, ad) in enumerate(actions):
            v2 = lat.next_v[av, v]
            s2 = lat.s_next(s, v, v2)
            d2 = lat.next_d[ad, d] if s2 > s else d
            r, hit = lat.reward(t, s, d2, v, v2, s2)
            ret += disc * float(r)
            if hit:
                return ret, True
            disc *= cfg.gamma
            d, v, s = d2, v2, s2
        return ret, False


class _Lattice:
    """Grids, transitions and rewards shared by the dynamic programme and the explicit rollout."""

    def __init__(self, scene: Scene, cfg: QConfig):
        step_d = cfg.lat_rate * cfg.dt
        reach = max(max(abs(x) for x in cfg.lat_targets), min(abs(scene.d0), 2 * LANE_WIDTH))
        n_half = int(math.ceil(reach / step_d - 1e-9))
        self.d_grid = np.arange(-n_half, n_half + 1) * step_d
        self.v_grid = np.arange(0.0, cfg.v_max + 1e-9, cfg.v_res)
        self.s_grid = np.arange(0.0, cfg.v_max * cfg.horizon * cfg.dt + cfg.s_res + 1e-9, cfg.s_res)
        vt = np.array([int(np.argmin(np.abs(self.v_grid - x))) for x in cfg.speed_targets])
        dt_idx = np.array([int(np.argmin(np.abs(self.d_grid - x))) for x in cfg.lat_targets])
        up = max(1, int(math.floor(cfg.a_acc * cfg.dt / cfg.v_res + 1e-9)))
        dn = max(1, int(math.floor(cfg.a_dec * cfg.dt / cfg.v_res + 1e-9)))
        vi, di = np.arange(len(self.v_grid)), np.arange(len(self.d_grid))
        self.next_v = np.stack([np.where(t > vi, np.minimum(vi + up, t), np.maximum(vi - dn, t)) for t in vt])          # (n_vt, n_v)
        self.next_d = np.stack([di + np.sign(t - di) for t in dt_idx])                                                   # (n_dt, n_d)
        inc = 0.5 * (self.v_grid[:, None] + self.v_grid[None, :]) * cfg.dt                                              # (v, v')
        self.s_inc = np.floor(inc / cfg.s_res + 0.5).astype(int)
        self.d0 = int(np.argmin(np.abs(self.d_grid - scene.d0)))
        self.v0 = int(np.argmin(np.abs(self.v_grid - min(scene.v0, cfg.v_max))))
        end, mid = scene.collision_grids(cfg, self.d_grid, self.s_grid)
        self.coll_end = end                                                                                              # (H+1, n_d, n_s)
        # swept check: any colliding station in [s, s'] against the actors at mid-step, as a prefix sum along s (O(1) per transition)
        self.mid_cs = np.concatenate([np.zeros(mid.shape[:2] + (1,), dtype=np.int32), np.cumsum(mid, axis=2, dtype=np.int32)], axis=2)
        self.d_cost = (cfg.w_lateral * np.abs(self.d_grid) / LANE_WIDTH
                       + cfg.w_offroad * ~drivable_lateral_mask(self.d_grid - self.d_grid[self.d0], scene.ego_info))
        self.limit = np.asarray(scene.target_speed, dtype=np.float64) + cfg.overspeed_margin
        self.cfg, self.n_s = cfg, len(self.s_grid)

    def s_next(self, s, v, v2):
        return np.minimum(s + self.s_inc[v, v2], self.n_s - 1)

    def base(self, t, v, v2):
        c = self.cfg
        return (c.w_progress * self.s_inc[v, v2] * c.s_res / (c.v_max * c.dt)
                - c.w_overspeed * np.maximum(0.0, self.v_grid[v2] - self.limit[t + 1]) / c.v_max)

    def hit(self, t, s, d2, s2):
        s_lo = np.minimum(s + 1, s2)
        return (self.mid_cs[t + 1][d2, s2 + 1] - self.mid_cs[t + 1][d2, s_lo] > 0) | self.coll_end[t + 1][d2, s2]

    def reward(self, t, s, d2, v, v2, s2):
        """Reward of the step t -> t+1 and whether it ends in a collision; broadcasts over lattice index arrays."""
        h = self.hit(t, s, d2, s2)
        return self.base(t, v, v2) - self.d_cost[d2] + np.where(h, self.cfg.collision_penalty, 0.0), h


def solve_q(scene: Scene, cfg: QConfig, return_hits: bool = False):
    """Q(s0, a) for every (speed target, lateral target), shape (n_speed, n_lat); with return_hits also whether a's best sequence collides."""
    lat = _Lattice(scene, cfg)
    nd, nv, ns = len(lat.d_grid), len(lat.v_grid), lat.n_s
    c = cfg
    vI, sI = np.arange(nv)[:, None], np.arange(ns)[None, :]
    S_cur = np.broadcast_to(sI, (nv, ns)).ravel()
    # The lateral target is fixed per column; the speed continuation is re-chosen each step after the hold. The step is separable: the lateral
    # move depends on (d, lateral target) only, the speed/arc-length move on (v, s, speed target) only, so the (v, s) lookups are gathered once
    # per speed target and every d is a row gather.
    per_t = []
    for t in range(cfg.horizon - 1, cfg.hold - 1, -1):
        rows = []
        for av in range(len(cfg.speed_targets)):
            V2 = lat.next_v[av][vI]                                                     # (nv, 1)
            S2 = lat.s_next(sI, vI, V2)                                                 # (nv, ns)
            moving = (S2 > sI).ravel()
            base = np.broadcast_to(lat.base(t, vI, V2), (nv, ns)).ravel()
            S2f = S2.ravel()
            S_lo = np.minimum(S_cur + 1, S2f)
            hit = (lat.mid_cs[t + 1][:, S2f + 1] - lat.mid_cs[t + 1][:, S_lo] > 0) | lat.coll_end[t + 1][:, S2f]      # (nd, nv*ns)
            rows.append(((V2 * ns + S2).ravel(), base, hit, moving))
        per_t.append((t, rows))
    q = np.empty((len(cfg.speed_targets), len(cfg.lat_targets)))
    hits = np.zeros(q.shape, dtype=bool)
    for ad in range(len(cfg.lat_targets)):
        D2 = lat.next_d[ad]
        value, flag = np.zeros((nd, nv * ns)), np.zeros((nd, nv * ns), dtype=bool)
        for t, rows in per_t:
            best, bflag = np.full((nd, nv * ns), -np.inf), np.zeros((nd, nv * ns), dtype=bool)
            for idx, base, hit, moving in rows:
                h = np.where(moving[None, :], hit[D2], hit)
                dc = np.where(moving[None, :], lat.d_cost[D2][:, None], lat.d_cost[:, None])
                vn = np.where(moving[None, :], value[D2][:, idx], value[:, idx])
                fn = np.where(moving[None, :], flag[D2][:, idx], flag[:, idx])
                cand = base[None, :] - dc + np.where(h, c.collision_penalty, c.gamma * vn)
                cflag = h | fn
                better = cand > best
                best, bflag = np.where(better, cand, best), np.where(better, cflag, bflag)
            value, flag = best, bflag
        value, flag = value.reshape(nd, nv, ns), flag.reshape(nd, nv, ns)
        for av in range(len(cfg.speed_targets)):                # the held first action is a single deterministic path from s0
            d, v, s, ret, disc, hit = lat.d0, lat.v0, 0, 0.0, 1.0, False
            for t in range(cfg.hold):
                v2 = lat.next_v[av, v]
                s2 = lat.s_next(s, v, v2)
                d2 = lat.next_d[ad, d] if s2 > s else d
                r, h = lat.reward(t, s, d2, v, v2, s2)
                ret += disc * float(r)
                if h:
                    hit = True
                    break
                disc *= cfg.gamma
                d, v, s = d2, v2, s2
            if not hit and cfg.hold < cfg.horizon:
                ret, hit = ret + disc * value[d, v, s], bool(flag[d, v, s])
            q[av, ad], hits[av, ad] = ret, hit
    return (q.astype(np.float32), hits) if return_hits else q.astype(np.float32)


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
    tgt = np.array([float(frames[k + t][0].get("target_speed", frames[k + t][0].get("speed", 0.0))) for t in range(cfg.horizon + 1)])
    return Scene(route, boxes, ego_half, float(meas0.get("speed", 0.0)), tgt, ego_info, s0, d0)


def load_q_labels(route_dir: str, tag: Optional[str]) -> dict:
    """{frame id: flattened Q row} from `<route_dir>/<tag>.npz` (written by scripts/training/build_q_labels.py); rows with a NaN are dropped."""
    import os
    path = os.path.join(route_dir, f"{tag}.npz") if tag else ""
    if not tag or not os.path.exists(path):
        return {}
    z = np.load(path, allow_pickle=False)
    want = QConfig()
    if z["q"].shape[1:] != (len(want.speed_targets), len(want.lat_targets)):
        raise ValueError(f"{path}: Q labels of shape {z['q'].shape[1:]}, the policy's action grid is "
                         f"{(len(want.speed_targets), len(want.lat_targets))}: rebuild with build_q_labels.py --overwrite")
    if "config" in z.files:
        cfg = json.loads(str(z["config"]))
        if [round(v, 4) for v in cfg.get("speed_targets", want.speed_targets)] != [round(v, 4) for v in want.speed_targets] or \
                [round(v, 4) for v in cfg.get("lat_targets", want.lat_targets)] != [round(v, 4) for v in want.lat_targets]:
            raise ValueError(f"{path}: built for another action grid ({cfg.get('speed_targets')}, {cfg.get('lat_targets')})")
    q = z["q"].reshape(len(z["frame"]), -1).astype(np.float32)
    return {str(f): q[i] for i, f in enumerate(z["frame"]) if np.isfinite(q[i]).all()}


def check_q_label_coverage(samples: list, tag: Optional[str]) -> Optional[float]:
    """Share of the real-camera samples that carry a Q label; raises below WOR_Q_MIN_COVERAGE (default 0.5) when labels were asked for. A
    wrong tag or data path otherwise gives a loss of exactly 0, a head that stays at its zero init, and a run that looks converged."""
    import os
    if not tag:
        return None
    real = [s for s in samples if not s.get("is_recovery_augmented")]
    cov = sum(s.get("qsa") is not None for s in real) / max(len(real), 1)
    print(f"--> Q labels '{tag}': {cov:.1%} of {len(real)} real-camera samples labelled", flush=True)
    if cov < float(os.environ.get("WOR_Q_MIN_COVERAGE", "0.5")):
        raise RuntimeError(f"Q labels '{tag}' cover only {cov:.1%} of the samples (WOR_Q_MIN_COVERAGE): wrong --q_labels tag or data dir, "
                           f"or build_q_labels.py has not run on these routes")
    return cov


def label_route(frames: List[Tuple[dict, list]], cfg: QConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q labels for every frame of a route: (n_frames, n_speed, n_lat) float32 with NaN where a frame has too few logged successors, the mask,
    and the collision flags of each action's best sequence."""
    q = np.full((len(frames), len(cfg.speed_targets), len(cfg.lat_targets)), np.nan, dtype=np.float32)
    hit = np.zeros(q.shape, dtype=bool)
    ok = np.zeros(len(frames), dtype=bool)
    for k in range(len(frames)):
        scene = build_scene(frames, k, cfg)
        if scene is not None:
            q[k], hit[k] = solve_q(scene, cfg, return_hits=True)
            ok[k] = True
    return q, ok, hit
