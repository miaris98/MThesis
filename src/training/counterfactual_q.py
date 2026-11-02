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
    over-speed -w_overspeed * max(0, v - (limit at that frame + overspeed_margin)) / v_max. The limit is the expert's target speed where the
               expert slowed for a red light, a stop sign or the road (curves): those are only in the logs as the target speed dropping, so this
               term carries them without a stop-line geometry PDM-Lite does not log. Where PDM-Lite says an actor caused the slow-down
               (speed_reduced_by_obj_type vehicle / walker / static) the limit is the road's speed_limit instead: the actor is in the scene and
               the collision term judges it, and the expert's own reaction to it (waiting behind a parked car) must not be what the labels
               reward (review 2026-11-01 evening: every moving action was "over speed" while the expert waited, so passing was worth 0).
    lateral    -w_lateral * |d| / lane width
    off-road   -w_offroad where d leaves the lanes the frame's `ego_info` reports around the ego's own lane
    collision  collision_penalty, terminal: the ego box (+ margin) overlaps a logged actor box anywhere along the step's swept stations (actors at
               mid-step) or at its end (actors at the step's end); no tunnelling at 5 m per step. Same-lane followers are not obstacles: a logged
               follower never reacts, so it would make every braking action a rear-end collision.

Terminal value at the horizon (state d, v, s): gamma^(N/2) * max over d' of (w_progress * progress / (v_max * dt) - n_back * lateral and
off-road cost at d'), N = tail_s / dt, where progress is what the next tail_s seconds at min(v, limit_H) allow: |d' - d| / lat_rate seconds
capped by the free gap at d (the lane change after the horizon), the rest by the free gap at d' (logged actors at the horizon step). Without
it, 3 s from standstill is worth about one Q unit of progress, less than staying put once the lateral cost is paid.

`Scene.rollout_return` evaluates one explicit action sequence with the same transitions; the brute-force test checks the dynamic programme
against it.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

TF_SPEEDS = (0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0)   # = aux_heads.TARGET_SPEEDS (kept torch-free here)
LANE_WIDTH = 3.5
_IGNORED = ("ego_car", "ego_info", "weather", "landmark")
FLAT_HALF_HEIGHT = 0.2                  # m: static boxes lower than this (half-extent z) are road debris, not obstacles


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
    tail_s: float = 4.0                     # terminal value: progress over the next tail_s seconds, capped by the free gap and the limit
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
    if cls.startswith("static") and len(b["extent"]) > 2 and float(b["extent"][2]) < FLAT_HALF_HEIGHT:
        return False                                    # flat props the expert drives over (ControlLoss dirt debris, 0.07 m)
    return cls.startswith("static") or cls in ("car", "walker", "bicycle", "motorcycle", "truck", "bus", "vehicle")


def _half_extents(b: dict) -> Tuple[float, float]:
    """(half length, half width) as logged. data_agent.py already swaps CARLA's buggy static extent ('BUG: static_extent x and y are swapped',
    line 1109) before saving, so no class is swapped here; swapping again turned parked cars sideways on real logs (2026-11-01 check)."""
    e = b["extent"]
    return float(e[0]), float(e[1])


CRUISE_FRACTION, CRUISE_MAX = 0.72, 20.0        # PDM-Lite cruises at min(0.72 x speed_limit, 20 m/s), not at the limit


def red_light_ahead(meas: dict, boxes=()) -> bool:
    """A red or yellow light that affects the ego is logged (PDM-Lite's IDM may log the queue's lead car, not the light, as the cause)."""
    return bool(meas.get("light_hazard")) or any(b.get("class") == "traffic_light" and b.get("affects_ego")
                                                and str(b.get("state", "")).lower() in ("red", "yellow") for b in boxes)


def speed_limit_of(meas: dict, boxes=(), standing=frozenset()) -> float:
    """The over-speed limit of one logged frame: the expert's target speed, except where PDM-Lite records that it slowed for a standing obstacle
    (a static prop, or a vehicle whose id is in `standing`: still for the whole horizon), with no red light, stop sign, walker or junction
    involved: then PDM-Lite's cruise speed, and the collision term judges the obstacle (review 2026-11-01 evening: the road limit for any actor
    cause passed red-light queues through the oncoming lane, pedestrians at 4 m/s and junction yields)."""
    ts = float(meas.get("target_speed", meas.get("speed", 0.0)))
    cause = str(meas.get("speed_reduced_by_obj_type") or "")
    if (meas.get("speed_limit") is None or meas.get("stop_sign_hazard") or meas.get("walker_hazard") or meas.get("junction")
            or red_light_ahead(meas, boxes) or any(b.get("class") == "ego_info" and b.get("is_in_junction") for b in boxes)):
        return ts
    if cause.startswith("static") or (cause.startswith("vehicle") and meas.get("speed_reduced_by_obj_id") in standing):
        return max(ts, min(CRUISE_FRACTION * float(meas["speed_limit"]), CRUISE_MAX))
    return ts


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- scene and lattice

@dataclass
class Scene:
    route: RouteFrame
    actor_boxes: np.ndarray                 # (H+1, M, 5) in the frame-0 ego frame; NaN where the actor is not logged at that step
    ego_half: Tuple[float, float]
    v0: float
    target_speed: np.ndarray                # (H+1,) the over-speed limit at each future frame (speed_limit_of: the expert's target speed or the road's)
    ego_info: Optional[dict]
    s0: float = 0.0                         # the ego's projection onto the route
    d0: float = 0.0
    mask_anchor: Optional[float] = None     # route offset of the lane `ego_info` describes (None: the ego's own, d0)

    def _ego_boxes(self, cfg: QConfig, d_grid: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
        S, D = np.meshgrid(self.s0 + s_grid, d_grid, indexing="ij")
        x, y, yaw = self.route.pose(S.ravel(), D.ravel())
        hl, hw = self.ego_half[0] + cfg.margin, self.ego_half[1] + cfg.margin
        return np.stack([x, y, yaw, np.full_like(x, hl), np.full_like(x, hw)], -1)

    @staticmethod
    def _overlap(ego: np.ndarray, act: np.ndarray) -> np.ndarray:
        """(len(ego),) bool: each ego box overlaps any finite actor box (bounding-circle pre-check, then the exact test on candidate pairs)."""
        act = act[np.isfinite(act[:, 0])]
        hit = np.zeros(len(ego), dtype=bool)
        if not len(act):
            return hit
        r_ego = math.hypot(ego[0, 3], ego[0, 4])
        near = np.hypot(ego[:, None, 0] - act[None, :, 0], ego[:, None, 1] - act[None, :, 1]) <= r_ego + np.hypot(act[:, 3], act[:, 4])[None]
        ci, ai = np.nonzero(near)
        if len(ci):
            hit[ci[boxes_overlap(ego[ci], act[ai])]] = True
        return hit

    def tail_grids(self, cfg: QConfig, d_grid: np.ndarray, s_grid: np.ndarray, n_steps: int) -> np.ndarray:
        """(n_steps, n_d, n_s) bool: the ego at (s0 + s, d) overlaps an actor at H + j * dt (j = 1..n_steps), the actors moved on at the constant
        velocity of their last logged step (standing where that step is unknown; gone where not logged at H)."""
        H = self.actor_boxes.shape[0] - 1
        a1, a0 = self.actor_boxes[H], self.actor_boxes[max(H - 1, 0)]
        vel = np.where(np.isfinite(a0[:, :2]), a1[:, :2] - a0[:, :2], 0.0)          # metres per step
        ego = self._ego_boxes(cfg, d_grid, s_grid)
        out = np.zeros((n_steps, len(d_grid), len(s_grid)), dtype=bool)
        for j in range(1, n_steps + 1):
            act = a1.copy()
            act[:, :2] = a1[:, :2] + j * vel
            out[j - 1] = self._overlap(ego, act).reshape(len(s_grid), len(d_grid)).T
        return out

    def collision_grids(self, cfg: QConfig, d_grid: np.ndarray, s_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(end, mid), each (H+1, n_d, n_s) bool: the ego at (s0 + s, d) overlaps an actor at step t's end / at the middle of step t-1 -> t."""
        ego = self._ego_boxes(cfg, d_grid, s_grid)
        shape = (self.actor_boxes.shape[0], len(d_grid), len(s_grid))
        end, mid = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
        for t in range(1, self.actor_boxes.shape[0]):
            a1, a0 = self.actor_boxes[t], self.actor_boxes[t - 1]
            both = np.isfinite(a0[:, 0]) & np.isfinite(a1[:, 0])
            am = np.where(both[:, None], a1, a1)
            am = np.where(both[:, None], np.c_[(a0[:, :2] + a1[:, :2]) / 2, a0[:, 2] + _wrap(a1[:, 2] - a0[:, 2]) / 2, a1[:, 3:]], am)
            for grid, act in ((end, a1), (mid, am)):
                grid[t] = self._overlap(ego, act).reshape(len(s_grid), len(d_grid)).T
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
        return ret + (disc * float(lat.tail[d, v, s]) if len(actions) == cfg.horizon else 0.0), False


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
        anchor = self.d_grid[self.d0] if scene.mask_anchor is None else scene.mask_anchor
        self.d_cost = (cfg.w_lateral * np.abs(self.d_grid) / LANE_WIDTH
                       + cfg.w_offroad * ~drivable_lateral_mask(self.d_grid - anchor, scene.ego_info))
        self.limit = np.asarray(scene.target_speed, dtype=np.float64) + cfg.overspeed_margin
        self.cfg, self.n_s = cfg, len(self.s_grid)
        # terminal value (nd, nv, ns): the ego drives on at min(v, limit_H) for tail_s seconds and stops before the first step at which it would
        # overlap an actor moved on at constant velocity (an oncoming car keeps coming: review 2026-11-01 evening). The continuation may change
        # lanes d -> d' too (else a car in lane 40 m behind an obstacle looks blocked for good and the labels send it into the oncoming lane at
        # once): it stays in d for the |d' - d| / lat_rate seconds of the change, then drives in d'. tail(d) = the best d'; collision_penalty where
        # an actor still reaches the stopped ego.
        H, n_tail = cfg.horizon, int(round(cfg.tail_s / cfg.dt))
        s_ext = np.arange(0.0, cfg.v_max * (cfg.horizon * cfg.dt + cfg.tail_s) + cfg.s_res + 1e-9, cfg.s_res)  # stations the tail can reach
        grids = scene.tail_grids(cfg, self.d_grid, s_ext, n_tail)                                                      # (n_tail, nd, n_ext)
        v_eff = np.minimum(self.v_grid, max(float(scene.target_speed[H]), 0.0))                                        # (nv,)
        tau = cfg.dt * np.arange(1, n_tail + 1)
        at = [grids[j][:, np.minimum(np.arange(self.n_s)[None, :] + np.floor(v_eff[:, None] * tau[j] / cfg.s_res + 0.5).astype(int),
                                     len(s_ext) - 1)] for j in range(n_tail)]                                         # n_tail x (nd, nv, ns)
        n_back = int(round(LANE_WIDTH / cfg.lat_rate / cfg.dt))                                                       # steps to drive back to the lane
        best = np.full((len(self.d_grid), len(self.v_grid), self.n_s), -np.inf)
        for jd in range(len(self.d_grid)):
            t_sw = np.abs(self.d_grid - self.d_grid[jd]) / cfg.lat_rate                                                # (nd,)
            first = np.full(best.shape, n_tail + 1)
            for j in range(n_tail):
                h = np.where((tau[j] <= t_sw)[:, None, None], at[j], at[j][jd][None])
                first = np.where((first > n_tail) & h, j + 1, first)
            prog = v_eff[None, :, None] * cfg.dt * (first - 1)
            # stopping where it stands does not help against an actor coming at it (an oncoming car): if one still reaches the stopped ego within
            # tail_s, the tail is a collision
            t_stop = cfg.dt * (first - 1)
            lane = np.where(t_stop <= t_sw[:, None, None], np.arange(len(self.d_grid))[:, None, None], jd)
            st = np.minimum(np.arange(self.n_s)[None, None, :] + np.floor(prog / cfg.s_res + 0.5).astype(int), len(s_ext) - 1)
            crash = np.zeros(best.shape, dtype=bool)
            for j in range(n_tail):
                crash |= (j + 1 >= first) & grids[j][lane, st]
            val = np.where(crash, cfg.collision_penalty, cfg.w_progress * prog / (cfg.v_max * cfg.dt) - n_back * self.d_cost[jd])
            best = np.maximum(best, val)
        self.tail = cfg.gamma ** (n_tail / 2) * best

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
        value, flag = lat.tail.reshape(nd, nv * ns).copy(), np.zeros((nd, nv * ns), dtype=bool)
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
            if not hit:
                ret, hit = ret + disc * value[d, v, s], bool(flag[d, v, s])        # value = the terminal value when hold == horizon
            q[av, ad], hits[av, ad] = ret, hit
    return (q.astype(np.float32), hits) if return_hits else q.astype(np.float32)


# moved to their own modules (2026-11-02: the file had grown past the 500-line limit); re-exported so existing imports keep working
from src.training.q_scene import _extended_route, _route_lane_info, _route_points, build_scene, logged_step_reward  # noqa: E402,F401
from src.training.q_label_io import (_open_labels, check_q_label_coverage, label_route, load_q_labels,  # noqa: E402,F401
                                     load_q_transitions)
