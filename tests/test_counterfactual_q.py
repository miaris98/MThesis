"""TODO A71 (Gemini review 2026-11-01): counterfactual Q labels on PDM-Lite logs, no simulator.

Covers the geometry (oriented-box overlap, route frame, actor tracks moved into the current frame), the finite-horizon Bellman pass
(checked against brute force over every action sequence on a tiny lattice), and the driving behaviour the labels must encode.
"""
import itertools
import math

import numpy as np
import pytest

from src.training.counterfactual_q import (
    QConfig, RouteFrame, boxes_overlap, build_scene, drivable_lateral_mask, label_route, solve_q)

LANE = 3.5


def _ego_matrix(x=0.0, y=0.0, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _straight_route(n=20):
    return [[2.5 + i, 0.0] for i in range(n)]


def _frames(n, speed=8.0, target_speed=8.0, actors=None, ego_info=None, route_y=0.0):
    """n logged frames of an ego driving straight along +x at `speed`; `actors(t)` returns that frame's boxes in world coordinates (extents as
    PDM-Lite logs them, true [x, y] for statics too: data_agent.py fixes CARLA's swapped static extent before saving). `target_speed` may be a
    function of t. `route_y` puts route_original that far right of the ego."""
    out = []
    for t in range(n):
        x = speed * 0.25 * t
        ts = target_speed(t) if callable(target_speed) else target_speed
        meas = {"speed": speed, "target_speed": ts, "ego_matrix": _ego_matrix(x)[:3], "route_original": [[p[0], p[1] + route_y] for p in _straight_route()],
                "route": _straight_route(), "light_hazard": False, "stop_sign_hazard": False, "junction": False}
        boxes = [{"class": "ego_car", "extent": [2.45, 0.92, 0.75], "position": [0, 0, 0], "yaw": 0.0, "id": 1, "matrix": _ego_matrix(x)},
                 {"class": "ego_info", "num_lanes_same_direction": 2, "num_lanes_opposite_direction": 1, "ego_lane_number": 0, "is_in_junction": False,
                  **(ego_info or {})}]
        for b in (actors(t) if actors else []):
            wx, wy, wyaw = b.pop("world")
            rel = np.linalg.inv(np.array(_ego_matrix(x))) @ np.array(_ego_matrix(wx, wy, wyaw))
            b.update(position=[float(rel[0, 3]), float(rel[1, 3]), 0.0], yaw=float(math.atan2(rel[1, 0], rel[0, 0])))
            if "id" in b:
                b["matrix"] = _ego_matrix(wx, wy, wyaw)
            boxes.append(b)
        out.append((meas, boxes))
    return out


CFG = QConfig()


# ---------------------------------------------------------------- geometry

def test_box_overlap_axis_aligned_rotated_and_separated():
    a = np.array([0.0, 0.0, 0.0, 2.0, 1.0])           # cx, cy, yaw, half-length, half-width
    assert boxes_overlap(a, np.array([3.5, 0.0, 0.0, 2.0, 1.0]))          # 0.5 m overlap along x
    assert not boxes_overlap(a, np.array([4.5, 0.0, 0.0, 2.0, 1.0]))      # 0.5 m gap
    assert not boxes_overlap(a, np.array([0.0, 2.5, 0.0, 2.0, 1.0]))
    assert boxes_overlap(a, np.array([0.0, 2.5, math.pi / 2, 2.0, 1.0]))  # rotated 90 degrees: its 2 m half-length now reaches y = 0.5
    # vectorised: one ego cell against many actors
    many = np.array([[3.5, 0, 0, 2, 1], [4.5, 0, 0, 2, 1], [0, 2.5, math.pi / 2, 2, 1]])
    assert boxes_overlap(a[None], many).tolist() == [True, False, True]


def test_route_frame_places_offsets_on_the_correct_side_and_extends_past_the_logged_route():
    rf = RouteFrame(np.array(_straight_route()))
    x, y, yaw = rf.pose(np.array([0.0, 10.0, 60.0]), np.array([0.0, LANE, -LANE]))
    assert np.allclose(x, [0.0, 10.0, 60.0], atol=1e-6)        # s = 0 is the ego, 60 m extrapolates past the 22 m logged route
    assert np.allclose(y, [0.0, LANE, -LANE], atol=1e-6)        # +d is +y (CARLA's ego frame: y points right)
    assert np.allclose(yaw, 0.0, atol=1e-6)
    curved = RouteFrame(np.array([[2.5 + 10 * math.sin(i / 10), 10 - 10 * math.cos(i / 10)] for i in range(20)]))   # a left/right arc
    _, _, yaw_c = curved.pose(np.array([15.0]), np.array([0.0]))
    assert abs(float(yaw_c[0])) > 0.5


def test_actor_tracks_are_moved_into_the_current_ego_frame():
    """A parked car 30 m ahead in world coordinates stays 30 m ahead of frame 0 in every future step, although its per-frame ego-relative position shrinks."""
    fr = _frames(14, speed=8.0, actors=lambda t: [{"class": "car", "id": 7, "extent": [2.4, 1.0, 0.8], "speed": 0.0, "world": (30.0, 0.0, 0.0)}])
    scene = build_scene(fr, 0, CFG)
    assert scene is not None
    tracks = scene.actor_boxes                                   # (H+1, M, 5) in frame-0 coordinates, NaN where unknown
    assert tracks.shape[0] == CFG.horizon + 1
    assert np.allclose(tracks[:, 0, 0], 30.0, atol=1e-5) and np.allclose(tracks[:, 0, 1], 0.0, atol=1e-5)


def test_static_boxes_without_id_are_held_fixed_and_non_obstacles_are_ignored():
    def actors(t):
        return [{"class": "static_car", "extent": [2.0, 0.9, 0.8], "world": (25.0, 0.0, 0.0)},
                {"class": "traffic_light", "id": 99, "extent": [1.5, 1.5, 0.5], "world": (20.0, 0.0, 0.0)},
                {"class": "landmark", "id": 98, "world": (15.0, 0.0, 0.0)}]
    scene = build_scene(_frames(CFG.horizon + 2, actors=actors), 0, CFG)
    finite = scene.actor_boxes[~np.isnan(scene.actor_boxes[..., 0])]
    assert len(np.unique(np.round(finite[:, 0], 3))) == 1 and np.allclose(finite[:, 0], 25.0, atol=1e-5)   # only the parked car, at a fixed place
    assert np.allclose(finite[:, 3], 2.0) and np.allclose(finite[:, 4], 0.9)     # extents used as logged (data_agent.py already fixed CARLA's x/y swap)


def test_statics_that_appear_later_are_included_from_then_on():
    def actors(t):
        return [{"class": "static", "extent": [0.6, 0.6, 0.9], "world": (30.0, 0.0, 0.0)}] if t >= 3 else []
    scene = build_scene(_frames(CFG.horizon + 2, actors=actors), 0, CFG)
    col = scene.actor_boxes[:, 0, 0]
    assert np.isnan(col[:3]).all() and np.allclose(col[3:], 30.0, atol=1e-5)


def test_route_is_extended_with_later_frames_routes():
    """The logged route reaches ~22 m; at 20 m/s the lattice reaches 60 m, so later frames' route_original must extend it (curves!)."""
    scene = build_scene(_frames(CFG.horizon + 2, speed=8.0), 0, CFG)
    assert scene.route.cum[-1] > 40.0


def test_ego_off_the_route_starts_the_lattice_at_its_own_offset():
    """Review 2026-11-01: mid-bypass the ego drives one lane left of route_original; a parked car stands in the route_original lane. Keeping the
    left lane must not be labelled a collision, and returning into the parked car's lane at speed must be."""
    stopped = {"class": "static_car", "extent": [2.0, 0.9, 0.8], "world": (14.0, LANE, 0.0)}
    scene = build_scene(_frames(CFG.horizon + 2, speed=8.0, actors=lambda t: [dict(stopped)], route_y=LANE), 0, CFG)
    assert abs(scene.d0 + LANE) < 0.05                           # the ego sits one lane left (-y) of route_original
    q, hit = solve_q(scene, CFG, return_hits=True)
    keep_left, back_in = _idx(8, -LANE), _idx(16, 0)
    assert not hit[keep_left] and hit[back_in]
    assert q[keep_left] > q[back_in]


def test_same_lane_followers_do_not_make_braking_a_collision():
    follower = lambda t: [{"class": "car", "id": 9, "extent": [2.4, 1.0, 0.8], "speed": 8.0, "world": (-16.0 + 2.0 * t, 0.0, 0.0)}]
    q_f, hit_f = solve_q(build_scene(_frames(CFG.horizon + 2, speed=8.0, actors=follower), 0, CFG), CFG, return_hits=True)
    q_0 = solve_q(build_scene(_frames(CFG.horizon + 2, speed=8.0), 0, CFG), CFG)
    assert not hit_f[_idx(0, 0)] and abs(q_f[_idx(0, 0)] - q_0[_idx(0, 0)]) < 1e-5


def test_late_appearing_same_lane_follower_is_ignored():
    """A follower that enters sensor range at t > 0 behind the ego in the same lane must also be excluded."""
    follower = lambda t: [{"class": "car", "id": 19, "extent": [2.4, 1.0, 0.8], "speed": 8.0, "world": (-14.0 + 2.0 * t, 0.0, 0.0)}] if t >= 2 else []
    _, hit = solve_q(build_scene(_frames(CFG.horizon + 2, speed=0.0, target_speed=0.0, actors=follower), 0, CFG), CFG, return_hits=True)
    assert not hit[_idx(0, 0)]


def test_stationary_ego_does_not_crabwalk_into_adjacent_lane():
    """A stopped car (v=0, target_speed=0) cannot slide sideways into an oncoming car in the adjacent lane."""
    oncoming = lambda t: [{"class": "car", "id": 21, "extent": [2.4, 1.0, 0.8], "speed": 8.0, "world": (12.0 - 2.0 * t, -2.6, math.pi)}]
    scene = build_scene(_frames(CFG.horizon + 2, speed=0.0, target_speed=0.0, actors=oncoming), 0, CFG)
    assert scene.rollout_return([(_idx(0, -1.75)) for _ in range(CFG.horizon)], CFG)[1] is False


def test_swept_check_does_not_collide_at_start_station_with_end_of_step_lateral_offset():
    """Merging back toward d=0 while moving forward past a parked car's front bumper must not check s=0 at d2."""
    parked = {"class": "static_car", "extent": [2.4, 1.0, 0.8], "world": (-4.7, 0.9, 0.0)}
    scene = build_scene(_frames(CFG.horizon + 2, speed=10.0, target_speed=10.0, actors=lambda t: [dict(parked)], route_y=1.3125), 0, CFG)
    _, hit = solve_q(scene, CFG, return_hits=True)
    assert not hit[_idx(10, 0)]


def test_too_few_future_frames_gives_no_label():
    fr = _frames(CFG.horizon)          # frame 0 has only horizon - 1 successors
    assert build_scene(fr, 0, CFG) is None


def test_drivable_lateral_mask_uses_lane_counts():
    """ego_lane_number counts from the leftmost same-direction lane; opposite lanes lie further left (-y)."""
    d = np.array([-2 * LANE, -LANE, 0.0, LANE, 2 * LANE])
    m = drivable_lateral_mask(d, {"num_lanes_same_direction": 2, "num_lanes_opposite_direction": 1, "ego_lane_number": 0, "is_in_junction": False})
    assert m.tolist() == [False, True, True, True, False]      # one oncoming lane left, one same-direction lane right
    m = drivable_lateral_mask(d, {"num_lanes_same_direction": 1, "num_lanes_opposite_direction": 0, "ego_lane_number": 0, "is_in_junction": False})
    assert m.tolist() == [False, False, True, False, False]
    m = drivable_lateral_mask(d, None)                           # no lane information: only the own lane is trusted
    assert m.tolist() == [False, False, True, False, False]


# ---------------------------------------------------------------- Bellman pass

def _tiny(hold):
    return QConfig(horizon=4, hold=hold, speed_targets=(0.0, 4.0, 10.0), lat_targets=(-LANE, 0.0, LANE), v_max=10.0, v_res=2.0, a_acc=8.0, a_dec=8.0,
                   s_res=1.0, lat_rate=LANE / 0.25 / 2)


def _brute_force_q(scene, cfg):
    """Every action sequence enumerated explicitly: the lateral target of a0 is kept for the whole horizon, its speed target held for `hold`
    steps, then any speed continuation; Q(a0) = the best continuation's return, plus whether that best sequence collides."""
    best = np.full((len(cfg.speed_targets), len(cfg.lat_targets)), -np.inf)
    hits = np.zeros(best.shape, dtype=bool)
    for av, ad in itertools.product(range(len(cfg.speed_targets)), range(len(cfg.lat_targets))):
        for cont in itertools.product(range(len(cfg.speed_targets)), repeat=cfg.horizon - cfg.hold):
            ret, hit = scene.rollout_return([(av, ad)] * cfg.hold + [(c, ad) for c in cont], cfg)
            if ret > best[av, ad] + 1e-9:
                best[av, ad], hits[av, ad] = ret, hit
    return best, hits


@pytest.mark.parametrize("hold", [1, 2])
@pytest.mark.parametrize("scene_kind", ["empty", "static_near", "oncoming"])
def test_dp_equals_brute_force_on_a_tiny_lattice(scene_kind, hold):
    """Moving actor, a target speed that changes per frame (over-speed fires), a speed cap above the target and a one-lane road (off-road cells):
    each term of the reward and the time indexing of the collision grid are exercised (review 2026-11-01: the first version passed mutated code)."""
    cfg = _tiny(hold)
    actors = {"empty": None,
              "static_near": lambda t: [{"class": "car", "id": 5, "extent": [2.0, 0.9, 0.8], "speed": 0.0, "world": (7.0, 0.0, 0.0)}],
              "oncoming": lambda t: [{"class": "car", "id": 6, "extent": [2.0, 0.9, 0.8], "speed": 8.0, "world": (16.0 - 2.0 * t, 0.0, math.pi)}]}[scene_kind]
    scene = build_scene(_frames(cfg.horizon + 2, speed=4.0, target_speed=lambda t: (8.0, 6.0, 2.0, 0.0, 4.0, 4.0)[t], actors=actors,
                                ego_info={"num_lanes_same_direction": 1, "num_lanes_opposite_direction": 0}), 0, cfg)
    q_dp, hit_dp = solve_q(scene, cfg, return_hits=True)
    q_bf, hit_bf = _brute_force_q(scene, cfg)
    assert np.allclose(q_dp, q_bf, atol=1e-4), f"DP\n{q_dp}\nbrute force\n{q_bf}"
    assert (hit_dp == hit_bf).all()
    if scene_kind != "empty":
        assert q_dp.max() > q_dp.min() + 1.0, "a close or oncoming car must separate the actions"


@pytest.mark.parametrize("cone_x", [12.4, 13.1, 13.9, 14.6, 15.3])
def test_thin_obstacles_are_not_tunnelled_at_high_speed(cone_x):
    """Gemini review 2026-11-01: at 20 m/s the ego moves 5 m per step; a 0.3 m cone between the sampled stations must still be hit."""
    cone = lambda t: [{"class": "static", "extent": [0.15, 0.15, 0.5], "world": (cone_x, 0.0, 0.0)}]
    scene = build_scene(_frames(CFG.horizon + 2, speed=20.0, target_speed=20.0, actors=cone), 0, CFG)
    _, hit = solve_q(scene, CFG, return_hits=True)
    assert hit[_idx(20, 0)], f"cone at {cone_x} m tunnelled at 20 m/s"


def test_lateral_targets_stay_distinct_under_the_hold():
    """Review 2026-11-01: with the lateral target released after the hold, the +-1.75 and +-3.5 columns came out identical."""
    q = _q(actors=lambda t: [{"class": "car", "id": 3, "extent": [2.4, 1.0, 0.8], "speed": 0.0, "world": (30.0, 1.75, 0.0)}], speed=8.0)
    assert not np.allclose(q[:, _idx(0, 1.75)[1]], q[:, _idx(0, LANE)[1]])


# ---------------------------------------------------------------- behaviour

def _q(actors=None, speed=8.0, target_speed=8.0, ego_info=None, cfg=CFG):
    scene = build_scene(_frames(cfg.horizon + 2, speed=speed, target_speed=target_speed, actors=actors, ego_info=ego_info), 0, cfg)
    return solve_q(scene, cfg)


def _idx(v, d, cfg=CFG):
    return int(np.argmin(np.abs(np.array(cfg.speed_targets) - v))), int(np.argmin(np.abs(np.array(cfg.lat_targets) - d)))


def test_empty_road_prefers_the_expert_speed_in_lane():
    q = _q(speed=8.0, target_speed=8.0)
    v_best, d_best = np.unravel_index(np.argmax(q), q.shape)
    assert d_best == _idx(0, 0)[1]
    assert abs(CFG.speed_targets[v_best] - 8.0) <= 2.0
    assert q[_idx(8, 0)] > q[_idx(0, 0)]              # driving beats standing still
    assert q[_idx(8, 0)] > q[_idx(20, 0)]             # and speeding far over the expert's target is penalised


def test_stopped_car_ahead_makes_full_speed_in_lane_the_worst_choice():
    actors = lambda t: [{"class": "car", "id": 3, "extent": [2.4, 1.0, 0.8], "speed": 0.0, "world": (20.0, 0.0, 0.0)}]
    q = _q(actors=actors, speed=10.0, target_speed=10.0)
    assert q[_idx(20, 0)] < q[_idx(0, 0)]              # braking beats driving into it
    assert q[_idx(20, 0)] < CFG.collision_penalty * 0.5
    assert np.argmax(q) != np.ravel_multi_index(_idx(20, 0), q.shape)


def test_lateral_bypass_is_used_only_into_a_drivable_free_lane():
    stopped = {"class": "car", "id": 3, "extent": [2.4, 1.0, 0.8], "speed": 0.0, "world": (18.0, 0.0, 0.0)}
    q = _q(actors=lambda t: [dict(stopped)], speed=8.0, target_speed=8.0)
    right_free = q[:, _idx(0, LANE)[1]].max()
    assert right_free > q[:, _idx(0, 0)[1]].max()      # passing in the free same-direction lane on the right beats stopping behind it
    # the same scene with the right lane occupied by a moving car alongside: the bypass loses its value
    blocker = lambda t: [dict(stopped), {"class": "car", "id": 4, "extent": [2.4, 1.0, 0.8], "speed": 8.0, "world": (2.0 * t + 4.0, LANE, 0.0)}]
    q_blocked = _q(actors=blocker, speed=8.0, target_speed=8.0)
    assert q_blocked[:, _idx(0, LANE)[1]].max() < right_free


def test_expert_target_speed_zero_makes_stopping_best():
    """A red light or stop sign shows up only as the expert's target speed dropping to 0; the over-speed term must carry it."""
    q = _q(speed=4.0, target_speed=0.0)
    v_best, _ = np.unravel_index(np.argmax(q), q.shape)
    assert CFG.speed_targets[v_best] == 0.0


def test_label_route_returns_one_row_per_frame_with_nan_at_the_tail():
    fr = _frames(CFG.horizon + 6)
    q, ok, hit = label_route(fr, CFG)
    assert q.shape == (len(fr), len(CFG.speed_targets), len(CFG.lat_targets)) and hit.shape == q.shape
    assert ok[:6].all() and not ok[-CFG.horizon:].any()
    assert np.isnan(q[~ok]).all() and np.isfinite(q[ok]).all()


def test_config_rejects_a_hold_longer_than_the_horizon():
    with pytest.raises(ValueError):
        QConfig(horizon=4, hold=5)
    with pytest.raises(ValueError):
        QConfig(hold=0)
