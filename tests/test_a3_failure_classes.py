"""TODO A3 (scripts/analysis/a3_failure_classes.py): the per-run failure class, the paired transition table against a
reference arm, `--arm` for labels a9_merge.py does not know, and the A44 clip taxonomy (clips written by the real recorder)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
a3 = pytest.importorskip("a3_failure_classes")
a9_merge = pytest.importorskip("a9_merge")


def _rec(route, status="Completed", rc=100.0, pen=1.0, **infractions):
    inf = {k: [] for k in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle", "red_light", "stop_infraction",
                           "outside_route_lanes", "min_speed_infractions", "yield_emergency_vehicle_infractions",
                           "scenario_timeouts", "route_dev", "vehicle_blocked", "route_timeout")}
    inf.update({k: [f"{k} event"] * n for k, n in infractions.items()})
    return {"route_id": f"RouteScenario_{route}", "status": status, "infractions": inf,
            "scores": {"score_route": rc, "score_penalty": pen, "score_composed": rc * pen}}


def test_clean_run():
    assert a3.primary_class(a3.split_run(_rec(2050))) == "clean"


def test_one_vehicle_collision_on_a_finished_route():
    x = a3.split_run(_rec(26401, pen=0.6, collisions_vehicle=1))
    assert x["lost_pen"] == pytest.approx(40.0)
    assert a3.primary_class(x) == "vehicle_collision"


def test_tick_cap_is_its_own_class_and_beats_a_small_penalty():
    x = a3.split_run(_rec(2509, status="Failed - TickRuntime", rc=30.0, pen=0.7, red_light=1))
    assert x["reason"] == "time_cap"
    assert a3.primary_class(x) == "time_cap"          # 70 points of completion against 9 of penalty


def test_blocked_uses_the_infraction_not_the_status():
    x = a3.split_run(_rec(3457, status="Failed - Agent got blocked", rc=40.0, vehicle_blocked=1))
    assert a3.primary_class(x) == "blocked"


def test_penalty_points_are_shared_by_log_penalty():
    x = a3.split_run(_rec(1, pen=0.6 * 0.7, collisions_vehicle=1, red_light=1))
    share = x["pen"]["collisions_vehicle"] / x["lost_pen"]
    assert share == pytest.approx(np.log(1 / 0.6) / (np.log(1 / 0.6) + np.log(1 / 0.7)))
    assert a3.primary_class(x) == "vehicle_collision"


def test_route_class_is_the_most_frequent_run_class_ties_to_the_costlier():
    runs = [a3.split_run(r) for r in (_rec(5, pen=0.6, collisions_vehicle=1), _rec(5, status="Failed - TickRuntime", rc=20.0))]
    assert a3.route_class(runs) == "time_cap"       # one each: the 80-point loss beats the 40-point one
    runs.append(a3.split_run(_rec(5, pen=0.6, collisions_vehicle=1)))
    assert a3.route_class(runs) == "vehicle_collision"


def test_class_table_lost_points_sum_to_the_arm_loss():
    runs = {1: [a3.split_run(_rec(1))], 2: [a3.split_run(_rec(2, pen=0.6, collisions_vehicle=1))],
            3: [a3.split_run(_rec(3, status="Failed - TickRuntime", rc=50.0))]}
    tab = a3.class_table(runs)
    assert sum(r["lost"] for r in tab.values()) == pytest.approx((0 + 40 + 50) / 3)
    assert list(tab)[0] == "time_cap"


def test_transitions_rank_cells_by_points_moved():
    ref = {r: [a3.split_run(_rec(r))] for r in (1, 2, 3)}
    arm = {1: [a3.split_run(_rec(1, pen=0.6, collisions_vehicle=1))], 2: [a3.split_run(_rec(2, pen=0.6, collisions_vehicle=1))],
           3: [a3.split_run(_rec(3))], 4: [a3.split_run(_rec(4))]}
    tr = a3.transitions(arm, ref)
    assert tr[0]["ref"] == "clean" and tr[0]["arm"] == "vehicle_collision"
    assert tr[0]["n"] == 2 and tr[0]["sum_diff"] == pytest.approx(-80.0)
    assert sum(t["n"] for t in tr) == 3                 # route 4 has no reference run


def test_arm_option_matches_before_the_builtin_table():
    extra = a9_merge.parse_arm_specs(["A57LOW=^c19v4_220(?!.*epic)", "A57EPIC=^c19v4_220.*epic"])
    assert a9_merge.arm_of("c19v4_220_l3_x.json", extra) == "A57LOW"
    assert a9_merge.arm_of("c19v4_220_crash2_epicr_x.json", extra) == "A57EPIC"
    assert a9_merge.arm_of("a9_E15_c01_x.json", extra) == "E15"
    assert a9_merge.arm_of("c19v4_220_l3_x.json") is None
    with pytest.raises(SystemExit):
        a9_merge.parse_arm_specs(["no-equals-sign"])


def _clip(tmp_path, label, route, kind, event, pre_frames, t0=0.0):
    """A clip written by the real A44 recorder (CARLA-free)."""
    from src.eval.clip_recorder import ClipRecorder
    d = tmp_path / "clips" / label
    rec = ClipRecorder(str(d), f"RouteScenario_{route}_rep0_Town12_HardBreakRoute_1_h1", hz=4.0, before_s=2.0, after_s=0.5)
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    t = t0
    for f in pre_frames:                                 # one stored frame per 5 ticks at 4 Hz
        for _ in range(5):
            t = round(t + 0.05, 2)
            rec.push(t, rgb, f)
    rec.trigger(t, kind, event)
    for _ in range(20):
        t = round(t + 0.05, 2)
        rec.push(t, rgb, {"speed": 0.0, "throttle": 0.0, "brake": 1.0, "steer": 0.0, "target_kmh": 0.0, "probs": [0.97, 0.0], "waypoints": None})
    rec.close()
    return rec.written[0]


def _frame(speed, p_stop, tgt, brake=0.0):
    return {"speed": speed, "throttle": 0.5, "brake": brake, "steer": 0.0, "target_kmh": tgt, "probs": [p_stop, 0.1], "waypoints": None}


def test_rear_end_into_a_stopped_car_with_no_stop_request(tmp_path):
    ev = {"other_type": "vehicle.audi.a2", "other_id": 7, "ego_speed": 9.1, "other_speed": 0.0, "other_rel_fwd": 4.7,
          "other_rel_right": 1.1, "ego_xy": [0, 0, 90.0], "actors": [{"type": "vehicle.audi.a2", "id": 7, "xy": [0, 5, 90.0], "speed": 0.0}]}
    row = a3.clip_row(_clip(tmp_path, "e15_nC1_2000", 2509, "vehicle", ev, [_frame(9.0, 0.01, 36.0)] * 8))
    assert row["route"] == 2509 and row["geometry"] == "rear_end" and row["stationary_other"] is True
    assert row["asked_stop"] is False and row["stop_lead_s"] == 0.0     # the after-frames' stop request does not count


def test_late_stop_request_and_its_lead_time(tmp_path):
    ev = {"other_id": 7, "ego_speed": 6.0, "other_speed": 2.0, "other_rel_fwd": 5.0, "other_rel_right": 0.2, "ego_xy": [0, 0, 0.0], "actors": []}
    frames = [_frame(8.0, 0.02, 30.0)] * 6 + [_frame(7.0, 0.8, 0.0, brake=1.0)] * 2   # asked for the last two frames (0.25 s apart)
    row = a3.clip_row(_clip(tmp_path, "e15_nC1_2000", 2509, "vehicle", ev, frames))
    assert row["asked_stop"] is True and row["stop_lead_s"] == pytest.approx(0.25)
    assert row["max_brake_1s"] == 1.0


@pytest.mark.parametrize("right, other_yaw, other_speed, fwd, expect", [
    (3.5, 0.0, 8.0, 1.0, "side_same_direction"), (3.0, 90.0, 8.0, 2.0, "crossing"), (2.5, 180.0, 8.0, 3.0, "oncoming"),
    (0.3, 0.0, 12.0, -4.0, "hit_from_behind")])
def test_geometry_classes(right, other_yaw, other_speed, fwd, expect):
    ev = {"kind": "vehicle", "other_id": 7, "ego_speed": 6.0, "other_speed": other_speed, "other_rel_fwd": fwd, "other_rel_right": right,
          "ego_xy": [0, 0, 10.0], "actors": [{"id": 7, "xy": [0, 0, 10.0 + other_yaw], "speed": other_speed}]}
    assert a3.collision_geometry(ev) == expect


def test_clips_take_their_arm_from_the_job_label_folder(tmp_path):
    ev = {"other_id": 7, "ego_speed": 9.0, "other_speed": 0.0, "other_rel_fwd": 4.0, "other_rel_right": 0.5, "ego_xy": [0, 0, 0.0], "actors": []}
    _clip(tmp_path, "a9_E15_c01", 2509, "vehicle", ev, [_frame(9.0, 0.0, 36.0)] * 4)
    _clip(tmp_path, "mystery", 3457, "vehicle", ev, [_frame(9.0, 0.0, 36.0)] * 4)
    clips = a3.load_clips([str(tmp_path / "clips")])
    assert set(clips) == {"E15", "unassigned"}
    s = a3.clip_summary(clips["E15"])
    assert s["vehicle"] == 1 and s["geometry"] == {"rear_end": 100.0} and s["asked_stop_pct"] == 0.0


def test_pushing_a_parked_car_counts_one_impact(tmp_path):
    impact = {"other_id": 7, "ego_speed": 9.0, "other_speed": 0.0, "other_rel_fwd": 4.0, "other_rel_right": 0.5, "ego_xy": [0, 0, 0.0], "actors": []}
    pushing = dict(impact, ego_speed=0.0)
    _clip(tmp_path, "a9_E15_c01", 2509, "vehicle", impact, [_frame(9.0, 0.0, 36.0)] * 4)
    _clip(tmp_path, "a9_E15_c01", 2509, "vehicle", pushing, [_frame(0.0, 0.9, 0.0)] * 4, t0=20.0)   # same run, same car, 20 s later
    rows = a3.load_clips([str(tmp_path / "clips")])["E15"]
    assert sorted(r["first_contact"] for r in rows) == [False, True]
    s = a3.clip_summary(rows)
    assert s["vehicle"] == 1 and s["repeat_contacts"] == 1 and s["ego_speed_median"] == 9.0


def test_cli_end_to_end_on_result_files(tmp_path):
    out = tmp_path / "live_test"
    out.mkdir()
    ref = [_rec(2050), _rec(26401), _rec(2509, status="Failed - TickRuntime", rc=30.0)]
    arm = [_rec(2050), _rec(26401, pen=0.6, collisions_vehicle=1), _rec(2509, status="Failed - TickRuntime", rc=25.0)]
    for name, recs in (("a9_WOR_c01_x.json", ref), ("c19v4_220_l1_x.json", arm)):
        (out / name).write_text(json.dumps({"_checkpoint": {"progress": [3, 3], "records": recs}}))
    js = tmp_path / "a3.json"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "analysis" / "a3_failure_classes.py"), str(out),
                          "--arm", "A57=^c19v4_220", "--arms", "A57,WOR", "--json-out", str(js)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "clean -> A57 vehicle_collision" in " ".join(res.stdout.split())
    data = json.loads(js.read_text())
    assert data["route_classes"]["A57"] == {"2050": "clean", "26401": "vehicle_collision", "2509": "time_cap"}
    assert data["transitions"]["A57-WOR"][0]["sum_diff"] == pytest.approx(-40.0)
