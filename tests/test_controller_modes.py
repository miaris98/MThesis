"""2026-11-20 controller study (WOR_CONTROLLER): the original PID stays bit-identical, the PDM-Lite / TF++ variants match the
carla_garage source they were ported from (read from Carla-utils/carla_garage when it is present), and every mode keeps the
control ranges."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.models.world_on_rails.pid_controller import (CONTROLLERS, PDM_LAT, PDM_LONG_PARAMS, PIDController, controller_mode,
                                                       pdm_lookahead_m, pdm_throttle_brake, point_along, tfpp_brake_rule)

ROOT = Path(__file__).resolve().parents[1]
TEAM = ROOT / "Carla-utils" / "carla_garage" / "team_code"


class _OriginalPID:
    """The controller as it was before 2026-11-20 (pid_controller.py at 50fe410), for the bit-identity check."""

    def __init__(self):
        self.kp_steer, self.ki_steer, self.kd_steer = 0.75, 0.05, 0.15
        self.kp_speed, self.ki_speed, self.kd_speed = 1.0, 0.05, 0.10
        self.si = self.sp = self.vi = self.vp = 0.0

    def step(self, wps, v, tgt):
        a = wps[min(2, len(wps) - 1)]
        e = math.atan2(float(a[1]), max(float(a[0]), 0.5))
        self.si = np.clip(self.si + e, -1.0, 1.0)
        d = e - self.sp
        self.sp = e
        steer = float(np.clip(self.kp_steer * e + self.ki_steer * self.si + self.kd_steer * d, -1.0, 1.0))
        se = (tgt - v) / 3.6
        self.vi = np.clip(self.vi + se, -10.0, 10.0)
        sd = se - self.vp
        self.vp = se
        acc = self.kp_speed * se + self.ki_speed * self.vi + self.kd_speed * sd
        return (steer, float(np.clip(acc, 0.0, 1.0)), 0.0) if acc >= 0 else (steer, 0.0, float(np.clip(-acc, 0.0, 1.0)))


def _episode(seed, n=120):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        v = float(rng.uniform(0, 60))
        wps = np.cumsum(rng.normal([v / 3.6 * 0.25, 0.0], [0.3, 0.4], size=(5, 2)), axis=0)
        yield wps, v, float(rng.choice([0.0, 10.0, 30.0, 50.0, rng.uniform(0, 70)]))


def test_default_mode_is_pid_and_bit_identical_to_the_original(monkeypatch):
    monkeypatch.delenv("WOR_CONTROLLER", raising=False)
    for seed in range(5):
        new, old = PIDController(), _OriginalPID()
        assert new.mode == "pid"
        for wps, v, tgt in _episode(seed):
            assert new.control_from_waypoints(wps, v, tgt) == old.step(wps, v, tgt)


def test_mode_comes_from_the_environment_and_rejects_typos(monkeypatch):
    monkeypatch.setenv("WOR_CONTROLLER", "PDM ")
    assert PIDController().mode == "pdm"
    monkeypatch.setenv("WOR_CONTROLLER", "tfpp")
    with pytest.raises(ValueError):
        PIDController()
    assert controller_mode("pdm_lat") == "pdm_lat"


@pytest.mark.parametrize("mode", CONTROLLERS)
def test_every_mode_stays_in_the_control_ranges(mode):
    c = PIDController(mode=mode)
    for wps, v, tgt in _episode(7, 300):
        s, t, b = c.control_from_waypoints(wps, v, tgt)
        assert -1.0 <= s <= 1.0 and 0.0 <= t <= 1.0 and 0.0 <= b <= 1.0
        assert not (t > 0 and b > 0)


def test_mode_logs_one_line(capsys):
    c = PIDController(mode="pdm_long")
    for wps, v, tgt in _episode(1, 5):
        c.control_from_waypoints(wps, v, tgt)
    assert capsys.readouterr().out.count("[controller] WOR_CONTROLLER=pdm_long") == 1


def test_lookahead_is_the_experts():
    assert pdm_lookahead_m(0.0) == pytest.approx(2.4)
    assert pdm_lookahead_m(30.0) == pytest.approx((0.9755321901954155 * 30 + 1.9152884533402488) / 10)
    assert pdm_lookahead_m(200.0) == pytest.approx(10.5)


def test_point_along_interpolates_and_falls_back_to_the_last_waypoint():
    wps = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(point_along(wps, 1.5), [1.5, 0.0])
    np.testing.assert_allclose(point_along(wps, 2.5), [2.0, 0.5])    # around the corner
    np.testing.assert_allclose(point_along(wps, 4.5), [2.0, 2.5])    # across the zero-length segment
    np.testing.assert_allclose(point_along(wps, 50.0), [2.0, 3.0])
    np.testing.assert_allclose(point_along(np.zeros((5, 2)), 2.4), [0.0, 0.0])


def test_stopped_car_holds_the_wheel_straight():
    c = PIDController(mode="pdm_lat")
    s, _, _ = c.control_from_waypoints(np.array([[0.01, -0.02]] * 5), 0.0, 0.0)
    assert s == 0.0


def test_brake_mode_is_the_tfpp_rule():
    c, ref = PIDController(mode="brake"), PIDController(mode="pid")
    for wps, v, tgt in _episode(3, 200):
        s, t, b = c.control_from_waypoints(wps, v, tgt)
        s0, t0, b0 = ref.control_from_waypoints(wps, v, tgt)
        assert s == s0
        assert (t, b) == tfpp_brake_rule(t0, b0, v / 3.6, tgt)


def test_brake_lat_is_pdm_lat_steering_with_the_brake_rule():
    c, lat, brk = PIDController(mode="brake_lat"), PIDController(mode="pdm_lat"), PIDController(mode="brake")
    for wps, v, tgt in _episode(4, 200):
        s, t, b = c.control_from_waypoints(wps, v, tgt)
        assert s == lat.control_from_waypoints(wps, v, tgt)[0]
        assert (t, b) == brk.control_from_waypoints(wps, v, tgt)[1:]


def test_reset_clears_the_lateral_window():
    c = PIDController(mode="pdm")
    for wps, v, tgt in _episode(2, 10):
        c.control_from_waypoints(wps, v, tgt)
    c.reset()
    assert c._lat_window == []


def _load_get_throttle():
    """nav_planner.get_throttle without nav_planner's carla import."""
    src = (TEAM / "nav_planner.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "get_throttle")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "nav_planner.py", "exec"), ns)
    return ns["get_throttle"]


@pytest.mark.skipif(not (TEAM / "nav_planner.py").exists(), reason="carla_garage not checked out")
def test_longitudinal_matches_tfpp_direct_mode():
    get_throttle = _load_get_throttle()
    cfg = SimpleNamespace(longitudinal_params=PDM_LONG_PARAMS, longitudinal_max_acceleration=1.89)
    for tgt in np.linspace(0.0, 20.0, 41):
        for v in np.linspace(0.0, 22.0, 45):
            brake = tgt < 0.01 or (v / tgt) > 1.1 if tgt > 0 else True          # model.py:control_pid_direct
            th, cb = get_throttle(cfg, brake, tgt, v)
            mine = pdm_throttle_brake(tgt, v)
            assert mine == pytest.approx((float(np.clip(th, 0.0, 1.0)), 1.0 if cb else 0.0)), (tgt, v)


@pytest.mark.skipif(not (TEAM / "lateral_controller.py").exists(), reason="carla_garage not checked out")
@pytest.mark.parametrize("v_kmh", [0.0, 10.0, 15.0, 120.0])   # the lookahead saturates at 24 / 105 points: no 10 cm rounding
def test_lateral_matches_the_expert_on_a_10cm_route(v_kmh):
    spec = importlib.util.spec_from_file_location("pdm_lateral_controller", TEAM / "lateral_controller.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = SimpleNamespace(lateral_pid_kp=PDM_LAT["kp"], lateral_pid_kd=PDM_LAT["kd"], lateral_pid_ki=PDM_LAT["ki"],
                          lateral_pid_speed_scale=PDM_LAT["speed_scale"], lateral_pid_speed_offset=PDM_LAT["speed_offset"],
                          lateral_pid_default_lookahead=24, lateral_pid_speed_threshold=23.15, lateral_pid_window_size=6,
                          lateral_pid_minimum_lookahead_distance=24, lateral_pid_maximum_lookahead_distance=105, route_points=10)
    expert, mine = mod.LateralPIDController(cfg), PIDController(mode="pdm_lat")
    rng = np.random.default_rng(int(v_kmh))
    for _ in range(25):
        # a smooth curve 15 m long; the expert reads it as 10 cm route points starting at the ego position
        k = rng.uniform(-0.08, 0.08)
        s = np.linspace(0, 15, 151)
        curve = np.stack([np.sin(k * s) / k if abs(k) > 1e-6 else s, (1 - np.cos(k * s)) / k if abs(k) > 1e-6 else 0 * s], 1)
        wps = curve[[30, 60, 90, 120, 150]]
        dense = np.vstack([point_along(wps, d) for d in np.arange(0, 15.0, 0.1)])
        want = expert.step(dense, v_kmh / 3.6, np.zeros(2), 0.0)
        got, _, _ = mine.control_from_waypoints(wps, v_kmh, 30.0)
        assert got == pytest.approx(want, abs=1e-9)
