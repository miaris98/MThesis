#!/usr/bin/env python3
"""Bench2Drive adapter for this project's World-on-Rails policies.

WHY THIS EXISTS
---------------
Every closed-loop number this project has produced is scored by its own harness
(`eval_wor_closed_loop.py` + `src/eval/driving_metrics.py`), on its own routes. That makes
the internal comparison rigorous (13.31) but leaves it incomparable to any published method.
The CARLA Leaderboard 1.0 route that would have fixed that is a dead end: the scoring server
is retired and its pinned simulator, CARLA 0.9.10.1, is no longer downloadable anywhere
official (see TODO_leaderboard_benchmark.md).

Bench2Drive (NeurIPS 2024 D&B) runs on **CARLA 0.9.15 - this project's exact version** - and
publishes a leaderboard including TCP, UniAD, VAD and DriveTransformer. Its harness computes
the score, so nothing here reimplements a metric: this file only adapts our policy to its
agent interface, and Bench2Drive does the measuring.

Deploy by copying/symlinking into `<Bench2Drive>/leaderboard/team_code/` and pointing
TEAM_AGENT at it. TEAM_CONFIG carries the checkpoint as `<arch>:<path/to/best_model.pth>`,
e.g. `qwen30m:/workspace/checkpoints/wor_cl_arms/qwen30m_geom_s0/best_model.pth`.
"""
import os
import sys

import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import carla

# This project's root, so `src.agents` and `scripts.eval.eval_wor` import. The harness runs
# with its own cwd inside the Bench2Drive checkout, so relying on cwd would not work, and
# this file is deployed by copy/symlink so `__file__` does not point into the repo either.
MTHESIS_ROOT = os.environ.get("MTHESIS_ROOT", "/workspace/MThesis")
if MTHESIS_ROOT not in sys.path:
    sys.path.insert(0, MTHESIS_ROOT)

from src.agents.wor_agent import WorldOnRailsAgent  # noqa: E402

# Imported rather than re-derived. `_ego_frame_route` is the exact ego-frame projection the
# policy was trained against (`_index_pdm_lite_route`'s ego_matrix convention) and is already
# the one every other evaluation path in this repo uses - a second copy here would be free to
# drift out of agreement with the dataset without anything failing loudly.
from scripts.eval.eval_wor import (  # noqa: E402
    _ego_frame_route,
    WOR_LANEFOLLOW_COMMAND,
)

# `_ego_frame_route` takes a fixed *number* of upcoming points (WOR_ROUTE_LOOKAHEAD = 20), so
# the lookahead *distance* it represents is set entirely by the spacing of the route list it
# is handed. This used to be 2.0 m (~40 m of lookahead) on the belief that it matched
# training - it did not: PDM-Lite's own `route` field measures at ~1.0 m spacing (20 points,
# ~19 m lookahead), checked across 8 routes spanning 6 towns. Bench2Drive's plan has its own
# spacing regardless, so it is resampled to this value before use - otherwise the policy would
# silently receive a route on a different scale than the one it was trained on, the same class
# of mismatch the camera-parity fix (11.4) addressed for the pixels.
ROUTE_SPACING_M = 1.0


def get_entry_point():
    return "WorB2DAgent"


def _resample_locations(locations, spacing=ROUTE_SPACING_M):
    """Resamples a polyline of carla.Location to ~uniform arc-length spacing."""
    if len(locations) < 2:
        return list(locations)

    out = [locations[0]]
    carry = 0.0  # distance already accumulated toward the next sample
    for a, b in zip(locations[:-1], locations[1:]):
        seg = a.distance(b)
        if seg <= 1e-9:
            continue
        d = spacing - carry
        while d <= seg:
            r = d / seg
            out.append(carla.Location(x=a.x + (b.x - a.x) * r,
                                      y=a.y + (b.y - a.y) * r,
                                      z=a.z + (b.z - a.z) * r))
            d += spacing
        carry = seg - (d - spacing)
    return out


class WorB2DAgent(AutonomousAgent):
    """Wraps `WorldOnRailsAgent` so Bench2Drive can score it.

    `WorldOnRailsAgent` is already close to Leaderboard-shaped - it declares `sensors()` and
    returns a `carla.VehicleControl` from `run_step` - so this class exists for three
    specific mismatches, each of which degrades driving silently rather than raising:

      1. Colour order. The Leaderboard delivers BGRA; the policy trained on RGB.
         `WorldOnRailsAgent.run_step` only drops the alpha channel, which leaves BGR. The
         same mistake produced degenerate steering once already (challenges 11.6), so the
         swap is done here explicitly.
      2. Command. `run_step` defaults to `2` (= STRAIGHT) when no command is supplied, but
         every frame of the training data is LANEFOLLOW (= 3). Passed explicitly.
      3. Route. The policy's steering comes almost entirely from the `route` input - the
         command embedding is constant across the whole training set, so it carries no
         turn information. `_ego_frame_route` needs a dense list of `carla.Location` and a
         monotonically advancing index, neither of which the Leaderboard supplies, so both
         are built and maintained here.
    """

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS

        # The evaluator appends '+<save_name>' to `args.agent_config` IN PLACE
        # (leaderboard_evaluator.py: `args.agent_config = args.agent_config + '+' + save_name`)
        # and that line runs once per route without ever resetting args.agent_config back to
        # the original string - so by route 2 the value passed here is already
        # "<arch>:<ckpt>+<route1_save_name>", and rsplit("+", 1) (stripping only the LAST
        # suffix) leaves "<arch>:<ckpt>+<route1_save_name>" behind, which is not a real file.
        # os.path.exists() on that then fails, load_wor_model() falls back to a fresh
        # ImageNet-backbone/random-head model, and every route past the first is silently
        # evaluated on an untrained policy - discovered when a real eval run showed exactly
        # one "Merged N frozen backbone tensors" line followed by "ImageNet Pretrained: True"
        # on every subsequent route setup. Splitting on the FIRST '+' instead is correct
        # regardless of how many suffixes have accumulated, since our own config string is
        # built here and never legitimately contains a '+'.
        cfg = path_to_conf_file.split("+", 1)[0]
        arch, _, checkpoint = cfg.partition(":")
        if not checkpoint:
            raise ValueError(
                f"TEAM_CONFIG must be '<arch>:<checkpoint.pth>', got {path_to_conf_file!r}")

        self._inner = WorldOnRailsAgent(
            checkpoint_path=checkpoint,
            policy_arch=arch,
            backbone_name=os.environ.get("WOR_BACKBONE", "resnet34"),
        )
        self._route_idx = 0
        self._route_locations = None  # built on first step, once the map is certainly loaded

    def sensors(self):
        return self._inner.sensors()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        # The base class downsamples the world-coordinate plan (`downsample_route(..., 50)`)
        # for the benefit of agents that want a sparse plan. We want the dense one - it is
        # already interpolated along the actual driven path - so keep a reference to it
        # before that happens.
        self._dense_plan_world_coord = global_plan_world_coord
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def _build_route_locations(self):
        plan = getattr(self, "_dense_plan_world_coord", None) or self._global_plan_world_coord
        locations = []
        for entry in plan:
            transform = entry[0] if isinstance(entry, (tuple, list)) else entry
            locations.append(transform.location
                             if hasattr(transform, "location") else transform)
        return _resample_locations(locations)

    def _get_hero(self):
        """The ego actor, looked up in a way that works on both evaluators.

        `hero_actor` / `get_hero()` are additions Bench2Drive made to its fork of
        AutonomousAgent; the vanilla Leaderboard base class this project also runs against
        (run_leaderboard_official.sh) has neither, so touching `self.hero_actor` there raises
        AttributeError on the first step of every route - the agent crashes, and the route is
        still scored, as a 0% completion rather than as an error. CarlaDataProvider is the
        scenario_runner API underneath both forks and is populated by the time run_step is
        first called, so prefer it and keep the fork's accessor only as a fallback.
        """
        hero = CarlaDataProvider.get_hero_actor()
        if hero is None and hasattr(self, "get_hero"):
            self.get_hero()
            hero = getattr(self, "hero_actor", None)
        if hero is None:
            raise RuntimeError(
                "No hero actor: CarlaDataProvider.get_hero_actor() returned None and no "
                "fork-specific accessor was available. The ego is not spawned/registered.")
        return hero

    def run_step(self, input_data, timestamp):
        if self._route_locations is None:
            self._route_locations = self._build_route_locations()
        hero = self._get_hero()

        rgb = input_data["rgb_front"][1]
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, [2, 1, 0]]  # BGRA -> RGB, not a bare alpha drop

        # self._inner.route_points: adopted by WorldOnRailsAgent from the checkpoint's
        # own run_config.json - see _ego_frame_route's docstring for why this must not
        # be the module's WOR_ROUTE_POINTS default.
        route, self._route_idx, _ = _ego_frame_route(
            hero, self._route_locations, self._route_idx,
            route_points=self._inner.route_points)

        control = self._inner.run_step({
            "rgb_front": rgb,
            "speed": input_data["speed"],  # (frame, {'speed': m/s}) - already the trained unit
            "command": WOR_LANEFOLLOW_COMMAND,
            "route": route,
        })
        if os.environ.get("WOR_CREEP") == "1":
            control = self._creep(input_data, control)
        if os.environ.get("B2D_TELEMETRY_DIR"):
            self._log_telemetry(hero, control, timestamp)
        return control

    def _creep(self, input_data, control):
        """TF++ stuck recovery (sensor_agent.py): after stuck_threshold=1100 ticks below 0.1 m/s,
        force throttle >= 0.4 without brake for creep_duration=20 ticks. TF++ guards the creep with
        a LiDAR safety box; this agent is RGB-only, so the creep here is unguarded."""
        spd = input_data["speed"][1]
        spd = float(spd.get("speed", 0.0)) if isinstance(spd, dict) else float(spd)
        self._stuck = getattr(self, "_stuck", 0) + 1 if spd < 0.1 else 0
        if self._stuck > 1100:
            self._force_move = 20
        if getattr(self, "_force_move", 0) > 0:
            control.throttle = max(0.4, control.throttle)
            control.brake = 0.0
            self._force_move -= 1
        return control

    # ---- optional per-tick telemetry (B2D_TELEMETRY_DIR); off by default ------------------------
    # Answers the two questions route results cannot: after a collision, is the car stopped or
    # crawling (and what speed does the policy ask for), and in a vehicle collision, who hit whom
    # (ego velocity vs the other actor's, and where the other actor was in the ego frame).
    def _log_telemetry(self, hero, control, timestamp):
        import csv, json, time
        if getattr(self, "_tele", None) is None:
            d = os.environ["B2D_TELEMETRY_DIR"]
            os.makedirs(d, exist_ok=True)
            stem = os.path.join(d, f"{time.strftime('%Y%m%d-%H%M%S')}_hero{hero.id}")
            self._tele_f = open(stem + "_ticks.csv", "w", newline="")
            self._tele = csv.writer(self._tele_f)
            self._tele.writerow(["t", "x", "y", "yaw", "speed_mps", "target_kmh", "throttle", "brake", "steer"])
            self._coll_f = open(stem + "_collisions.jsonl", "w")
            bp = CarlaDataProvider.get_world().get_blueprint_library().find("sensor.other.collision")
            self._coll_sensor = CarlaDataProvider.get_world().spawn_actor(bp, carla.Transform(), attach_to=hero)

            def on_collision(ev, hero=hero, f=self._coll_f):
                ht, hv, ov = hero.get_transform(), hero.get_velocity(), ev.other_actor.get_velocity()
                rel = ev.other_actor.get_location() - ht.location
                fwd, right = ht.get_forward_vector(), ht.get_right_vector()
                f.write(json.dumps({
                    "t": ev.timestamp, "other_type": ev.other_actor.type_id, "other_id": ev.other_actor.id,
                    "ego_speed": (hv.x ** 2 + hv.y ** 2) ** 0.5, "other_speed": (ov.x ** 2 + ov.y ** 2) ** 0.5,
                    # other actor in the ego frame: +fwd = ahead of us, +right = to our right
                    "other_rel_fwd": rel.x * fwd.x + rel.y * fwd.y, "other_rel_right": rel.x * right.x + rel.y * right.y,
                    # closing speeds along the ego->other line: >0 = that party moves toward the other
                    "ego_closing": (hv.x * rel.x + hv.y * rel.y) / max((rel.x ** 2 + rel.y ** 2) ** 0.5, 1e-3),
                    "other_closing": -(ov.x * rel.x + ov.y * rel.y) / max((rel.x ** 2 + rel.y ** 2) ** 0.5, 1e-3),
                    "impulse": [ev.normal_impulse.x, ev.normal_impulse.y, ev.normal_impulse.z],
                }) + "\n")
                f.flush()
            self._coll_sensor.listen(on_collision)
        tr, v = hero.get_transform(), hero.get_velocity()
        tgt = getattr(getattr(getattr(self._inner, "net", None), "controller", None), "target_speed", None)
        self._tele.writerow([f"{timestamp:.2f}", f"{tr.location.x:.2f}", f"{tr.location.y:.2f}", f"{tr.rotation.yaw:.1f}",
                             f"{(v.x ** 2 + v.y ** 2) ** 0.5:.2f}", "" if tgt is None else f"{float(tgt):.1f}",
                             f"{control.throttle:.2f}", f"{control.brake:.2f}", f"{control.steer:.3f}"])

    def destroy(self):
        if getattr(self, "_coll_sensor", None) is not None:
            try:
                self._coll_sensor.stop(); self._coll_sensor.destroy()
            except Exception:
                pass
        for f in (getattr(self, "_tele_f", None), getattr(self, "_coll_f", None)):
            if f is not None:
                f.close()
        # Resets the PID's integrator state. The evaluator builds a fresh agent per route, so
        # this is belt-and-braces rather than load-bearing, but leaving stale integrator state
        # behind is exactly the kind of cross-route leakage that would bias a comparison.
        if getattr(self, "_inner", None) is not None:
            self._inner.destroy()
