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

# 2026-11-21 A87 target-point arm: a checkpoint trained with --route_mode dense_tp reads its route_points - 2 near route points plus TF++'s
# target point and next target point; the eval job must set WOR_ROUTE_MODE=dense_tp (the agent then builds that layout).
_ROUTE_MODE = os.environ.get("WOR_ROUTE_MODE", "dense") or "dense"

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
        # the evaluator appends '+<save_name>' per route (see above): the last suffix is the current route (A44 clip names)
        self._route_tag = path_to_conf_file.rsplit("+", 1)[-1] if "+" in path_to_conf_file else "route"

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

    def _tfpp_target_points(self, hero):
        """(2, 2) ego-frame TF++ target point and next target point (src/agents/tfpp_target_point.py: carla_garage's RoutePlanner on the
        sparse plan, projected like every route point here)."""
        if getattr(self, "_tp", None) is None:
            from src.agents.tfpp_target_point import TfppTargetPoints
            self._tp = TfppTargetPoints(self._global_plan_world_coord)
        tf = hero.get_transform()
        return self._tp.step((tf.location.x, tf.location.y, tf.location.z), tf.get_inverse_matrix())

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

    def _privileged_tokens(self, hero, m):
        """(m, BOX_FEATS) object tokens for the privileged teacher (src/agents/privileged_boxes.py). The traffic-light stop waypoints
        are built once per agent instance (the evaluator sets the agent up once per route). The first tick's tokens are printed, so a
        lane log shows what the teacher was given."""
        if getattr(self, "_priv", None) is None:
            from src.agents.privileged_boxes import PrivilegedBoxes
            self._priv = PrivilegedBoxes(CarlaDataProvider.get_world(), CarlaDataProvider.get_map())
            self._priv_logged = False
        variant = str(getattr(self._inner, "privileged_variant", "") or "")
        tok = self._priv.tokens(hero, m, variant, float(getattr(self._inner, "crop_bottom_frac", 0.0) or 0.0))
        if not self._priv_logged:
            from scripts.training.build_box_targets import BOX_CLASSES, VALID_SLOT
            valid = tok[:, VALID_SLOT] > 0.5
            first = [(BOX_CLASSES[int(np.argmax(r[9:15]))], round(float(np.hypot(r[0] * 20.0, r[1] * 10.0)), 1)) for r in tok[valid][:3]]
            print(f"[privileged] variant {variant or 'all'}: {int(valid.sum())}/{m} object tokens on tick 1, "
                  f"{len(self._priv.lights)} traffic lights on the map; "
                  f"nearest: {first}", flush=True)
            self._priv_logged = True
        dump = os.environ.get("B2D_PRIV_DUMP")   # 2026-11-18: per-tick tokens, to compare with the PDM-Lite log tokens of the same scenario
        if dump:
            self._priv_ticks = getattr(self, "_priv_ticks", [])
            self._priv_ticks.append(tok.astype(np.float16))
            if len(self._priv_ticks) % 50 == 1:   # rewritten as the route goes, so a crash keeps what was driven
                os.makedirs(dump, exist_ok=True)
                name = os.path.basename(str(getattr(self, "_route_tag", "") or "")) or f"route_{id(self)}"
                np.savez_compressed(os.path.join(dump, f"{name}_tokens.npz"), tokens=np.stack(self._priv_ticks), variant=variant)
        return tok

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
            route_points=self._inner.route_points - (2 if _ROUTE_MODE == "dense_tp" else 0))
        if getattr(self._inner, "route_mode", "dense") in ("dense", "dense_tp") and getattr(self._inner, "route_mode", "dense") != _ROUTE_MODE:
            raise RuntimeError(f"the checkpoint was trained with route_mode {self._inner.route_mode!r} but WOR_ROUTE_MODE is {_ROUTE_MODE!r}: "
                               "set the eval job's WOR_ROUTE_MODE to the training value")
        if _ROUTE_MODE == "dense_tp":   # 2026-11-21 A87: + TF++'s target point and next target point (its RoutePlanner rule)
            route = np.concatenate([np.asarray(route, dtype=np.float32).reshape(-1, 2), self._tfpp_target_points(hero)], axis=0)

        if os.environ.get("WOR_RL"):
            self._rl_pre_step(hero, timestamp)                  # TODO A74: last tick's reward first, then act() picks this tick's maneuver
        if os.environ.get("B2D_VIDEO_DIR") and getattr(self, "_probe", None) is None:
            try:  # 2026-11-12: read-only attention hooks, installed before the first forward so tick 0 has its attention too
                from src.eval.attention_probe import PolicyAttentionProbe
                self._probe = PolicyAttentionProbe(self._inner.net)
            except Exception as e:
                self._probe = False
                print(f"[video] attention probe off for this route: {e!r}", flush=True)
        inner_input = {
            "rgb_front": rgb,
            "speed": input_data["speed"],  # (frame, {'speed': m/s}) - already the trained unit
            "command": WOR_LANEFOLLOW_COMMAND,
            "route": route,
        }
        m_priv = int(getattr(self._inner.net, "privileged_boxes", 0) or 0)
        if m_priv:  # 2026-11-17 ceiling diagnostic: the privileged teacher's object tokens, from the simulator (never an RGB-only agent)
            inner_input["privileged_tokens"] = self._privileged_tokens(hero, m_priv)
        control = self._inner.run_step(inner_input)
        if os.environ.get("WOR_CREEP") == "1":
            control = self._creep(input_data, control)
        if os.environ.get("B2D_TELEMETRY_DIR"):
            self._log_telemetry(hero, control, timestamp)
        if os.environ.get("B2D_CLIPS_DIR") and not getattr(self, "_clips_failed", False):
            try:  # a recorder bug must never cost a route
                self._clip_step(hero, rgb, control, timestamp, input_data)
            except Exception as e:
                self._clips_failed = True
                print(f"[A44] clip recorder disabled for this route: {e!r}", flush=True)
        if os.environ.get("B2D_VIDEO_DIR") and not getattr(self, "_vid_failed", False):
            try:  # the video recorder reads only; a recorder bug must never cost a route or change a control
                self._video_step(hero, rgb, control, timestamp, input_data, route)
            except Exception as e:
                self._vid_failed = True
                print(f"[video] recorder disabled for this route: {e!r}", flush=True)
        return control

    def _rl_pre_step(self, hero, timestamp):
        """TODO A74 (WOR_RL): on a route's first tick attach a maneuver controller to the policy and a collision sensor to the ego (rewards
        only; the policy still sees the camera alone); every tick, credit the reward of the transition the previous maneuver caused
        (maneuver_rl.RLEpisode) before act() picks this tick's maneuver."""
        import queue
        from src.models.world_on_rails.maneuver_rl import RLEpisode, controller_from_env
        if getattr(self, "_rl_off", False):
            return
        if getattr(self, "_rl_ep", None) is None:
            net = self._inner.net
            if not hasattr(net, "embed_dim"):
                raise RuntimeError(f"WOR_RL needs the qwen policy's maneuver hook; {type(net).__name__} has none")
            made = controller_from_env(net.embed_dim, seed=(os.getpid() * 1000003 + hero.id) % (2 ** 32))
            if made is None:                                     # WOR_RL="0": off, the plain decode
                self._rl_off = True
                return
            ctl, cfg = made
            net._rl = ctl
            xy = np.array([[loc.x, loc.y] for loc in self._route_locations], dtype=np.float64)
            self._rl_ep = RLEpisode(ctl, cfg, xy, self._route_tag)
            self._rl_q = queue.Queue()
            world = CarlaDataProvider.get_world()
            self._rl_sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=hero)
            self._rl_sensor.listen(lambda ev, q=self._rl_q: q.put((ev.other_actor.type_id, ev.other_actor.id)))
            print(f"[A74] maneuver RL on: {cfg}", flush=True)
        coll = []
        while not self._rl_q.empty():
            coll.append(self._rl_q.get_nowait())
        loc = hero.get_location()
        self._rl_ep.tick(loc.x, loc.y, float(timestamp), coll)

    # ---- A44 collision-clip recorder (B2D_CLIPS_DIR); off by default ------------------------------------------------------------
    # A 6 s ring buffer at 4 Hz (frame, speed, control, speed posterior, target speed, waypoints) written as one npz per collision / long stand-still,
    # with the other actors' poses for analysis only (never an input). Loader and format: src/eval/clip_recorder.py.
    def _clip_step(self, hero, rgb, control, timestamp, input_data):
        import queue
        from src.eval.clip_recorder import ClipRecorder
        if getattr(self, "_clips", None) is None:
            self._clips = ClipRecorder(os.environ["B2D_CLIPS_DIR"], f"{self._route_tag}_h{hero.id}")
            self._clip_q = queue.Queue()
            world = CarlaDataProvider.get_world()
            self._clip_sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=hero)

            def on_collision(ev, hero=hero, q=self._clip_q):
                hv, ov = hero.get_velocity(), ev.other_actor.get_velocity()
                rel, ht = ev.other_actor.get_location() - hero.get_location(), hero.get_transform()
                fwd, right = ht.get_forward_vector(), ht.get_right_vector()
                kind = "vehicle" if ev.other_actor.type_id.startswith("vehicle") else "walker" if ev.other_actor.type_id.startswith("walker") else "layout"
                q.put((kind, {"other_type": ev.other_actor.type_id, "other_id": ev.other_actor.id, "ego_speed": (hv.x ** 2 + hv.y ** 2) ** 0.5,
                              "other_speed": (ov.x ** 2 + ov.y ** 2) ** 0.5, "other_rel_fwd": rel.x * fwd.x + rel.y * fwd.y,
                              "other_rel_right": rel.x * right.x + rel.y * right.y, "ego_xy": [ht.location.x, ht.location.y, ht.rotation.yaw]}))
            self._clip_sensor.listen(on_collision)
            self._clip_stand = 0.0
        spd = input_data["speed"][1]
        spd = float(spd.get("speed", 0.0)) if isinstance(spd, dict) else float(spd)
        dbg = getattr(getattr(self._inner, "net", None), "last_debug", None) or {}
        self._clips.push(timestamp, rgb, {"speed": spd, "throttle": float(control.throttle), "brake": float(control.brake), "steer": float(control.steer),
                                          "target_kmh": dbg.get("target_kmh"), "probs": None if dbg.get("probs") is None else [round(float(x), 4) for x in dbg["probs"]],
                                          "waypoints": None if dbg.get("waypoints") is None else [[round(float(a), 2) for a in w] for w in dbg["waypoints"]],
                                          "force_brake": dbg.get("force_brake"),
                                          "danger_p": dbg.get("danger_p")})   # TODO A84: did the danger head see the crash coming?
        self._clip_stand = self._clip_stand + 0.05 if spd < 0.1 else 0.0
        events = []
        while not self._clip_q.empty():
            events.append(self._clip_q.get())
        if self._clip_stand > 60.0:
            events.append(("stuck", {"standing_s": self._clip_stand})); self._clip_stand = 0.0
        for kind, ev in events:
            actors = []
            _all = CarlaDataProvider.get_world().get_actors()
            for a in list(_all.filter("*vehicle*")) + list(_all.filter("*walker*")):
                if a.id != hero.id and a.get_location().distance(hero.get_location()) < 50.0:
                    t, v = a.get_transform(), a.get_velocity()
                    actors.append({"type": a.type_id, "id": a.id, "xy": [t.location.x, t.location.y, t.rotation.yaw], "speed": (v.x ** 2 + v.y ** 2) ** 0.5})
            self._clips.trigger(timestamp, kind, {**ev, "actors": actors})

    # ---- per-tick video recorder (B2D_VIDEO_DIR); off by default ----------------------------------------------------------------
    # One JPEG of the front camera as the policy receives it, and one JSON line per tick with what the policy was given (speed,
    # command, the four route points in the ego frame) and what it returned (predicted waypoints, target speed, control).
    # scripts/eval/compose_eval_video.py turns the folder into an mp4 with a HUD on top. Read-only: it never changes a control.
    def _video_step(self, hero, rgb, control, timestamp, input_data, route):
        import json
        import cv2

        def _np(x):
            if x is None:
                return None
            if hasattr(x, "detach"):
                x = x.detach().float().cpu().numpy()
            return np.asarray(x, dtype=float)

        probe = getattr(self, "_probe", None) or None
        if getattr(self, "_vid_f", None) is None:
            d = os.path.join(os.environ["B2D_VIDEO_DIR"], self._route_tag)
            os.makedirs(d, exist_ok=True)
            self._vid_dir = d
            self._vid_f = open(os.path.join(d, "ticks.jsonl"), "w")
            self._vid_n = 0
            # B2D_VIDEO_EVERY=k keeps every k-th tick (the composer then plays at 20/k frames per simulated second)
            self._vid_every = max(1, int(os.environ.get("B2D_VIDEO_EVERY", "1")))
            meta = {"tag": self._route_tag, "checkpoint": os.environ.get("WOR_CHECKPOINT") or getattr(self._inner, "checkpoint_path", None),
                    "label": os.environ.get("B2D_VIDEO_LABEL", ""), "every": self._vid_every, "tick_dt": 0.05}
            if probe:
                meta["layout"] = probe.layout()
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f)
        n = self._vid_n
        self._vid_n += 1
        if n % self._vid_every:
            return
        # the frame the policy received (resized to its input size, route overlay drawn when the checkpoint uses one), so the attention
        # grid lies on exactly what the network saw; the raw camera frame only when the agent does not expose it
        pin = getattr(self._inner, "_last_input", None)
        frame = pin if pin is not None and getattr(pin, "ndim", 0) == 3 else rgb
        cv2.imwrite(os.path.join(self._vid_dir, f"{n:06d}.jpg"), cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        spd = input_data["speed"][1]
        spd = float(spd.get("speed", 0.0)) if isinstance(spd, dict) else float(spd)
        dbg = getattr(getattr(self._inner, "net", None), "last_debug", None) or {}
        wps = _np(dbg.get("waypoints"))
        tgt = dbg.get("target_kmh")
        tgt = None if tgt is None else float(_np(tgt).reshape(-1)[0])
        probs = dbg.get("probs")
        tr = hero.get_transform()
        row = {
            "n": n, "t": round(float(timestamp), 3), "tag": self._route_tag,
            "speed_mps": round(spd, 3), "command": WOR_LANEFOLLOW_COMMAND,
            "route_ego": _np(route).round(3).tolist(),
            "waypoints_pred": None if wps is None else wps.reshape(-1, 2).round(3).tolist(),
            "target_kmh": None if tgt is None else round(tgt, 2),
            "throttle": round(float(control.throttle), 4), "brake": round(float(control.brake), 4), "steer": round(float(control.steer), 4),
            "ego_xy": [round(tr.location.x, 2), round(tr.location.y, 2), round(tr.rotation.yaw, 1)],
            "probs": None if probs is None else [round(float(p), 4) for p in np.asarray(probs).reshape(-1)],
            "force_brake": dbg.get("force_brake"),
            "danger_p": dbg.get("danger_p"),   # TODO A84
        }
        if probe:  # what the policy token attended to on this tick, and how the speed head (and its experts) decided
            row["attn"] = probe.attention()
            row["dec"] = probe.decision(spd)
        self._vid_f.write(json.dumps(row) + "\n")
        self._vid_f.flush()

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
        if getattr(self, "_rl_ep", None) is not None:          # TODO A74: write the route's episode for the learner
            try:
                self._rl_sensor.stop(); self._rl_sensor.destroy()
            except Exception:
                pass
            print(f"[A74] episode {self._rl_ep.finish()}", flush=True)
            self._inner.net._rl = None
            self._rl_ep = None
        if getattr(self, "_clips", None) is not None:
            try:
                self._clip_sensor.stop(); self._clip_sensor.destroy()
                self._clips.close()
            except Exception:
                pass
        if getattr(self, "_coll_sensor", None) is not None:
            try:
                self._coll_sensor.stop(); self._coll_sensor.destroy()
            except Exception:
                pass
        for f in (getattr(self, "_tele_f", None), getattr(self, "_coll_f", None), getattr(self, "_vid_f", None)):
            if f is not None:
                f.close()
        if getattr(self, "_probe", None):
            self._probe.close()
        # Resets the PID's integrator state. The evaluator builds a fresh agent per route, so
        # this is belt-and-braces rather than load-bearing, but leaving stale integrator state
        # behind is exactly the kind of cross-route leakage that would bias a comparison.
        if getattr(self, "_inner", None) is not None:
            self._inner.destroy()
