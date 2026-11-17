"""Idea 19e / T5: our policy drives, the PDM-Lite expert takes over when it is about to go wrong (SafeDAgger-style), 2026-11-08.

Design: docs/todo/DESIGN_19E_EXPERT_CORRECTIONS_2026-11-08.md.

How it works
------------
This is PDM-Lite's own DataAgent (expert + recorder, same folder layout as the training data), with one change: the control that is
APPLIED to the car comes from our policy unless the expert has taken over. The expert's planner runs every tick from the state our car is
actually in (it plans from the live world, not from a recorded trajectory), so it is a valid shadow expert in states the original expert
never visited.

Labels
------
Our training labels are the car's own future poses (wor_dataset._index_pdm_lite_route: ego_matrix of the next pred_len=5 saved frames,
saved every data_save_freq=5 ticks -> 25 ticks = 1.25 s). A saved frame is therefore a correct expert label only if the EXPERT drove for
the whole label horizon after it. Every tick's control source is logged to dagger_sources.json; scripts/datagen/dagger_prune.py then
writes "dagger_keep" into each frame's measurements (true only when the expert drove from that frame to KEEP_HORIZON ticks after it), and
the loader skips frames with dagger_keep == false.

Takeover rule (expert takes control when any holds; DESIGN_19E section "Takeover rule"):
  hazard   the expert brakes for a hazard (brake, no throttle) while our policy is not braking and the car moves > 1 m/s, for 3 ticks
  stuck    the car has stood still > STUCK_S while the expert wants to drive (throttle > 0, no brake)
  lateral  the car is > LAT_M from the expert's (obstacle-shifted) route
Hand back after at least HOLD_S of expert control AND AGREE_S of agreement between the two controls (steer within 0.1, same brake).
HOLD_S >= the label horizon so every takeover yields kept frames. RoG-DAgger (arXiv 2608.24525) found expert plans requested just before
a crash still crash 30-43% of the time: the triggers are meant to fire early, and frames before a takeover are never kept.

Run (CARLA 0.9.15 + carla_garage leaderboard, like PDM-Lite data collection):
  TEAM_AGENT=/workspace/MThesis/scripts/datagen/dagger_agent.py  TEAM_CONFIG=<carla_garage config dir>  SAVE_PATH=<out>  DATAGEN=1
  WOR_DAGGER_CKPT=<arch>:<checkpoint.pth>   (our policy, same format as bench2drive_agent.py's TEAM_CONFIG)
  PYTHONPATH must contain carla_garage/team_code, its leaderboard and scenario_runner, and MTHESIS_ROOT (default /workspace/MThesis).
"""
import gzip
import json
import math
import os
import sys

import numpy as np

MTHESIS_ROOT = os.environ.get("MTHESIS_ROOT", "/workspace/MThesis")
if MTHESIS_ROOT not in sys.path:
    sys.path.insert(0, MTHESIS_ROOT)

import carla  # noqa: E402
from data_agent import DataAgent  # noqa: E402  (carla_garage/team_code)

from src.agents.wor_agent import WorldOnRailsAgent  # noqa: E402
from scripts.eval.eval_wor import _ego_frame_route, WOR_LANEFOLLOW_COMMAND  # noqa: E402

FPS = 20
HAZARD_TICKS = int(os.environ.get("DAGGER_HAZARD_TICKS", 3))
STUCK_S = float(os.environ.get("DAGGER_STUCK_S", 3.0))
LAT_M = float(os.environ.get("DAGGER_LAT_M", 1.0))
HOLD_S = float(os.environ.get("DAGGER_HOLD_S", 4.0))
AGREE_S = float(os.environ.get("DAGGER_AGREE_S", 2.0))
POLICY_ONLY = os.environ.get("DAGGER_POLICY_ONLY", "0") == "1"   # parity / sanity runs: never take over (records what our policy does)
ROUTE_SPACING_M = 1.0


def get_entry_point():
    return "DaggerAgent"


def _resample(locations, spacing=ROUTE_SPACING_M):
    """carla.Location polyline -> ~uniform spacing (same as bench2drive_agent._resample_locations, the spacing the policy trained on)."""
    out, carry = [locations[0]], 0.0
    for a, b in zip(locations[:-1], locations[1:]):
        seg = a.distance(b)
        if seg <= 1e-9:
            continue
        d = spacing - carry
        while d <= seg:
            r = d / seg
            out.append(carla.Location(x=a.x + (b.x - a.x) * r, y=a.y + (b.y - a.y) * r, z=a.z + (b.z - a.z) * r))
            d += spacing
        carry = seg - (d - spacing)
    return out


class DaggerAgent(DataAgent):

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager=traffic_manager)
        cfg = os.environ["WOR_DAGGER_CKPT"]
        arch, _, ckpt = cfg.partition(":")
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(f"WOR_DAGGER_CKPT must be '<arch>:<existing checkpoint>', got {cfg!r}")   # never drive an untrained policy
        self._policy = WorldOnRailsAgent(checkpoint_path=ckpt, policy_arch=arch,
                                         backbone_name=os.environ.get("WOR_BACKBONE", "regnety_032"))
        self._route_locs, self._route_idx = None, 0
        self._expert_on, self._expert_ticks, self._agree_ticks = False, 0, 0
        self._hazard_ticks, self._still_ticks = 0, 0
        self._sources = []          # per tick: [step, "expert"|"policy", reason]
        self._n_takeovers = 0

    # ------------------------------------------------------------------ helpers
    def _policy_control(self, input_data):
        if self._route_locs is None:
            plan = self.org_dense_route_world_coord
            self._route_locs = _resample([(e[0] if isinstance(e, (tuple, list)) else e).location for e in plan])
        route, self._route_idx, _ = _ego_frame_route(self._vehicle, self._route_locs, self._route_idx,
                                                     route_points=self._policy.route_points)
        rgb = input_data["rgb"][1]
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, [2, 1, 0]]   # BGRA -> RGB (bench2drive_agent.py, challenges 11.6)
        return self._policy.run_step({"rgb_front": rgb, "speed": input_data["speed"], "command": WOR_LANEFOLLOW_COMMAND, "route": route})

    def _lateral_error(self):
        wp = self._waypoint_planner
        i = int(getattr(wp, "route_index", 0))
        pts = np.asarray(wp.route_points[max(i - 5, 0):i + 15])[:, :2]   # the obstacle-shifted route the expert follows
        loc = self._vehicle.get_location()
        return float(np.min(np.hypot(pts[:, 0] - loc.x, pts[:, 1] - loc.y))) if len(pts) else 0.0

    def _trigger(self, expert, ours, speed):
        expert_brakes = expert.brake > 0.5 and expert.throttle < 0.05
        ours_brakes = ours.brake > 0.5
        self._hazard_ticks = self._hazard_ticks + 1 if (expert_brakes and not ours_brakes and speed > 1.0) else 0
        if self._hazard_ticks >= HAZARD_TICKS:
            return "hazard"
        self._still_ticks = self._still_ticks + 1 if speed < 0.1 else 0
        if self._still_ticks > STUCK_S * FPS and expert.throttle > 0.05 and expert.brake < 0.5:
            return "stuck"
        if self._lateral_error() > LAT_M:
            return "lateral"
        return ""

    # ------------------------------------------------------------------ step
    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        expert = super().run_step(input_data, timestamp, sensors=sensors, plant=plant)   # expert plan + recording (all frames)
        if plant or not self.initialized:
            return expert
        ours = self._policy_control(input_data)
        speed = float(self._vehicle.get_velocity().length())
        if not self._expert_on:
            reason = "" if POLICY_ONLY else self._trigger(expert, ours, speed)
            if reason:
                self._expert_on, self._expert_ticks, self._agree_ticks = True, 0, 0
                self._n_takeovers += 1
                print(f"[dagger] step {self.step}: expert takes over ({reason}), takeover {self._n_takeovers}", flush=True)
        else:
            self._expert_ticks += 1
            agree = abs(expert.steer - ours.steer) < 0.1 and (expert.brake > 0.5) == (ours.brake > 0.5)
            self._agree_ticks = self._agree_ticks + 1 if agree else 0
            if self._expert_ticks >= HOLD_S * FPS and self._agree_ticks >= AGREE_S * FPS:
                self._expert_on, self._hazard_ticks, self._still_ticks = False, 0, 0
                print(f"[dagger] step {self.step}: hand back to the policy", flush=True)
        src = "expert" if self._expert_on else "policy"
        self._sources.append([int(self.step), src])
        return expert if self._expert_on else ours

    def destroy(self, results=None):
        if self.save_path is not None and self.datagen:
            with gzip.open(self.save_path / "dagger_sources.json.gz", "wt", encoding="utf-8") as f:
                json.dump({"data_save_freq": int(self.config.data_save_freq), "takeovers": self._n_takeovers,
                           "policy_only": POLICY_ONLY, "sources": self._sources}, f)
            print(f"[dagger] route done: {self._n_takeovers} takeovers, expert ticks "
                  f"{sum(1 for s in self._sources if s[1] == 'expert')} of {len(self._sources)}", flush=True)
        super().destroy(results)
