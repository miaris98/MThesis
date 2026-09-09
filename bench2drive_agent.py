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

# This project's root, so `src.agents` and `eval_wor` import. The harness runs with its own
# cwd inside the Bench2Drive checkout, so relying on cwd would not work.
MTHESIS_ROOT = os.environ.get("MTHESIS_ROOT", "/workspace/MThesis")
if MTHESIS_ROOT not in sys.path:
    sys.path.insert(0, MTHESIS_ROOT)

from src.agents.wor_agent import WorldOnRailsAgent  # noqa: E402

# Imported rather than re-derived. `_ego_frame_route` is the exact ego-frame projection the
# policy was trained against (`_index_pdm_lite_route`'s ego_matrix convention) and is already
# the one every other evaluation path in this repo uses - a second copy here would be free to
# drift out of agreement with the dataset without anything failing loudly.
from eval_wor import (  # noqa: E402
    _ego_frame_route,
    WOR_LANEFOLLOW_COMMAND,
)

# `_ego_frame_route` takes a fixed *number* of upcoming points (WOR_ROUTE_LOOKAHEAD = 20), so
# the lookahead *distance* it represents is set entirely by the spacing of the route list it
# is handed. Everything else in this project feeds it a GlobalRoutePlanner route sampled at
# 2.0 m, i.e. ~40 m of lookahead. Bench2Drive's plan has its own spacing, so it is resampled
# to this value before use - otherwise the policy would silently receive a route on a
# different scale than the one it was trained on.
ROUTE_SPACING_M = 2.0


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

        # The evaluator appends '+<save_name>' to whatever TEAM_CONFIG was passed
        # (leaderboard_evaluator.py: `args.agent_config = args.agent_config + '+' + save_name`),
        # so strip that suffix back off before parsing our own config out of it.
        cfg = path_to_conf_file.rsplit("+", 1)[0]
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

    def run_step(self, input_data, timestamp):
        if self._route_locations is None:
            self._route_locations = self._build_route_locations()
        if self.hero_actor is None:
            # The base class looks the hero up at construction time, which can precede the
            # ego actually being spawned.
            self.get_hero()

        rgb = input_data["rgb_front"][1]
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, [2, 1, 0]]  # BGRA -> RGB, not a bare alpha drop

        route, self._route_idx, _ = _ego_frame_route(
            self.hero_actor, self._route_locations, self._route_idx)

        return self._inner.run_step({
            "rgb_front": rgb,
            "speed": input_data["speed"],  # (frame, {'speed': m/s}) - already the trained unit
            "command": WOR_LANEFOLLOW_COMMAND,
            "route": route,
        })

    def destroy(self):
        # Resets the PID's integrator state. The evaluator builds a fresh agent per route, so
        # this is belt-and-braces rather than load-bearing, but leaving stale integrator state
        # behind is exactly the kind of cross-route leakage that would bias a comparison.
        if getattr(self, "_inner", None) is not None:
            self._inner.destroy()
