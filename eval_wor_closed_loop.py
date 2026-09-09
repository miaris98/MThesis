#!/usr/bin/env python3
"""Closed-loop CARLA evaluation for World on Rails policies.

Every number in Challenge Group 13 - the 13.6% margin, all three replication seeds, the
size curve, and 13.29's geometry-loss A/B - is held-out L1 waypoint error on a fixed
dataset. None of it has ever been shown to correspond to driving. This script produces
the number that does.

WHY THIS IS A SEPARATE SCRIPT FROM `eval_wor.py`
------------------------------------------------
`eval_wor.py` is a demo recorder: one episode, a random spawn point, an HD video with a
HUD, and a raw count of collision-sensor callbacks. Three of those properties make it
unusable as a measurement:

  * The spawn point is `random.choice(spawn_points[:10])` with no seed, and the
    destination is "whichever spawn point is farthest away". Two checkpoints evaluated
    with it drive **different routes**, so the comparison is confounded before it starts.
  * `--episodes` is parsed and then never used - the rollout loop runs exactly once.
  * A raw collision count is not a score (see `src/eval/driving_metrics.py`).

This script fixes the measurement properties and drops the video. Determinism is the
whole point: the same `--route_seed` yields byte-identical route and traffic layouts, so
two checkpoints can be compared **paired over routes**, which is the methodology the
open-loop side of this project already settled on (13.17, 13.19) after learning that
unpaired comparisons on a small route set resolve nothing.

USAGE
-----
    # one arm
    python eval_wor_closed_loop.py --checkpoint .../qwen30m_geom_s0/best_model.pth \\
        --policy_arch qwen30m --town Town01 --routes 20 --route_seed 0 \\
        --out results/closed_loop/qwen30m_geom_s0.json

    # then the paired comparison
    python compare_closed_loop.py --a .../baseline_s0.json --b .../geom_s0.json
"""
import argparse
import glob
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import numpy as np
import torch

# CARLA PythonAPI discovery. Deliberately NOT eval_wor.py's version, which prepends every
# `carla-*-py3*.egg` it finds to sys.path unconditionally. CARLA 0.9.15 ships only cp27 and
# cp37 artifacts, so on any newer interpreter that egg is both unloadable and ahead of a
# perfectly good pip-installed client - it shadows the working import with a broken one.
# Prefer whatever is already importable; fall back to the shipped egg only if there is
# nothing else, which keeps the original py3.7 conda environment working.
carla_root = os.environ.get("CARLA_ROOT", "/workspace/carla")

# The `agents.navigation` helpers are not packaged in the wheel and only ever live in the
# server tree, so this path is needed regardless of where `carla` itself comes from.
_api_path = os.path.join(carla_root, "PythonAPI", "carla")
if os.path.isdir(_api_path) and _api_path not in sys.path:
    sys.path.append(_api_path)

try:
    import carla  # noqa: E402
except ImportError:
    _dist = os.path.join(carla_root, "PythonAPI", "carla", "dist")
    for _p in glob.glob(os.path.join(_dist, "carla-*-py3*.egg")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import carla  # noqa: E402
from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: E402

from src.agents.wor_agent import WorldOnRailsAgent  # noqa: E402
from src.eval.driving_metrics import DrivingMetrics, TerminationReason, aggregate  # noqa: E402
from src.eval.rollout_video import RouteVideoRecorder  # noqa: E402

# Imported rather than re-derived: these define the exact ego-frame route encoding the
# policy was trained on, and a second copy of them here would be free to drift out of
# agreement with the dataset without anything failing loudly.
from eval_wor import (  # noqa: E402
    _ego_frame_route,
    WOR_ROUTE_POINTS,
    WOR_LANEFOLLOW_COMMAND,
)

# Route planner sampling. 2.0 m matches eval_wor.py; route completion is computed as an
# index fraction, which is only a distance fraction because this spacing is uniform.
ROUTE_SAMPLING_RESOLUTION = 2.0
# A route must be at least this long to be worth scoring - two adjacent spawn points
# would otherwise produce a "route" the car completes by rolling forward three metres.
MIN_ROUTE_LENGTH_M = 100.0
# Leaderboard's route-deviation threshold.
MAX_ROUTE_DEVIATION_M = 30.0
# Below this the vehicle is treated as not driving through a light, so a light turning
# red over a stationary car is not counted as a violation.
RED_LIGHT_MIN_SPEED_MPS = 0.5

# CARLA's 14 standard presets. The leaderboard varies weather per route and does not publish
# the assignment, so there is no "correct" mapping to reproduce - this cycles through all 14
# by route position instead of leaving every route on the server's default. It is a function
# of route position alone (not of policy_arch, checkpoint, or wall-clock time), so every arm
# sees the identical weather sequence for the identical --route_seed manifest and the paired
# comparison stays valid.
WEATHER_PRESETS = [
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "MidRainyNoon",
    "HardRainNoon", "SoftRainNoon", "ClearSunset", "CloudySunset", "WetSunset",
    "WetCloudySunset", "MidRainSunset", "HardRainSunset", "SoftRainSunset",
]


def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop CARLA evaluation of a WoR policy")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth to evaluate")
    p.add_argument("--policy_arch", type=str, default="qwen30m",
                   choices=["cnn", "qwen10m", "qwen30m", "qwen100m", "qwen500m", "qwen900m"],
                   help="Must match the --policy_arch this checkpoint was trained with")
    p.add_argument("--backbone", type=str, default="resnet34")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm_port", type=int, default=0,
                   help="Traffic manager RPC port. 0 (default) derives --port + 8000, which "
                        "reduces to the client library's own hardcoded default (8000) only "
                        "when run against a lone server on --port 0 - anywhere else it must "
                        "be set. get_trafficmanager() ALWAYS binds a fixed port unless told "
                        "otherwise, regardless of which CARLA server --port the client "
                        "connected to, so two concurrent evaluations against two different "
                        "CARLA servers still collide here unless each passes a distinct "
                        "--tm_port - the CARLA server's own RPC/streaming ports (--port, "
                        "--port+1, --port+2) do not protect against this at all, it is a "
                        "wholly separate service with its own port")
    p.add_argument("--town", type=str, default="Town01")
    p.add_argument("--routes", type=int, default=20,
                   help="Number of routes to drive. Routes are the independent unit for the "
                        "paired bootstrap, so this - not episode length - is what sets the "
                        "resolution of the comparison")
    p.add_argument("--route_source", type=str, default="random", choices=["random", "official"],
                   help="'random' (default) draws --routes random origin/destination pairs "
                        "via build_route_manifest, as this script always has. 'official' "
                        "loads hand-authored routes for --town from --route_file (a CARLA "
                        "Leaderboard 1.0 routes_*.xml) via load_official_routes instead - "
                        "the closest local proxy to the leaderboard's own protocol. See "
                        "TODO_leaderboard_benchmark.md for what it still does not reproduce")
    p.add_argument("--route_file", type=str, default="",
                   help="Path to a Leaderboard 1.0 routes_*.xml (e.g. "
                        "leaderboard/data/routes_testing.xml). Required when "
                        "--route_source official; ignored otherwise")
    p.add_argument("--route_seed", type=int, default=0,
                   help="Seeds route generation AND traffic. Two runs sharing this value drive "
                        "identical routes through identical traffic, which is what makes them "
                        "comparable route-by-route")
    p.add_argument("--max_steps", type=int, default=3000,
                   help="Per-route step cap at 20 FPS (3000 = 150 s of simulation)")
    p.add_argument("--blocked_timeout_s", type=float, default=60.0,
                   help="Stationary time before a route is abandoned. The leaderboard uses 90 s; "
                        "the default here is shorter because these routes are shorter")
    p.add_argument("--num_vehicles", type=int, default=30)
    p.add_argument("--num_walkers", type=int, default=0,
                   help="Pedestrians. Off by default: a walker spawned without an "
                        "ai.walker.controller just stands where it was placed, so it is a "
                        "static obstacle wearing a pedestrian's collision penalty rather "
                        "than traffic, and walker spawning is the most segfault-prone part "
                        "of the setup (Group 3). Enable only with the controller work done")
    p.add_argument("--out", type=str, required=True, help="Output JSON path")
    p.add_argument("--record_video", type=int, default=0,
                   help="Record a chase-camera video per route. Off by default: it costs "
                        "a 1280x720 sensor per step and the scored numbers do not depend on it")
    p.add_argument("--video_dir", type=str, default="",
                   help="Where to write videos (defaults to <out>_videos/)")
    p.add_argument("--video_routes", type=int, default=3,
                   help="Record only the first N routes, so a 20-route run does not "
                        "produce 20 videos nobody watches")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_route_manifest(world, num_routes: int, seed: int) -> List[Dict]:
    """Chooses `num_routes` (origin, destination) spawn-point index pairs deterministically.

    Indices into `get_spawn_points()` are used rather than raw coordinates because they
    are stable for a given town and CARLA version, and they make the manifest short
    enough to read. The RNG is local to this function so that traffic spawning later
    cannot shift the route draw.
    """
    rng = random.Random(seed)
    spawn_points = world.get_map().get_spawn_points()
    grp = GlobalRoutePlanner(world.get_map(), ROUTE_SAMPLING_RESOLUTION)

    manifest, attempts = [], 0
    n = len(spawn_points)
    while len(manifest) < num_routes and attempts < num_routes * 40:
        attempts += 1
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        try:
            route = grp.trace_route(spawn_points[i].location, spawn_points[j].location)
        except Exception:
            continue
        length_m = ROUTE_SAMPLING_RESOLUTION * max(0, len(route) - 1)
        if length_m < MIN_ROUTE_LENGTH_M:
            continue
        manifest.append({
            "route_id": f"r{len(manifest):03d}",
            "spawn_idx": i,
            "dest_idx": j,
            "num_points": len(route),
            "length_m": round(length_m, 1),
        })

    if len(manifest) < num_routes:
        print(f"[WARNING] Only {len(manifest)}/{num_routes} routes met the "
              f"{MIN_ROUTE_LENGTH_M} m minimum on this map.")
    return manifest


def load_official_routes(xml_path: str, town: str, num_routes: Optional[int],
                          seed: int) -> List[Dict]:
    """Loads routes for `town` from a CARLA Leaderboard 1.0 `routes_*.xml` file.

    These are hand-authored multi-waypoint routes (see `leaderboard/data/routes_training.xml`
    and `routes_testing.xml`, 50 + 26 routes across the 8 public towns), not the random
    origin/destination pairs `build_route_manifest` draws. This is the closest legitimate
    proxy available for the leaderboard's own protocol: the actual held-out test routes are
    secret and were only ever scoreable by the leaderboard's own (now-deprecated) online
    server, so this file's public route pool - what the leaderboard ships for local
    training/validation - is as close as a local run can get. See
    TODO_leaderboard_benchmark.md for what this still does not reproduce (no
    `scenario_runner`-driven scripted scenarios, no official weather assignment).

    Only (x, y) is kept from each `<waypoint>`. z/pitch/yaw/roll are not trusted as spawn
    geometry - some entries in the shipped files carry values like `pitch="360.0"`, and a
    raw z is not guaranteed to sit exactly on the road surface. `run_route` instead snaps
    each (x, y) to the nearest driving-lane waypoint via `get_waypoint(project_to_road=True)`,
    which is the standard CARLA pattern for turning an arbitrary 2D coordinate into a valid
    on-road transform.

    Deterministic subsampling: if `town` has more routes than `num_routes`, a `Random(seed)`
    instance picks which ones, so the same `--route_seed` always yields the same subset -
    the same determinism guarantee `build_route_manifest` gives the random-route path.
    """
    tree = ET.parse(xml_path)
    town_routes = [r for r in tree.getroot().findall("route") if r.get("town") == town]

    if num_routes is not None and len(town_routes) > num_routes:
        rng = random.Random(seed)
        town_routes = rng.sample(town_routes, num_routes)

    manifest = []
    for r in town_routes:
        waypoints = [(float(wp.get("x")), float(wp.get("y")))
                     for wp in r.findall("waypoint")]
        if len(waypoints) < 2:
            continue
        length_m = sum(
            ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            for (x1, y1), (x2, y2) in zip(waypoints[:-1], waypoints[1:])
        )
        manifest.append({
            "route_id": f"official_r{r.get('id')}",
            "waypoints_raw": waypoints,
            "length_m": round(length_m, 1),
        })

    if num_routes is not None and len(manifest) < num_routes:
        print(f"[WARNING] Town {town} has only {len(manifest)} official routes in "
              f"{xml_path}, fewer than the {num_routes} requested.")
    return manifest


def spawn_traffic(world, client, num_vehicles: int, num_walkers: int, seed: int,
                   tm_port: int = 8000):
    """Spawns background traffic under a fixed traffic-manager seed.

    Without `set_random_device_seed` the NPC vehicles behave differently on every run,
    which would put a different obstacle in front of each checkpoint and reintroduce
    exactly the variance the fixed route manifest exists to remove.

    `tm_port` matters more than it looks. `get_trafficmanager()` starts an RPC server of
    its own and binds it to a FIXED port (8000) unless told otherwise - unlike the CARLA
    server itself, whose --carla-port choice this call has no knowledge of at all. Two
    concurrent evaluations against two different CARLA servers on two different ports
    still collide here if neither passes an explicit `tm_port`: whichever calls
    `get_trafficmanager()` second gets `RuntimeError: ... bind error`, and depending on
    timing the first can instead take down the whole process with an uncatchable C++
    exception (`clmdep_msgpack::v1::type_error: std::bad_cast`, straight to
    `std::terminate` - no Python try/except reaches it). Found by running three arms of
    this evaluation concurrently in the same tmux host: it reproduced on the third arm
    both times, consistent with the third caller racing the first two for a port already
    taken.
    """
    actors = []
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)

    bp_lib = world.get_blueprint_library()
    rng = random.Random(seed)
    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)

    vehicle_bps = [b for b in bp_lib.filter("vehicle.*")
                   if int(b.get_attribute("number_of_wheels")) == 4]
    for sp in spawn_points[:num_vehicles]:
        bp = vehicle_bps[rng.randrange(len(vehicle_bps))]
        bp.set_attribute("role_name", "autopilot")
        v = world.try_spawn_actor(bp, sp)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            actors.append(v)

    if num_walkers > 0:
        world.set_pedestrians_seed(seed)
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        for _ in range(num_walkers):
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = walker_bps[rng.randrange(len(walker_bps))]
            w = world.try_spawn_actor(bp, carla.Transform(loc))
            if w is not None:
                actors.append(w)

    # Let the newly spawned actors register before anything else touches the world.
    # Spawning in synchronous mode without advancing the simulation leaves actors in a
    # half-initialised state that the next heavy operation can crash on.
    world.tick()
    return actors, tm


class RedLightWatcher:
    """Detects driving through a red light.

    This is an approximation of the leaderboard's `RunRedLight` criterion, which tests
    the vehicle's crossing of a light's trigger-volume stop line geometrically. Here the
    simulator's own "which light governs this vehicle" answer is used instead: a
    violation is a transition out of a light's influence while that light was red and the
    vehicle was moving.

    The honest limitation: if CARLA stops reporting a light as governing the vehicle
    slightly before the stop line is crossed, a violation can be missed. It cannot
    produce a false positive for a car waiting correctly at a red - waiting keeps the
    light governing the vehicle, and a light turning green clears the pending state.
    """

    def __init__(self):
        self._pending_red_id: Optional[int] = None

    def update(self, ego, speed_mps: float) -> Optional[int]:
        light = ego.get_traffic_light()
        state = ego.get_traffic_light_state()
        current_id = light.id if light is not None else None
        is_red = (state == carla.TrafficLightState.Red)

        violated = None
        if self._pending_red_id is not None:
            left_that_light = (current_id != self._pending_red_id)
            if left_that_light and speed_mps > RED_LIGHT_MIN_SPEED_MPS:
                violated = self._pending_red_id
                self._pending_red_id = None
            elif not is_red and current_id == self._pending_red_id:
                self._pending_red_id = None  # it turned green while we waited

        if is_red and current_id is not None:
            self._pending_red_id = current_id

        return violated


def run_route(world, client, agent, route_spec, spawn_points, grp, args, route_index: int,
              record_video: bool = False) -> Dict:
    """Drives one route and returns its metrics record."""
    bp_lib = world.get_blueprint_library()
    actors = []
    recorder = None

    # A pure function of route_index, not of policy_arch/checkpoint/wall-clock, so every
    # arm sees the identical weather for the identical manifest position - see
    # WEATHER_PRESETS.
    world.set_weather(getattr(carla.WeatherParameters,
                               WEATHER_PRESETS[(route_index - 1) % len(WEATHER_PRESETS)]))

    if "waypoints_raw" in route_spec:
        # Official-route path (load_official_routes): snap each hand-authored (x, y) to
        # the nearest driving lane, then chain trace_route() between consecutive snapped
        # points so the vehicle follows all of the route's control points, not just a
        # start/end pair - a multi-waypoint route through Town03/04 (up to 29 points, see
        # TODO_leaderboard_benchmark.md) means a single trace_route(first, last) call
        # would let the planner pick its own path and silently discard the route's shape.
        world_map = world.get_map()
        transforms = [
            world_map.get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True,
                                    lane_type=carla.LaneType.Driving).transform
            for x, y in route_spec["waypoints_raw"]
        ]
        origin = carla.Transform(
            carla.Location(x=transforms[0].location.x, y=transforms[0].location.y,
                            z=transforms[0].location.z + 0.3),  # lift clear of the road mesh
            transforms[0].rotation)
        route_locations = [transforms[0].location]
        for a, b in zip(transforms[:-1], transforms[1:]):
            try:
                seg = grp.trace_route(a.location, b.location)
            except Exception:
                continue
            route_locations.extend(wp.transform.location for wp, _ in seg)
    else:
        origin = spawn_points[route_spec["spawn_idx"]]
        dest = spawn_points[route_spec["dest_idx"]]
        route = grp.trace_route(origin.location, dest.location)
        route_locations = [wp.transform.location for wp, _ in route]

    metrics = DrivingMetrics(
        route_id=route_spec["route_id"],
        total_route_points=len(route_locations),
        route_length_m=route_spec.get("length_m", 0.0),
        blocked_timeout_s=args.blocked_timeout_s,
    )

    try:
        ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ego_bp.set_attribute("role_name", "hero")
        ego = world.try_spawn_actor(ego_bp, origin)
        if ego is None:
            metrics.terminate(TerminationReason.ERROR)
            return metrics.to_dict()
        actors.append(ego)

        collision_events = []
        col_bp = bp_lib.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        actors.append(col_sensor)
        col_sensor.listen(lambda e: collision_events.append(
            (e.other_actor.type_id, e.other_actor.id)))

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "256")
        cam_bp.set_attribute("image_size_y", "256")
        cam_bp.set_attribute("fov", "100")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.3, z=1.3)),
                                attach_to=ego)
        actors.append(cam)
        # BGRA -> true RGB via an explicit channel swap. A plain [:, :, :3] alpha-drop
        # leaves the frame in BGR and the policy was trained on RGB; eval_wor.py carries
        # the same note because that exact bug produced degenerate steering once already.
        rgb_buf = {"data": None}
        cam.listen(lambda img: rgb_buf.update({
            "data": np.frombuffer(img.raw_data, dtype=np.uint8)
                      .reshape((256, 256, 4))[:, :, [2, 1, 0]]}))

        if record_video:
            video_dir = args.video_dir or (os.path.splitext(args.out)[0] + "_videos")
            recorder = RouteVideoRecorder(
                world, ego,
                path=os.path.join(video_dir, f"{route_spec['route_id']}.mp4"),
                route_id=route_spec["route_id"])
            actors.extend(recorder.actors())

        # Settle under gravity before scoring, so an interpenetrating spawn is not
        # charged to the policy as a collision.
        for _ in range(20):
            world.tick()
        collision_events.clear()
        agent.destroy()  # resets the PID's internal state between routes

        world_map = world.get_map()
        red_watcher = RedLightWatcher()
        route_idx = 0
        t_s = 0.0

        for step in range(1, args.max_steps + 1):
            world.tick()
            t_s = step * 0.05

            if rgb_buf["data"] is None:
                continue

            tf = ego.get_transform()
            vel = ego.get_velocity()
            speed_mps = float(np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))

            ego_route, route_idx, reached_end = _ego_frame_route(ego, route_locations, route_idx)

            off_road = world_map.get_waypoint(
                tf.location, project_to_road=False,
                lane_type=carla.LaneType.Driving) is None

            metrics.tick(t_s=t_s, speed_mps=speed_mps, route_idx=route_idx, off_road=off_road)

            while collision_events:
                type_id, actor_id = collision_events.pop(0)
                metrics.add_collision(t_s, type_id, actor_id)

            ran = red_watcher.update(ego, speed_mps)
            if ran is not None:
                metrics.add_red_light_violation(t_s, ran)

            if reached_end:
                metrics.terminate(TerminationReason.COMPLETED)
                break
            if metrics.is_blocked():
                metrics.terminate(TerminationReason.BLOCKED)
                break
            if tf.location.distance(route_locations[route_idx]) > MAX_ROUTE_DEVIATION_M:
                metrics.terminate(TerminationReason.ROUTE_DEVIATION)
                break

            control = agent.run_step({
                "rgb_front": (step, rgb_buf["data"]),
                "speed": (step, speed_mps),
                "command": WOR_LANEFOLLOW_COMMAND,
                "route": ego_route,
            })
            ego.apply_control(control)

            if recorder is not None:
                recorder.capture(
                    speed_kmh=speed_mps * 3.6,
                    target_speed_kmh=agent.net.controller.target_speed,
                    steer=control.steer, throttle=control.throttle, brake=control.brake,
                    step=step, max_steps=args.max_steps,
                    command_name="LANEFOLLOW",
                    route_progress_pct=100.0 * metrics.raw_route_completion,
                    collisions=(metrics.to_dict()["collisions_vehicle"]
                                + metrics.to_dict()["collisions_static"]
                                + metrics.to_dict()["collisions_pedestrian"]),
                    elapsed_s=t_s,
                    driving_score=metrics.driving_score,
                )

        record = metrics.to_dict()
        if recorder is not None:
            written = recorder.close()
            if written:
                record["video"] = written
        return record

    finally:
        # Sensors must be stopped before they are destroyed. A listening sensor delivers
        # its callback on a C++ worker thread, and one that fires against an actor which
        # has just been destroyed raises an exception nowhere Python can catch it - it
        # reaches std::terminate and takes the process down after the run is otherwise
        # complete, which is exactly how the first working rollout ended.
        for a in actors:
            try:
                if getattr(a, "is_listening", False):
                    a.stop()
            except Exception:
                pass
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass


def main():
    args = parse_args()
    if args.route_source == "official" and not args.route_file:
        raise SystemExit("--route_file is required when --route_source official")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("=" * 70)
    print(" Closed-loop CARLA evaluation")
    print(f" Checkpoint : {args.checkpoint}")
    print(f" Arch       : {args.policy_arch} | Town: {args.town}")
    print(f" Routes     : {args.routes} (route_seed {args.route_seed})")
    print("=" * 70)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if args.town not in world.get_map().name:
        print(f"--> Loading {args.town} ...")
        world = client.load_world(args.town)
        # The map streams in asynchronously. Spawning into it before it has settled is
        # a reliable way to segfault the server, so give it real time in the async mode
        # it is still in - a tick here would block on a synchronous mode not yet set.
        time.sleep(5.0)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.tick()

    traffic_actors, tm = [], None
    try:
        if args.route_source == "official":
            manifest = load_official_routes(args.route_file, args.town, args.routes,
                                             args.route_seed)
            print(f"✓ Official route manifest: {len(manifest)} routes from "
                  f"{args.route_file} (town {args.town}), "
                  f"mean length {np.mean([r['length_m'] for r in manifest]):.0f} m "
                  f"(straight-segment estimate, not the traced driving distance)")
        else:
            manifest = build_route_manifest(world, args.routes, args.route_seed)
            print(f"✓ Route manifest: {len(manifest)} routes, "
                  f"mean length {np.mean([r['length_m'] for r in manifest]):.0f} m")

        tm_port = args.tm_port if args.tm_port else (args.port + 8000)
        traffic_actors, tm = spawn_traffic(world, client, args.num_vehicles,
                                           args.num_walkers, args.route_seed, tm_port)
        print(f"✓ Traffic: {len(traffic_actors)} actors (tm seed {args.route_seed}, "
              f"tm_port {tm_port})")

        agent = WorldOnRailsAgent(
            checkpoint_path=args.checkpoint,
            backbone_name=args.backbone,
            device=args.device,
            policy_arch=args.policy_arch,
            route_points=WOR_ROUTE_POINTS,
        )

        spawn_points = world.get_map().get_spawn_points()
        grp = GlobalRoutePlanner(world.get_map(), ROUTE_SAMPLING_RESOLUTION)

        records = []
        t_start = time.time()
        for k, spec in enumerate(manifest, 1):
            rec = run_route(world, client, agent, spec, spawn_points, grp, args, k,
                            record_video=bool(args.record_video) and k <= args.video_routes)
            records.append(rec)
            print(f"  [{k:03d}/{len(manifest)}] {spec['route_id']}  "
                  f"DS={rec['driving_score']:.3f}  RC={rec['route_completion']:.3f}  "
                  f"IP={rec['infraction_penalty']:.3f}  {rec['termination_reason']}")

        summary = aggregate(records)
        payload = {
            "checkpoint": args.checkpoint,
            "policy_arch": args.policy_arch,
            "town": args.town,
            "route_source": args.route_source,
            "route_file": args.route_file,
            "route_seed": args.route_seed,
            "num_vehicles": args.num_vehicles,
            "num_walkers": args.num_walkers,
            "max_steps": args.max_steps,
            "wall_time_s": round(time.time() - t_start, 1),
            "summary": summary,
            "routes": records,
            "manifest": manifest,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)

        print("-" * 70)
        print(f"  Driving Score    : {summary['driving_score']:.4f}")
        print(f"  Route Completion : {summary['route_completion']:.4f}")
        print(f"  Infraction Pen.  : {summary['infraction_penalty']:.4f}")
        print(f"  Terminations     : {summary['termination_reasons']}")
        print(f"✓ Wrote {args.out}")

    finally:
        # Order matters: drop the traffic manager's synchronous hold and return the world
        # to asynchronous mode *before* removing actors, so nothing is waiting on a tick
        # that will never come while the batch destroy runs.
        if tm is not None:
            try:
                tm.set_synchronous_mode(False)
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        if traffic_actors:
            try:
                client.apply_batch([carla.command.DestroyActor(a) for a in traffic_actors])
            except Exception:
                for a in traffic_actors:
                    try:
                        a.destroy()
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
