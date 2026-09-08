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


def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop CARLA evaluation of a WoR policy")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth to evaluate")
    p.add_argument("--policy_arch", type=str, default="qwen30m",
                   choices=["cnn", "qwen10m", "qwen30m", "qwen100m", "qwen500m", "qwen900m"],
                   help="Must match the --policy_arch this checkpoint was trained with")
    p.add_argument("--backbone", type=str, default="resnet34")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", type=str, default="Town01")
    p.add_argument("--routes", type=int, default=20,
                   help="Number of routes to drive. Routes are the independent unit for the "
                        "paired bootstrap, so this - not episode length - is what sets the "
                        "resolution of the comparison")
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


def spawn_traffic(world, client, num_vehicles: int, num_walkers: int, seed: int):
    """Spawns background traffic under a fixed traffic-manager seed.

    Without `set_random_device_seed` the NPC vehicles behave differently on every run,
    which would put a different obstacle in front of each checkpoint and reintroduce
    exactly the variance the fixed route manifest exists to remove.
    """
    actors = []
    tm = client.get_trafficmanager()
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


def run_route(world, client, agent, route_spec, spawn_points, grp, args,
              record_video: bool = False) -> Dict:
    """Drives one route and returns its metrics record."""
    bp_lib = world.get_blueprint_library()
    actors = []
    recorder = None
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
        manifest = build_route_manifest(world, args.routes, args.route_seed)
        print(f"✓ Route manifest: {len(manifest)} routes, "
              f"mean length {np.mean([r['length_m'] for r in manifest]):.0f} m")

        traffic_actors, tm = spawn_traffic(world, client, args.num_vehicles,
                                           args.num_walkers, args.route_seed)
        print(f"✓ Traffic: {len(traffic_actors)} actors (tm seed {args.route_seed})")

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
            rec = run_route(world, client, agent, spec, spawn_points, grp, args,
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
