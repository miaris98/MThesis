"""Vectorized Multi-CARLA Server Gymnasium Environment with process-level isolation."""
import os
import sys
import time
import warnings

# Completely silence runtime, deprecation, and gymnasium warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from typing import List, Callable, Dict, Any, Tuple, Optional
from dataclasses import replace
import random
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection
from concurrent.futures import ThreadPoolExecutor

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = object
        spaces = object

try:
    import carla
except ImportError:
    carla = None

from src.envs.camera_easycarla_env import CameraEasyCarlaEnv
from src.envs.carla_gym_env import CarlaGymEnv
from src.envs.base_env import wait_for_carla_server, safe_clear_owned_actors
from src.config.training_config import TrainingConfig


def _carla_worker(remote: Any, parent_remote: Any, env_factory: Any, worker_id: int = 0) -> None:
    """Worker process loop communicating with a single dedicated CARLA server instance."""
    import os
    import time
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    parent_remote.close()

    # Stagger connection to avoid port hammering
    time.sleep(worker_id * 1.5)

    env = None
    for attempt in range(10):
        try:
            env = env_factory()
            break
        except Exception as e:
            if attempt == 9:
                import traceback
                print(f"[Worker {worker_id} Fatal Init Error] {e}\n{traceback.format_exc()}", flush=True)
                raise
            time.sleep(2.0)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                action = data
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if done:
                    # Save terminal observation and execute seamless in-place auto-reset
                    info["terminal_observation"] = obs
                    obs, reset_info = env.reset()
                    info["reset_info"] = reset_info
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                seed = data.get("seed", None) if isinstance(data, dict) else None
                options = data.get("options", None) if isinstance(data, dict) else None
                obs, info = env.reset(seed=seed, options=options)
                remote.send((obs, info))
            elif cmd == "set_curriculum_factor":
                factor = data
                if hasattr(env, "set_curriculum_factor"):
                    env.set_curriculum_factor(factor)
                remote.send(True)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command received by worker: {cmd}")
    except Exception as e:
        import traceback
        print(f"[Worker {worker_id} Error] {e}\n{traceback.format_exc()}", flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass


class SubprocCarlaVectorEnv:
    """
    Subprocess-based Vectorized CARLA Environment.
    Runs each CARLA server client in its own dedicated Python process to eliminate GIL
    bottlenecks and prevent server tick latency jitter between instances.
    """
    def __init__(self, env_factories: List[Any]):
        self.num_envs = len(env_factories)
        self.closed = False
        
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [
            ctx.Process(target=_carla_worker, args=(work_remote, remote, factory, i), daemon=True)
            for i, (work_remote, remote, factory) in enumerate(zip(self.work_remotes, self.remotes, env_factories))
        ]
        
        for p in self.ps:
            p.start()
        for work_remote in self.work_remotes:
            work_remote.close()

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        """Reset all environments in parallel and return batched observations."""
        for i, remote in enumerate(self.remotes):
            env_seed = seed + i if seed is not None else None
            remote.send(("reset", {"seed": env_seed}))
        
        results = [remote.recv() for remote in self.remotes]
        obs_list, info_list = zip(*results)
        
        batched_obs = {
            "image": np.stack([o["image"] for o in obs_list]),
            "speed": np.stack([o["speed"] for o in obs_list])
        }
        return batched_obs, list(info_list)

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments in parallel with batched actions (num_envs, action_dim)."""
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        
        results = [remote.recv() for remote in self.remotes]
        obs_list, rew_list, term_list, trunc_list, info_list = zip(*results)
        
        batched_obs = {
            "image": np.stack([o["image"] for o in obs_list]),
            "speed": np.stack([o["speed"] for o in obs_list])
        }
        rewards = np.array(rew_list, dtype=np.float32)
        terminated = np.array(term_list, dtype=bool)
        truncated = np.array(trunc_list, dtype=bool)
        
        return batched_obs, rewards, terminated, truncated, list(info_list)

    def set_curriculum_factor(self, factor: float) -> None:
        """Broadcast curriculum factor update to all environment workers."""
        for remote in self.remotes:
            remote.send(("set_curriculum_factor", factor))
        for remote in self.remotes:
            remote.recv()

    def close(self) -> None:
        """Clean up all worker processes and close IPC connections."""
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


class DummyCarlaVectorEnv:
    """Single-process vectorized wrapper for single environment execution or debugging."""
    def __init__(self, env_factories: List[Any]):
        self.envs = [factory() for factory in env_factories]
        self.num_envs = len(self.envs)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        obs_list, info_list = [], []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)
        return {
            "image": np.stack([o["image"] for o in obs_list]),
            "speed": np.stack([o["speed"] for o in obs_list])
        }, info_list

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                info["terminal_observation"] = obs
                obs, reset_info = env.reset()
                info["reset_info"] = reset_info
            obs_list.append(obs)
            rew_list.append(reward)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return (
            {
                "image": np.stack([o["image"] for o in obs_list]),
                "speed": np.stack([o["speed"] for o in obs_list])
            },
            np.array(rew_list, dtype=np.float32),
            np.array(term_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            info_list
        )

    def set_curriculum_factor(self, factor: float) -> None:
        for env in self.envs:
            if hasattr(env, "set_curriculum_factor"):
                env.set_curriculum_factor(factor)

    def close(self) -> None:
        for env in self.envs:
            env.close()


class SharedServerCarlaVectorEnv:
    """
    Single-GPU vectorized CARLA environment: ONE CARLA server/world, N vehicle actors
    sharing it, driven by one coordinator that owns synchronous-mode/tick. Replaces
    SubprocCarlaVectorEnv's N-separate-server-process model on single-GPU machines,
    where N independent UE4 processes each duplicating map/asset load + holding their
    own GPU context saturate the rasterizer well before raw compute is the bottleneck.
    Runs entirely in-process (no subprocess workers): slots share a small, capped pool
    of dedicated carla.Client connections (round-robin, see shared_server_max_connections),
    and each connection's group of slots is dispatched to its own worker thread via
    ThreadPoolExecutor (CPython releases the GIL during blocking socket I/O, so this
    genuinely parallelizes round-trip latency across groups) - only world.tick() itself
    stays a single sequential call owned by this coordinator's own connection.
    """

    def __init__(self, cfg: TrainingConfig, port: int):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.frame_skip = cfg.frame_skip
        self.closed = False
        # Lightweight step()-phase timing breakdown, printed periodically - apply/tick/
        # read isolate CARLA-side cost, "other" catches this method's own aggregation
        # overhead (np.stack etc). Not for anything outside this method (e.g. the PPO
        # forward pass in ppo_trainer.py happens before step() is even called).
        self._prof_apply = 0.0
        self._prof_tick = 0.0
        self._prof_read = 0.0
        self._prof_wall = 0.0
        self._prof_calls = 0
        self._prof_stale_max = 0
        self._prof_stale_sum = 0.0
        self._prof_stale_n = 0
        self._prof_reset_stale_max = 0
        self._prof_reset_stale_sum = 0.0
        self._prof_reset_stale_n = 0
        self._prof_resets = 0
        # Matches CameraEasyCarlaEnv._apply_sub_action's action mapping: throttle =
        # (a0+1)/2 -> 0.0, steer = a1 -> 0.0, brake gated on (a2>0.4 and throttle<0.3) -> 1.0.
        self._brake_action = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        self._npc_owned_ids: List[int] = []

        wait_for_carla_server(port, max_wait=60)
        self.client = carla.Client('127.0.0.1', port)
        self.client.set_timeout(120.0)
        try:
            curr_world = self.client.get_world()
            if cfg.town.lower() in curr_world.get_map().name.lower():
                self.world = curr_world
            else:
                self.world = self.client.load_world(cfg.town)
        except Exception:
            self.world = self.client.load_world(cfg.town)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = float(getattr(cfg, "sim_dt", 0.05))
        self.world.apply_settings(self.settings)

        all_spawns = list(self.world.get_map().get_spawn_points())
        random.shuffle(all_spawns)
        spawn_chunks = [all_spawns[i::self.num_envs] for i in range(self.num_envs)]

        self._spawn_shared_npc_pool()

        # Slots share a small, capped pool of dedicated carla.Client connections
        # (round-robin), rather than either one connection per slot or one shared
        # connection for all. One-per-slot hit a hard ceiling in practice (confirmed
        # empirically: 4 dedicated connections is fine, 20 fails with "Resource
        # temporarily unavailable" on the bare carla.Client(...) constructor - some
        # resource ceiling below 20, cause unconfirmed, possibly CARLA-server-side or
        # a container-level limit invisible to `ulimit`). A single shared connection
        # would serialize every RPC at the socket layer regardless of Python-level
        # threading. This pool gets most of the latency-hiding benefit (each
        # connection's group of slots still parallelizes against every OTHER
        # connection's group) while capping how many raw connections/threads get
        # opened - safe at any num_envs. Each connection's own slots are only ever
        # touched sequentially, from a single dedicated worker thread, never
        # concurrently, so no serialization is reintroduced within a group.
        num_connections = max(1, min(self.num_envs, int(getattr(cfg, 'shared_server_max_connections', 8))))
        self._groups: List[List[int]] = [
            [i for i in range(self.num_envs) if i % num_connections == g]
            for g in range(num_connections)
        ]

        self.slots: List[Any] = [None] * self.num_envs
        for group in self._groups:
            factory = CarlaEnvFactory(cfg, port)
            # Opening many client connections back-to-back can transiently overrun
            # CARLA's RPC server accept path ("Resource temporarily unavailable"),
            # so retry with backoff and stagger successive connections slightly.
            group_client = None
            for attempt in range(10):
                try:
                    group_client = carla.Client('127.0.0.1', port)
                    group_client.set_timeout(120.0)
                    _ = group_client.get_server_version()
                    break
                except Exception:
                    if attempt == 9:
                        raise
                    time.sleep(0.5)
            group_world = group_client.get_world()  # attaches to the already-loaded world
            time.sleep(0.05)
            for i in group:
                params_override = {
                    'shared_mode': True,
                    'external_client': group_client,
                    'external_world': group_world,
                    'spawn_point_subset': spawn_chunks[i] if spawn_chunks[i] else all_spawns,
                    'number_of_vehicles': 0,
                    'number_of_walkers': 0,
                }
                self.slots[i] = factory(params_override=params_override)

        # world.tick() stays a single sequential call owned by self.client - only the
        # per-group actor RPCs (apply control, read state, reset) are parallelized,
        # one worker thread per connection group.
        self._pool = ThreadPoolExecutor(max_workers=num_connections, thread_name_prefix="carla-slot-group")

    def _spawn_shared_npc_pool(self) -> None:
        """Spawn background traffic exactly once for the whole shared world, instead of
        once per vehicle-slot per episode reset (which would thrash the actor pool and
        respawn traffic every ~15-25 steps)."""
        num_vehicles = self.cfg.num_vehicles
        num_walkers = self.cfg.num_walkers

        if num_vehicles > 0:
            spawn_points = list(self.world.get_map().get_spawn_points())
            random.shuffle(spawn_points)
            count = num_vehicles
            for sp in spawn_points:
                if count <= 0:
                    break
                vehicle = self._try_spawn_npc_vehicle(sp)
                if vehicle is not None:
                    self._npc_owned_ids.append(vehicle.id)
                    vehicle.set_autopilot()
                    count -= 1

        if num_walkers > 0:
            walker_spawns = []
            for _ in range(num_walkers):
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    walker_spawns.append(carla.Transform(loc))
            count = num_walkers
            for sp in walker_spawns:
                if count <= 0:
                    break
                if self._try_spawn_npc_walker(sp):
                    count -= 1

        # Freeze traffic lights once for the whole shared world - each vehicle-slot's
        # own reset() skips this in shared mode (see CarlaEnv.reset()).
        for actor in self.world.get_actors().filter('traffic.traffic_light*'):
            actor.set_state(carla.TrafficLightState.Green)
            actor.freeze(True)

    def _try_spawn_npc_vehicle(self, transform: Any) -> Optional[Any]:
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprint_library = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        if not blueprint_library:
            return None
        blueprint = random.choice(blueprint_library)
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
        blueprint.set_attribute('role_name', 'autopilot')
        return self.world.try_spawn_actor(blueprint, transform)

    def _try_spawn_npc_walker(self, transform: Any) -> bool:
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
        if walker_actor is None:
            return False
        self._npc_owned_ids.append(walker_actor.id)
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        controller_actor = self.world.spawn_actor(controller_bp, carla.Transform(), walker_actor)
        self._npc_owned_ids.append(controller_actor.id)
        controller_actor.start()
        controller_actor.go_to_location(self.world.get_random_location_from_navigation())
        controller_actor.set_max_speed(1 + random.random())
        return True

    # Each of these processes one connection GROUP - i.e. one dedicated carla.Client -
    # sequentially over that group's slots. They're dispatched one-per-worker-thread via
    # self._pool, so different groups' RPCs genuinely run in parallel (separate
    # connections), while calls within a single group never overlap (same connection,
    # same thread).
    def _apply_group(self, group: List[int], actions: List[np.ndarray]) -> None:
        for idx, action in zip(group, actions):
            self.slots[idx]._apply_sub_action(action)

    def _read_group(self, group: List[int]) -> List[Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]]:
        return [self.slots[idx]._read_sub_result() for idx in group]

    def _begin_group_reset(self, group: List[int], seeds: List[Optional[int]]) -> None:
        for idx, s in zip(group, seeds):
            self.slots[idx].begin_reset(seed=s)

    def _finish_group_reset(self, group: List[int]) -> List[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        return [self.slots[idx].finish_reset() for idx in group]

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        seeds = [(seed + i if seed is not None else None) for i in range(self.num_envs)]
        seeds_per_group = [[seeds[idx] for idx in group] for group in self._groups]
        list(self._pool.map(self._begin_group_reset, self._groups, seeds_per_group))
        self.world.tick()

        group_results = list(self._pool.map(self._finish_group_reset, self._groups))
        obs_list: List[Any] = [None] * self.num_envs
        info_list: List[Any] = [None] * self.num_envs
        for group, results in zip(self._groups, group_results):
            for idx, (obs, info) in zip(group, results):
                obs_list[idx] = obs
                info_list[idx] = info

        return {
            "image": np.stack([o["image"] for o in obs_list]),
            "speed": np.stack([o["speed"] for o in obs_list])
        }, info_list

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        _step_t0 = time.time()
        n = self.num_envs
        total_reward = np.zeros(n, dtype=np.float32)
        total_cost = np.zeros(n, dtype=np.float32)
        final_obs: List[Any] = [None] * n
        final_info: List[Any] = [None] * n
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)

        for _ in range(self.frame_skip):
            # Already-done slots keep braking rather than coasting on their last
            # throttle, so they don't drift into a neighboring slot's still-active
            # ego during the remaining ticks this shared world must still advance for.
            effective_actions = [
                self._brake_action if (terminated[i] or truncated[i]) else actions[i]
                for i in range(n)
            ]
            actions_per_group = [[effective_actions[idx] for idx in group] for group in self._groups]
            _t0 = time.time()
            list(self._pool.map(self._apply_group, self._groups, actions_per_group))

            _t1 = time.time()
            self.world.tick()  # single tick shared across every group's sub-step

            _t2 = time.time()
            group_results = list(self._pool.map(self._read_group, self._groups))
            _t3 = time.time()
            self._prof_apply += _t1 - _t0
            self._prof_tick += _t2 - _t1
            self._prof_read += _t3 - _t2
            for group, results in zip(self._groups, group_results):
                for idx, (obs, reward, term, trunc, info) in zip(group, results):
                    if terminated[idx] or truncated[idx]:
                        continue  # already done this outer step; result discarded above
                    total_reward[idx] += reward
                    total_cost[idx] += info.get("cost", 0.0)
                    final_obs[idx] = obs
                    final_info[idx] = info
                    if term or trunc:
                        terminated[idx] = term
                        truncated[idx] = trunc

        # Observation staleness check. With sensor_tick = dt * frame_skip the cameras only
        # render once per step() call. If that render lands on the final sub-step the obs
        # just read is current (gap 0); if a reset shifts the phase onto an earlier
        # sub-step, every returned obs is silently a tick or more old. Sampled here, before
        # the reset block below ticks the world again and muddies the frame numbers.
        try:
            world_frame = self.world.get_snapshot().frame
            gaps = [
                world_frame - slot.sensor_mgr.last_capture_frame
                for slot in self.slots
                if getattr(slot, "sensor_mgr", None) is not None
                and slot.sensor_mgr.last_capture_frame > 0
            ]
            if gaps:
                self._prof_stale_max = max(self._prof_stale_max, max(gaps))
                self._prof_stale_sum += sum(gaps) / len(gaps)
                self._prof_stale_n += 1
        except Exception:
            pass

        for i in range(n):
            final_info[i]["cost"] = float(total_cost[i])
            final_info[i]["frame_skip"] = self.frame_skip

        reset_idxs = [i for i in range(n) if terminated[i] or truncated[i]]
        if reset_idxs:
            self._prof_resets += 1
            reset_idx_set = set(reset_idxs)
            reset_groups = [[idx for idx in group if idx in reset_idx_set] for group in self._groups]
            reset_groups = [g for g in reset_groups if g]
            for i in reset_idxs:
                final_info[i]["terminal_observation"] = final_obs[i]
            list(self._pool.map(self._begin_group_reset, reset_groups, [[None] * len(g) for g in reset_groups]))
            self.world.tick()
            reset_group_results = list(self._pool.map(self._finish_group_reset, reset_groups))
            for group, results in zip(reset_groups, reset_group_results):
                for idx, (obs, reset_info) in zip(group, results):
                    final_obs[idx] = obs
                    final_info[idx]["reset_info"] = reset_info

            # Same staleness question for the reset path, which is the riskier one: the
            # single settle tick above only yields a fresh frame if it happens to be a
            # render tick under sensor_tick = dt * frame_skip. If it isn't, a just-respawned
            # vehicle's first observation is the image from where the previous episode
            # ended - wrong position entirely, not merely one tick late.
            try:
                world_frame = self.world.get_snapshot().frame
                reset_gaps = [
                    world_frame - self.slots[idx].sensor_mgr.last_capture_frame
                    for group in reset_groups for idx in group
                    if getattr(self.slots[idx], "sensor_mgr", None) is not None
                    and self.slots[idx].sensor_mgr.last_capture_frame > 0
                ]
                if reset_gaps:
                    self._prof_reset_stale_max = max(self._prof_reset_stale_max, max(reset_gaps))
                    self._prof_reset_stale_sum += sum(reset_gaps) / len(reset_gaps)
                    self._prof_reset_stale_n += 1
            except Exception:
                pass

        batched_obs = {
            "image": np.stack([o["image"] for o in final_obs]),
            "speed": np.stack([o["speed"] for o in final_obs])
        }

        self._prof_wall += time.time() - _step_t0
        self._prof_calls += 1
        if self._prof_calls % 50 == 0:
            wall = max(self._prof_wall, 1e-9)
            other = max(0.0, wall - self._prof_apply - self._prof_tick - self._prof_read)
            print(
                f"[PROFILE last {self._prof_calls} step() calls] "
                f"apply={self._prof_apply*1000:.0f}ms({100*self._prof_apply/wall:.0f}%) "
                f"tick={self._prof_tick*1000:.0f}ms({100*self._prof_tick/wall:.0f}%) "
                f"read={self._prof_read*1000:.0f}ms({100*self._prof_read/wall:.0f}%) "
                f"other={other*1000:.0f}ms({100*other/wall:.0f}%) "
                f"wall={wall*1000:.0f}ms "
                f"obs_stale(avg/max ticks)={self._prof_stale_sum/max(self._prof_stale_n,1):.2f}/{self._prof_stale_max} "
                f"reset_stale={self._prof_reset_stale_sum/max(self._prof_reset_stale_n,1):.2f}/{self._prof_reset_stale_max} "
                f"reset_ticks={self._prof_resets}/50",
                flush=True,
            )
            self._prof_apply = self._prof_tick = self._prof_read = self._prof_wall = 0.0
            self._prof_stale_max = 0
            self._prof_stale_sum = 0.0
            self._prof_stale_n = 0
            self._prof_reset_stale_max = 0
            self._prof_reset_stale_sum = 0.0
            self._prof_reset_stale_n = 0
            self._prof_resets = 0

        return batched_obs, total_reward, terminated, truncated, final_info

    def set_curriculum_factor(self, factor: float) -> None:
        for slot in self.slots:
            if hasattr(slot, "set_curriculum_factor"):
                slot.set_curriculum_factor(factor)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._pool.shutdown(wait=True)
        for slot in self.slots:
            try:
                slot.close()
            except Exception:
                pass
        safe_clear_owned_actors(self.world, self.client, self._npc_owned_ids)
        try:
            self.settings.synchronous_mode = False
            self.world.apply_settings(self.settings)
        except Exception:
            pass


class CarlaEnvFactory:
    """Picklable top-level factory creating an isolated CARLA environment instance."""
    def __init__(self, cfg: TrainingConfig, port: int):
        self.cfg = cfg
        self.port = port

    def __call__(self, params_override: Optional[Dict[str, Any]] = None) -> Any:
        if self.cfg.env_type == "camera_easycarla":
            easy_params = {
                'number_of_vehicles': self.cfg.num_vehicles,
                'number_of_walkers': self.cfg.num_walkers,
                'frame_skip': self.cfg.frame_skip,
                'dt': float(getattr(self.cfg, 'sim_dt', 0.05)),
                'ego_vehicle_filter': 'vehicle.tesla.model3',
                'surrounding_vehicle_spawned_randomly': True,
                'port': self.port,
                'town': self.cfg.town,
                'max_time_episode': self.cfg.rollout_steps * self.cfg.frame_skip,
                'max_waypoints': 12,
                'visualize_waypoints': False,
                'desired_speed': 8,
                'max_ego_spawn_times': 200,
                'view_mode': 'top',
                'traffic': 'off',
                'lidar_max_range': 50.0,
                'max_nearby_vehicles': 5,
                'img_width': self.cfg.img_width,
                'img_height': self.cfg.img_height,
                'reward_fn': getattr(self.cfg, 'reward_fn', 'custom_1'),
            }
            if params_override:
                easy_params.update(params_override)
            return CameraEasyCarlaEnv(params=easy_params)
        return CarlaGymEnv(host=self.cfg.host, port=self.port, img_width=self.cfg.img_width, img_height=self.cfg.img_height, max_steps=self.cfg.rollout_steps)


class MultiServerSharedCarlaVectorEnv:
    """
    K independent CARLA servers on one GPU, each hosting num_envs/K vehicle-actor slots.

    Motivation: profiling a single shared server showed GPU utilization at ~22% while
    world.tick() accounted for ~75-85% of wall time. The tick is a serialized dispatch
    chain (the server walks each camera sensor in turn, CPU-side setup plus a round-trip
    per sensor) rather than GPU-bound work, so the card sits idle waiting. Splitting the
    slots across separate server processes gives the machine K independent chains that
    can genuinely progress at once.

    The critical detail is that the coordinators must be stepped CONCURRENTLY. Each owns
    its own client, world and tick, so they are already independent - but calling their
    step()s back to back would cost tickA + tickB, exactly what one server doing all the
    work costs today, defeating the entire point. They are therefore driven from a thread
    pool: world.tick() blocks on socket I/O to a separate server process and CPython
    releases the GIL for the duration, so the two waits overlap in wall-clock time.

    VRAM is the limiting factor, not compute: each server loads its own copy of the map
    and engine (~5.3GB observed on Town10HD_Opt), which is why 3 servers collapsed on a
    12GB card - 3 x 5.3GB oversubscribes it and thrashes. Camera render targets are
    negligible by comparison (~4MB total at 128px), so slots-per-server barely affects
    the footprint; the server COUNT is what has to fit.
    """

    def __init__(self, cfg: TrainingConfig, ports: List[int]):
        self.cfg = cfg
        self.num_servers = len(ports)
        self.num_envs = cfg.num_envs
        self.frame_skip = cfg.frame_skip
        self.closed = False

        if self.num_servers < 1:
            raise ValueError("MultiServerSharedCarlaVectorEnv requires at least one port")
        if cfg.num_envs < self.num_servers:
            raise ValueError(
                f"num_envs ({cfg.num_envs}) must be >= number of servers ({self.num_servers})"
            )

        # Distribute slots as evenly as possible; the first `remainder` servers take one extra.
        base, remainder = divmod(cfg.num_envs, self.num_servers)
        counts = [base + (1 if i < remainder else 0) for i in range(self.num_servers)]

        self._slices: List[Tuple[int, int]] = []
        start = 0
        for count in counts:
            self._slices.append((start, start + count))
            start += count

        self.coordinators: List[SharedServerCarlaVectorEnv] = []
        for port, count in zip(ports, counts):
            print(f"--> Initializing shared CARLA server on port {port} with {count} vehicle-envs")
            sub_cfg = replace(cfg, num_envs=count)
            self.coordinators.append(SharedServerCarlaVectorEnv(sub_cfg, port=port))

        self._pool = ThreadPoolExecutor(
            max_workers=self.num_servers, thread_name_prefix="carla-server"
        )

    def _merge(self, per_server: List[Tuple[Dict[str, np.ndarray], Any]]) -> Tuple[Dict[str, np.ndarray], List[Any]]:
        """Concatenate per-coordinator batched observations back into one batch."""
        images = np.concatenate([obs["image"] for obs, _ in per_server], axis=0)
        speeds = np.concatenate([obs["speed"] for obs, _ in per_server], axis=0)
        infos: List[Any] = []
        for _, info_list in per_server:
            infos.extend(info_list)
        return {"image": images, "speed": speeds}, infos

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        # Offset each coordinator's seed by its slot range so the per-slot seeds stay
        # globally distinct rather than repeating the same sequence on every server.
        seeds = [
            (seed + start if seed is not None else None) for start, _ in self._slices
        ]
        results = list(self._pool.map(
            lambda pair: pair[0].reset(seed=pair[1]),
            list(zip(self.coordinators, seeds)),
        ))
        return self._merge(results)

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        per_server_actions = [actions[start:end] for start, end in self._slices]
        results = list(self._pool.map(
            lambda pair: pair[0].step(pair[1]),
            list(zip(self.coordinators, per_server_actions)),
        ))

        images = np.concatenate([r[0]["image"] for r in results], axis=0)
        speeds = np.concatenate([r[0]["speed"] for r in results], axis=0)
        rewards = np.concatenate([r[1] for r in results], axis=0)
        terminated = np.concatenate([r[2] for r in results], axis=0)
        truncated = np.concatenate([r[3] for r in results], axis=0)
        infos: List[Any] = []
        for r in results:
            infos.extend(r[4])

        return {"image": images, "speed": speeds}, rewards, terminated, truncated, infos

    def set_curriculum_factor(self, factor: float) -> None:
        for coordinator in self.coordinators:
            coordinator.set_curriculum_factor(factor)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._pool.shutdown(wait=True)
        for coordinator in self.coordinators:
            try:
                coordinator.close()
            except Exception:
                pass


def create_vector_carla_env(cfg: TrainingConfig) -> Any:
    """Instantiate vectorized environment based on configuration."""
    if getattr(cfg, "shared_server", False):
        shared_ports = cfg.get_ports() if cfg.carla_ports else [cfg.port]
        if len(shared_ports) > 1:
            print(
                f"--> Initializing {cfg.num_envs} Vehicle-Envs across "
                f"{len(shared_ports)} shared CARLA servers (ports: {shared_ports})"
            )
            return MultiServerSharedCarlaVectorEnv(cfg, ports=shared_ports)
        print(f"--> Initializing {cfg.num_envs} Parallel Vehicle-Envs on a single shared CARLA server (port {cfg.port})")
        return SharedServerCarlaVectorEnv(cfg, port=cfg.port)

    ports = cfg.get_ports()
    print(f"--> Initializing {len(ports)} Parallel CARLA Environment Workers on ports: {ports}")
    factories = [CarlaEnvFactory(cfg, p) for p in ports]

    if len(ports) > 1:
        return SubprocCarlaVectorEnv(factories)
    return DummyCarlaVectorEnv(factories)
