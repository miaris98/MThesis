"""Vectorized Multi-CARLA Server Gymnasium Environment with process-level isolation."""
import os
import sys
import time
import warnings

# Completely silence runtime, deprecation, and gymnasium warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from typing import List, Callable, Dict, Any, Tuple, Optional
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
    Runs entirely in-process (no subprocess workers): each slot has its own
    carla.Client connection to the one shared server, and per-slot apply/read RPCs are
    dispatched through a ThreadPoolExecutor (CPython releases the GIL during blocking
    socket I/O, so this genuinely parallelizes round-trip latency) - only world.tick()
    itself stays a single sequential call owned by this coordinator's own connection.
    """

    def __init__(self, cfg: TrainingConfig, port: int):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.frame_skip = cfg.frame_skip
        self.closed = False
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
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)

        all_spawns = list(self.world.get_map().get_spawn_points())
        random.shuffle(all_spawns)
        spawn_chunks = [all_spawns[i::self.num_envs] for i in range(self.num_envs)]

        self._spawn_shared_npc_pool()

        # Each slot gets its OWN carla.Client connection (CARLA supports many
        # simultaneous connections to one server) rather than sharing self.client -
        # a shared connection would serialize every RPC at the socket layer
        # regardless of how many Python threads dispatch them. With separate
        # connections, ThreadPoolExecutor genuinely parallelizes the per-slot
        # apply/read RPC round-trips below, since CPython releases the GIL during
        # blocking socket I/O.
        self.slots: List[Any] = []
        for i in range(self.num_envs):
            factory = CarlaEnvFactory(cfg, port)
            slot_client = carla.Client('127.0.0.1', port)
            slot_client.set_timeout(120.0)
            slot_world = slot_client.get_world()  # attaches to the already-loaded world
            params_override = {
                'shared_mode': True,
                'external_client': slot_client,
                'external_world': slot_world,
                'spawn_point_subset': spawn_chunks[i] if spawn_chunks[i] else all_spawns,
                'number_of_vehicles': 0,
                'number_of_walkers': 0,
            }
            self.slots.append(factory(params_override=params_override))

        # world.tick() stays a single sequential call owned by self.client - only the
        # per-slot actor RPCs (apply control, read state, reset) are parallelized.
        self._pool = ThreadPoolExecutor(max_workers=self.num_envs, thread_name_prefix="carla-slot")

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

    @staticmethod
    def _apply_slot_action(slot: Any, action: np.ndarray) -> None:
        slot._apply_sub_action(action)

    @staticmethod
    def _read_slot_result(slot: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        return slot._read_sub_result()

    @staticmethod
    def _begin_slot_reset(slot: Any, seed: Optional[int]) -> None:
        slot.begin_reset(seed=seed)

    @staticmethod
    def _finish_slot_reset(slot: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return slot.finish_reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        seeds = [(seed + i if seed is not None else None) for i in range(self.num_envs)]
        list(self._pool.map(self._begin_slot_reset, self.slots, seeds))
        self.world.tick()

        results = list(self._pool.map(self._finish_slot_reset, self.slots))
        obs_list = [obs for obs, _ in results]
        info_list = [info for _, info in results]

        return {
            "image": np.stack([o["image"] for o in obs_list]),
            "speed": np.stack([o["speed"] for o in obs_list])
        }, info_list

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
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
            list(self._pool.map(self._apply_slot_action, self.slots, effective_actions))

            self.world.tick()  # single tick shared across every slot's sub-step

            results = list(self._pool.map(self._read_slot_result, self.slots))
            for i, (obs, reward, term, trunc, info) in enumerate(results):
                if terminated[i] or truncated[i]:
                    continue  # already done this outer step; result discarded above
                total_reward[i] += reward
                total_cost[i] += info.get("cost", 0.0)
                final_obs[i] = obs
                final_info[i] = info
                if term or trunc:
                    terminated[i] = term
                    truncated[i] = trunc

        for i in range(n):
            final_info[i]["cost"] = float(total_cost[i])
            final_info[i]["frame_skip"] = self.frame_skip

        reset_idxs = [i for i in range(n) if terminated[i] or truncated[i]]
        if reset_idxs:
            reset_slots = [self.slots[i] for i in reset_idxs]
            for i in reset_idxs:
                final_info[i]["terminal_observation"] = final_obs[i]
            list(self._pool.map(self._begin_slot_reset, reset_slots, [None] * len(reset_idxs)))
            self.world.tick()
            reset_results = list(self._pool.map(self._finish_slot_reset, reset_slots))
            for i, (obs, reset_info) in zip(reset_idxs, reset_results):
                final_obs[i] = obs
                final_info[i]["reset_info"] = reset_info

        batched_obs = {
            "image": np.stack([o["image"] for o in final_obs]),
            "speed": np.stack([o["speed"] for o in final_obs])
        }
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
                'dt': 0.05,
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
                'img_width': 256,
                'img_height': 256,
                'reward_fn': getattr(self.cfg, 'reward_fn', 'custom_1'),
            }
            if params_override:
                easy_params.update(params_override)
            return CameraEasyCarlaEnv(params=easy_params)
        return CarlaGymEnv(host=self.cfg.host, port=self.port, img_width=256, img_height=256, max_steps=self.cfg.rollout_steps)


def create_vector_carla_env(cfg: TrainingConfig) -> Any:
    """Instantiate vectorized environment based on configuration."""
    if getattr(cfg, "shared_server", False):
        print(f"--> Initializing {cfg.num_envs} Parallel Vehicle-Envs on a single shared CARLA server (port {cfg.port})")
        return SharedServerCarlaVectorEnv(cfg, port=cfg.port)

    ports = cfg.get_ports()
    print(f"--> Initializing {len(ports)} Parallel CARLA Environment Workers on ports: {ports}")
    factories = [CarlaEnvFactory(cfg, p) for p in ports]

    if len(ports) > 1:
        return SubprocCarlaVectorEnv(factories)
    return DummyCarlaVectorEnv(factories)
