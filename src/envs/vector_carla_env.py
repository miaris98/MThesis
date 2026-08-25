"""Vectorized Multi-CARLA Server Gymnasium Environment with process-level isolation."""
import os
import sys
import time
import warnings

# Completely silence runtime, deprecation, and gymnasium warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from typing import List, Callable, Dict, Any, Tuple, Optional
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection

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

from src.envs.camera_easycarla_env import CameraEasyCarlaEnv
from src.envs.carla_gym_env import CarlaGymEnv
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
            print(f"⚠️ [Worker {worker_id} Port {getattr(env_factory, 'port', 'unknown')} Attempt {attempt+1}/10 Error]: {e}", flush=True)
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


class CarlaEnvFactory:
    """Picklable top-level factory creating an isolated CARLA environment instance."""
    def __init__(self, cfg: TrainingConfig, port: int):
        self.cfg = cfg
        self.port = port

    def __call__(self) -> Any:
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
            }
            return CameraEasyCarlaEnv(params=easy_params)
        return CarlaGymEnv(host=self.cfg.host, port=self.port, img_width=256, img_height=256, max_steps=self.cfg.rollout_steps)


def create_vector_carla_env(cfg: TrainingConfig) -> Any:
    """Instantiate vectorized multi-server environment based on configuration."""
    ports = cfg.get_ports()
    print(f"--> Initializing {len(ports)} Parallel CARLA Environment Workers on ports: {ports}")
    factories = [CarlaEnvFactory(cfg, p) for p in ports]
    
    if len(ports) > 1:
        return SubprocCarlaVectorEnv(factories)
    return DummyCarlaVectorEnv(factories)
