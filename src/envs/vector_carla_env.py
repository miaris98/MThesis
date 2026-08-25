"""Vectorized Multi-CARLA Server Gymnasium Environment with process-level isolation."""
import os
import sys
import time
from typing import List, Callable, Dict, Any, Tuple, Optional
import numpy as np
import multiprocessing as mp

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


def _carla_worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper) -> None:
    """Worker process loop communicating with a single dedicated CARLA server instance."""
    parent_remote.close()
    env = env_fn_wrapper.x()
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
        print(f"[Worker Error] {e}\n{traceback.format_exc()}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass


class CloudpickleWrapper:
    """Wrapper to enable pickling of lambda/factory callables for multiprocessing."""
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import pickle
        return pickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocCarlaVectorEnv:
    """
    Subprocess-based Vectorized CARLA Environment.
    Runs each CARLA server client in its own dedicated Python process to eliminate GIL
    bottlenecks and prevent server tick latency jitter between instances.
    """
    def __init__(self, env_fns: List[Callable[[], Any]]):
        self.num_envs = len(env_fns)
        self.closed = False
        
        ctx = mp.get_context("spawn" if sys.platform == "win32" else "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [
            ctx.Process(target=_carla_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)), daemon=True)
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
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
    def __init__(self, env_fns: List[Callable[[], Any]]):
        self.envs = [fn() for fn in env_fns]
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


def make_carla_env(cfg: TrainingConfig, port: int) -> Callable[[], Any]:
    """Factory helper creating an isolated CARLA environment instance bound to a specific port."""
    def _thunk() -> Any:
        if cfg.env_type == "camera_easycarla":
            easy_params = {
                'number_of_vehicles': cfg.num_vehicles,
                'number_of_walkers': cfg.num_walkers,
                'frame_skip': cfg.frame_skip,
                'dt': 0.05,
                'ego_vehicle_filter': 'vehicle.tesla.model3',
                'surrounding_vehicle_spawned_randomly': True,
                'port': port,
                'town': cfg.town,
                'max_time_episode': cfg.rollout_steps * cfg.frame_skip,
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
        return CarlaGymEnv(host=cfg.host, port=port, img_width=256, img_height=256, max_steps=cfg.rollout_steps)
    return _thunk


def create_vector_carla_env(cfg: TrainingConfig) -> Any:
    """Instantiate vectorized multi-server environment based on configuration."""
    ports = cfg.get_ports()
    print(f"--> Initializing {len(ports)} Parallel CARLA Environment Workers on ports: {ports}")
    env_fns = [make_carla_env(cfg, p) for p in ports]
    
    if len(ports) > 1:
        return SubprocCarlaVectorEnv(env_fns)
    return DummyCarlaVectorEnv(env_fns)
