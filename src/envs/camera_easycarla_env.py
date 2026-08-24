"""CameraEasyCarlaEnv: Gymnasium wrapper for vision-based RL with zero-latency in-place reset."""
import os
import time
import math
import random
import threading
from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        class DummySpaces:
            Box = object
            Dict = dict
        spaces = DummySpaces()
        class DummyGym:
            Env = object
            spaces = spaces
        gym = DummyGym()

try:
    import carla
except ImportError:
    carla = None

try:
    from easycarla.envs.carla_env import CarlaEnv
except ImportError:
    try:
        from CarlaEnv import CarlaEnv
    except ImportError:
        CarlaEnv = object

from src.envs.base_env import wait_for_carla_server, safe_clear_carla_actors
from src.envs.camera_sensor import CameraSensorManager
from src.envs.reward_calculator import RewardCalculator


class CameraEasyCarlaEnv(gym.Env):
    """
    Zero-latency 3-Camera Gymnasium Environment Wrapper around EasyCarla-RL.
    Mounts Left, Center, Right RGB cameras and optimizes synchronous CARLA world ticks.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        default_params = {
            'number_of_vehicles': 3,
            'number_of_walkers': 10,
            'display_size': 256,
            'max_past_step': 1,
            'dt': 0.05,
            'discrete': False,
            'discrete_acc': [-3.0, 1.5, 3.0],
            'discrete_steer': [-0.2, 0.0, 0.2],
            'continuous_accel_range': [-3.0, 3.0],
            'continuous_steer_range': [-0.3, 0.3],
            'ego_vehicle_filter': 'vehicle.tesla.model3',
            'port': 2000,
            'town': 'Town10HD_Opt',
            'max_time_episode': 250,
            'max_waypoints': 12,
            'visualize_waypoints': False,
            'desired_speed': 8,
            'max_ego_spawn_times': 200,
            'view_mode': 'top',
            'traffic': 'off',
            'lidar_max_range': 50.0,
            'max_nearby_vehicles': 5,
            'surrounding_vehicle_spawned_randomly': True,
            'img_width': 256,
            'img_height': 256,
            'frame_skip': 2,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.img_width = self.params.get('img_width', 256)
        self.img_height = self.params.get('img_height', 256)
        self.frame_skip = int(self.params.get('frame_skip', 2))
        self.curriculum_factor = 1.0
        self.episode_count = 0

        self.sensor_mgr = CameraSensorManager(self.img_width, self.img_height)
        self.reward_calc = RewardCalculator(desired_speed=self.params.get('desired_speed', 25.0))

        port = self.params.get('port', 2000)
        wait_for_carla_server(port, max_wait=60)

        self.carla_client = carla.Client('127.0.0.1', port) if carla is not None else None
        if self.carla_client:
            self.carla_client.set_timeout(120.0)

        self._init_easy_env(self.params)
        self._optimize_easy_env()

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width * 3, 3), dtype=np.uint8),
            "speed": spaces.Box(low=0.0, high=150.0, shape=(1,), dtype=np.float32)
        })

    def _init_easy_env(self, params: Dict[str, Any]) -> None:
        """Initialize EasyCarla environment with safe 120s timeout and map preservation."""
        if carla is None:
            self.easy_env = None
            return

        orig_set_timeout = carla.Client.set_timeout
        carla.Client.set_timeout = lambda s, t: orig_set_timeout(s, max(t, 120.0))
        try:
            self.easy_env = CarlaEnv(params)
        finally:
            carla.Client.set_timeout = orig_set_timeout

    def _optimize_easy_env(self) -> None:
        """Optimize EasyCarla: minimize unused LiDAR raycasting and bypass unused math."""
        if not hasattr(self, 'easy_env') or self.easy_env is None:
            return

        if hasattr(self.easy_env, 'lidar_bp') and self.easy_env.lidar_bp is not None:
            try:
                if self.easy_env.lidar_bp.has_attribute('points_per_second'):
                    self.easy_env.lidar_bp.set_attribute('points_per_second', '1000')
            except Exception:
                pass

        self.easy_env._get_obs = lambda: {
            'ego_state': np.zeros(9, dtype=np.float32),
            'lane_info': np.zeros(2, dtype=np.float32),
            'lidar': np.zeros(240, dtype=np.float32),
            'nearby_vehicles': np.zeros(20, dtype=np.float32),
            'waypoints': np.zeros(36, dtype=np.float32)
        }

        self.easy_env._clear_all_actors = lambda filters: safe_clear_carla_actors(
            self.easy_env.world, self.carla_client, filters
        )

    def set_curriculum_factor(self, factor: float) -> None:
        """Set dynamic penalty scaling factor (in [0.2, 1.0]) for curriculum training."""
        self.curriculum_factor = max(0.2, min(1.0, float(factor)))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Return dict with panoramic stitched RGB camera buffer and speed scalar."""
        speed_kmh = 0.0
        try:
            if hasattr(self.easy_env, 'ego') and self.easy_env.ego is not None:
                vel = self.easy_env.ego.get_velocity()
                speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        except Exception:
            pass
        return {
            "image": self.sensor_mgr.panorama_buffer,
            "speed": np.array([speed_kmh], dtype=np.float32)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment using zero-latency in-place vehicle repositioning."""
        if seed is not None:
            np.random.seed(seed)

        self.episode_count += 1
        self.reward_calc.reset_episode_tracking()

        ego_alive = self.sensor_mgr.are_all_alive(getattr(self.easy_env, 'ego', None))

        # Fast in-place reset (99% of episodes) to prevent RPC socket teardown bottleneck
        if ego_alive and (self.episode_count % 100 != 0):
            try:
                self.easy_env.ego.set_simulate_physics(False)
                spawn_points = self.easy_env.map.get_spawn_points()
                if spawn_points:
                    sp = random.choice(spawn_points)
                    sp.location.z += 0.5
                    self.easy_env.ego.set_transform(sp)

                self.easy_env.ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                self.easy_env.ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                self.easy_env._is_collision = False
                self.easy_env._is_off_road = False
                self.easy_env.time_step = 0
                self.easy_env.total_reward = 0.0
                self.easy_env.ego.set_simulate_physics(True)

                if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                    try:
                        self.easy_env.world.tick()
                    except Exception:
                        pass
                time.sleep(0.02)
                return self._get_obs(), {}
            except Exception:
                pass

        # Full reset fallback
        self.sensor_mgr.cleanup_cameras()
        _wd_reset = threading.Timer(90.0, lambda: os._exit(1))
        _wd_reset.daemon = True
        _wd_reset.start()
        try:
            for _ in range(3):
                try:
                    self.easy_env.reset()
                    break
                except Exception:
                    time.sleep(1.0)
        finally:
            _wd_reset.cancel()

        self.sensor_mgr.setup_cameras(self.easy_env.world, self.easy_env.ego)
        try:
            if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                self.easy_env.world.tick()
        except Exception:
            pass
        time.sleep(0.05)
        return self._get_obs(), {}

    def _sub_step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute single physics simulation tick with watchdog protection."""
        throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        brake = float(np.clip((action[2] - 0.2) / 0.8, 0.0, 1.0)) if action[2] > 0.2 else 0.0
        scaled_action = [throttle, steer, brake]

        cost, done = 0.0, False
        _watchdog = threading.Timer(90.0, lambda: os._exit(1))
        _watchdog.daemon = True
        _watchdog.start()
        try:
            _, _, cost, done, _ = self.easy_env.step(scaled_action)
        except Exception:
            cost = 1.0
            done = True
            self.easy_env._is_collision = True
        finally:
            _watchdog.cancel()

        obs = self._get_obs()
        speed_kmh = float(obs["speed"][0])

        state = {
            "speed_kmh": speed_kmh, "heading_cos": 1.0, "heading_cos_far": 1.0,
            "lateral_dist": 0.0, "curve_factor": 1.0, "is_junction": False,
            "steer": steer, "throttle": throttle, "brake": brake,
            "is_at_red_light": False, "min_obs_dist": 99.0, "is_pedestrian": False,
            "ttc_seconds": 99.0, "is_collision": self.easy_env._is_collision,
            "is_off_road": self.easy_env._is_off_road, "time_step": self.easy_env.time_step
        }

        if hasattr(self.easy_env, 'ego') and self.easy_env.ego is not None:
            try:
                tf = self.easy_env.ego.get_transform()
                wp = self.easy_env.map.get_waypoint(tf.location)
                if wp:
                    fwd = tf.get_forward_vector()
                    wp_fwd = wp.transform.get_forward_vector()
                    state["heading_cos"] = fwd.x * wp_fwd.x + fwd.y * wp_fwd.y
                    state["lateral_dist"] = tf.location.distance(wp.transform.location)
            except Exception:
                pass

        reward, sub_info = self.reward_calc.compute_reward(state, self.curriculum_factor)
        terminated = bool(done or self.easy_env._is_collision or self.easy_env._is_off_road or sub_info["is_stalled"])
        truncated = bool(self.easy_env.time_step >= self.easy_env.max_time_episode)

        reason = "Active"
        if sub_info["is_stalled"]:
            reason = "Stalled / No Movement"
        elif self.easy_env._is_collision:
            reason = "Collision"
        elif self.easy_env._is_off_road:
            reason = "Lane Deviation / Off-Road"
        elif truncated:
            reason = "Max Steps Reached"

        info = {
            "cost": cost, "is_collision": self.easy_env._is_collision,
            "is_off_road": self.easy_env._is_off_road, "termination_reason": reason,
            "speed_kmh": speed_kmh, **sub_info
        }
        return obs, reward, terminated, truncated, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step environment with continuous action and frame-skip (action repeat)."""
        total_reward, total_cost = 0.0, 0.0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self._sub_step(action)
            total_reward += reward
            total_cost += info.get("cost", 0.0)
            if terminated or truncated:
                break
        info["cost"] = total_cost
        info["frame_skip"] = self.frame_skip
        return obs, total_reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up 3 camera sensors and close environment."""
        self.sensor_mgr.cleanup_cameras()
        if hasattr(self, 'easy_env') and self.easy_env is not None:
            try:
                self.easy_env.close()
            except Exception:
                pass
