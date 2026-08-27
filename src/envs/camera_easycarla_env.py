"""CameraEasyCarlaEnv: Gymnasium wrapper for vision-based RL with zero-latency in-place reset."""
import os
import sys
import time
import math
import random
import warnings
from typing import Dict, Any, Tuple, Optional
import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

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
    CarlaEnv._on_collision = lambda self, event: (setattr(self, '_is_collision', True), self.collision_hist.append(event) if getattr(self, 'collision_hist', None) is not None else None)
    CarlaEnv._on_lane_invasion = lambda self, event: setattr(self, '_is_off_road', True)
    CarlaEnv._on_invasion = lambda self, event: setattr(self, '_is_off_road', True)
except ImportError:
    try:
        from CarlaEnv import CarlaEnv
        CarlaEnv._on_collision = lambda self, event: setattr(self, '_is_collision', True)
        CarlaEnv._on_lane_invasion = lambda self, event: setattr(self, '_is_off_road', True)
    except ImportError:
        CarlaEnv = object

from src.envs.base_env import wait_for_carla_server, safe_clear_carla_actors
from src.envs.camera_sensor import CameraSensorManager
from src.envs.reward_calculator import RewardCalculator


class CameraEasyCarlaEnv(gym.Env):
    """Zero-latency 3-Camera Gymnasium Environment Wrapper around EasyCarla-RL."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        default_params = {
            'number_of_vehicles': 3, 'number_of_walkers': 10, 'display_size': 256,
            'max_past_step': 1, 'dt': 0.05, 'discrete': False,
            'ego_vehicle_filter': 'vehicle.tesla.model3', 'port': 2000,
            'town': 'Town10HD_Opt', 'max_time_episode': 250, 'max_waypoints': 12,
            'visualize_waypoints': False, 'desired_speed': 8, 'max_ego_spawn_times': 200,
            'view_mode': 'top', 'traffic': 'off', 'lidar_max_range': 50.0,
            'max_nearby_vehicles': 5, 'surrounding_vehicle_spawned_randomly': True,
            'img_width': 256, 'img_height': 256, 'frame_skip': 2,
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

        self.carla_client = None
        if carla is not None:
            for attempt in range(10):
                try:
                    self.carla_client = carla.Client('127.0.0.1', port)
                    self.carla_client.set_timeout(120.0)
                    _ = self.carla_client.get_server_version()
                    break
                except Exception:
                    if attempt == 9:
                        raise
                    time.sleep(1.0)

        self._init_easy_env(self.params)
        self._optimize_easy_env()

        self.action_space = spaces.Box(low=np.array([0.0, -1.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width * 3, 3), dtype=np.uint8),
            "speed": spaces.Box(low=0.0, high=150.0, shape=(1,), dtype=np.float32)
        })

    def _init_easy_env(self, params: Dict[str, Any]) -> None:
        """Initialize EasyCarla environment with map-reuse and safe timeout."""
        if carla is None:
            self.easy_env = None
            return

        orig_set_timeout = carla.Client.set_timeout
        orig_load_world = carla.Client.load_world

        def safe_load_world(client_self, town_name, *args, **kwargs):
            try:
                curr_world = client_self.get_world()
                if town_name.lower() in curr_world.get_map().name.lower():
                    return curr_world
            except Exception:
                pass
            return orig_load_world(client_self, town_name, *args, **kwargs)

        carla.Client.set_timeout = lambda s, t: orig_set_timeout(s, max(t, 120.0))
        carla.Client.load_world = safe_load_world
        try:
            for attempt in range(10):
                try:
                    self.easy_env = CarlaEnv(params)
                    break
                except Exception:
                    if attempt == 9:
                        raise
                    time.sleep(2.0)
        finally:
            carla.Client.set_timeout = orig_set_timeout
            carla.Client.load_world = orig_load_world

    def _optimize_easy_env(self) -> None:
        """Optimize EasyCarla: filter false-positive ground collisions and eliminate unformatted stdout spam."""
        if not hasattr(self, 'easy_env') or self.easy_env is None:
            return
        if hasattr(self.easy_env, 'lidar_bp') and self.easy_env.lidar_bp is not None:
            try:
                if self.easy_env.lidar_bp.has_attribute('points_per_second'):
                    self.easy_env.lidar_bp.set_attribute('points_per_second', '1000')
            except Exception:
                pass

        self.easy_env.view_mode = 'none'
        self.easy_env._get_obs = lambda: {
            'ego_state': np.zeros(9, dtype=np.float32), 'lane_info': np.zeros(2, dtype=np.float32),
            'lidar': np.zeros(240, dtype=np.float32), 'nearby_vehicles': np.zeros(20, dtype=np.float32),
            'waypoints': np.zeros(36, dtype=np.float32)
        }
        self.world_map = self.easy_env.world.get_map() if hasattr(self.easy_env, 'world') and self.easy_env.world is not None else None
        self.spawn_points = list(self.world_map.get_spawn_points()) if self.world_map is not None else []

        def _safe_on_collision(event):
            impulse = getattr(event, 'normal_impulse', None)
            intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2) if impulse else 0.0
            other = getattr(event, 'other_actor', None)
            other_type = getattr(other, 'type_id', '').lower() if other else ''
            if 'road' in other_type or 'ground' in other_type or 'static.road' in other_type:
                return
            if intensity > 250.0 or any(k in other_type for k in ['vehicle', 'walker', 'pedestrian', 'prop', 'building', 'pole', 'wall', 'fence']):
                self.easy_env._is_collision = True
                if hasattr(self.easy_env, 'collision_hist') and self.easy_env.collision_hist is not None:
                    self.easy_env.collision_hist.append(event)

        self.easy_env._on_collision = _safe_on_collision
        self.easy_env._on_lane_invasion = lambda event: setattr(self.easy_env, '_is_off_road', True)
        self.easy_env._on_invasion = lambda event: setattr(self.easy_env, '_is_off_road', True)
        self.easy_env._terminal = lambda: bool(self.easy_env._is_collision or self.easy_env._is_off_road or (self.easy_env.time_step >= self.easy_env.max_time_episode))
        self.easy_env._clear_all_actors = lambda filters: safe_clear_carla_actors(self.easy_env.world, self.carla_client, filters)

    def set_curriculum_factor(self, factor: float) -> None:
        self.curriculum_factor = max(0.2, min(1.0, float(factor)))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        speed_kmh = 0.0
        try:
            if hasattr(self.easy_env, 'ego') and self.easy_env.ego is not None:
                vel = self.easy_env.ego.get_velocity()
                speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        except Exception:
            pass
        return {"image": self.sensor_mgr.panorama_buffer, "speed": np.array([speed_kmh], dtype=np.float32)}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self.episode_count += 1
        self.reward_calc.reset_episode_tracking()

        ego_alive = self.sensor_mgr.are_all_alive(getattr(self.easy_env, 'ego', None))
        if ego_alive:
            try:
                self.easy_env.ego.set_simulate_physics(False)
                if not self.spawn_points and self.world_map is not None:
                    self.spawn_points = list(self.world_map.get_spawn_points())
                if self.spawn_points:
                    sp = random.choice(self.spawn_points)
                    sp.location.z += 0.5
                    self.easy_env.ego.set_transform(sp)
                self.easy_env.ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                self.easy_env.ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                self.easy_env._is_collision = False
                self.easy_env._is_off_road = False
                if hasattr(self.easy_env, 'collision_hist'):
                    self.easy_env.collision_hist = []
                self.easy_env.time_step = 0
                self.easy_env.total_reward = 0.0
                self.easy_env.ego.set_simulate_physics(True)
                if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                    try:
                        self.easy_env.world.tick()
                    except Exception:
                        pass
                return self._get_obs(), {}
            except Exception:
                pass

        self.sensor_mgr.cleanup_cameras()
        try:
            for _ in range(3):
                try:
                    self.easy_env.reset()
                    break
                except Exception:
                    time.sleep(0.5)
        except Exception:
            pass
        self._optimize_easy_env()
        self.sensor_mgr.setup_cameras(self.easy_env.world, self.easy_env.ego)
        try:
            if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                self.easy_env.world.tick()
        except Exception:
            pass
        return self._get_obs(), {}

    def _sub_step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0)) if action[2] > 0.4 and throttle < 0.3 else 0.0

        cost, done = 0.0, False
        try:
            _, _, cost, done, _ = self.easy_env.step([throttle, steer, brake])
        except Exception:
            cost, done = 1.0, True
            self.easy_env._is_collision = True

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
        if hasattr(self.easy_env, 'ego') and self.easy_env.ego is not None and self.world_map is not None:
            try:
                tf = self.easy_env.ego.get_transform()
                wp_exact = self.world_map.get_waypoint(tf.location, project_to_road=False, lane_type=carla.LaneType.Driving)
                if wp_exact is None:
                    self.easy_env._is_off_road = True

                wp = self.world_map.get_waypoint(tf.location, project_to_road=True)
                if wp:
                    fwd, wp_fwd = tf.get_forward_vector(), wp.transform.get_forward_vector()
                    state["heading_cos"] = float(np.clip(fwd.x * wp_fwd.x + fwd.y * wp_fwd.y, -1.0, 1.0))
                    wp_right = wp.transform.get_right_vector()
                    dx = tf.location.x - wp.transform.location.x
                    dy = tf.location.y - wp.transform.location.y
                    lat_cross = abs(dx * wp_right.x + dy * wp_right.y)
                    state["lateral_dist"] = float(min(3.0, lat_cross))
                    if not wp.is_junction and lat_cross > ((wp.lane_width / 2.0) + 0.8):
                        self.easy_env._is_off_road = True
                    if not wp.is_junction and state["heading_cos"] < -0.2:
                        self.easy_env._is_off_road = True
            except Exception:
                pass

        state["is_off_road"] = bool(self.easy_env._is_off_road)
        state["is_collision"] = bool(self.easy_env._is_collision)
        reward, sub_info = self.reward_calc.compute_reward(state, self.curriculum_factor)
        terminated = bool(done or self.easy_env._is_collision or self.easy_env._is_off_road or sub_info["is_stalled"])
        truncated = bool(self.easy_env.time_step >= self.easy_env.max_time_episode)
        reason = "Stalled" if sub_info["is_stalled"] else ("Collision" if self.easy_env._is_collision else ("Off-Road" if self.easy_env._is_off_road else ("Max Steps" if truncated else "Active")))

        info = {"cost": cost, "is_collision": self.easy_env._is_collision, "is_off_road": self.easy_env._is_off_road, "termination_reason": reason, "speed_kmh": speed_kmh, **sub_info}
        return obs, reward, terminated, truncated, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
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
        self.sensor_mgr.cleanup_cameras()
        if hasattr(self, 'easy_env') and self.easy_env is not None:
            try:
                self.easy_env.close()
            except Exception:
                pass
