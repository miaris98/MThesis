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
from src.envs.driving_state import DrivingStateExtractor
from src.envs.rewards import make_reward


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
            'img_width': 256, 'img_height': 256, 'frame_skip': 2, 'reward_fn': 'custom_1',
            # _optimize_easy_env replaces easy_env._get_obs with a zeros stub, so nothing
            # ever reads the LIDAR scan - but the sensor would still ray-cast and dispatch
            # a callback on every tick, per ego vehicle. Don't spawn it at all.
            'enable_lidar': False,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.img_width = self.params.get('img_width', 256)
        self.img_height = self.params.get('img_height', 256)
        self.frame_skip = int(self.params.get('frame_skip', 2))
        self.dt = float(self.params.get('dt', 0.05))
        self.curriculum_factor = 1.0
        self.episode_count = 0
        # shared_mode: this instance is one of N vehicle-envs sharing a single CARLA
        # server/world owned by a SharedServerCarlaVectorEnv coordinator, instead of
        # owning its own dedicated server process.
        self.shared_mode = bool(self.params.get('shared_mode', False))

        # sensor_tick = dt * frame_skip so cameras render once per outer step() call (on
        # the final frame_skip sub-step) instead of every world tick - intermediate
        # sub-steps' frames are computed and immediately overwritten in step()'s loop and
        # were never observed, just fully rendered and thrown away.
        self.sensor_mgr = CameraSensorManager(self.img_width, self.img_height, sensor_tick=self.dt * self.frame_skip)
        # EasyCarla expresses 'desired_speed' in m/s while RewardCalculator compares against km/h.
        desired_speed_ms = float(self.params.get('desired_speed', 8.0))
        self.reward_fn_name = str(self.params.get('reward_fn', 'custom_1'))
        self.reward_calc = make_reward(self.reward_fn_name, desired_speed=desired_speed_ms * 3.6)
        self.state_extractor = DrivingStateExtractor()

        if self.shared_mode:
            # The coordinator already confirmed server readiness and owns the client
            # connection; reuse it instead of opening a second connection to the same
            # server.
            self.carla_client = self.params['external_client']
        else:
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

        # Raw policy output is Tanh-bounded [-1, 1] on every axis; _apply_sub_action maps it
        # to CARLA controls (throttle = (a0 + 1) / 2, steer = a1, brake = a2 when a2 > 0.4,
        # which then overrides throttle to 0).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
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
        spawn_point_subset = self.params.get('spawn_point_subset')
        if spawn_point_subset:
            # Coordinator-assigned, non-overlapping subset - this is what the fast
            # in-place teleport reset path actually reads, so it's the load-bearing
            # spawn-coordination surface for shared mode.
            self.spawn_points = list(spawn_point_subset)
        else:
            self.spawn_points = list(self.world_map.get_spawn_points()) if self.world_map is not None else []
        self.state_extractor.bind(getattr(self.easy_env, 'world', None), self.world_map)

        def _safe_on_collision(event):
            impulse = getattr(event, 'normal_impulse', None)
            intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2) if impulse else 0.0
            other = getattr(event, 'other_actor', None)
            other_type = getattr(other, 'type_id', '').lower() if other else ''
            # Only genuine road-surface contact is ignored, and only when gentle - a hard
            # curb slam still counts. Every other actor (vehicle, walker, pole, building,
            # fence, traffic light, unclassified static mesh) terminates on the first event,
            # with no impulse threshold to delay it by a physics tick or two.
            if any(k in other_type for k in ('road', 'ground', 'terrain', 'sidewalk')) and intensity < 400.0:
                return
            self.easy_env._is_collision = True
            if hasattr(self.easy_env, 'collision_hist') and self.easy_env.collision_hist is not None:
                self.easy_env.collision_hist.append(event)

        self.easy_env._on_collision = _safe_on_collision
        self.easy_env._on_lane_invasion = lambda event: setattr(self.easy_env, '_is_off_road', True)
        self.easy_env._on_invasion = lambda event: setattr(self.easy_env, '_is_off_road', True)
        self.easy_env._terminal = lambda: bool(self.easy_env._is_collision or self.easy_env._is_off_road or (self.easy_env.time_step >= self.easy_env.max_time_episode))
        if not self.shared_mode:
            # In shared mode, reset() routes to easy_env._clear_owned_actors() directly
            # and never calls _clear_all_actors - this monkeypatch would be dead but
            # misleading if left assigned.
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

    def begin_reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """Everything a reset needs to do before the settling world.tick(). Call
        finish_reset() afterward to get the resulting observation. Split out so a
        shared-server coordinator can tick once for every vehicle-env resetting together,
        instead of each one ticking independently."""
        if seed is not None:
            np.random.seed(seed)
        self.episode_count += 1
        self.reward_calc.reset_episode_tracking()
        self.state_extractor.reset()

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
                zero = carla.Vector3D(0, 0, 0)
                # Physics is disabled here, so both the target and the actual rigid-body
                # velocity must be cleared - otherwise the ego inherits the speed it
                # crashed at (observed: ~16 km/h at step 1 of every episode).
                self.easy_env.ego.set_target_velocity(zero)
                self.easy_env.ego.set_target_angular_velocity(zero)
                try:
                    self.easy_env.ego.enable_constant_velocity(zero)
                    self.easy_env.ego.disable_constant_velocity()
                except Exception:
                    pass
                self.easy_env.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                self.easy_env._is_collision = False
                self.easy_env._is_off_road = False
                if hasattr(self.easy_env, 'collision_hist'):
                    self.easy_env.collision_hist = []
                self.easy_env.time_step = 0
                self.easy_env.total_reward = 0.0
                self.easy_env.ego.set_simulate_physics(True)
                return
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

    def finish_reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Pairs with begin_reset() - call after the settling world.tick() has happened."""
        return self._get_obs(), {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.begin_reset(seed=seed, options=options)
        if not self.shared_mode:
            try:
                if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                    self.easy_env.world.tick()
            except Exception:
                pass
        return self.finish_reset()

    def _apply_sub_action(self, action: np.ndarray) -> None:
        """Apply control for one frame_skip sub-step. Does not tick - reset() in
        non-shared mode ticks right after this; a shared-server coordinator ticks once
        across all vehicle-envs' _apply_sub_action calls before reading any of them."""
        throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        # Brake and throttle stay mutually exclusive, but the brake axis decides that on its
        # own and cuts throttle when it engages. Previously both had to hold on the SAME step
        # (action[2] > 0.4 AND throttle < 0.3), which made braking effectively unreachable:
        # measured over a 10k-step run, the brake condition fired on 3.3% of steps and the
        # throttle condition on 4.0%, but both together on 0.1% - so the policy was carrying
        # a third action dimension it could not actually apply, at a median throttle of 0.73.
        # Matches the signed-acceleration convention in models/world_on_rails/wor_policy.py.
        brake = float(np.clip(action[2], 0.0, 1.0)) if action[2] > 0.4 else 0.0
        if brake > 0.0:
            throttle = 0.0
        self._last_sub_action = (throttle, steer, brake)
        self._last_apply_failed = False
        try:
            self.easy_env._apply_action([throttle, steer, brake])
        except Exception:
            self._last_apply_failed = True

    def _read_sub_result(self) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Pairs with _apply_sub_action() - call after the world.tick() for this
        sub-step has happened."""
        throttle, steer, brake = self._last_sub_action
        cost, done = 0.0, False
        if self._last_apply_failed:
            cost, done = 1.0, True
            self.easy_env._is_collision = True
        else:
            try:
                _, _, cost, done, _ = self.easy_env._post_tick()
            except Exception:
                cost, done = 1.0, True
                self.easy_env._is_collision = True

        obs = self._get_obs()
        speed_kmh = float(obs["speed"][0])
        state = self.state_extractor.extract(
            ego=getattr(self.easy_env, 'ego', None), speed_kmh=speed_kmh,
            time_step=self.easy_env.time_step, throttle=throttle, steer=steer, brake=brake,
            is_collision=self.easy_env._is_collision, is_off_road=self.easy_env._is_off_road
        )
        self.easy_env._is_off_road = state["is_off_road"]
        reward, sub_info = self.reward_calc.compute_reward(state, self.curriculum_factor, dt=self.dt)
        is_stalled = bool(sub_info.get("is_stalled", False))
        terminated = bool(done or self.easy_env._is_collision or self.easy_env._is_off_road or is_stalled)
        truncated = bool(self.easy_env.time_step >= self.easy_env.max_time_episode)
        reason = "Stalled" if is_stalled else ("Collision" if self.easy_env._is_collision else ("Off-Road" if self.easy_env._is_off_road else ("Max Steps" if truncated else "Active")))

        info = {
            "cost": cost, "is_collision": self.easy_env._is_collision, "is_off_road": self.easy_env._is_off_road,
            "termination_reason": reason, "speed_kmh": speed_kmh,
            "is_at_red_light": state["is_at_red_light"], "min_obs_dist": state["min_obs_dist"],
            "ttc_seconds": state["ttc_seconds"], "lateral_dist": state["lateral_dist"],
            "lane_width": state["lane_width"], "is_junction": state["is_junction"],
            "heading_cos": state["heading_cos"], **sub_info
        }
        return obs, reward, terminated, truncated, info

    def _sub_step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self._apply_sub_action(action)
        if not self.shared_mode:
            try:
                self.easy_env.world.tick()
            except Exception:
                pass
        return self._read_sub_result()

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
            if self.shared_mode and hasattr(self.easy_env, '_clear_owned_actors'):
                # A recovered/recreated coordinator is likely to reuse the still-alive
                # world rather than reload it, so leftover actors must be cleaned up
                # explicitly here or they'd leak across recovery cycles.
                try:
                    self.easy_env._clear_owned_actors()
                except Exception:
                    pass
            try:
                self.easy_env.close()
            except Exception:
                pass
