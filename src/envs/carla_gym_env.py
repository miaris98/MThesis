"""Standalone CarlaGymEnv implementation for native CARLA clients."""
import os
import sys
import time
import math
import random
import warnings

# Completely silence runtime, deprecation, and gymnasium warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

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

from src.envs.base_env import wait_for_carla_server


class CarlaGymEnv(gym.Env):
    """
    Gymnasium Environment for Autonomous Driving in standalone CARLA 0.9.15.
    Observation: RGB image (256x256x3) and speed kinematics scalar.
    Action: Continuous [steering, throttle, brake].
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        town: str = "Town10HD_Opt",
        img_width: int = 256,
        img_height: int = 256,
        max_steps: int = 500,
        target_speed: float = 30.0,
        synchronous_mode: bool = True,
        delta_seconds: float = 0.05
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.town = town
        self.img_width = img_width
        self.img_height = img_height
        self.max_steps = max_steps
        self.target_speed = target_speed
        self.synchronous_mode = synchronous_mode
        self.delta_seconds = delta_seconds

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
            "speed": spaces.Box(low=0.0, high=150.0, shape=(1,), dtype=np.float32)
        })

        self.client = None
        self.world = None
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.actor_list = []
        self.latest_image = None
        self.has_collided = False
        self.step_count = 0

        self._connect_to_server()

    def _connect_to_server(self) -> None:
        if carla is None:
            return
        wait_for_carla_server(self.port, max_wait=30)
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(120.0)
        try:
            self.world = self.client.get_world()
        except Exception:
            pass

    def _setup_sensors(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.img_width))
        cam_bp.set_attribute('image_size_y', str(self.img_height))
        cam_bp.set_attribute('fov', '90')

        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera_sensor)

        def _cam_callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = np.reshape(arr, (self.img_height, self.img_width, 4))
            self.latest_image = arr[:, :, :3]

        self.camera_sensor.listen(_cam_callback)

        col_bp = bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: setattr(self, 'has_collided', True))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        if self.latest_image is None:
            img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            img = self.latest_image.copy()

        speed_kmh = 0.0
        if self.vehicle is not None:
            v = self.vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        return {
            "image": img,
            "speed": np.array([speed_kmh], dtype=np.float32)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self._cleanup()
        self.step_count = 0
        self.has_collided = False

        if self.world is not None:
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.filter('vehicle.lincoln.mkz_2020')[0]
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.actor_list.append(self.vehicle)
            self._setup_sensors()

            if self.synchronous_mode:
                self.world.tick()

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])

        if self.vehicle is not None:
            ctrl = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            self.vehicle.apply_control(ctrl)

        if self.synchronous_mode and self.world is not None:
            self.world.tick()

        obs = self._get_obs()
        speed_kmh = float(obs["speed"][0])
        
        reward = 1.0 - abs(speed_kmh - self.target_speed) / max(1.0, self.target_speed)
        if self.has_collided:
            reward -= 20.0

        terminated = bool(self.has_collided)
        truncated = bool(self.step_count >= self.max_steps)
        info = {"speed_kmh": speed_kmh, "is_collision": self.has_collided}

        return obs, reward, terminated, truncated, info

    def _cleanup(self) -> None:
        for actor in self.actor_list:
            if actor is not None and getattr(actor, 'is_alive', False):
                try:
                    if hasattr(actor, 'stop'):
                        actor.stop()
                    actor.destroy()
                except Exception:
                    pass
        self.actor_list = []
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None

    def close(self) -> None:
        self._cleanup()
