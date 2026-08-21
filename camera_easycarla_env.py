import os
import sys
import glob
import time
import math
import numpy as np

# Auto-add local CARLA 0.9.15 client package if present
carla_root = os.environ.get("CARLA_ROOT", "/workspace/carla")
carla_dist_path = os.path.join(carla_root, "PythonAPI", "carla", "dist")
if os.path.exists(carla_dist_path):
    eggs = glob.glob(os.path.join(carla_dist_path, "carla-*-py3*.egg"))
    for p in eggs:
        if p not in sys.path:
            sys.path.insert(0, p)
    if os.path.join(carla_root, "PythonAPI", "carla") not in sys.path:
        sys.path.insert(0, os.path.join(carla_root, "PythonAPI", "carla"))

import carla
import gymnasium as gym
from gymnasium import spaces

try:
    from easycarla.envs.carla_env import CarlaEnv
except ImportError:
    # Fallback import if easycarla is in PYTHONPATH
    from envs.carla_env import CarlaEnv

class CameraEasyCarlaEnv(gym.Env):
    """
    Camera-Only Gymnasium Environment Wrapper around EasyCarla-RL.
    
    Observation Space:
        - 'image': RGB Camera Frame of shape (height, width, 3), uint8 [0, 255]
        - 'speed': Float array [speed_kmh]
        
    Action Space:
        - Box(low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 1.0])
        - [throttle, steer, brake]
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, params=None):
        super(CameraEasyCarlaEnv, self).__init__()

        if params is None:
            params = {
                'number_of_vehicles': 10,
                'number_of_walkers': 0,
                'dt': 0.05,
                'ego_vehicle_filter': 'vehicle.tesla.model3',
                'surrounding_vehicle_spawned_randomly': True,
                'port': 2000,
                'town': 'Town10HD_Opt',
                'max_time_episode': 500,
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

        self.params = params
        self.img_width = params.get('img_width', 256)
        self.img_height = params.get('img_height', 256)
        
        # Pre-connect to CARLA to unstick server if left in synchronous mode by a previous process
        port = params.get('port', 2000)
        try:
            temp_client = carla.Client('127.0.0.1', port)
            temp_client.set_timeout(10.0)
            temp_world = temp_client.get_world()
            temp_settings = temp_world.get_settings()
            if temp_settings.synchronous_mode:
                temp_settings.synchronous_mode = False
                temp_world.apply_settings(temp_settings)
            active_map = temp_world.get_map().name.split('/')[-1]
            if params['town'] in active_map:
                params['town'] = active_map
        except Exception:
            pass

        # Instantiate underlying EasyCarla environment
        self.easy_env = CarlaEnv(params)
        
        # Ensure underlying client timeout is 60s
        if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
            try:
                self.easy_env.world.get_client().set_timeout(60.0)
            except Exception:
                pass
        
        # Define Action Space: [throttle (0 to 1), steer (-1 to 1), brake (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Define Observation Space: Camera RGB Image + Speed Kinematics
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
            "speed": spaces.Box(low=0.0, high=150.0, shape=(1,), dtype=np.float32)
        })

        self.camera_sensor = None
        self.latest_image = None

    def _setup_camera(self):
        """Attach RGB camera sensor to ego vehicle."""
        if self.camera_sensor is not None:
            try:
                if hasattr(self.camera_sensor, 'is_listening') and self.camera_sensor.is_listening:
                    self.camera_sensor.stop()
                if hasattr(self.camera_sensor, 'is_alive') and self.camera_sensor.is_alive:
                    self.camera_sensor.destroy()
            except Exception:
                pass
            self.camera_sensor = None

        world = self.easy_env.world
        bp_library = world.get_blueprint_library()
        cam_bp = bp_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.img_width))
        cam_bp.set_attribute("image_size_y", str(self.img_height))
        cam_bp.set_attribute("fov", "90")

        # Mount camera on hood / front windshield
        cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=-8.0))
        self.camera_sensor = world.spawn_actor(cam_bp, cam_transform, attach_to=self.easy_env.ego)

        def _camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4)) # BGRA
            rgb_array = array[:, :, :3][:, :, ::-1] # Convert BGRA -> RGB
            self.latest_image = rgb_array

        self.camera_sensor.listen(_camera_callback)

    def _get_speed_kmh(self):
        """Calculate ego vehicle speed in km/h."""
        if self.easy_env.ego is None:
            return 0.0
        vel = self.easy_env.ego.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def _get_obs(self):
        """Return dict containing latest RGB image and speed."""
        if self.latest_image is None:
            image_obs = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            image_obs = self.latest_image.copy()

        speed_kmh = self._get_speed_kmh()
        return {
            "image": image_obs,
            "speed": np.array([speed_kmh], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        """Reset environment and attach camera sensor."""
        if seed is not None:
            np.random.seed(seed)

        # Stop and destroy camera sensor before resetting underlying EasyCarla env
        if self.camera_sensor is not None:
            try:
                if hasattr(self.camera_sensor, 'is_listening') and self.camera_sensor.is_listening:
                    self.camera_sensor.stop()
                if hasattr(self.camera_sensor, 'is_alive') and self.camera_sensor.is_alive:
                    self.camera_sensor.destroy()
            except Exception:
                pass
            self.camera_sensor = None

        self.latest_image = None
        
        # Call underlying EasyCarla reset with 60s timeout & automatic retry logic
        for attempt in range(3):
            try:
                if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                    self.easy_env.world.get_client().set_timeout(60.0)
                self.easy_env.reset()
                break
            except Exception as e:
                print(f"Warning: CARLA environment reset attempt {attempt+1}/3 failed ({e}). Retrying reset...")
                time.sleep(2.0)
        
        # Attach camera sensor to newly spawned ego vehicle
        self._setup_camera()

        # Tick world once to allow camera sensor callback to receive initial frame
        self.easy_env.world.tick()
        time.sleep(0.05)

        obs = self._get_obs()
        info = {
            "reset_step": self.easy_env.reset_step,
            "time_step": self.easy_env.time_step
        }
        return obs, info

    def step(self, action):
        """Step environment with continuous action [throttle, steer, brake]."""
        # Map policy Tanh output [-1, 1] to vehicle control ranges:
        # action[0] (throttle): [-1, 1] -> [0.0, 1.0] (neutral 0.0 maps to 0.5 gas)
        # action[1] (steer):    [-1, 1] -> [-1.0, 1.0]
        # action[2] (brake):    [-1, 1] -> [0.0, 1.0] (only active if > 0.2)
        scaled_action = [
            float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0)),
            float(np.clip(action[1], -1.0, 1.0)),
            float(np.clip((action[2] - 0.2) / 0.8, 0.0, 1.0)) if action[2] > 0.2 else 0.0
        ]

        # Step underlying EasyCarla environment
        easy_obs, reward, cost, done, easy_info = self.easy_env.step(scaled_action)

        # Tick world to trigger camera update
        self.easy_env.world.tick()

        obs = self._get_obs()
        
        # Combine Gym signals
        terminated = bool(done or self.easy_env._is_collision or self.easy_env._is_off_road)
        truncated = bool(self.easy_env.time_step >= self.easy_env.max_time_episode)

        # Determine exact human-readable termination reason
        reason = "Active"
        if self.easy_env._is_collision:
            reason = "Collision"
        elif self.easy_env._is_off_road:
            reason = "Lane Deviation / Off-Road"
        elif truncated:
            reason = "Max Steps Reached"
        elif done:
            reason = "Episode Done"

        info = {
            "cost": cost,
            "is_collision": self.easy_env._is_collision,
            "is_off_road": self.easy_env._is_off_road,
            "termination_reason": reason,
            "speed_kmh": self._get_speed_kmh()
        }
        
        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up camera sensor and close environment."""
        if self.camera_sensor is not None:
            try:
                self.camera_sensor.stop()
                self.camera_sensor.destroy()
            except Exception:
                pass
            self.camera_sensor = None

        if hasattr(self, 'easy_env') and self.easy_env is not None:
            self.easy_env.close()

if __name__ == "__main__":
    print("CameraEasyCarlaEnv module defined successfully!")
