import os
import sys
import time
import math
import glob
import random
import numpy as np

# Auto-add local CARLA 0.9.15 client package if present
carla_root = os.environ.get("CARLA_ROOT", "/workspace/carla")
carla_dist_path = os.path.join(carla_root, "PythonAPI", "carla", "dist")
if os.path.exists(carla_dist_path):
    eggs = glob.glob(os.path.join(carla_dist_path, "carla-*-py3*.egg"))
    for p in eggs:
        if p not in sys.path:
            sys.path.insert(0, p)

import carla
import gymnasium as gym
from gymnasium import spaces

class CarlaGymEnv(gym.Env):
    """
    Custom Gymnasium Environment for Autonomous Driving in CARLA 0.9.15.
    
    Observation Space:
        - 'image': RGB Camera Frame of shape (height, width, 3), uint8
        - 'speed': Float array [speed_kmh]
        
    Action Space:
        - Box(low=[-1.0, 0.0, 0.0], high=[1.0, 1.0, 1.0])
        - [steering, throttle, brake]
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        host="127.0.0.1",
        port=2000,
        town="Town10HD_Opt",
        img_width=256,
        img_height=256,
        max_steps=500,
        target_speed=30.0,
        synchronous_mode=True,
        delta_seconds=0.05
    ):
        super(CarlaGymEnv, self).__init__()
        
        self.host = host
        self.port = port
        self.town = town
        self.img_width = img_width
        self.img_height = img_height
        self.max_steps = max_steps
        self.target_speed = target_speed
        self.synchronous_mode = synchronous_mode
        self.delta_seconds = delta_seconds

        # Action Space: [steering (-1 to 1), throttle (0 to 1), brake (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space: Dictionary containing RGB image and speed kinematics
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

        self._init_carla()

    def _init_carla(self):
        """Connect to CARLA server."""
        print(f"[CarlaGymEnv] Connecting to CARLA server at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        
        # Enable Synchronous Mode for deterministic RL steps
        if self.synchronous_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.delta_seconds
            self.world.apply_settings(settings)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()

        self.step_count = 0
        self.has_collided = False
        self.latest_image = None

        blueprint_library = self.world.get_blueprint_library()

        # 1. Spawn Ego Vehicle
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found on map!")
            
        spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        
        while self.vehicle is None:
            spawn_point = random.choice(spawn_points)
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        self.actor_list.append(self.vehicle)

        # 2. Attach RGB Camera Sensor
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.img_width))
        camera_bp.set_attribute("image_size_y", str(self.img_height))
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(self._on_camera_image)
        self.actor_list.append(self.camera_sensor)

        # 3. Attach Collision Sensor
        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)
        self.actor_list.append(self.collision_sensor)

        # Tick simulation to process initial frame
        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Wait for camera observation to arrive
        time_start = time.time()
        while self.latest_image is None:
            if self.synchronous_mode:
                self.world.tick()
            else:
                time.sleep(0.01)
            if time.time() - time_start > 5.0:
                # Fallback black image if camera timeout
                self.latest_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        obs = self._get_obs()
        info = {}
        return obs, info

    def _on_camera_image(self, image):
        """Callback for incoming RGB camera images."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        # Convert BGRA to RGB with contiguous memory stride
        self.latest_image = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])

    def _on_collision(self, event):
        """Callback for collision events."""
        self.has_collided = True

    def _get_speed_kmh(self):
        """Calculate current vehicle speed in km/h."""
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def _get_obs(self):
        """Build observation dictionary."""
        speed = np.array([self._get_speed_kmh()], dtype=np.float32)
        image = self.latest_image if self.latest_image is not None else np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        return {
            "image": image,
            "speed": speed
        }

    def step(self, action):
        self.step_count += 1
        
        # Extract action components: [steering, throttle, brake]
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        # Apply vehicle control
        control = carla.VehicleControl(
            steer=steering,
            throttle=throttle,
            brake=brake,
            hand_brake=False,
            reverse=False
        )
        self.vehicle.apply_control(control)

        # Tick simulation
        if self.synchronous_mode:
            self.world.tick()

        speed_kmh = self._get_speed_kmh()
        obs = self._get_obs()

        # --- Compute Reward Function ---
        # 1. Speed Reward: Encourage driving near target_speed
        speed_reward = 1.0 - abs(speed_kmh - self.target_speed) / self.target_speed
        speed_reward = max(speed_reward, -1.0)

        # 2. Steering Smoothness Penalty: Penalize sharp high-speed steering turns
        steering_penalty = -0.5 * (steering ** 2) if speed_kmh > 10.0 else 0.0

        # 3. Collision Penalty
        collision_penalty = -100.0 if self.has_collided else 0.0

        reward = speed_reward + steering_penalty + collision_penalty

        # Check termination conditions
        terminated = self.has_collided
        truncated = self.step_count >= self.max_steps
        
        info = {
            "speed_kmh": speed_kmh,
            "has_collided": self.has_collided,
            "step_count": self.step_count
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Return latest RGB camera frame for rendering."""
        return self.latest_image

    def _cleanup(self):
        """Destroy spawned actors cleanly."""
        for actor in reversed(self.actor_list):
            if actor is not None:
                try:
                    if hasattr(actor, "stop"):
                        actor.stop()
                    if actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass
        self.actor_list.clear()

    def close(self):
        """Close environment and clean up CARLA settings."""
        self._cleanup()
        if self.world and self.synchronous_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        print("[CarlaGymEnv] Environment closed cleanly.")

if __name__ == "__main__":
    print("Testing CarlaGymEnv Gymnasium Environment...")
    env = CarlaGymEnv(host="127.0.0.1", port=2000, img_width=256, img_height=256, max_steps=50)
    obs, info = env.reset()
    print(f"Reset Successful! Obs Image Shape: {obs['image'].shape}, Speed: {obs['speed']}")
    
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1:02d} | Action: Steer={action[0]:+.2f}, Throttle={action[1]:.2f} | Speed: {info['speed_kmh']:.1f} km/h | Reward: {reward:+.2f}")
        if terminated or truncated:
            break
            
    env.close()
    print(f"Test Completed cleanly! Total Reward: {total_reward:.2f}")
