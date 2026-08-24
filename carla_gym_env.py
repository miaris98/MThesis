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

try:
    import carla
except ImportError:
    carla = None

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
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.curriculum_factor = 1.0

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

    def set_curriculum_factor(self, factor):
        """Set dynamic reward curriculum factor in range [0.2, 1.0] matching literature annealing schedules."""
        self.curriculum_factor = float(np.clip(factor, 0.2, 1.0))

    def _get_obs(self):
        """Build observation dictionary."""
        speed = np.array([self._get_speed_kmh()], dtype=np.float32)
        image = self.latest_image if self.latest_image is not None else np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        return {
            "image": image,
            "speed": speed
        }

    def _get_lane_alignment(self):
        """
        Calculate heading alignment cos(delta_yaw), lateral distance to lane centerline,
        ahead curvature factor, junction status, and far horizon heading alignment (10m).
        Returns: (heading_cos, heading_cos_far, lateral_dist, curve_factor, is_junction)
        """
        try:
            if self.vehicle is None or self.world is None:
                return 1.0, 1.0, 0.0, 1.0, False
            ego_tf = self.vehicle.get_transform()
            ego_loc = ego_tf.location
            carla_map = self.world.get_map()
            wpt = carla_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wpt is None:
                return 1.0, 1.0, 0.0, 1.0, False
            wpt_tf = wpt.transform
            is_junction = bool(wpt.is_junction)
            
            ego_yaw_rad = math.radians(ego_tf.rotation.yaw)
            wpt_yaw_rad = math.radians(wpt_tf.rotation.yaw)
            heading_cos = math.cos(ego_yaw_rad - wpt_yaw_rad)
            
            heading_cos_far = heading_cos
            curve_factor = 1.0
            next_wpts = wpt.next(5.0)
            if next_wpts and len(next_wpts) > 0:
                ahead_yaw_rad = math.radians(next_wpts[0].transform.rotation.yaw)
                curve_factor = max(0.4, math.cos(ego_yaw_rad - ahead_yaw_rad))
                next_10m = next_wpts[0].next(5.0)
                if next_10m and len(next_10m) > 0:
                    far_yaw_rad = math.radians(next_10m[0].transform.rotation.yaw)
                    heading_cos_far = math.cos(ego_yaw_rad - far_yaw_rad)
                
            lateral_dist = ego_loc.distance(wpt_tf.location)
            return heading_cos, heading_cos_far, lateral_dist, curve_factor, is_junction
        except Exception:
            return 1.0, 1.0, 0.0, 1.0, False

    def _get_front_obstacle_info(self, max_dist=15.0):
        """
        Scan for pedestrians and vehicles in front of ego vehicle within max_dist meters.
        Returns: (min_dist, is_pedestrian, ttc_seconds)
        """
        try:
            if self.vehicle is None or self.world is None:
                return max_dist, False, 99.0
            ego_tf = self.vehicle.get_transform()
            ego_loc = ego_tf.location
            ego_fwd = ego_tf.get_forward_vector()
            ego_vel = self.vehicle.get_velocity()
            
            actors = self.world.get_actors()
            min_dist = max_dist
            is_pedestrian = False
            ttc_min = 99.0
            
            for actor in actors:
                if actor.id == self.vehicle.id:
                    continue
                a_type = actor.type_id
                if not (a_type.startswith('walker.pedestrian') or a_type.startswith('vehicle.')):
                    continue
                loc = actor.get_location()
                dist = ego_loc.distance(loc)
                if dist < min_dist and dist > 0.5:
                    vec = carla.Vector3D(loc.x - ego_loc.x, loc.y - ego_loc.y, loc.z - ego_loc.z)
                    norm = math.sqrt(vec.x**2 + vec.y**2 + vec.z**2) + 1e-6
                    dot = (vec.x * ego_fwd.x + vec.y * ego_fwd.y + vec.z * ego_fwd.z) / norm
                    if dot > 0.707:
                        min_dist = dist
                        if a_type.startswith('walker.pedestrian'):
                            is_pedestrian = True
                        
                        obs_vel = actor.get_velocity()
                        closing_speed_mps = (ego_vel.x - obs_vel.x) * ego_fwd.x + (ego_vel.y - obs_vel.y) * ego_fwd.y
                        if closing_speed_mps > 0.1:
                            ttc = dist / closing_speed_mps
                            if ttc < ttc_min:
                                ttc_min = ttc
                                
            return min_dist, is_pedestrian, ttc_min
        except Exception:
            return max_dist, False, 99.0

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
        if self.vehicle is not None:
            self.vehicle.apply_control(control)

        # Tick simulation
        if self.synchronous_mode and self.world is not None:
            self.world.tick()

        speed_kmh = self._get_speed_kmh()
        obs = self._get_obs()

        # 1. Traffic Light status check (Red & Yellow)
        is_at_red_light = False
        try:
            if self.vehicle is not None and self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                if tl is not None and tl.get_state() in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                    is_at_red_light = True
        except Exception:
            is_at_red_light = False

        # 2. Dual-Horizon Heading alignment, Lane centering & Gaussian Potential Well
        heading_cos, heading_cos_far, lateral_dist, curve_factor, is_junction = self._get_lane_alignment()
        r_heading = 0.35 * heading_cos + 0.15 * heading_cos_far
        
        # Gaussian Lane Potential Well: +0.5 at exact center line, smoothly decreasing to -0.5 near boundaries
        r_lateral = 1.0 * (math.exp(-(lateral_dist ** 2) / (2.0 * (0.6 ** 2))) - 0.5)
        r_boundary = -1.0 * (max(0.0, lateral_dist - 1.2) ** 2)

        # 3. Directional Velocity Projection Progress & Curvature-Adaptive Target Speed
        base_target = self.target_speed
        adaptive_target_speed = base_target * curve_factor
        if is_junction:
            adaptive_target_speed = min(adaptive_target_speed, 15.0)
            
        # Directional forward velocity along lane tangent
        v_proj = speed_kmh * max(0.0, heading_cos)
        speed_diff = abs(v_proj - adaptive_target_speed)
        
        if not is_at_red_light:
            if speed_diff <= 3.0:
                r_speed = 1.5
            else:
                r_speed = 1.5 * max(0.0, 1.0 - (speed_diff - 3.0) / adaptive_target_speed)
        else:
            r_speed = 0.0

        # 4. Steering Smoothness, Rate & Dynamic Envelope Regularization
        steer_diff = abs(steering - self.prev_steer)
        self.prev_steer = steering
        r_steer_rate = -0.3 * steer_diff
        r_steer_mag = -0.2 * (steering ** 2) if speed_kmh > 10.0 else 0.0
        
        # Velocity-dynamic steering magnitude limit
        steer_max_allowed = max(0.15, 30.0 / (speed_kmh + 5.0))
        r_steer_envelope = -2.0 * (max(0.0, abs(steering) - steer_max_allowed) ** 2)
        r_steer = r_steer_rate + r_steer_mag + r_steer_envelope

        # 5. Comfort & Throttle-Brake Conflict / Jitter Penalty
        throttle_diff = abs(throttle - self.prev_throttle)
        self.prev_throttle = throttle
        r_comfort = -0.5 * (throttle * brake) - 0.2 * throttle_diff

        # 6. Wrong-Way / Reverse Driving Penalty
        r_wrong_way = -3.0 * max(0.0, -heading_cos) * min(speed_kmh / 5.0, 1.0)

        # 7. Traffic Light Compliance
        if is_at_red_light:
            if speed_kmh < 2.0 or brake > 0.2:
                r_light = 1.5
            else:
                r_light = -5.0
        else:
            r_light = 0.0

        # 8. Obstacle & Pedestrian Proximity Barrier Function & Time-To-Collision (TTC)
        min_obs_dist, is_pedestrian, ttc_seconds = self._get_front_obstacle_info(max_dist=15.0)
        r_obstacle = 0.0
        if min_obs_dist < 10.0:
            barrier_scale = 1.0 - (min_obs_dist / 10.0)
            multiplier = 2.0 if is_pedestrian else 1.0
            if brake > 0.2 or speed_kmh < 2.0:
                r_obstacle = 1.5 * barrier_scale * multiplier
            elif throttle > 0.2:
                r_obstacle = -4.0 * (barrier_scale ** 2) * multiplier

        r_ttc = -3.0 * (max(0.0, (2.0 - ttc_seconds) / 2.0) ** 2) if ttc_seconds < 2.0 else 0.0

        # 9. Collision Penalty
        r_collision = -100.0 if self.has_collided else 0.0

        # Apply literature-aligned dynamic curriculum scaling (alpha in [0.2, 1.0])
        alpha = self.curriculum_factor
        r_boundary_s = r_boundary * alpha
        r_wrong_way_s = r_wrong_way * alpha
        r_light_s = r_light if r_light > 0 else (r_light * alpha)
        r_obstacle_s = r_obstacle if r_obstacle > 0 else (r_obstacle * alpha)
        r_ttc_s = r_ttc * alpha
        r_collision_s = r_collision * alpha

        reward = (r_speed + r_heading + r_lateral + r_boundary_s + r_steer + 
                  r_comfort + r_wrong_way_s + r_light_s + r_obstacle_s + r_ttc_s + r_collision_s)

        # Check termination conditions
        terminated = self.has_collided
        truncated = self.step_count >= self.max_steps
        
        info = {
            "speed_kmh": speed_kmh,
            "has_collided": self.has_collided,
            "step_count": self.step_count,
            "r_speed": r_speed,
            "r_heading": r_heading,
            "r_lateral": r_lateral,
            "r_boundary": r_boundary,
            "r_steer": r_steer,
            "r_comfort": r_comfort,
            "r_wrong_way": r_wrong_way,
            "r_light": r_light,
            "r_obstacle": r_obstacle,
            "r_collision": r_collision
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Return latest RGB camera frame for rendering."""
        return self.latest_image

    def _cleanup(self):
        """Destroy spawned actors cleanly via server batch command."""
        for sensor in [self.camera_sensor, self.collision_sensor]:
            if sensor is not None:
                try:
                    if sensor.is_listening:
                        sensor.stop()
                except Exception:
                    pass
                try:
                    sensor.destroy()
                except Exception:
                    pass

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass

        if self.client and self.actor_list:
            destroy_cmds = [carla.command.DestroyActor(a.id) for a in self.actor_list if a is not None and a.is_alive]
            try:
                self.client.apply_batch_sync(destroy_cmds)
            except Exception:
                pass
        self.actor_list.clear()
        self.camera_sensor = None
        self.collision_sensor = None
        self.vehicle = None

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
