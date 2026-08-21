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


def _wait_for_carla_server(port=2000, max_wait=45):
    """Poll CARLA server until RPC socket, world, and map geometry are fully initialized and responding."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            c = carla.Client('127.0.0.1', port)
            c.set_timeout(4.0)
            ver = c.get_server_version()
            if ver:
                world = c.get_world()
                map_obj = world.get_map()
                spawn_pts = map_obj.get_spawn_points()
                if len(spawn_pts) > 0:
                    time.sleep(2.0)  # Grace period for rendering pipeline
                    return True
        except (Exception, BaseException):
            time.sleep(1.0)
    return False


class CameraEasyCarlaEnv(gym.Env):
    """
    Camera-Only Gymnasium Environment Wrapper around EasyCarla-RL.
    
    Wraps EasyCarla-RL and mounts a front-facing RGB camera sensor on the ego vehicle.
    Returns observations containing (256, 256, 3) RGB images and vehicle speed scalar,
    supporting end-to-end RL policies with vision backbones.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, params=None):
        super().__init__()

        if params is None:
            params = {
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
                'ego_vehicle_filter': 'vehicle.lincoln.mkz_2020',
                'port': 2000,
                'town': 'Town10HD_Opt',
                'max_time_episode': 250,
                'max_waypoints': 12,
                'visualize_waypoints': False,
                'desired_speed': 8,
                'max_ego_spawn_times': 200,
                'view_mode': 'follow',
                'traffic': 'off',
                'lidar_max_range': 50.0,
                'max_nearby_vehicles': 5,
                'surrounding_vehicle_spawned_randomly': True,
                'img_width': 256,
                'img_height': 256,
            }

        self.params = params
        self.img_width = params.get('img_width', 256)
        self.img_height = params.get('img_height', 256)
        self.stalled_steps = 0
        
        # Check if CARLA server is responsive and running the requested town
        town = params.get('town', 'Town10HD_Opt')
        port = params.get('port', 2000)
        server_ok = False
        try:
            test_c = carla.Client('127.0.0.1', port)
            test_c.set_timeout(5.0)
            cur_map = test_c.get_world().get_map().name
            if town in cur_map:
                server_ok = True
            else:
                print(f"--> Current server map is {cur_map}, but {town} was requested. Restarting server on {town}...")
                server_ok = False
        except Exception:
            server_ok = False

        if not server_ok and os.path.exists("/workspace/carla/CarlaUE4.sh"):
            print(f"--> Starting CARLA server session with map {town}...")
            os.system("pkill -9 -f CarlaUE4 2>/dev/null || true")
            os.system("tmux kill-session -t carla_server 2>/dev/null || true")
            os.system(f"tmux new-session -d -s carla_server \"su carlauser -c '/workspace/carla/CarlaUE4.sh /Game/Carla/Maps/{town} -carla-port={port} -RenderOffScreen -nosound -vulkan -quality-level=Low -benchmark -fps=20' > /workspace/carla_server.log 2>&1\"")
            _wait_for_carla_server(port, max_wait=45)

        self.carla_client = carla.Client('127.0.0.1', port)
        self.carla_client.set_timeout(60.0)
        try:
            temp_world = self.carla_client.get_world()
            try:
                temp_world.tick()
            except Exception:
                pass
            temp_settings = temp_world.get_settings()
            if temp_settings.synchronous_mode:
                temp_settings.synchronous_mode = False
                temp_world.apply_settings(temp_settings)
        except Exception:
            pass

        # Monkey-patch carla.Client during EasyCarla initialization to use 60s timeout and re-use active world map
        orig_set_timeout = carla.Client.set_timeout
        orig_load_world = carla.Client.load_world

        def patched_set_timeout(client_self, timeout):
            orig_set_timeout(client_self, max(timeout, 60.0))

        def patched_load_world(client_self, town_name, reset_settings=True):
            orig_set_timeout(client_self, 60.0)
            for map_attempt in range(3):
                try:
                    current_map = client_self.get_world().get_map().name
                    if town_name in current_map:
                        print(f"✓ CARLA server is already running {current_map}. Re-using active world!")
                        return client_self.get_world()
                    print(f"--> Loading CARLA map {town_name}...")
                    new_world = orig_load_world(client_self, town_name, reset_settings)
                    try:
                        settings = new_world.get_settings()
                        if settings.synchronous_mode:
                            settings.synchronous_mode = False
                            new_world.apply_settings(settings)
                        new_world.tick()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    return new_world
                except (Exception, BaseException) as load_err:
                    print(f"Waiting for simulator world... ({load_err})")
                    time.sleep(2.0)
            try:
                return client_self.get_world()
            except Exception:
                return None

        carla.Client.set_timeout = patched_set_timeout
        carla.Client.load_world = patched_load_world

        try:
            # Instantiate underlying EasyCarla environment
            self.easy_env = CarlaEnv(params)
        finally:
            # Restore original methods
            carla.Client.set_timeout = orig_set_timeout
            carla.Client.load_world = orig_load_world

        # Global protection against short 10s timeouts in underlying CarlaEnv
        orig_set_timeout = carla.Client.set_timeout
        def safe_set_timeout(client_self, timeout):
            orig_set_timeout(client_self, max(timeout, 60.0))
        carla.Client.set_timeout = safe_set_timeout

        # Attach safe atomic batch actor destruction with controller stopping and render pipeline draining
        def safe_clear_all_actors(actor_filters):
            # 1. Stop all controllers and sensors first
            for actor_filter in actor_filters:
                for actor in self.easy_env.world.get_actors().filter(actor_filter):
                    try:
                        if hasattr(actor, 'stop') and callable(actor.stop):
                            actor.stop()
                    except (Exception, BaseException):
                        pass

            # 2. Drain render thread in-flight frames before deleting memory
            try:
                self.easy_env.world.tick()
            except (Exception, BaseException):
                pass

            # 3. Gather batch of destroy commands for unique living actors
            batch = []
            seen_ids = set()
            for actor_filter in actor_filters:
                for actor in self.easy_env.world.get_actors().filter(actor_filter):
                    if actor.id not in seen_ids:
                        seen_ids.add(actor.id)
                        batch.append(carla.command.DestroyActor(actor.id))

            if batch and hasattr(self, 'carla_client') and self.carla_client is not None:
                try:
                    self.carla_client.apply_batch(batch)
                except (Exception, BaseException):
                    pass

        self.easy_env._clear_all_actors = safe_clear_all_actors
        
        # Define Action Space: [throttle (0 to 1), steer (-1 to 1), brake (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Define Observation Space: 3-Camera RGB Image (256x768x3 stitched: Left | Center | Right) + Speed Kinematics
        self.num_cameras = 3
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width * self.num_cameras, 3), dtype=np.uint8),
            "speed": spaces.Box(low=0.0, high=150.0, shape=(1,), dtype=np.float32)
        })

        self.camera_sensors = {"left": None, "center": None, "right": None}
        self.latest_images = {"left": None, "center": None, "right": None}

    def _setup_camera(self):
        """Attach 3 synchronized RGB camera sensors (Left, Center, Right) to ego vehicle."""
        for cam_key in ["left", "center", "right"]:
            if self.camera_sensors[cam_key] is not None:
                try:
                    if hasattr(self.camera_sensors[cam_key], 'is_listening') and self.camera_sensors[cam_key].is_listening:
                        self.camera_sensors[cam_key].stop()
                    if hasattr(self.camera_sensors[cam_key], 'is_alive') and self.camera_sensors[cam_key].is_alive:
                        self.camera_sensors[cam_key].destroy()
                except Exception:
                    pass
                self.camera_sensors[cam_key] = None

        if not hasattr(self.easy_env, 'ego') or self.easy_env.ego is None:
            print("Warning: Ego vehicle is None, skipping camera attachment.")
            return

        world = self.easy_env.world
        bp_library = world.get_blueprint_library()
        cam_bp = bp_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.img_width))
        cam_bp.set_attribute("image_size_y", str(self.img_height))
        cam_bp.set_attribute("fov", "90")

        # 1. Center Front Camera (yaw=0.0)
        center_tf = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.4), carla.Rotation(pitch=-8.0, yaw=0.0))
        self.camera_sensors["center"] = world.spawn_actor(cam_bp, center_tf, attach_to=self.easy_env.ego)

        def _center_callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            arr = np.reshape(arr, (image.height, image.width, 4))
            self.latest_images["center"] = arr[:, :, :3][:, :, ::-1].copy()

        self.camera_sensors["center"].listen(_center_callback)

        # 2. Left Front Camera (yaw=-55.0)
        left_tf = carla.Transform(carla.Location(x=1.3, y=-0.4, z=1.4), carla.Rotation(pitch=-8.0, yaw=-55.0))
        self.camera_sensors["left"] = world.spawn_actor(cam_bp, left_tf, attach_to=self.easy_env.ego)

        def _left_callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            arr = np.reshape(arr, (image.height, image.width, 4))
            self.latest_images["left"] = arr[:, :, :3][:, :, ::-1].copy()

        self.camera_sensors["left"].listen(_left_callback)

        # 3. Right Front Camera (yaw=+55.0)
        right_tf = carla.Transform(carla.Location(x=1.3, y=0.4, z=1.4), carla.Rotation(pitch=-8.0, yaw=55.0))
        self.camera_sensors["right"] = world.spawn_actor(cam_bp, right_tf, attach_to=self.easy_env.ego)

        def _right_callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            arr = np.reshape(arr, (image.height, image.width, 4))
            self.latest_images["right"] = arr[:, :, :3][:, :, ::-1].copy()

        self.camera_sensors["right"].listen(_right_callback)

    def _get_speed_kmh(self):
        """Calculate ego vehicle speed in km/h."""
        if not hasattr(self.easy_env, 'ego') or self.easy_env.ego is None:
            return 0.0
        vel = self.easy_env.ego.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def _get_obs(self):
        """Return dict containing 3-camera stitched RGB panorama [Left | Center | Right] and speed."""
        blank_cam = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        img_left = self.latest_images["left"] if self.latest_images["left"] is not None else blank_cam
        img_center = self.latest_images["center"] if self.latest_images["center"] is not None else blank_cam
        img_right = self.latest_images["right"] if self.latest_images["right"] is not None else blank_cam

        # Stitched 3-camera horizontal panorama: shape (256, 768, 3)
        panorama_obs = np.ascontiguousarray(np.hstack([img_left, img_center, img_right]))

        speed_kmh = self._get_speed_kmh()
        return {
            "image": panorama_obs,
            "speed": np.array([speed_kmh], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        """Reset environment and attach camera sensor."""
        if seed is not None:
            np.random.seed(seed)

        # 1. Temporarily switch OFF synchronous mode so all cleanup and destruction happens asynchronously in 0.01s!
        if hasattr(self, 'carla_client') and self.carla_client is not None:
            try:
                self.carla_client.set_timeout(10.0)
                temp_world = self.carla_client.get_world()
                settings = temp_world.get_settings()
                if settings.synchronous_mode:
                    settings.synchronous_mode = False
                    temp_world.apply_settings(settings)
            except (Exception, BaseException):
                pass

        # 2. Stop and destroy 3-camera sensors safely via batch command
        cam_destroy_cmds = []
        for cam_key in ["left", "center", "right"]:
            sensor = self.camera_sensors.get(cam_key)
            if sensor is not None:
                try:
                    if hasattr(sensor, 'is_listening') and sensor.is_listening:
                        sensor.stop()
                    if hasattr(sensor, 'is_alive') and sensor.is_alive:
                        cam_destroy_cmds.append(carla.command.DestroyActor(sensor.id))
                except (Exception, BaseException):
                    pass
                self.camera_sensors[cam_key] = None

        if cam_destroy_cmds and hasattr(self, 'carla_client') and self.carla_client is not None:
            try:
                self.carla_client.apply_batch(cam_destroy_cmds)
            except (Exception, BaseException):
                pass

        self.latest_images = {"left": None, "center": None, "right": None}
        self.stalled_steps = 0
        
        # 3. Call underlying EasyCarla reset with automatic retry logic & server auto-restart
        for attempt in range(3):
            try:
                self.easy_env.reset()
                break
            except (Exception, BaseException) as e:
                print(f"Warning: CARLA reset attempt {attempt+1}/3 failed ({e}). Auto-restarting CARLA engine...")
                if os.path.exists("/workspace/carla/CarlaUE4.sh"):
                    port = self.params.get('port', 2000)
                    town = self.params.get('town', 'Town10HD_Opt')
                    os.system("pkill -9 -f CarlaUE4 2>/dev/null || true")
                    os.system("tmux kill-session -t carla_server 2>/dev/null || true")
                    os.system(f"tmux new-session -d -s carla_server \"su carlauser -c '/workspace/carla/CarlaUE4.sh /Game/Carla/Maps/{town} -carla-port={port} -RenderOffScreen -nosound -vulkan -quality-level=Low -benchmark -fps=20' > /workspace/carla_server.log 2>&1\"")
                    _wait_for_carla_server(port, max_wait=45)
                    self.carla_client = carla.Client('127.0.0.1', port)
                    self.carla_client.set_timeout(60.0)

                    # Re-create underlying CarlaEnv on fresh server instance with patch applied
                    orig_set_timeout = carla.Client.set_timeout
                    orig_load_world = carla.Client.load_world

                    def patched_set_timeout(client_self, timeout):
                        orig_set_timeout(client_self, max(timeout, 60.0))

                    def patched_load_world(client_self, town_name, reset_settings=True):
                        orig_set_timeout(client_self, 60.0)
                        for map_attempt in range(3):
                            try:
                                current_map = client_self.get_world().get_map().name
                                if town_name in current_map:
                                    return client_self.get_world()
                                return orig_load_world(client_self, town_name, reset_settings)
                            except (Exception, BaseException):
                                time.sleep(2.0)
                        try:
                            return client_self.get_world()
                        except Exception:
                            return None

                    carla.Client.set_timeout = patched_set_timeout
                    carla.Client.load_world = patched_load_world

                    try:
                        self.easy_env = CarlaEnv(self.params)
                        def safe_clear_all_actors(actor_filters):
                            for actor_filter in actor_filters:
                                for actor in self.easy_env.world.get_actors().filter(actor_filter):
                                    try:
                                        if hasattr(actor, 'stop') and callable(actor.stop):
                                            actor.stop()
                                    except (Exception, BaseException):
                                        pass
                            try:
                                self.easy_env.world.tick()
                            except (Exception, BaseException):
                                pass
                            batch = []
                            seen_ids = set()
                            for actor_filter in actor_filters:
                                for actor in self.easy_env.world.get_actors().filter(actor_filter):
                                    if actor.id not in seen_ids:
                                        seen_ids.add(actor.id)
                                        batch.append(carla.command.DestroyActor(actor.id))
                            if batch and hasattr(self, 'carla_client') and self.carla_client is not None:
                                try:
                                    self.carla_client.apply_batch(batch)
                                except (Exception, BaseException):
                                    pass
                        self.easy_env._clear_all_actors = safe_clear_all_actors
                    except (Exception, BaseException) as re_err:
                        print(f"Re-initialization error: {re_err}")
                    finally:
                        carla.Client.set_timeout = orig_set_timeout
                        carla.Client.load_world = orig_load_world
        
        # Attach camera sensor to newly spawned ego vehicle
        self._setup_camera()

        # Tick world once to allow camera sensor callback to receive initial frame
        try:
            if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                self.easy_env.world.tick()
        except (Exception, BaseException):
            pass
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

        # Step underlying EasyCarla environment (which already ticks the world in synchronous mode)
        easy_obs, reward, cost, done, easy_info = self.easy_env.step(scaled_action)

        obs = self._get_obs()
        speed_kmh = float(obs["speed"][0])
        
        # Check traffic light status affecting ego vehicle
        is_at_red_light = False
        try:
            if hasattr(self.easy_env, 'ego') and self.easy_env.ego is not None:
                if self.easy_env.ego.is_at_traffic_light():
                    tl = self.easy_env.ego.get_traffic_light()
                    if tl is not None:
                        tl_state = tl.get_state()
                        if tl_state in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                            is_at_red_light = True
        except Exception:
            is_at_red_light = False

        if is_at_red_light:
            if speed_kmh < 2.0:
                # Legally stopped at red light! Freeze stall counter, grant compliance bonus
                self.stalled_steps = 0
                reward += 0.5  # Reward for waiting cleanly at red light
            else:
                # Vehicle is moving through an active red light!
                reward -= 2.0  # Violation penalty for running red light
        else:
            # On open road: strictly penalize idle behavior
            if speed_kmh < 2.0:
                self.stalled_steps += 1
                reward -= 0.5  # Strict continuous idle penalty
            else:
                self.stalled_steps = 0

        # Terminate episode if car remains stationary/stalled on open road for >= 25 steps (1.25s)
        is_stalled = bool(self.stalled_steps >= 25)
        if is_stalled:
            reward -= 30.0  # Strict penalty for refusing to drive on open road

        # Positive bonus for forward movement (up to +1.5 reward per step)
        reward += 1.5 * min(speed_kmh / 25.0, 1.0)

        # Combine Gym signals
        terminated = bool(done or self.easy_env._is_collision or self.easy_env._is_off_road or is_stalled)
        truncated = bool(self.easy_env.time_step >= self.easy_env.max_time_episode)

        # Determine exact human-readable termination reason
        reason = "Active"
        if is_stalled:
            reason = "Stalled / No Movement"
        elif self.easy_env._is_collision:
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
            "is_at_red_light": is_at_red_light,
            "termination_reason": reason,
            "speed_kmh": speed_kmh
        }
        
        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up 3 camera sensors and close environment."""
        for cam_key in ["left", "center", "right"]:
            if self.camera_sensors.get(cam_key) is not None:
                try:
                    self.camera_sensors[cam_key].stop()
                    self.camera_sensors[cam_key].destroy()
                except Exception:
                    pass
                self.camera_sensors[cam_key] = None

        if hasattr(self, 'easy_env') and self.easy_env is not None:
            self.easy_env.close()

if __name__ == "__main__":
    print("CameraEasyCarlaEnv module defined successfully!")
