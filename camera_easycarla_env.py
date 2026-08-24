import os
import sys
import glob
import time
import math
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
    if os.path.join(carla_root, "PythonAPI", "carla") not in sys.path:
        sys.path.insert(0, os.path.join(carla_root, "PythonAPI", "carla"))

# Auto-add local EasyCarla-RL package if present
easycarla_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Carla-utils", "EasyCarla-RL")
if os.path.exists(easycarla_path) and easycarla_path not in sys.path:
    sys.path.insert(0, easycarla_path)

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

try:
    from easycarla.envs.carla_env import CarlaEnv
except ImportError:
    try:
        from envs.carla_env import CarlaEnv
    except ImportError:
        CarlaEnv = object


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
        self.frame_skip = int(params.get('frame_skip', 2))
        self.stalled_steps = 0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.curriculum_factor = 1.0
        
        # Connect to CARLA server on requested port
        town = params.get('town', 'Town10HD_Opt')
        port = params.get('port', 2000)

        # 1. Wait for server (started by supervisor / run_training_loop.sh) to open port
        print(f"--> Connecting to CARLA server on port {port}...")
        server_ok = _wait_for_carla_server(port, max_wait=60)

        # 2. If server is not running at all, launch background server (OpenGL fallback)
        if not server_ok and os.path.exists("/workspace/carla/CarlaUE4.sh"):
            print(f"--> No active server found on port {port}. Starting CARLA server (-opengl)...")
            os.system(f"tmux new-session -d -s carla_server \"su carlauser -c '/workspace/carla/CarlaUE4.sh -carla-port={port} -RenderOffScreen -nosound -opengl -quality-level=Low' > /workspace/carla_server.log 2>&1\"")
            _wait_for_carla_server(port, max_wait=45)

        # 3. Check active town map and load requested town via Python RPC if needed
        try:
            test_c = carla.Client('127.0.0.1', port)
            test_c.set_timeout(30.0)
            cur_map = test_c.get_world().get_map().name
            if town in cur_map:
                print(f"✓ CARLA server is already running map '{cur_map}'. Re-using active world!")
            else:
                print(f"--> Current map is '{cur_map}'. Switching to requested map '{town}' via Python API...")
                test_c.load_world(town)
                print(f"✓ Successfully loaded map '{town}'!")
        except Exception as e:
            print(f"--> Map check notice: {e}")

        self.carla_client = carla.Client('127.0.0.1', port)
        self.carla_client.set_timeout(120.0)
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

        # Monkey-patch carla.Client during EasyCarla initialization to use 120s timeout and re-use active world map
        orig_set_timeout = carla.Client.set_timeout
        orig_load_world = carla.Client.load_world

        def patched_set_timeout(client_self, timeout):
            orig_set_timeout(client_self, max(timeout, 120.0))

        def patched_load_world(client_self, town_name, reset_settings=True):
            orig_set_timeout(client_self, 120.0)
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
                    self.carla_client.apply_batch_sync(batch, False)
                except (Exception, BaseException):
                    try:
                        self.carla_client.apply_batch(batch)
                    except Exception:
                        pass

        self.easy_env._clear_all_actors = safe_clear_all_actors
        self._optimize_underlying_easy_env(self.easy_env)
        
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
        # Zero-copy pre-allocated panorama buffer (256, 768, 3) in contiguous memory
        self.panorama_buffer = np.zeros((self.img_height, self.img_width * self.num_cameras, 3), dtype=np.uint8)

    def _optimize_underlying_easy_env(self, easy_env):
        """Optimize underlying EasyCarla environment: minimize sensor overhead and bypass unused observation math."""
        # 1. Minimize LiDAR raycasting overhead without setting blueprint to None
        if hasattr(easy_env, 'lidar_bp') and easy_env.lidar_bp is not None:
            try:
                if easy_env.lidar_bp.has_attribute('points_per_second'):
                    easy_env.lidar_bp.set_attribute('points_per_second', '1000')
            except Exception:
                pass

        # 2. Fast stub for unused EasyCarla _get_obs
        easy_env._get_obs = lambda: {
            'ego_state': np.zeros(9, dtype=np.float32),
            'lane_info': np.zeros(2, dtype=np.float32),
            'lidar': np.zeros(240, dtype=np.float32),
            'nearby_vehicles': np.zeros(20, dtype=np.float32),
            'waypoints': np.zeros(36, dtype=np.float32)
        }

        # 3. Track spawned walkers for fast local obstacle queries without world.get_actors() RPC scans
        easy_env.spawned_walkers = []
        orig_spawn_walker = easy_env._try_spawn_random_walker_at
        def tracked_spawn_walker(transform):
            walker_bp = random.choice(easy_env.world.get_blueprint_library().filter('walker.*'))
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            walker_actor = easy_env.world.try_spawn_actor(walker_bp, transform)
            if walker_actor is not None:
                easy_env.spawned_walkers.append(walker_actor)
                walker_controller_bp = easy_env.world.get_blueprint_library().find('controller.ai.walker')
                walker_controller_actor = easy_env.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
                walker_controller_actor.start()
                walker_controller_actor.go_to_location(easy_env.world.get_random_location_from_navigation())
                walker_controller_actor.set_max_speed(1 + random.random())
                return True
            return False
        easy_env._try_spawn_random_walker_at = tracked_spawn_walker

    def _setup_camera(self):
        """Attach 3 synchronized RGB camera sensors (Left, Center, Right) to ego vehicle with zero-copy buffer slices."""
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

        w = self.img_width

        # 1. Center Front Camera (yaw=0.0) -> Slice [w : 2*w]
        center_tf = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.4), carla.Rotation(pitch=-8.0, yaw=0.0))
        self.camera_sensors["center"] = world.spawn_actor(cam_bp, center_tf, attach_to=self.easy_env.ego)

        def _center_callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            self.panorama_buffer[:, w:2*w, :] = arr[:, :, [2, 1, 0]]

        self.camera_sensors["center"].listen(_center_callback)

        # 2. Left Front Camera (yaw=-55.0) -> Slice [0 : w]
        left_tf = carla.Transform(carla.Location(x=1.3, y=-0.4, z=1.4), carla.Rotation(pitch=-8.0, yaw=-55.0))
        self.camera_sensors["left"] = world.spawn_actor(cam_bp, left_tf, attach_to=self.easy_env.ego)

        def _left_callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            self.panorama_buffer[:, 0:w, :] = arr[:, :, [2, 1, 0]]

        self.camera_sensors["left"].listen(_left_callback)

        # 3. Right Front Camera (yaw=+55.0) -> Slice [2*w : 3*w]
        right_tf = carla.Transform(carla.Location(x=1.3, y=0.4, z=1.4), carla.Rotation(pitch=-8.0, yaw=55.0))
        self.camera_sensors["right"] = world.spawn_actor(cam_bp, right_tf, attach_to=self.easy_env.ego)

        def _right_callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            self.panorama_buffer[:, 2*w:3*w, :] = arr[:, :, [2, 1, 0]]

        self.camera_sensors["right"].listen(_right_callback)

    def set_curriculum_factor(self, factor):
        """Set dynamic reward curriculum factor in range [0.2, 1.0] matching literature annealing schedules."""
        self.curriculum_factor = float(np.clip(factor, 0.2, 1.0))

    def _get_speed_kmh(self):
        """Calculate ego vehicle speed in km/h."""
        if not hasattr(self.easy_env, 'ego') or self.easy_env.ego is None:
            return 0.0
        vel = self.easy_env.ego.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def _get_lane_alignment(self):
        """
        Calculate heading alignment cos(delta_yaw), lateral distance to lane centerline,
        ahead curvature factor, junction status, and far horizon heading alignment (10m).
        Returns: (heading_cos, heading_cos_far, lateral_dist, curve_factor, is_junction)
        """
        try:
            if not hasattr(self.easy_env, 'ego') or self.easy_env.ego is None:
                return 1.0, 1.0, 0.0, 1.0, False
            ego_tf = self.easy_env.ego.get_transform()
            ego_loc = ego_tf.location
            
            if hasattr(self.easy_env, 'map') and self.easy_env.map is not None:
                carla_map = self.easy_env.map
            elif hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                carla_map = self.easy_env.world.get_map()
            else:
                return 1.0, 1.0, 0.0, 1.0, False
                
            wpt = carla_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wpt is None:
                return 1.0, 1.0, 0.0, 1.0, False
                
            wpt_tf = wpt.transform
            is_junction = bool(wpt.is_junction)
            
            # Near horizon heading angle error (0m)
            ego_yaw_rad = math.radians(ego_tf.rotation.yaw)
            wpt_yaw_rad = math.radians(wpt_tf.rotation.yaw)
            heading_cos = math.cos(ego_yaw_rad - wpt_yaw_rad)
            
            # Far horizon heading angle error (10m predictive lookahead)
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
            
            # Lateral distance to lane center line
            lateral_dist = ego_loc.distance(wpt_tf.location)
            return heading_cos, heading_cos_far, lateral_dist, curve_factor, is_junction
        except Exception:
            return 1.0, 1.0, 0.0, 1.0, False

    def _get_front_obstacle_info(self, max_dist=15.0):
        """
        Fast front obstacle scan using tracked spawned vehicles and walkers.
        Avoids expensive world.get_actors() RPC round-trips over TCP.
        Returns: (min_dist, is_pedestrian, ttc_seconds)
        """
        try:
            if not hasattr(self.easy_env, 'ego') or self.easy_env.ego is None:
                return max_dist, False, 99.0

            ego = self.easy_env.ego
            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_fwd = ego_tf.get_forward_vector()
            ego_vel = ego.get_velocity()
            
            min_dist = max_dist
            is_pedestrian = False
            ttc_min = 99.0
            
            candidate_actors = []
            if hasattr(self.easy_env, 'spawned_vehicles') and self.easy_env.spawned_vehicles:
                candidate_actors.extend(self.easy_env.spawned_vehicles)
            if hasattr(self.easy_env, 'spawned_walkers') and self.easy_env.spawned_walkers:
                candidate_actors.extend(self.easy_env.spawned_walkers)

            if not candidate_actors and hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                candidate_actors = self.easy_env.world.get_actors()
            
            for actor in candidate_actors:
                if not getattr(actor, 'is_alive', True) or actor.id == ego.id:
                    continue
                a_type = getattr(actor, 'type_id', '')
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
                        
                        # Calculate relative closing velocity along forward vector
                        obs_vel = actor.get_velocity()
                        closing_speed_mps = (ego_vel.x - obs_vel.x) * ego_fwd.x + (ego_vel.y - obs_vel.y) * ego_fwd.y
                        if closing_speed_mps > 0.1:
                            ttc = dist / closing_speed_mps
                            if ttc < ttc_min:
                                ttc_min = ttc
                            
            return min_dist, is_pedestrian, ttc_min
        except Exception:
            return max_dist, False, 99.0

    def _get_obs(self):
        """Return dict containing zero-copy 3-camera stitched RGB panorama [Left | Center | Right] and speed."""
        speed_kmh = self._get_speed_kmh()
        return {
            "image": self.panorama_buffer,
            "speed": np.array([speed_kmh], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        """Reset environment safely using zero-latency in-place repositioning."""
        if seed is not None:
            np.random.seed(seed)

        self.episode_count = getattr(self, 'episode_count', 0) + 1
        self.stalled_steps = 0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0

        # Check if cameras and ego are already alive and working (Fast in-place reset!)
        all_cams_alive = (
            all(self.camera_sensors.get(k) is not None and getattr(self.camera_sensors[k], 'is_alive', False) for k in ["left", "center", "right"])
            and hasattr(self.easy_env, 'ego') and self.easy_env.ego is not None and getattr(self.easy_env.ego, 'is_alive', False)
        )

        # Do fast in-place reset (99% of the time) to prevent RPC socket bottleneck
        if all_cams_alive and (self.episode_count % 100 != 0):
            try:
                # Temporarily disable physics simulation to avoid collision sensor triggering on reposition
                self.easy_env.ego.set_simulate_physics(False)

                # Pick a valid spawn point from map (with +0.5m z-offset to ensure clean ground clearance)
                spawn_points = self.easy_env.map.get_spawn_points()
                if spawn_points:
                    sp = random.choice(spawn_points)
                    sp.location.z += 0.5
                    self.easy_env.ego.set_transform(sp)
                
                self.easy_env.ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                self.easy_env.ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                
                # Reset easy_env state flags
                self.easy_env._is_collision = False
                self.easy_env._is_off_road = False
                self.easy_env.time_step = 0
                self.easy_env.total_reward = 0.0
                
                # Re-enable physics simulation
                self.easy_env.ego.set_simulate_physics(True)

                # Tick world once to update camera frames
                if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                    try:
                        self.easy_env.world.tick()
                    except Exception:
                        pass
                    
                time.sleep(0.02)
                return self._get_obs(), {}
            except Exception:
                pass # Fallback to full reset on any error

        # Full reset fallback (on startup, camera loss, or periodic maintenance)
        for cam_key in ["left", "center", "right"]:
            sensor = self.camera_sensors.get(cam_key)
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass
                self.camera_sensors[cam_key] = None

        self.panorama_buffer.fill(0)
        
        # Call underlying EasyCarla reset
        for attempt in range(3):
            try:
                self.easy_env.reset()
                break
            except Exception as e:
                time.sleep(1.0)
        
        # Attach camera sensor to newly spawned ego vehicle
        self._setup_camera()

        # Tick world once to allow camera sensor callback to receive initial frame
        try:
            if hasattr(self.easy_env, 'world') and self.easy_env.world is not None:
                self.easy_env.world.tick()
        except (Exception, BaseException):
            pass
        time.sleep(0.05)

        return self._get_obs(), {}

    def _sub_step(self, action):
        """Execute single physics simulation tick with continuous action [throttle, steer, brake]."""
        # Map policy Tanh output [-1, 1] to vehicle control ranges:
        # action[0] (throttle): [-1, 1] -> [0.0, 1.0] (neutral 0.0 maps to 0.5 gas)
        # action[1] (steer):    [-1, 1] -> [-1.0, 1.0]
        # action[2] (brake):    [-1, 1] -> [0.0, 1.0] (only active if > 0.2)
        throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        brake = float(np.clip((action[2] - 0.2) / 0.8, 0.0, 1.0)) if action[2] > 0.2 else 0.0
        
        scaled_action = [throttle, steer, brake]

        # Step underlying EasyCarla environment (which ticks world in synchronous mode)
        easy_obs, easy_reward, cost, done, easy_info = self.easy_env.step(scaled_action)

        obs = self._get_obs()
        speed_kmh = float(obs["speed"][0])
        
        # 1. Traffic light status check (Red & Yellow)
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

        # 2. Dual-Horizon Heading alignment, Lane centering & Gaussian Potential Well
        heading_cos, heading_cos_far, lateral_dist, curve_factor, is_junction = self._get_lane_alignment()
        r_heading = 0.35 * max(-1.0, min(1.0, heading_cos)) + 0.15 * max(-1.0, min(1.0, heading_cos_far))
        
        # Gaussian Lane Potential Well: +1.0 at exact center line, decreasing quadratically with lateral deviation
        r_lateral = 1.0 * math.exp(-(lateral_dist ** 2) / (2.0 * (0.45 ** 2))) - 0.8 * (lateral_dist ** 2)
        r_boundary = -2.0 * (max(0.0, lateral_dist - 0.9) ** 2)

        # 3. Directional Velocity Projection Progress & Curvature-Adaptive Target Speed
        base_target = self.params.get('desired_speed', 25.0)
        adaptive_target_speed = base_target * curve_factor
        if is_junction:
            adaptive_target_speed = min(adaptive_target_speed, 15.0)
            
        # Directional forward velocity along lane tangent
        v_proj = speed_kmh * max(0.0, heading_cos)
        speed_diff = abs(v_proj - adaptive_target_speed)
        
        # Scale speed reward by lane centering quality so driving in circles off-center gives 0 speed reward
        lane_centering_gate = max(0.0, 1.0 - (lateral_dist / 1.2)) * max(0.0, heading_cos)
        if not is_at_red_light:
            if speed_diff <= 3.0:
                raw_speed_r = 1.5
            else:
                raw_speed_r = 1.5 * max(0.0, 1.0 - (speed_diff - 3.0) / adaptive_target_speed)
            r_speed = raw_speed_r * lane_centering_gate
        else:
            r_speed = 0.0

        # 4. Steering Smoothness, Rate & Dynamic Envelope Regularization (Active at all speeds)
        steer_diff = abs(steer - self.prev_steer)
        self.prev_steer = steer
        r_steer_rate = -0.3 * steer_diff
        r_steer_mag = -0.35 * (steer ** 2)
        
        # Penalize excessive steering angle beyond 0.35 rad (approx 20 deg) during straight cruising
        steer_max_allowed = max(0.20, min(0.60, 15.0 / (speed_kmh + 10.0)))
        r_steer_envelope = -2.0 * (max(0.0, abs(steer) - steer_max_allowed) ** 2)
        r_steer = r_steer_rate + r_steer_mag + r_steer_envelope

        # 5. Comfort & Throttle-Brake Conflict / Jitter Penalty
        throttle_diff = abs(throttle - self.prev_throttle)
        self.prev_throttle = throttle
        r_comfort = -0.5 * (throttle * brake) - 0.2 * throttle_diff

        # 6. Wrong-Way / Reverse Driving Penalty
        r_wrong_way = -3.0 * max(0.0, -heading_cos) * min(speed_kmh / 5.0, 1.0)

        # 7. Traffic Light (Red & Yellow) Compliance
        if is_at_red_light:
            if speed_kmh < 2.0 or brake > 0.2:
                self.stalled_steps = 0
                r_light = 1.5  # Compliance reward for stopping/braking at Red or Yellow light
            else:
                r_light = -5.0 # Penalty for running through active Red or Yellow light
        else:
            r_light = 0.0

        # 8. Pedestrian & Vehicle Proximity Barrier Function & Time-To-Collision (TTC)
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

        # 9. Idle & Stall Penalties on Open Road (with 30-step initial acceleration grace period)
        if self.easy_env.time_step > 30 and not is_at_red_light and min_obs_dist >= 10.0:
            if speed_kmh < 2.0:
                self.stalled_steps += 1
                r_idle = -0.5
            else:
                self.stalled_steps = 0
                r_idle = 0.0
        else:
            r_idle = 0.0

        is_stalled = bool(self.stalled_steps >= 120)

        # 10. Terminal Failure Penalties (Stall, Off-road, Collision)
        r_terminal = 0.0
        if self.easy_env._is_collision:
            r_terminal = -25.0
        elif self.easy_env._is_off_road:
            r_terminal = -20.0
        elif is_stalled:
            r_terminal = -20.0

        # Apply literature-aligned dynamic curriculum scaling (alpha in [0.2, 1.0])
        alpha = self.curriculum_factor
        r_boundary_s = r_boundary * alpha
        r_wrong_way_s = r_wrong_way * alpha
        r_light_s = r_light if r_light > 0 else (r_light * alpha)
        r_obstacle_s = r_obstacle if r_obstacle > 0 else (r_obstacle * alpha)
        r_ttc_s = r_ttc * alpha
        r_terminal_s = r_terminal * alpha

        # Combine step reward terms
        reward = (r_speed + r_heading + r_lateral + r_boundary_s + r_steer + 
                  r_comfort + r_wrong_way_s + r_light_s + r_obstacle_s + r_ttc_s + r_idle + r_terminal_s)

        # Combine Gym termination signals
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
            "speed_kmh": speed_kmh,
            "r_speed": r_speed,
            "r_heading": r_heading,
            "r_lateral": r_lateral,
            "r_boundary": r_boundary,
            "r_steer": r_steer,
            "r_comfort": r_comfort,
            "r_wrong_way": r_wrong_way,
            "r_light": r_light,
            "r_obstacle": r_obstacle,
            "r_idle": r_idle
        }
        
        return obs, reward, terminated, truncated, info

    def step(self, action):
        """Step environment with continuous action [throttle, steer, brake] and frame-skip (action repeat)."""
        total_reward = 0.0
        total_cost = 0.0
        
        for k in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self._sub_step(action)
            total_reward += reward
            total_cost += info.get("cost", 0.0)
            if terminated or truncated:
                break
                
        info["cost"] = total_cost
        info["frame_skip"] = self.frame_skip
        return obs, total_reward, terminated, truncated, info

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
