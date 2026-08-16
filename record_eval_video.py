import os
import sys
import time
import glob
import random
import numpy as np
import cv2

# Auto-add local CARLA 0.9.15 client package if present
carla_root = os.environ.get("CARLA_ROOT", "/workspace/carla")
carla_dist_path = os.path.join(carla_root, "PythonAPI", "carla", "dist")
if os.path.exists(carla_dist_path):
    eggs = glob.glob(os.path.join(carla_dist_path, "carla-*-py3*.egg"))
    for p in eggs:
        if p not in sys.path:
            sys.path.insert(0, p)

import carla

def record_multiview_eval(
    host="127.0.0.1",
    port=2000,
    steps=150,
    img_width=400,
    img_height=300,
    output_video="/workspace/output_screenshots/driving_multiview.mp4",
    num_npc_vehicles=20
):
    print(f"--- Starting Multi-Sensor Evaluation & Video Recording ---")
    print(f"Connecting to CARLA at {host}:{port}...")

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    
    # 1. Enable Synchronous Mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 2. Setup TrafficManager for NPC vehicles
    traffic_manager = client.get_trafficmanager(port + 6000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    actor_list = []
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # 3. Spawn NPC Traffic Vehicles
    print(f"Spawning {num_npc_vehicles} NPC traffic vehicles...")
    for _ in range(num_npc_vehicles):
        sp = random.choice(spawn_points)
        npc_bp = random.choice(blueprint_library.filter("vehicle.*"))
        npc = world.try_spawn_actor(npc_bp, sp)
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            actor_list.append(npc)

    # 4. Spawn Ego Vehicle
    ego_bp = blueprint_library.find("vehicle.tesla.model3")
    ego_spawn = random.choice(spawn_points)
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn)
    while ego_vehicle is None:
        ego_spawn = random.choice(spawn_points)
        ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn)

    ego_vehicle.set_autopilot(True, traffic_manager.get_port())
    actor_list.append(ego_vehicle)

    # 5. Attach 3 Synchronized Multi-Sensors to Ego Vehicle
    # Sensor 1: RGB Camera
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(img_width))
    rgb_bp.set_attribute("image_size_y", str(img_height))
    rgb_bp.set_attribute("fov", "90")
    tf = carla.Transform(carla.Location(x=1.6, z=1.7))
    rgb_cam = world.spawn_actor(rgb_bp, tf, attach_to=ego_vehicle)
    actor_list.append(rgb_cam)

    # Sensor 2: Depth Camera
    depth_bp = blueprint_library.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(img_width))
    depth_bp.set_attribute("image_size_y", str(img_height))
    depth_bp.set_attribute("fov", "90")
    depth_cam = world.spawn_actor(depth_bp, tf, attach_to=ego_vehicle)
    actor_list.append(depth_cam)

    # Sensor 3: Semantic Segmentation Camera
    sem_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    sem_bp.set_attribute("image_size_x", str(img_width))
    sem_bp.set_attribute("image_size_y", str(img_height))
    sem_bp.set_attribute("fov", "90")
    sem_cam = world.spawn_actor(sem_bp, tf, attach_to=ego_vehicle)
    actor_list.append(sem_cam)

    # Sensor Buffers
    frames = {"rgb": None, "depth": None, "sem": None}

    def process_rgb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (img.height, img.width, 4))
        frames["rgb"] = arr[:, :, :3]  # BGR

    def process_depth(img):
        # Convert depth buffer to logarithmic color map
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (img.height, img.width, 4))
        r, g, b = arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]
        normalized_depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
        depth_gray = (normalized_depth * 255.0).astype(np.uint8)
        frames["depth"] = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)

    def process_sem(img):
        # Convert raw city-scapes tags to colorized representation
        img.convert(carla.ColorConverter.CityScapesPalette)
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (img.height, img.width, 4))
        frames["sem"] = arr[:, :, :3]  # BGR

    rgb_cam.listen(process_rgb)
    depth_cam.listen(process_depth)
    sem_cam.listen(process_sem)

    # Initialize Video Writer
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    canvas_w = img_width * 3
    canvas_h = img_height + 40  # 40px top banner for stats
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (canvas_w, canvas_h))

    print(f"Executing {steps} synchronized steps with 3 multi-sensors...")

    try:
        for i in range(steps):
            world.tick()

            while frames["rgb"] is None or frames["depth"] is None or frames["sem"] is None:
                time.sleep(0.005)

            v = ego_vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

            # Combine 3 views horizontally: [RGB Camera | Depth Camera | Semantic Segmentation]
            combined = np.hstack((frames["rgb"], frames["depth"], frames["sem"]))

            # Add Top Banner
            banner = np.zeros((40, canvas_w, 3), dtype=np.uint8)
            cv2.putText(banner, f"CARLA Multi-Sensor Eval | Step: {i+1:03d}/{steps} | Ego Speed: {speed_kmh:.1f} km/h | Active Traffic: {len(actor_list)-4}",
                        (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Stack banner + camera view
            final_frame = np.vstack((banner, combined))
            video_writer.write(final_frame)

            if (i + 1) % 25 == 0:
                print(f"[Step {i+1:03d}/{steps}] Speed: {speed_kmh:.1f} km/h | Frame rendered to video.")

    finally:
        print("Cleaning up actors and stopping video writer...")
        video_writer.release()
        for actor in reversed(actor_list):
            if actor is not None and actor.is_alive:
                actor.destroy()
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print(f"--- Evaluation Complete! Video saved to: {os.path.abspath(output_video)} ---")

if __name__ == "__main__":
    record_multiview_eval()
