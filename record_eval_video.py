import os
import sys
import time
import math
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
    num_npc_vehicles=20,
    checkpoint="/workspace/checkpoints/ppo_carla_best.pth",
    backbone="lav"
):
    print(f"--- Starting Multi-Sensor Evaluation & Video Recording ---")
    print(f"Connecting to CARLA at {host}:{port}...")

    client = carla.Client(host, port)
    client.set_timeout(60.0)
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

    actor_list.append(ego_vehicle)

    # 5. Load Trained PyTorch PPO Model or Fallback to Autopilot
    agent = None
    device = None
    if checkpoint and os.path.exists(checkpoint):
        try:
            import torch
            from train_rl_agent import ActorCriticPPO
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            agent = ActorCriticPPO(action_dim=3, backbone_name=backbone).to(device)
            agent.load_state_dict(torch.load(checkpoint, map_location=device))
            agent.eval()
            print(f"Driving vehicle using Trained PyTorch PPO Policy Model: {checkpoint} (Backbone: {backbone}, Device: {device})")
            ego_vehicle.set_autopilot(False)
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Defaulting to Autopilot.")
            ego_vehicle.set_autopilot(True, traffic_manager.get_port())
    else:
        print("Driving vehicle using CARLA Autopilot (Pass --checkpoint to evaluate PyTorch policy model).")
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

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

            # Apply actions from trained PyTorch PPO policy if loaded
            if agent is not None:
                rgb_resized = cv2.resize(frames["rgb"][:, :, ::-1], (256, 256))
                img_tensor = torch.tensor(rgb_resized.copy(), dtype=torch.uint8).unsqueeze(0).to(device)
                spd_tensor = torch.tensor([speed_kmh], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(img_tensor, spd_tensor)
                act = action.cpu().numpy()[0]
                control = carla.VehicleControl(
                    steer=float(np.clip(act[0], -1.0, 1.0)),
                    throttle=float(np.clip(act[1], 0.0, 1.0)),
                    brake=float(np.clip(act[2], 0.0, 1.0))
                )
                ego_vehicle.apply_control(control)

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
        # Stop camera listeners
        for s in [rgb_cam, depth_cam, sem_cam]:
            try:
                s.stop()
            except Exception:
                pass

        # Batch destroy all actors safely on server
        if actor_list:
            destroy_cmds = [carla.command.DestroyActor(a.id) for a in actor_list if a is not None]
            client.apply_batch(destroy_cmds)
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print(f"--- Evaluation Complete! Raw Video saved to: {os.path.abspath(output_video)} ---")

        # Convert to H.264 HTML5 browser-compatible video via ffmpeg
        h264_video = output_video.replace(".mp4", "_h264.mp4")
        try:
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", output_video,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                h264_video
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(h264_video):
                print(f"--- H.264 Web Browser Video saved to: {os.path.abspath(h264_video)} ---")
        except Exception:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record Multi-Sensor Evaluation Video in CARLA.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--steps", type=int, default=150, help="Number of simulation steps to record")
    parser.add_argument("--img-width", type=int, default=400, help="Camera width per view")
    parser.add_argument("--img-height", type=int, default=300, help="Camera height per view")
    parser.add_argument("--output-video", type=str, default="/workspace/output_screenshots/driving_multiview.mp4", help="Output MP4 path")
    parser.add_argument("--npc-vehicles", type=int, default=20, help="Number of NPC traffic vehicles")
    parser.add_argument("--backbone", type=str, default="lav", choices=["lav", "erfnet", "resnet18", "resnet34"], help="Vision backbone used during training")
    parser.add_argument("--checkpoint", type=str, default="/workspace/checkpoints/ppo_carla_best.pth", help="Path to PyTorch PPO model checkpoint")

    args = parser.parse_args()

    record_multiview_eval(
        host=args.host,
        port=args.port,
        steps=args.steps,
        img_width=args.img_width,
        img_height=args.img_height,
        output_video=args.output_video,
        num_npc_vehicles=args.npc_vehicles,
        checkpoint=args.checkpoint,
        backbone=args.backbone
    )
