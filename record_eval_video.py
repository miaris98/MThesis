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
    if os.path.join(carla_root, "PythonAPI", "carla") not in sys.path:
        sys.path.insert(0, os.path.join(carla_root, "PythonAPI", "carla"))

import carla


def draw_hud(canvas, x_offset, y_offset, panel_w, panel_h, model_rgb, speed_kmh, throttle, steer, brake, step, total_steps, backbone_name):
    """Draw a rich telemetry and model-input dashboard panel."""
    # Dark panel background
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (25, 28, 36), -1)
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (55, 65, 81), 2)

    # Panel Header
    cv2.putText(canvas, "MODEL INPUT & TELEMETRY", (x_offset + 20, y_offset + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (x_offset + 20, y_offset + 40), (x_offset + panel_w - 20, y_offset + 40), (75, 85, 99), 1)

    # 1. Model Input: 256x256 Front RGB Camera Feed
    cv2.putText(canvas, "Input 1: Front Camera (256x256 RGB)", (x_offset + 20, y_offset + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1, cv2.LINE_AA)
    
    cam_box_x = x_offset + (panel_w - 256) // 2
    cam_box_y = y_offset + 80
    if model_rgb is not None:
        canvas[cam_box_y:cam_box_y + 256, cam_box_x:cam_box_x + 256] = model_rgb
    cv2.rectangle(canvas, (cam_box_x - 2, cam_box_y - 2), (cam_box_x + 258, cam_box_y + 258), (59, 130, 246), 2)

    # 2. Model Input: Speed State
    info_y = cam_box_y + 280
    cv2.putText(canvas, "Input 2: Speed Kinematics", (x_offset + 20, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1, cv2.LINE_AA)
    
    speed_text = f"{speed_kmh:.1f} km/h"
    norm_speed_text = f"(Normalized: {speed_kmh/50.0:.2f})"
    cv2.putText(canvas, speed_text, (x_offset + 30, info_y + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (16, 185, 129), 2, cv2.LINE_AA)
    cv2.putText(canvas, norm_speed_text, (x_offset + 190, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (156, 163, 175), 1, cv2.LINE_AA)

    # 3. Model Output: Policy Action Controls
    action_y = info_y + 70
    cv2.line(canvas, (x_offset + 20, action_y - 15), (x_offset + panel_w - 20, action_y - 15), (75, 85, 99), 1)
    cv2.putText(canvas, f"Policy Actions ({backbone_name.upper()} Backbone)", (x_offset + 20, action_y + 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    bar_w = 200
    bar_h = 16
    bar_x = x_offset + 110

    # Throttle Bar
    t_y = action_y + 35
    cv2.putText(canvas, "Throttle:", (x_offset + 20, t_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, t_y), (bar_x + bar_w, t_y + bar_h), (55, 65, 81), -1)
    fill_t = int(bar_w * np.clip(throttle, 0.0, 1.0))
    cv2.rectangle(canvas, (bar_x, t_y), (bar_x + fill_t, t_y + bar_h), (34, 197, 94), -1)
    cv2.putText(canvas, f"{throttle*100:.0f}%", (bar_x + bar_w + 10, t_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 197, 94), 1, cv2.LINE_AA)

    # Steer Bar (Center-indexed)
    s_y = t_y + 30
    cv2.putText(canvas, "Steer:", (x_offset + 20, s_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, s_y), (bar_x + bar_w, s_y + bar_h), (55, 65, 81), -1)
    center_x = bar_x + bar_w // 2
    steer_val = np.clip(steer, -1.0, 1.0)
    steer_px = int((bar_w // 2) * steer_val)
    if steer_px >= 0:
        cv2.rectangle(canvas, (center_x, s_y), (center_x + steer_px, s_y + bar_h), (59, 130, 246), -1)
    else:
        cv2.rectangle(canvas, (center_x + steer_px, s_y), (center_x, s_y + bar_h), (249, 115, 22), -1)
    cv2.line(canvas, (center_x, s_y - 2), (center_x, s_y + bar_h + 2), (255, 255, 255), 1)
    cv2.putText(canvas, f"{steer:+.2f}", (bar_x + bar_w + 10, s_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (59, 130, 246), 1, cv2.LINE_AA)

    # Brake Bar
    b_y = s_y + 30
    cv2.putText(canvas, "Brake:", (x_offset + 20, b_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, b_y), (bar_x + bar_w, b_y + bar_h), (55, 65, 81), -1)
    fill_b = int(bar_w * np.clip(brake, 0.0, 1.0))
    cv2.rectangle(canvas, (bar_x, b_y), (bar_x + fill_b, b_y + bar_h), (239, 68, 68), -1)
    cv2.putText(canvas, f"{brake*100:.0f}%", (bar_x + bar_w + 10, b_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 1, cv2.LINE_AA)

    # Step & Evaluation Progress Footer
    foot_y = panel_h - 25
    cv2.line(canvas, (x_offset + 20, foot_y - 15), (x_offset + panel_w - 20, foot_y - 15), (75, 85, 99), 1)
    cv2.putText(canvas, f"Evaluation Step: {step:03d}/{total_steps}  |  20 FPS Fixed Delta",
                (x_offset + 20, foot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (156, 163, 175), 1, cv2.LINE_AA)


def record_eval_video(
    host="127.0.0.1",
    port=2000,
    steps=300,
    output_video="/workspace/output_screenshots/driving_eval_model_input.mp4",
    num_npc_vehicles=3,
    checkpoint="/workspace/checkpoints/ppo_carla_best.pth",
    backbone="lav",
    town="Town10HD_Opt"
):
    print(f"==============================================================")
    print(f"   🎥 Starting CARLA PPO Model Evaluation & Video Recording   ")
    print(f"==============================================================")
    print(f"Checkpoint: {checkpoint}")
    print(f"Backbone: {backbone.upper()} | Map: {town} | Recording Steps: {steps}")
    print(f"Output Video: {output_video}")

    client = carla.Client(host, port)
    client.set_timeout(60.0)
    
    # Verify current world
    world = client.get_world()
    cur_map = world.get_map().name
    if town not in cur_map:
        print(f"Switching map to {town}...")
        world = client.load_world(town)
    
    # 1. Enable Synchronous Simulation Mode (20 FPS / 50ms)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 2. Setup TrafficManager for surrounding NPC vehicles
    traffic_manager = client.get_trafficmanager(port + 6000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)

    actor_list = []
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # 3. Spawn Surrounding NPC Vehicles
    print(f"Spawning {num_npc_vehicles} surrounding NPC traffic vehicles...")
    for _ in range(num_npc_vehicles):
        sp = random.choice(spawn_points)
        npc_bp = random.choice(blueprint_library.filter("vehicle.*"))
        npc = world.try_spawn_actor(npc_bp, sp)
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            actor_list.append(npc)

    # 4. Spawn Ego Vehicle
    ego_bp = blueprint_library.find("vehicle.lincoln.mkz_2020")
    if ego_bp is None:
        ego_bp = blueprint_library.find("vehicle.tesla.model3")
    
    ego_spawn = random.choice(spawn_points)
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn)
    while ego_vehicle is None:
        ego_spawn = random.choice(spawn_points)
        ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn)

    actor_list.append(ego_vehicle)
    ego_vehicle.set_autopilot(False)

    # 5. Load Trained PyTorch PPO Model
    import torch
    from train_rl_agent import ActorCriticPPO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ActorCriticPPO(action_dim=3, backbone_name=backbone).to(device)

    if checkpoint and os.path.exists(checkpoint):
        try:
            agent.load_state_dict(torch.load(checkpoint, map_location=device))
            agent.eval()
            print(f"✓ Successfully loaded trained PPO checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Using fresh policy weights.")
    else:
        print(f"Warning: Checkpoint path not found ({checkpoint}).")

    # 6. Mount Sensors
    # Sensor A: Main Third-Person Follow / Chase Camera (880x660)
    chase_w, chase_h = 880, 660
    chase_bp = blueprint_library.find("sensor.camera.rgb")
    chase_bp.set_attribute("image_size_x", str(chase_w))
    chase_bp.set_attribute("image_size_y", str(chase_h))
    chase_bp.set_attribute("fov", "95")
    chase_tf = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-12.0))
    chase_cam = world.spawn_actor(chase_bp, chase_tf, attach_to=ego_vehicle)
    actor_list.append(chase_cam)

    # Sensor B: Front RGB Camera (256x256) - EXACT MODEL INPUT
    model_bp = blueprint_library.find("sensor.camera.rgb")
    model_bp.set_attribute("image_size_x", "256")
    model_bp.set_attribute("image_size_y", "256")
    model_bp.set_attribute("fov", "90")
    model_tf = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=-8.0))
    model_cam = world.spawn_actor(model_bp, model_tf, attach_to=ego_vehicle)
    actor_list.append(model_cam)

    # Sensor Buffers
    frames = {"chase": None, "model": None}

    def process_chase(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (img.height, img.width, 4))
        frames["chase"] = arr[:, :, :3].copy() # BGR

    def process_model(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (img.height, img.width, 4))
        frames["model"] = arr[:, :, :3].copy() # BGR

    chase_cam.listen(process_chase)
    model_cam.listen(process_model)

    # Video Dimensions
    panel_w = 400
    canvas_w = chase_w + panel_w
    canvas_h = chase_h + 50 # 50px top header banner
    
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (canvas_w, canvas_h))

    print(f"Recording {steps} evaluation steps at 20 FPS...")

    # Action State
    throttle_val = 0.0
    steer_val = 0.0
    brake_val = 0.0

    try:
        for i in range(steps):
            world.tick()

            while frames["chase"] is None or frames["model"] is None:
                time.sleep(0.002)

            # Get vehicle velocity & speed
            vel = ego_vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            # Model Inference
            # Model takes: (256, 256, 3) RGB image + speed scalar
            model_rgb_input = np.ascontiguousarray(frames["model"][:, :, ::-1])  # BGR -> RGB contiguous
            img_tensor = torch.as_tensor(model_rgb_input, dtype=torch.uint8, device=device).unsqueeze(0)
            spd_tensor = torch.as_tensor([speed_kmh], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.inference_mode():
                action, _, _, _ = agent.get_action_and_value(img_tensor, spd_tensor)
            
            act = action.cpu().numpy()[0]
            throttle_val = float(np.clip((act[0] + 1.0) / 2.0, 0.0, 1.0))
            steer_val = float(np.clip(act[1], -1.0, 1.0))
            brake_val = float(np.clip((act[2] - 0.2) / 0.8, 0.0, 1.0)) if act[2] > 0.2 else 0.0

            # Apply Vehicle Control
            control = carla.VehicleControl(
                throttle=throttle_val,
                steer=steer_val,
                brake=brake_val
            )
            ego_vehicle.apply_control(control)

            # Build Full Frame Canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # 1. Top Global Header Banner
            header = np.zeros((50, canvas_w, 3), dtype=np.uint8)
            cv2.rectangle(header, (0, 0), (canvas_w, 50), (17, 24, 39), -1)
            cv2.putText(header, "CARLA Deep RL Autonomous Driving Evaluation", (20, 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(header, f"Model: PPO + {backbone.upper()}  |  Map: {town}  |  Checkpoint: {os.path.basename(checkpoint)}",
                        (620, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (156, 163, 175), 1, cv2.LINE_AA)

            canvas[0:50, 0:canvas_w] = header

            # 2. Main View: Chase Camera (Left)
            canvas[50:50 + chase_h, 0:chase_w] = frames["chase"]

            # 3. Telemetry & Model Input Side Panel (Right)
            draw_hud(
                canvas=canvas,
                x_offset=chase_w,
                y_offset=50,
                panel_w=panel_w,
                panel_h=chase_h,
                model_rgb=frames["model"],
                speed_kmh=speed_kmh,
                throttle=throttle_val,
                steer=steer_val,
                brake=brake_val,
                step=i + 1,
                total_steps=steps,
                backbone_name=backbone
            )

            video_writer.write(canvas)

            if (i + 1) % 50 == 0 or i == steps - 1:
                print(f"[Step {i+1:03d}/{steps}] Speed: {speed_kmh:4.1f} km/h | Throttle: {throttle_val:.2f} | Steer: {steer_val:+.2f} | Brake: {brake_val:.2f}")

    finally:
        print("Finalizing video recording and releasing sensors...")
        video_writer.release()

        # Stop camera listeners
        for s in [chase_cam, model_cam]:
            try:
                s.stop()
            except Exception:
                pass

        # Drain render pipeline
        try:
            world.tick()
        except Exception:
            pass

        # Batch destroy all actors safely
        if actor_list:
            destroy_cmds = [carla.command.DestroyActor(a.id) for a in actor_list if a is not None]
            client.apply_batch(destroy_cmds)

        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        print(f"--- Evaluation Video Saved: {os.path.abspath(output_video)} ---")

        # Convert to Web-Compatible H.264 MP4 via ffmpeg
        h264_video = output_video.replace(".mp4", "_h264.mp4")
        try:
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", output_video,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                h264_video
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(h264_video):
                print(f"--- Web-Optimized H.264 Video Saved: {os.path.abspath(h264_video)} ---")
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record Model Input & Driving Telemetry Video in CARLA.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps to record (default: 300 steps = 15s)")
    parser.add_argument("--output-video", type=str, default="/workspace/output_screenshots/driving_eval_model_input.mp4", help="Output MP4 path")
    parser.add_argument("--npc-vehicles", type=int, default=3, help="Number of NPC traffic vehicles")
    parser.add_argument("--backbone", type=str, default="lav", choices=["lav", "erfnet", "resnet18", "resnet34"], help="Vision backbone used during training")
    parser.add_argument("--checkpoint", type=str, default="/workspace/checkpoints/ppo_carla_best.pth", help="Path to PyTorch PPO model checkpoint")
    parser.add_argument("--town", type=str, default="Town10HD_Opt", help="CARLA map town")

    args = parser.parse_args()

    record_eval_video(
        host=args.host,
        port=args.port,
        steps=args.steps,
        output_video=args.output_video,
        num_npc_vehicles=args.npc_vehicles,
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        town=args.town
    )
