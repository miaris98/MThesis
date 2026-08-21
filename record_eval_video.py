import os
import sys
import time
import math
import glob
import random
import numpy as np
import cv2
import torch

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
from camera_easycarla_env import CameraEasyCarlaEnv
from train_rl_agent import ActorCriticPPO


def draw_hud(canvas, x_offset, y_offset, panel_w, panel_h, model_rgb_256, speed_kmh, throttle, steer, brake, step, total_steps, episode_count, ep_reward, ep_avg_speed, backbone_name, status_str="Active"):
    """Draw clean telemetry panel showing exact model inputs and policy outputs."""
    # Dark card background
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (22, 27, 34), -1)
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (48, 54, 61), 2)

    # 1. Section Header: Model Inputs
    cv2.putText(canvas, "MODEL SENSOR INPUTS", (x_offset + 20, y_offset + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (x_offset + 20, y_offset + 38), (x_offset + panel_w - 20, y_offset + 38), (65, 75, 90), 1)

    # Input 1: Front RGB Camera Feed (256x256)
    cv2.putText(canvas, "Input 1: Front Camera (256x256 RGB)", (x_offset + 20, y_offset + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (209, 213, 219), 1, cv2.LINE_AA)
    
    cam_x = x_offset + (panel_w - 256) // 2
    cam_y = y_offset + 72
    if model_rgb_256 is not None:
        canvas[cam_y:cam_y + 256, cam_x:cam_x + 256] = model_rgb_256
    cv2.rectangle(canvas, (cam_x - 2, cam_y - 2), (cam_x + 258, cam_y + 258), (34, 197, 94), 2)

    # Input 2: Speed Kinematics
    spd_y = cam_y + 278
    cv2.putText(canvas, "Input 2: Speed Kinematics", (x_offset + 20, spd_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (209, 213, 219), 1, cv2.LINE_AA)
    
    speed_text = f"{speed_kmh:4.1f} km/h"
    norm_text = f"Normalized: {speed_kmh/50.0:.2f}"
    cv2.putText(canvas, speed_text, (x_offset + 25, spd_y + 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (16, 185, 129), 2, cv2.LINE_AA)
    cv2.putText(canvas, norm_text, (x_offset + 230, spd_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (156, 163, 175), 1, cv2.LINE_AA)

    # 2. Section Header: Policy Actions
    act_y = spd_y + 65
    cv2.line(canvas, (x_offset + 20, act_y - 12), (x_offset + panel_w - 20, act_y - 12), (65, 75, 90), 1)
    cv2.putText(canvas, f"MODEL POLICY ACTIONS ({backbone_name.upper()})", (x_offset + 20, act_y + 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    bar_w = 200
    bar_h = 16
    bar_x = x_offset + 110

    # Throttle Bar
    t_y = act_y + 32
    cv2.putText(canvas, "Throttle:", (x_offset + 20, t_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, t_y), (bar_x + bar_w, t_y + bar_h), (55, 65, 81), -1)
    fill_t = int(bar_w * np.clip(throttle, 0.0, 1.0))
    cv2.rectangle(canvas, (bar_x, t_y), (bar_x + fill_t, t_y + bar_h), (34, 197, 94), -1)
    cv2.putText(canvas, f"{throttle*100:3.0f}%", (bar_x + bar_w + 10, t_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 197, 94), 1, cv2.LINE_AA)

    # Steer Bar (Centered)
    s_y = t_y + 28
    cv2.putText(canvas, "Steer:", (x_offset + 20, s_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, s_y), (bar_x + bar_w, s_y + bar_h), (55, 65, 81), -1)
    center_x = bar_x + bar_w // 2
    steer_val = np.clip(steer, -1.0, 1.0)
    steer_px = int((bar_w // 2) * steer_val)
    if steer_px >= 0:
        cv2.rectangle(canvas, (center_x, s_y), (center_x + steer_px, s_y + bar_h), (59, 130, 246), -1)
    else:
        cv2.rectangle(canvas, (center_x + steer_px, s_y), (center_x, s_y + bar_h), (249, 115, 22), -1)
    cv2.line(canvas, (center_x, s_y - 2), (center_x, s_y + bar_h + 2), (255, 255, 255), 2)
    cv2.putText(canvas, f"{steer:+.2f}", (bar_x + bar_w + 10, s_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (59, 130, 246), 1, cv2.LINE_AA)

    # Brake Bar
    b_y = s_y + 28
    cv2.putText(canvas, "Brake:", (x_offset + 20, b_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, b_y), (bar_x + bar_w, b_y + bar_h), (55, 65, 81), -1)
    fill_b = int(bar_w * np.clip(brake, 0.0, 1.0))
    cv2.rectangle(canvas, (bar_x, b_y), (bar_x + fill_b, b_y + bar_h), (239, 68, 68), -1)
    cv2.putText(canvas, f"{brake*100:3.0f}%", (bar_x + bar_w + 10, b_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 1, cv2.LINE_AA)

    # 3. Episode & Status Footer
    foot_y = panel_h - 40
    cv2.line(canvas, (x_offset + 20, foot_y - 10), (x_offset + panel_w - 20, foot_y - 10), (65, 75, 90), 1)
    cv2.putText(canvas, f"Episode: #{episode_count}  |  Reward: {ep_reward:+.1f}  |  Avg Speed: {ep_avg_speed:.1f} km/h",
                (x_offset + 20, foot_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 158, 11) if ep_reward >= 0 else (239, 68, 68), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Step: {step:04d}/{total_steps}  |  Status: {status_str}",
                (x_offset + 20, foot_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (156, 163, 175), 1, cv2.LINE_AA)


def record_eval_video(
    port=2000,
    steps=600,
    output_video="/workspace/output_screenshots/driving_eval_model_input.mp4",
    num_npc_vehicles=3,
    checkpoint="/workspace/checkpoints/ppo_carla_best.pth",
    backbone="lav",
    town="Town10HD_Opt"
):
    print(f"==============================================================")
    print(f"   🎥 Starting CARLA PPO Driving Evaluation & Video Studio   ")
    print(f"==============================================================")
    print(f"Checkpoint: {checkpoint}")
    print(f"Vision Backbone: {backbone.upper()} | Map: {town} | Total Steps: {steps}")
    print(f"Output Video: {output_video}")

    # 1. Instantiate the Exact Training Environment (CameraEasyCarlaEnv)
    easy_params = {
        'number_of_vehicles': num_npc_vehicles,
        'number_of_walkers': 0,
        'display_size': 256,
        'max_past_step': 1,
        'dt': 0.05,
        'discrete': False,
        'discrete_acc': [-3.0, 1.5, 3.0],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-0.3, 0.3],
        'ego_vehicle_filter': 'vehicle.lincoln.mkz_2020',
        'port': port,
        'town': town,
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

    env = CameraEasyCarlaEnv(params=easy_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Trained Policy Checkpoint
    agent = ActorCriticPPO(action_dim=3, backbone_name=backbone).to(device)
    if checkpoint and os.path.exists(checkpoint):
        try:
            agent.load_state_dict(torch.load(checkpoint, map_location=device))
            agent.eval()
            print(f"✓ Successfully loaded trained PPO policy: {checkpoint}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Using initialized weights.")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint}.")

    # 3. Mount Spectator / Chase Camera on the Ego Vehicle for the Third-Person View
    chase_w, chase_h = 800, 650
    world = env.easy_env.world
    blueprint_library = world.get_blueprint_library()
    
    chase_cam_holder = [None]
    chase_frame_buffer = [None]

    def setup_chase_camera():
        if chase_cam_holder[0] is not None:
            try:
                chase_cam_holder[0].stop()
                chase_cam_holder[0].destroy()
            except Exception:
                pass
            chase_cam_holder[0] = None

        chase_bp = blueprint_library.find("sensor.camera.rgb")
        chase_bp.set_attribute("image_size_x", str(chase_w))
        chase_bp.set_attribute("image_size_y", str(chase_h))
        chase_bp.set_attribute("fov", "95")
        chase_tf = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-12.0))
        chase_cam = world.spawn_actor(chase_bp, chase_tf, attach_to=env.easy_env.ego)

        def _on_chase_img(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            arr = np.reshape(arr, (img.height, img.width, 4))
            chase_frame_buffer[0] = arr[:, :, :3].copy() # BGR

        chase_cam.listen(_on_chase_img)
        chase_cam_holder[0] = chase_cam

    # 4. Canvas Dimensions (1280 x 720 HD)
    panel_w = 480
    canvas_w = chase_w + panel_w
    canvas_h = chase_h + 50 # 50px header banner
    
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (canvas_w, canvas_h))

    print(f"Recording {steps} evaluation steps at 20 FPS...")

    # Initial Environment Reset
    obs, info = env.reset()
    setup_chase_camera()

    episode_count = 1
    episode_reward = 0.0
    episode_speeds = []

    throttle_val = 0.0
    steer_val = 0.0
    brake_val = 0.0
    status_str = "Active Driving"

    try:
        for step in range(steps):
            # Model Input 1: (256, 256, 3) RGB Front Camera Frame
            model_rgb_input = obs["image"] # uint8 [0, 255] RGB
            # Model Input 2: Speed Scalar
            speed_kmh = float(obs["speed"][0])
            episode_speeds.append(speed_kmh)

            # Model Inference
            img_tensor = torch.as_tensor(model_rgb_input, dtype=torch.uint8, device=device).unsqueeze(0)
            spd_tensor = torch.as_tensor([speed_kmh], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.inference_mode():
                action, _, _, _ = agent.get_action_and_value(img_tensor, spd_tensor)
            
            act = action.cpu().numpy()[0]
            throttle_val = float(np.clip((act[0] + 1.0) / 2.0, 0.0, 1.0))
            steer_val = float(np.clip(act[1], -1.0, 1.0))
            brake_val = float(np.clip((act[2] - 0.2) / 0.8, 0.0, 1.0)) if act[2] > 0.2 else 0.0

            # Step Environment using the exact continuous action space
            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            episode_reward += float(reward)

            # Wait for Chase camera frame
            while chase_frame_buffer[0] is None:
                time.sleep(0.002)

            # Build Video Frame (1280 x 720)
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # 1. Header Banner
            header = np.zeros((50, canvas_w, 3), dtype=np.uint8)
            cv2.rectangle(header, (0, 0), (canvas_w, 50), (15, 23, 42), -1)
            cv2.putText(header, "CARLA AUTONOMOUS DRIVING EVALUATION STUDIO", (25, 33),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(header, f"Model: PPO + {backbone.upper()}  |  Map: {town}  |  Checkpoint: {os.path.basename(checkpoint)}",
                        (640, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (156, 163, 175), 1, cv2.LINE_AA)
            canvas[0:50, 0:canvas_w] = header

            # 2. Main View: Chase Camera (Left)
            canvas[50:50 + chase_h, 0:chase_w] = chase_frame_buffer[0]
            # Overlay label on chase view
            cv2.rectangle(canvas, (10, 60), (280, 88), (0, 0, 0), -1)
            cv2.putText(canvas, "Third-Person Follow View", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Model Inputs & Telemetry Panel (Right)
            model_bgr_256 = cv2.cvtColor(model_rgb_input, cv2.COLOR_RGB2BGR)
            avg_spd = np.mean(episode_speeds) if len(episode_speeds) > 0 else speed_kmh

            draw_hud(
                canvas=canvas,
                x_offset=chase_w,
                y_offset=50,
                panel_w=panel_w,
                panel_h=chase_h,
                model_rgb_256=model_bgr_256,
                speed_kmh=speed_kmh,
                throttle=throttle_val,
                steer=steer_val,
                brake=brake_val,
                step=step + 1,
                total_steps=steps,
                episode_count=episode_count,
                ep_reward=episode_reward,
                ep_avg_speed=avg_spd,
                backbone_name=backbone,
                status_str=status_str
            )

            # If episode terminated on this step, show termination alert overlay
            if done:
                term_reason = info.get("termination_reason", "Episode Finished")
                alert_banner = np.zeros((60, chase_w, 3), dtype=np.uint8)
                bg_col = (185, 28, 28) if "Collision" in term_reason else ((217, 119, 6) if "Off-Road" in term_reason else (30, 64, 175))
                cv2.rectangle(alert_banner, (0, 0), (chase_w, 60), bg_col, -1)
                cv2.putText(alert_banner, f"EPISODE #{episode_count} TERMINATED: {term_reason.upper()}", (30, 26),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(alert_banner, f"Reward: {episode_reward:+.2f}  |  Avg Speed: {avg_spd:.1f} km/h  |  Respawning to New Route...", (30, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                canvas[50 + chase_h - 70:50 + chase_h - 10, 0:chase_w] = alert_banner

            video_writer.write(canvas)

            if done:
                term_reason = info.get("termination_reason", "Episode Finished")
                print(f"[Step {step+1:04d}/{steps}] Episode #{episode_count} Terminated: {term_reason} | Reward: {episode_reward:+.2f} | Avg Speed: {avg_spd:.1f} km/h")
                
                # Write 10 duplicate alert frames so the termination banner is clearly readable in the video
                for _ in range(10):
                    video_writer.write(canvas)

                # Reset environment for next episode
                obs, info = env.reset()
                setup_chase_camera()
                episode_count += 1
                episode_reward = 0.0
                episode_speeds = []
                status_str = "Active Driving"
            else:
                obs = next_obs

            if (step + 1) % 100 == 0:
                print(f"[Step {step+1:04d}/{steps}] Speed: {speed_kmh:4.1f} km/h | Throttle: {throttle_val:.2f} | Steer: {steer_val:+.2f} | Brake: {brake_val:.2f} | Ep: #{episode_count}")

    finally:
        print("Finalizing video recording and closing environment...")
        video_writer.release()

        if chase_cam_holder[0] is not None:
            try:
                chase_cam_holder[0].stop()
                chase_cam_holder[0].destroy()
            except Exception:
                pass

        env.close()

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
    parser = argparse.ArgumentParser(description="Record PPO Model Evaluation Video in CARLA.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--steps", type=int, default=600, help="Number of simulation steps to record (default: 600 steps = 30s)")
    parser.add_argument("--output-video", type=str, default="/workspace/output_screenshots/driving_eval_model_input.mp4", help="Output MP4 path")
    parser.add_argument("--npc-vehicles", type=int, default=3, help="Number of NPC traffic vehicles")
    parser.add_argument("--backbone", type=str, default="lav", choices=["lav", "erfnet", "resnet18", "resnet34"], help="Vision backbone used during training")
    parser.add_argument("--checkpoint", type=str, default="/workspace/checkpoints/ppo_carla_best.pth", help="Path to PyTorch PPO model checkpoint")
    parser.add_argument("--town", type=str, default="Town10HD_Opt", help="CARLA map town")

    args = parser.parse_args()

    record_eval_video(
        port=args.port,
        steps=args.steps,
        output_video=args.output_video,
        num_npc_vehicles=args.npc_vehicles,
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        town=args.town
    )
