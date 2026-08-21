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


def draw_hud(canvas, x_offset, y_offset, panel_w, panel_h, speed_kmh, throttle, steer, brake, step, total_steps, episode_count, ep_reward, ep_avg_speed, backbone_name, is_at_red_light=False, num_vehicles=3, num_walkers=10, status_str="Active"):
    """Draw clean telemetry panel showing traffic lights, speed kinematics, and policy controls."""
    # Dark card background
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (22, 27, 34), -1)
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (48, 54, 61), 2)

    # 1. Section Header: Telemetry & Traffic State
    cv2.putText(canvas, "REAL-TIME TELEMETRY & POLICY CONTROLS", (x_offset + 25, y_offset + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (x_offset + 25, y_offset + 40), (x_offset + panel_w - 25, y_offset + 40), (65, 75, 90), 1)

    # Traffic Light Status Badge
    badge_y = y_offset + 60
    if is_at_red_light:
        cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (30, 41, 59), -1)
        cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (239, 68, 68), 2)
        cv2.circle(canvas, (x_offset + 45, badge_y + 17), 7, (239, 68, 68), -1)
        cv2.putText(canvas, "TRAFFIC LIGHT: RED (Legal Stop Active - No Stall Penalty)", (x_offset + 65, badge_y + 23),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (239, 68, 68), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (30, 41, 59), -1)
        cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (34, 197, 94), 2)
        cv2.circle(canvas, (x_offset + 45, badge_y + 17), 7, (34, 197, 94), -1)
        cv2.putText(canvas, "TRAFFIC LIGHT: GREEN / OPEN ROAD NAVIGATION", (x_offset + 65, badge_y + 23),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (34, 197, 94), 1, cv2.LINE_AA)

    # 2. Left Column: Speed Kinematics
    col1_x = x_offset + 25
    col1_y = badge_y + 55
    cv2.putText(canvas, "Kinematics State:", (col1_x, col1_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1, cv2.LINE_AA)
    
    speed_text = f"{speed_kmh:4.1f} km/h"
    norm_text = f"Normalized: {speed_kmh/50.0:.2f}"
    cv2.putText(canvas, speed_text, (col1_x, col1_y + 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (16, 185, 129), 2, cv2.LINE_AA)
    cv2.putText(canvas, norm_text, (col1_x + 5, col1_y + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)

    # Environment Stats
    cv2.putText(canvas, f"NPC Traffic: {num_vehicles} Cars | {num_walkers} Walkers", (col1_x, col1_y + 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (59, 130, 246), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Episode: #{episode_count}  |  Reward: {ep_reward:+.1f}", (col1_x, col1_y + 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 158, 11) if ep_reward >= 0 else (239, 68, 68), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Avg Speed: {ep_avg_speed:.1f} km/h", (col1_x, col1_y + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)

    # Vertical Divider
    mid_x = x_offset + 300
    cv2.line(canvas, (mid_x, badge_y + 50), (mid_x, y_offset + panel_h - 25), (55, 65, 81), 1)

    # 3. Right Column: Policy Controls
    col2_x = mid_x + 25
    cv2.putText(canvas, f"Model Output ({backbone_name.upper()} Backbone):", (col2_x, col1_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1, cv2.LINE_AA)

    bar_w = 200
    bar_h = 16
    bar_x = col2_x + 90

    # Throttle Bar
    t_y = col1_y + 20
    cv2.putText(canvas, "Throttle:", (col2_x, t_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, t_y), (bar_x + bar_w, t_y + bar_h), (55, 65, 81), -1)
    fill_t = int(bar_w * np.clip(throttle, 0.0, 1.0))
    cv2.rectangle(canvas, (bar_x, t_y), (bar_x + fill_t, t_y + bar_h), (34, 197, 94), -1)
    cv2.putText(canvas, f"{throttle*100:3.0f}%", (bar_x + bar_w + 10, t_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 197, 94), 1, cv2.LINE_AA)

    # Steer Bar (Centered)
    s_y = t_y + 35
    cv2.putText(canvas, "Steer:", (col2_x, s_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
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
    b_y = s_y + 35
    cv2.putText(canvas, "Brake:", (col2_x, b_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, b_y), (bar_x + bar_w, b_y + bar_h), (55, 65, 81), -1)
    fill_b = int(bar_w * np.clip(brake, 0.0, 1.0))
    cv2.rectangle(canvas, (bar_x, b_y), (bar_x + fill_b, b_y + bar_h), (239, 68, 68), -1)
    cv2.putText(canvas, f"{brake*100:3.0f}%", (bar_x + bar_w + 10, b_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 1, cv2.LINE_AA)

    # Footer
    foot_y = y_offset + panel_h - 20
    cv2.line(canvas, (x_offset + 25, foot_y - 12), (x_offset + panel_w - 25, foot_y - 12), (65, 75, 90), 1)
    cv2.putText(canvas, f"Frame: {step:04d}/{total_steps} (20 FPS Fixed Delta)  |  Status: {status_str}",
                (x_offset + 25, foot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (156, 163, 175), 1, cv2.LINE_AA)


def record_eval_video(
    port=2000,
    steps=600,
    max_episode_steps=500,
    min_speed=8.0,
    output_video="/workspace/output_screenshots/driving_eval_model_input.mp4",
    num_npc_vehicles=3,
    num_walkers=10,
    checkpoint="/workspace/checkpoints/ppo_carla_best.pth",
    backbone="lav",
    town="Town10HD_Opt"
):
    print(f"==============================================================")
    print(f"   🎥 Starting 3-Camera CARLA PPO Driving Evaluation Studio  ")
    print(f"==============================================================")
    print(f"Checkpoint: {checkpoint}")
    print(f"Vision Backbone: {backbone.upper()} | Map: {town}")
    print(f"Sensors: 3 RGB Cameras (Left, Center, Right) + Speed")
    print(f"NPC Traffic: {num_npc_vehicles} Vehicles | {num_walkers} Pedestrians")
    print(f"Output Video: {output_video}")

    # 1. Instantiate 3-Camera Environment (CameraEasyCarlaEnv)
    easy_params = {
        'number_of_vehicles': num_npc_vehicles,
        'number_of_walkers': num_walkers,
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
        'max_time_episode': max_episode_steps,
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

    # 3. Mount Spectator / Chase Camera for Third-Person Follow View
    chase_w, chase_h = 580, 420
    world = env.easy_env.world
    blueprint_library = world.get_blueprint_library()
    
    chase_cam_holder = [None]
    chase_frame_buffer = [None]

    def setup_chase_camera():
        chase_frame_buffer[0] = None
        if chase_cam_holder[0] is not None:
            try:
                chase_cam_holder[0].stop()
            except Exception:
                pass
            chase_cam_holder[0] = None

        chase_bp = blueprint_library.find("sensor.camera.rgb")
        chase_bp.set_attribute("image_size_x", str(chase_w))
        chase_bp.set_attribute("image_size_y", str(chase_h))
        chase_bp.set_attribute("fov", "95")
        chase_tf = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-12.0))
        chase_cam = env.easy_env.world.spawn_actor(chase_bp, chase_tf, attach_to=env.easy_env.ego)

        def _on_chase_img(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            arr = np.reshape(arr, (img.height, img.width, 4))
            chase_frame_buffer[0] = arr[:, :, :3].copy()

        chase_cam.listen(_on_chase_img)
        chase_cam_holder[0] = chase_cam

    # 4. Canvas Dimensions (1280 x 720 HD)
    canvas_w = 1280
    canvas_h = 720
    header_h = 50
    cam_row_h = 240
    bottom_row_h = 430
    cam_w = 426
    
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (canvas_w, canvas_h))

    print(f"Recording {steps} valid evaluation steps at 20 FPS...")

    target_valid_steps = steps
    recorded_valid_steps = 0
    saved_episodes = 0
    attempted_episodes = 0

    # Initial Environment Reset
    obs, info = env.reset()
    setup_chase_camera()

    throttle_val = 0.0
    steer_val = 0.0
    brake_val = 0.0
    status_str = "Active Driving"

    try:
        while recorded_valid_steps < target_valid_steps:
            attempted_episodes += 1
            ep_frames = []
            episode_speeds = []
            episode_reward = 0.0
            done = False
            step_in_ep = 0

            while not done:
                step_in_ep += 1

                # Model Input 1: (256, 768, 3) 3-Camera RGB Panorama
                model_rgb_input = obs["image"]
                # Model Input 2: Speed Scalar
                speed_kmh = float(obs["speed"][0])
                episode_speeds.append(speed_kmh)

                # Model Inference (Deterministic Action)
                img_tensor = torch.as_tensor(model_rgb_input, dtype=torch.uint8, device=device).unsqueeze(0)
                spd_tensor = torch.as_tensor([speed_kmh], dtype=torch.float32, device=device).unsqueeze(0)

                with torch.inference_mode():
                    action, _, _, _ = agent.get_action_and_value(img_tensor, spd_tensor, deterministic=True)
                
                act = action.cpu().numpy()[0]
                throttle_val = float(np.clip((act[0] + 1.0) / 2.0, 0.0, 1.0))
                steer_val = float(np.clip(act[1], -1.0, 1.0))
                brake_val = float(np.clip((act[2] - 0.2) / 0.8, 0.0, 1.0)) if act[2] > 0.2 else 0.0

                # Step Environment
                next_obs, reward, terminated, truncated, info = env.step(act)
                done = terminated or truncated
                episode_reward += float(reward)
                is_at_red_light = bool(info.get("is_at_red_light", False))

                # Wait for Chase camera frame
                while chase_frame_buffer[0] is None:
                    time.sleep(0.002)

                # Build Full 1280x720 Canvas
                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

                # 1. Global Header Banner
                header = np.zeros((header_h, canvas_w, 3), dtype=np.uint8)
                cv2.rectangle(header, (0, 0), (canvas_w, header_h), (15, 23, 42), -1)
                cv2.putText(header, "CARLA 3-CAMERA AUTONOMOUS DRIVING EVALUATION STUDIO", (25, 33),
                            cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(header, f"Vision: 3x RGB + {backbone.upper()}  |  Map: {town}  |  Model: ppo_carla_best.pth",
                            (670, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)
                canvas[0:header_h, 0:canvas_w] = header

                # 2. Top Row: 3 Front RGB Cameras (Left | Center | Right)
                # Split the 256x768 panorama into 3x 256x256 images
                left_rgb = cv2.cvtColor(model_rgb_input[:, :256, :], cv2.COLOR_RGB2BGR)
                center_rgb = cv2.cvtColor(model_rgb_input[:, 256:512, :], cv2.COLOR_RGB2BGR)
                right_rgb = cv2.cvtColor(model_rgb_input[:, 512:, :], cv2.COLOR_RGB2BGR)

                f_left = cv2.resize(left_rgb, (cam_w, cam_row_h))
                cv2.rectangle(f_left, (0, 0), (cam_w, 26), (0, 0, 0), -1)
                cv2.putText(f_left, "1. Front-Left Camera (-55 deg)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (59, 130, 246), 1, cv2.LINE_AA)
                canvas[header_h:header_h + cam_row_h, 0:cam_w] = f_left

                f_center = cv2.resize(center_rgb, (cam_w, cam_row_h))
                cv2.rectangle(f_center, (0, 0), (cam_w, 26), (0, 0, 0), -1)
                cv2.putText(f_center, "2. Front-Center Camera (0 deg)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (34, 197, 94), 1, cv2.LINE_AA)
                canvas[header_h:header_h + cam_row_h, cam_w:cam_w * 2] = f_center

                r_w = canvas_w - cam_w * 2
                f_right = cv2.resize(right_rgb, (r_w, cam_row_h))
                cv2.rectangle(f_right, (0, 0), (r_w, 26), (0, 0, 0), -1)
                cv2.putText(f_right, "3. Front-Right Camera (+55 deg)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (249, 115, 22), 1, cv2.LINE_AA)
                canvas[header_h:header_h + cam_row_h, cam_w * 2:canvas_w] = f_right

                # 3. Bottom Row: Third-Person Follow View (Left) + Telemetry Dashboard (Right)
                row2_y = header_h + cam_row_h
                f_chase = cv2.resize(chase_frame_buffer[0], (chase_w, bottom_row_h))
                cv2.rectangle(f_chase, (0, 0), (chase_w, 26), (0, 0, 0), -1)
                cv2.putText(f_chase, f"4. Follow View (Traffic: {num_npc_vehicles} Cars, {num_walkers} Walkers)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                canvas[row2_y:canvas_h, 0:chase_w] = f_chase

                avg_spd = np.mean(episode_speeds) if len(episode_speeds) > 0 else speed_kmh
                draw_hud(
                    canvas=canvas,
                    x_offset=chase_w,
                    y_offset=row2_y,
                    panel_w=canvas_w - chase_w,
                    panel_h=bottom_row_h,
                    speed_kmh=speed_kmh,
                    throttle=throttle_val,
                    steer=steer_val,
                    brake=brake_val,
                    step=recorded_valid_steps + len(ep_frames) + 1,
                    total_steps=target_valid_steps,
                    episode_count=saved_episodes + 1,
                    ep_reward=episode_reward,
                    ep_avg_speed=avg_spd,
                    backbone_name=backbone,
                    is_at_red_light=is_at_red_light,
                    num_vehicles=num_npc_vehicles,
                    num_walkers=num_walkers,
                    status_str=status_str
                )

                ep_frames.append(canvas)
                obs = next_obs

            # Episode Finished - Determine if it was dynamic driving or stalled
            term_reason = info.get("termination_reason", "Episode Finished")
            max_spd = max(episode_speeds) if episode_speeds else 0.0
            avg_spd = np.mean(episode_speeds) if episode_speeds else 0.0
            is_valid_drive = (max_spd >= min_speed) and (avg_spd >= 4.0) and (step_in_ep >= 25) and ("Stalled" not in term_reason)

            if not is_valid_drive:
                print(f"[Attempt #{attempted_episodes} DISCARDED] Stalled/Slow (Max: {max_spd:.1f} km/h, Avg: {avg_spd:.1f} km/h, Steps: {step_in_ep}, Reason: {term_reason}) - Skipped.")
            else:
                saved_episodes += 1
                # Attach ending alert banner
                alert_canvas = ep_frames[-1].copy()
                alert_banner = np.zeros((60, chase_w, 3), dtype=np.uint8)
                bg_col = (185, 28, 28) if "Collision" in term_reason else ((217, 119, 6) if "Off-Road" in term_reason else (30, 64, 175))
                cv2.rectangle(alert_banner, (0, 0), (chase_w, 60), bg_col, -1)
                cv2.putText(alert_banner, f"EPISODE #{saved_episodes} FINISHED: {term_reason.upper()}", (30, 26),
                            cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(alert_banner, f"Reward: {episode_reward:+.2f}  |  Avg Speed: {avg_spd:.1f} km/h (Max: {max_spd:.1f})", (30, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                alert_canvas[row2_y + bottom_row_h - 70:row2_y + bottom_row_h - 10, 0:chase_w] = alert_banner

                # Write all valid episode frames to video
                for f in ep_frames:
                    video_writer.write(f)
                for _ in range(12):  # 0.6s hold on end screen
                    video_writer.write(alert_canvas)

                recorded_valid_steps += len(ep_frames)
                print(f"✓ [SAVED TO VIDEO] Episode #{saved_episodes} ({term_reason}) | Steps: {len(ep_frames)} | Max Speed: {max_spd:.1f} km/h | Avg: {avg_spd:.1f} km/h | Total Video Frames: {recorded_valid_steps}/{target_valid_steps}")

            # Reset environment for next episode
            obs, info = env.reset()
            setup_chase_camera()

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
    parser = argparse.ArgumentParser(description="Record 3-Camera PPO Model Evaluation Video in CARLA.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--steps", type=int, default=600, help="Total number of valid driving frames to record (default: 600 steps = 30s)")
    parser.add_argument("--max-episode-steps", type=int, default=500, help="Maximum steps allowed per episode before timeout (default: 500 steps = 25s)")
    parser.add_argument("--min-speed", type=float, default=8.0, help="Minimum peak speed (km/h) required for an episode to be saved to video")
    parser.add_argument("--output-video", type=str, default="/workspace/output_screenshots/driving_eval_model_input.mp4", help="Output MP4 path")
    parser.add_argument("--num-vehicles", "--npc-vehicles", dest="npc_vehicles", type=int, default=3, help="Number of NPC traffic vehicles")
    parser.add_argument("--num-walkers", type=int, default=10, help="Number of pedestrian walkers in the environment")
    parser.add_argument("--backbone", type=str, default="lav", choices=["lav", "erfnet", "resnet18", "resnet34"], help="Vision backbone used during training")
    parser.add_argument("--checkpoint", type=str, default="/workspace/checkpoints/ppo_carla_best.pth", help="Path to PyTorch PPO model checkpoint")
    parser.add_argument("--town", type=str, default="Town10HD_Opt", help="CARLA map town")

    args = parser.parse_args()

    record_eval_video(
        port=args.port,
        steps=args.steps,
        max_episode_steps=args.max_episode_steps,
        min_speed=args.min_speed,
        output_video=args.output_video,
        num_npc_vehicles=args.npc_vehicles,
        num_walkers=args.num_walkers,
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        town=args.town
    )
