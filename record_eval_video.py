"""Evaluation and video recording script for trained CARLA RL policies."""
import os
import sys
import argparse
import time
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.actor_critic import ActorCriticPPO
from src.envs.camera_easycarla_env import CameraEasyCarlaEnv
from src.utils.video_renderer import VideoRenderer


def record_evaluation_video():
    parser = argparse.ArgumentParser(description="Record CARLA RL Evaluation Video with HUD Dashboard.")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="lav")
    parser.add_argument("--policy-arch", type=str, default="qwen100m")
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--town", type=str, default="Town10HD_Opt")
    parser.add_argument("--num-vehicles", type=int, default=3)
    parser.add_argument("--num-walkers", type=int, default=10)
    parser.add_argument("--output-video", type=str, default="/workspace/eval_video.mp4")
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Auto-detect pretrained backbone weights
    if not args.weights_path:
        for wp in [
            "/workspace/pretrained_carla/model_0030_0.pth",
            "./papers_and_code/LAV/lav_pretrained.pth",
            "/workspace/MThesis/papers_and_code/LAV/lav_pretrained.pth"
        ]:
            if os.path.exists(wp):
                args.weights_path = wp
                break

    # Auto-detect checkpoint
    if not args.checkpoint_path:
        for cp in [
            "/workspace/checkpoints/ppo_carla_best.pth",
            "/workspace/checkpoints/ppo_carla_latest.pth",
            "./checkpoints/ppo_carla_best.pth",
            "./checkpoints/ppo_carla_latest.pth"
        ]:
            if os.path.exists(cp):
                args.checkpoint_path = cp
                break

    # 1. Initialize Environment
    params = {
        'number_of_vehicles': args.num_vehicles,
        'number_of_walkers': args.num_walkers,
        'frame_skip': 2,
        'port': args.port,
        'town': args.town,
        'max_time_episode': args.eval_steps,
        'desired_speed': 8,
        'img_width': 256,
        'img_height': 256
    }
    env = CameraEasyCarlaEnv(params=params)

    # 2. Load Agent Policy
    agent = ActorCriticPPO(
        action_dim=3,
        features_dim=512,
        backbone_name=args.backbone,
        policy_arch=args.policy_arch,
        freeze_backbone=True,
        use_pretrained=True,
        weights_path=args.weights_path
    ).to(device)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"--> Loading trained checkpoint: {args.checkpoint_path}")
        agent.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)
    else:
        print("⚠️  No checkpoint file found; recording with baseline model.")
    agent.eval()

    # 3. Setup Video Writer
    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    canvas_w = 768
    canvas_h = 256 + 200
    video_writer = cv2.VideoWriter(args.output_video, fourcc, 20.0, (canvas_w, canvas_h))

    print(f"--> Recording evaluation to: {args.output_video} ({args.eval_steps} steps)...")
    obs, _ = env.reset()
    ep_reward = 0.0
    ep_speeds = []
    ep_count = 1

    for step in range(args.eval_steps):
        img_t = torch.as_tensor(obs["image"], dtype=torch.uint8, device=device).unsqueeze(0)
        spd_t = torch.as_tensor(obs["speed"], dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(image=img_t, speed=spd_t, deterministic=True)

        action_np = action.cpu().numpy()[0]
        obs, reward, term, trunc, info = env.step(action_np)
        ep_reward += reward
        speed_val = float(info.get("speed_kmh", obs["speed"][0]))
        ep_speeds.append(speed_val)

        # Composite video frame
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:256, :768, :] = cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR)

        VideoRenderer.draw_hud(
            canvas=canvas,
            x_offset=0,
            y_offset=256,
            panel_w=768,
            panel_h=200,
            speed_kmh=speed_val,
            throttle=float(np.clip((action_np[0] + 1.0) / 2.0, 0.0, 1.0)),
            steer=float(action_np[1]),
            brake=float(action_np[2]) if action_np[2] > 0.2 else 0.0,
            step=step,
            total_steps=args.eval_steps,
            episode_count=ep_count,
            ep_reward=ep_reward,
            ep_avg_speed=float(np.mean(ep_speeds)),
            backbone_name=args.backbone,
            is_at_red_light=bool(info.get("is_at_red_light", False)),
            num_vehicles=args.num_vehicles,
            num_walkers=args.num_walkers
        )

        video_writer.write(canvas)

        if term or trunc:
            ep_count += 1
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_speeds = []

    video_writer.release()
    env.close()
    print(f"✓ Evaluation Video successfully recorded & saved to {args.output_video}!")


if __name__ == "__main__":
    record_evaluation_video()
