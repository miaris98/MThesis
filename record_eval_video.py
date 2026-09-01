"""CARLA 3-Camera Autonomous Driving Evaluation Studio (1280x720 HD 4-View Layout)."""
import os
import sys
import time
import glob
import argparse
import subprocess
from typing import Optional
import numpy as np
import cv2
import torch

carla_root = os.environ.get("CARLA_ROOT", "/workspace/carla")
carla_dist_path = os.path.join(carla_root, "PythonAPI", "carla", "dist")
if os.path.exists(carla_dist_path):
    eggs = glob.glob(os.path.join(carla_dist_path, "carla-*-py3*.egg"))
    for p in eggs:
        if p not in sys.path:
            sys.path.insert(0, p)
    if os.path.join(carla_root, "PythonAPI", "carla") not in sys.path:
        sys.path.insert(0, os.path.join(carla_root, "PythonAPI", "carla"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import carla
except ImportError:
    carla = None

from src.envs.camera_easycarla_env import CameraEasyCarlaEnv
from src.models.actor_critic import ActorCriticPPO
from src.utils.evaluation_studio import draw_hud


def _upload_to_mlflow(video_path: str, experiment_name: str = "CARLA_PPO_RL") -> None:
    """Attach the finished evaluation video to the most recent MLflow run's artifacts."""
    try:
        import mlflow
    except ImportError:
        return

    # The training supervisor picks the MLflow port dynamically (iptables-mapped ports
    # first, then a fixed fallback list), so probe the same candidates rather than
    # assuming one. An explicit MLFLOW_TRACKING_URI always wins.
    candidates = []
    if os.environ.get("MLFLOW_TRACKING_URI"):
        candidates.append(os.environ["MLFLOW_TRACKING_URI"])
    candidates += [f"http://127.0.0.1:{p}" for p in (10100, 10200, 9090, 7070, 4040, 5000)]

    for uri in candidates:
        try:
            mlflow.set_tracking_uri(uri)
            exp = mlflow.get_experiment_by_name(experiment_name)
            if not exp:
                continue
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=1)
            if runs.empty:
                continue
            run_id = runs.iloc[0].run_id
            mlflow.tracking.MlflowClient().log_artifact(run_id, video_path)
            print(f"📦 [MLflow] Synced {os.path.basename(video_path)} to run {run_id} at {uri}")
            return
        except Exception:
            continue
    print("ℹ️  [MLflow] No reachable tracking server found; video saved locally only.")


def record_eval_video(
    port: int = 2000, steps: int = 600, max_episode_steps: int = 500,
    min_speed: float = 0.0, output_video: str = "/workspace/eval_video.mp4",
    num_npc_vehicles: int = 3, num_walkers: int = 10,
    checkpoint: Optional[str] = None, backbone: str = "lav",
    policy_arch: str = "qwen100m", weights_path: Optional[str] = None,
    town: str = "Town10HD_Opt", algo: str = "ppo", sac_policy_arch: str = "mlp"
) -> None:
    if not weights_path:
        for wp in ["/workspace/pretrained_carla/model_0030_0.pth", "./papers_and_code/LAV/lav_pretrained.pth", "/workspace/MThesis/papers_and_code/LAV/lav_pretrained.pth"]:
            if os.path.exists(wp):
                weights_path = wp
                break

    if not checkpoint:
        prefix = "sac" if algo == "sac" else "ppo"
        for cp in [f"/workspace/checkpoints/{prefix}_carla_best.pth", f"/workspace/checkpoints/{prefix}_carla_latest.pth",
                   f"./checkpoints/{prefix}_carla_best.pth", f"./checkpoints/{prefix}_carla_latest.pth"]:
            if os.path.exists(cp):
                checkpoint = cp
                break

    print("==============================================================")
    print("   🎥 Starting 3-Camera CARLA PPO Driving Evaluation Studio   ")
    print("==============================================================")
    print(f"Algo: {algo.upper()} | Checkpoint: {checkpoint} | Arch: {policy_arch} | Backbone: {backbone} | Map: {town} (Port: {port})")

    easy_params = {
        'number_of_vehicles': num_npc_vehicles, 'number_of_walkers': num_walkers, 'display_size': 256,
        'max_past_step': 1, 'dt': 0.05, 'discrete': False, 'ego_vehicle_filter': 'vehicle.tesla.model3',
        'port': port, 'town': town, 'max_time_episode': max_episode_steps, 'max_waypoints': 12,
        'visualize_waypoints': False, 'desired_speed': 8, 'max_ego_spawn_times': 200, 'view_mode': 'top',
        'traffic': 'off', 'lidar_max_range': 50.0, 'max_nearby_vehicles': 5,
        'surrounding_vehicle_spawned_randomly': True, 'img_width': 256, 'img_height': 256,
    }

    env = CameraEasyCarlaEnv(params=easy_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if algo == "sac":
        from src.models.sac_networks import SACActorCritic
        agent = SACActorCritic(
            action_dim=3, features_dim=512, backbone_name=backbone,
            freeze_backbone=True, use_pretrained=True, weights_path=weights_path,
            policy_arch=sac_policy_arch
        ).to(device)
    else:
        agent = ActorCriticPPO(
            action_dim=3, features_dim=512, backbone_name=backbone, policy_arch=policy_arch,
            freeze_backbone=True, use_pretrained=True, weights_path=weights_path
        ).to(device)

    if checkpoint and os.path.exists(checkpoint):
        try:
            agent.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
            print(f"✓ Successfully loaded trained {algo.upper()} policy: {checkpoint}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}).")
    agent.eval()

    chase_w, chase_h = 580, 430
    chase_cam_holder, chase_frame_buffer = [None], [None]

    def cleanup_chase_camera():
        chase_frame_buffer[0] = None
        if chase_cam_holder[0] is not None:
            try:
                if hasattr(chase_cam_holder[0], 'is_listening') and chase_cam_holder[0].is_listening:
                    chase_cam_holder[0].stop()
                if hasattr(chase_cam_holder[0], 'is_alive') and chase_cam_holder[0].is_alive:
                    chase_cam_holder[0].destroy()
            except Exception:
                pass
            chase_cam_holder[0] = None

    def setup_chase_camera():
        if chase_cam_holder[0] is not None and hasattr(chase_cam_holder[0], 'is_alive') and chase_cam_holder[0].is_alive:
            return
        cleanup_chase_camera()
        if not hasattr(env, 'easy_env') or not hasattr(env.easy_env, 'ego') or env.easy_env.ego is None:
            return
        bp = env.easy_env.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(chase_w))
        bp.set_attribute("image_size_y", str(chase_h))
        bp.set_attribute("fov", "95")
        tf = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-12.0))
        try:
            cam = env.easy_env.world.spawn_actor(bp, tf, attach_to=env.easy_env.ego)
            cam.listen(lambda img: chase_frame_buffer.__setitem__(0, np.reshape(np.frombuffer(img.raw_data, dtype=np.uint8), (img.height, img.width, 4))[:, :, :3].copy()))
            chase_cam_holder[0] = cam
        except Exception:
            pass

    canvas_w, canvas_h, header_h, cam_row_h, bottom_row_h, cam_w = 1280, 720, 50, 240, 430, 426
    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    raw_path = output_video.replace(".mp4", "_raw.mp4")
    video_writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (canvas_w, canvas_h))

    recorded_valid_steps, saved_episodes = 0, 0
    obs, info = env.reset()
    setup_chase_camera()

    try:
        while recorded_valid_steps < steps:
            ep_frames, episode_speeds, episode_reward, done, step_in_ep = [], [], 0.0, False, 0
            while not done:
                step_in_ep += 1
                model_rgb = obs["image"]
                spd = float(obs["speed"][0])
                episode_speeds.append(spd)

                img_t = torch.as_tensor(model_rgb, dtype=torch.uint8, device=device).unsqueeze(0)
                spd_t = torch.as_tensor([spd], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.inference_mode():
                    if algo == "sac":
                        vis = agent.extract_visual_features(img_t)
                        action, _ = agent.sample_action(vis, spd_t, deterministic=True)
                    else:
                        action, _, _, _ = agent.get_action_and_value(img_t, spd_t, deterministic=True)
                act = action.cpu().numpy()[0]
                t_val = float(np.clip((act[0] + 1.0) / 2.0, 0.0, 1.0))
                s_val = float(np.clip(act[1], -1.0, 1.0))
                b_val = float(np.clip(act[2], 0.0, 1.0)) if act[2] > 0.4 and t_val < 0.3 else 0.0

                next_obs, reward, term, trunc, info = env.step(act)
                done = term or trunc
                episode_reward += float(reward)

                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                header = np.zeros((header_h, canvas_w, 3), dtype=np.uint8)
                cv2.rectangle(header, (0, 0), (canvas_w, header_h), (15, 23, 42), -1)
                cv2.putText(header, "CARLA 3-CAMERA AUTONOMOUS DRIVING EVALUATION STUDIO", (25, 33), cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(header, f"Vision: 3x RGB + {backbone.upper()} | Map: {town}", (750, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)
                canvas[0:header_h, 0:canvas_w] = header

                left_bgr = cv2.cvtColor(model_rgb[:, :256, :], cv2.COLOR_RGB2BGR)
                center_bgr = cv2.cvtColor(model_rgb[:, 256:512, :], cv2.COLOR_RGB2BGR)
                right_bgr = cv2.cvtColor(model_rgb[:, 512:, :], cv2.COLOR_RGB2BGR)
                canvas[header_h:header_h + cam_row_h, 0:cam_w] = cv2.resize(left_bgr, (cam_w, cam_row_h))
                canvas[header_h:header_h + cam_row_h, cam_w:cam_w * 2] = cv2.resize(center_bgr, (cam_w, cam_row_h))
                canvas[header_h:header_h + cam_row_h, cam_w * 2:canvas_w] = cv2.resize(right_bgr, (canvas_w - cam_w * 2, cam_row_h))

                row2_y = header_h + cam_row_h
                f_chase = cv2.resize(chase_frame_buffer[0], (chase_w, bottom_row_h)) if chase_frame_buffer[0] is not None else np.zeros((bottom_row_h, chase_w, 3), dtype=np.uint8)
                canvas[row2_y:canvas_h, 0:chase_w] = f_chase

                avg_spd = np.mean(episode_speeds) if episode_speeds else spd
                draw_hud(
                    canvas=canvas, x_offset=chase_w, y_offset=row2_y, panel_w=canvas_w - chase_w, panel_h=bottom_row_h,
                    speed_kmh=spd, throttle=t_val, steer=s_val, brake=b_val, step=recorded_valid_steps + len(ep_frames) + 1,
                    total_steps=steps, episode_count=saved_episodes + 1, ep_reward=episode_reward, ep_avg_speed=avg_spd,
                    backbone_name=backbone, is_at_red_light=bool(info.get("is_at_red_light", False)),
                    num_vehicles=num_npc_vehicles, num_walkers=num_walkers
                )
                ep_frames.append(canvas)
                obs = next_obs

            if step_in_ep >= 15:
                saved_episodes += 1
                for f in ep_frames:
                    video_writer.write(f)
                recorded_valid_steps += len(ep_frames)
                print(f"✓ [VIDEO] Ep #{saved_episodes} | Steps: {len(ep_frames)} | Total Video Frames: {recorded_valid_steps}/{steps}")

            try:
                obs, info = env.reset()
                setup_chase_camera()
            except Exception:
                break
    finally:
        video_writer.release()
        print("✓ Raw video frames written. Finalizing H.264 MP4 conversion...")
        if os.path.exists(raw_path):
            try:
                cmd = ["ffmpeg", "-y", "-i", raw_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "faststart", output_video]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(output_video):
                    os.remove(raw_path)
                    print(f"🎬 Successfully generated HD driving video: {output_video}")
                    _upload_to_mlflow(output_video)
            except Exception as e:
                print(f"ffmpeg notice: {e}")
        cleanup_chase_camera()
        try:
            env.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="CARLA 3-Camera PPO Driving Evaluation Studio")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--min-speed", type=float, default=0.0)
    parser.add_argument("--output-video", type=str, default="/workspace/eval_video.mp4")
    parser.add_argument("--num-vehicles", "--npc-vehicles", dest="npc_vehicles", type=int, default=3)
    parser.add_argument("--num-walkers", type=int, default=10)
    parser.add_argument("--backbone", type=str, default="lav")
    parser.add_argument("--policy-arch", type=str, default="qwen100m")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--town", type=str, default="Town10HD_Opt")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="Which trained policy to evaluate")
    parser.add_argument("--sac-policy-arch", type=str, default="mlp", choices=["mlp", "qwen100m", "qwen500m", "qwen900m"], help="Architecture the SAC checkpoint was trained with")
    args = parser.parse_args()

    record_eval_video(
        port=args.port, steps=args.steps, max_episode_steps=args.max_episode_steps,
        min_speed=args.min_speed, output_video=args.output_video, num_npc_vehicles=args.npc_vehicles,
        num_walkers=args.num_walkers, checkpoint=args.checkpoint, backbone=args.backbone,
        policy_arch=args.policy_arch, weights_path=args.weights_path, town=args.town, algo=args.algo, sac_policy_arch=args.sac_policy_arch
    )


if __name__ == "__main__":
    main()
