#!/usr/bin/env python3
"""World on Rails (WoR) Evaluation & Video Recording Studio.

Evaluates a trained or pretrained World on Rails agent inside CARLA simulator,
records HD evaluation videos with live HUD telemetry overlay, and syncs with MLflow.

Usage:
    python eval_wor.py --checkpoint /workspace/checkpoints/wor_10k/best_model.pth --record_video 1
"""
import argparse
import os
import sys
import time
import glob
import subprocess
import numpy as np
import cv2
import torch

# Ensure CARLA PythonAPI is accessible
carla_root = os.environ.get("CARLA_ROOT", "/workspace/carla")
carla_dist_path = os.path.join(carla_root, "PythonAPI", "carla", "dist")
if os.path.exists(carla_dist_path):
    eggs = glob.glob(os.path.join(carla_dist_path, "carla-*-py3*.egg"))
    for p in eggs:
        if p not in sys.path:
            sys.path.insert(0, p)
    if os.path.join(carla_root, "PythonAPI", "carla") not in sys.path:
        sys.path.insert(0, os.path.join(carla_root, "PythonAPI", "carla"))

try:
    import carla
except ImportError:
    carla = None

from src.agents.wor_agent import WorldOnRailsAgent

# The policy was trained on WorldOnRailsDataset._PDM_LITE_COMMAND_MAP, which remaps
# carla_garage/scenario_runner's raw RoadOption ids (1=LEFT..6=CHANGELANERIGHT) to a
# 0-indexed command space. "Just keep driving" is LANEFOLLOW (raw 4), which maps to
# index 3 - NOT 2 (index 2 is STRAIGHT). Sending the wrong index here biases the
# policy toward a turn it was never asked to make at this point in the route.
WOR_LANEFOLLOW_COMMAND = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate World on Rails Agent and Record Video in CARLA")
    parser.add_argument("--checkpoint", type=str, default="/workspace/checkpoints/wor_10k/best_model.pth", help="Path to custom model checkpoint (.pth)")
    parser.add_argument("--model_type", type=str, default="wor_nc", choices=["wor_nc", "wor_lb"], help="Pretrained PCLA model variant")
    parser.add_argument("--backbone", type=str, default="resnet34", help="Backbone architecture (resnet18/34/50)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA world port")
    parser.add_argument("--town", type=str, default="Town01", help="CARLA map/town")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=600, help="Max steps per episode (600 steps ~ 30s)")
    parser.add_argument("--record_video", type=int, default=1, help="Record evaluation video (1=True, 0=False)")
    parser.add_argument("--video_path", type=str, default="/workspace/wor_eval_video.mp4", help="Output MP4 video path")
    parser.add_argument("--num_vehicles", type=int, default=10, help="Number of NPC vehicles")
    parser.add_argument("--num_walkers", type=int, default=10, help="Number of pedestrians")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    return parser.parse_args()


def draw_eval_hud(frame: np.ndarray, speed_kmh: float, steer: float, throttle: float, brake: float, step: int, max_steps: int) -> np.ndarray:
    """Draws telemetry HUD overlay on evaluation video frames."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Semi-transparent top and bottom banner
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
    alpha = 0.65
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Telemetry text
    title_text = f"World on Rails (WoR) Evaluation | Step: {step:04d}/{max_steps:04d}"
    cv2.putText(frame, title_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    speed_text = f"Speed: {speed_kmh:4.1f} km/h"
    cv2.putText(frame, speed_text, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Control telemetry gauges
    steer_text = f"Steer: {steer:+.2f}"
    throt_text = f"Throttle: {throttle:.2f}"
    brake_text = f"Brake: {brake:.2f}"

    cv2.putText(frame, steer_text, (w - 480, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, throt_text, (w - 320, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2, cv2.LINE_AA)
    cv2.putText(frame, brake_text, (w - 150, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 255), 2, cv2.LINE_AA)

    return frame


def run_carla_evaluation(args, agent: WorldOnRailsAgent):
    """Executes live evaluation episode in CARLA and records MP4 video."""
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # Load target town if needed
    if args.town not in world.get_map().name:
        print(f"--> Loading map {args.town}...")
        world = client.load_world(args.town)

    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    blueprint_lib = world.get_blueprint_library()
    actor_list = []

    try:
        # 1. Spawn Ego Vehicle. spawn_points[0] is a fixed point that can happen to sit
        # too close to geometry on a given map/CARLA build - pick randomly among the
        # first few candidates instead of hard-coding index 0.
        import random
        ego_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
        ego_bp.set_attribute("role_name", "hero")
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points[:10]) if spawn_points else carla.Transform()
        ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
        ego_vehicle.set_simulate_physics(True)
        actor_list.append(ego_vehicle)
        print(f"✓ Ego vehicle spawned at {spawn_point.location}")

        # Collision sensor purely for diagnostics: a car stuck at ~0 km/h under full
        # throttle for the whole run is far more likely to be wedged against geometry
        # at spawn than a "bad" policy, since even a badly-trained policy drifts under
        # sustained throttle - this makes that distinguishable from the driving log.
        collision_log = {"count": 0, "first": None}
        col_bp = blueprint_lib.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego_vehicle)
        actor_list.append(col_sensor)

        def _on_collision(event):
            collision_log["count"] += 1
            if collision_log["first"] is None:
                collision_log["first"] = event.other_actor.type_id
        col_sensor.listen(_on_collision)

        # Let the vehicle settle under gravity for a few ticks before the eval loop
        # starts, so an interpenetrating spawn doesn't get mistaken for a bad policy.
        for _ in range(10):
            world.tick()
        if collision_log["count"] > 0:
            print(f"[WARNING] Ego vehicle collided with '{collision_log['first']}' during spawn settling "
                  f"({collision_log['count']} contacts) - it may be wedged against geometry.")

        # 2. Spawn Front RGB Camera for Agent (256x256)
        cam_agent_bp = blueprint_lib.find("sensor.camera.rgb")
        cam_agent_bp.set_attribute("image_size_x", "256")
        cam_agent_bp.set_attribute("image_size_y", "256")
        cam_agent_bp.set_attribute("fov", "100")
        cam_agent_transform = carla.Transform(carla.Location(x=1.3, z=1.3))
        cam_agent = world.spawn_actor(cam_agent_bp, cam_agent_transform, attach_to=ego_vehicle)
        actor_list.append(cam_agent)

        # CARLA raw_data is BGRA (see src/envs/camera_sensor.py) - reorder to true RGB
        # via the [2, 1, 0] channel swap, not a plain [:, :, :3] alpha-drop, which would
        # silently leave the image in BGR order. The policy was trained on PIL-loaded
        # (true RGB) frames, so feeding it BGR here means every eval run sees the world
        # through red/blue-swapped colors relative to training - sky-as-red, cars painted
        # in complementary colors, etc. - which is very plausibly part of why steering
        # looked so degenerate.
        agent_rgb_buffer = {"data": None}
        cam_agent.listen(lambda img: agent_rgb_buffer.update({"data": np.frombuffer(img.raw_data, dtype=np.uint8).reshape((256, 256, 4))[:, :, [2, 1, 0]]}))

        # 3. Spawn Third-Person Video Recording Camera (1280x720)
        cam_video_bp = blueprint_lib.find("sensor.camera.rgb")
        cam_video_bp.set_attribute("image_size_x", "1280")
        cam_video_bp.set_attribute("image_size_y", "720")
        cam_video_bp.set_attribute("fov", "90")
        cam_video_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0))
        cam_video = world.spawn_actor(cam_video_bp, cam_video_transform, attach_to=ego_vehicle)
        actor_list.append(cam_video)

        # Same BGRA raw buffer as the agent camera - keep it in native BGR order here
        # (drop alpha only) since that's exactly what cv2.VideoWriter/cv2.imshow expect,
        # instead of converting to RGB and then back to BGR for no reason.
        video_frame_buffer = {"data": None}
        cam_video.listen(lambda img: video_frame_buffer.update({"data": np.frombuffer(img.raw_data, dtype=np.uint8).reshape((720, 1280, 4))[:, :, :3]}))

        # Initialize VideoWriter. cv2's mp4v codec isn't broadly browser/player
        # compatible, so (matching record_eval_video.py) write raw mp4v frames first
        # and re-encode to H.264 via ffmpeg once recording finishes.
        video_writer = None
        raw_video_path = args.video_path.replace(".mp4", "_raw.mp4")
        if args.record_video:
            os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(raw_video_path, fourcc, 20.0, (1280, 720))
            print(f"📹 Recording evaluation video to: {args.video_path}")

        print(f"--> Running World on Rails evaluation for {args.max_steps} steps (~{args.max_steps/20:.1f}s)...")
        world.tick()

        for step in range(1, args.max_steps + 1):
            world.tick()

            # Extract agent sensor readings
            if agent_rgb_buffer["data"] is None:
                continue

            vel = ego_vehicle.get_velocity()
            speed_kmh = float(3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2))

            sensor_data = {
                "rgb_front": (step, agent_rgb_buffer["data"]),
                "speed": (step, speed_kmh / 3.6),
                "command": WOR_LANEFOLLOW_COMMAND
            }

            # Generate WoR Control
            control = agent.run_step(sensor_data)
            ego_vehicle.apply_control(control)

            # Record Video Frame with HUD. video_frame_buffer is already BGR (raw CARLA
            # buffer with alpha dropped) - it was previously run through
            # cv2.COLOR_RGB2BGR as if it were RGB, which re-swaps an already-BGR frame's
            # R/B channels and produced the reverted colors seen in the recorded video.
            if video_writer is not None and video_frame_buffer["data"] is not None:
                hud_frame = draw_eval_hud(
                    frame=video_frame_buffer["data"].copy(),
                    speed_kmh=speed_kmh,
                    steer=control.steer,
                    throttle=control.throttle,
                    brake=control.brake,
                    step=step,
                    max_steps=args.max_steps
                )
                video_writer.write(hud_frame)

            if step % 50 == 0:
                col_note = f" | Collisions: {collision_log['count']}" if collision_log["count"] else ""
                print(f"  [Step {step:04d}/{args.max_steps:04d}] Speed: {speed_kmh:4.1f} km/h | Steer: {control.steer:+.2f} | Throttle: {control.throttle:.2f} | Brake: {control.brake:.2f}{col_note}")

        if collision_log["count"] > 0:
            print(f"[WARNING] {collision_log['count']} total collision contacts during the run "
                  f"(first with '{collision_log['first']}') - low speed is likely a stuck/blocked "
                  f"vehicle, not necessarily a bad policy.")

        if video_writer is not None:
            video_writer.release()
            print("✓ Raw video frames written. Finalizing H.264 MP4 conversion...")
            if os.path.exists(raw_video_path):
                try:
                    cmd = ["ffmpeg", "-y", "-i", raw_video_path, "-vcodec", "libx264",
                           "-pix_fmt", "yuv420p", "-movflags", "faststart", args.video_path]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(args.video_path):
                        os.remove(raw_video_path)
                        print(f"🎬 Successfully generated HD driving video: {args.video_path}")
                except Exception as e:
                    print(f"ffmpeg notice: {e}")

    finally:
        print("--> Cleaning up simulation actors...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in actor_list:
            if actor is not None:
                actor.destroy()
        agent.destroy()
        print("✓ Cleanup complete.")


def run_standalone_test(agent: WorldOnRailsAgent):
    """Runs a standalone synthetic forward pass to verify agent control generation."""
    print("--> Running standalone inference test (No active CARLA server detected)...")
    dummy_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sensor_data = {
        "rgb_front": (0, dummy_rgb),
        "speed": (0, {"speed": 5.0}),
        "command": (0, WOR_LANEFOLLOW_COMMAND)
    }
    control = agent.run_step(sensor_data)
    print(f"✓ Agent generated control successfully: {control}")


def main():
    args = parse_args()

    print("=" * 65)
    print(" 🚗 World on Rails (WoR) Agent Evaluation & Video Recording")
    print(f" Checkpoint:      {args.checkpoint}")
    print(f" Target Town:     {args.town} | Max Steps: {args.max_steps}")
    print(f" Output Video:    {args.video_path if args.record_video else 'Disabled'}")
    print(f" Device:          {args.device.upper()}")
    print("=" * 65)

    # 1. Initialize PCLA Agent
    agent = WorldOnRailsAgent(
        checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None,
        model_type=args.model_type,
        backbone_name=args.backbone,
        device=args.device
    )

    # 2. Try connecting to CARLA
    if carla is not None:
        try:
            run_carla_evaluation(args, agent)
            return
        except Exception as e:
            print(f"[Notice] Could not connect to CARLA simulator at {args.host}:{args.port}: {e}")

    run_standalone_test(agent)


if __name__ == "__main__":
    main()
