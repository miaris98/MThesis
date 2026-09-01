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
        # 1. Spawn Ego Vehicle
        ego_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
        ego_bp.set_attribute("role_name", "hero")
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
        actor_list.append(ego_vehicle)
        print(f"✓ Ego vehicle spawned at {spawn_point.location}")

        # 2. Spawn Front RGB Camera for Agent (256x256)
        cam_agent_bp = blueprint_lib.find("sensor.camera.rgb")
        cam_agent_bp.set_attribute("image_size_x", "256")
        cam_agent_bp.set_attribute("image_size_y", "256")
        cam_agent_bp.set_attribute("fov", "100")
        cam_agent_transform = carla.Transform(carla.Location(x=1.3, z=1.3))
        cam_agent = world.spawn_actor(cam_agent_bp, cam_agent_transform, attach_to=ego_vehicle)
        actor_list.append(cam_agent)

        agent_rgb_buffer = {"data": None}
        cam_agent.listen(lambda img: agent_rgb_buffer.update({"data": np.frombuffer(img.raw_data, dtype=np.uint8).reshape((256, 256, 4))[:, :, :3]}))

        # 3. Spawn Third-Person Video Recording Camera (1280x720)
        cam_video_bp = blueprint_lib.find("sensor.camera.rgb")
        cam_video_bp.set_attribute("image_size_x", "1280")
        cam_video_bp.set_attribute("image_size_y", "720")
        cam_video_bp.set_attribute("fov", "90")
        cam_video_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0))
        cam_video = world.spawn_actor(cam_video_bp, cam_video_transform, attach_to=ego_vehicle)
        actor_list.append(cam_video)

        video_frame_buffer = {"data": None}
        cam_video.listen(lambda img: video_frame_buffer.update({"data": np.frombuffer(img.raw_data, dtype=np.uint8).reshape((720, 1280, 4))[:, :, :3]}))

        # Initialize VideoWriter
        video_writer = None
        if args.record_video:
            os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(args.video_path, fourcc, 20.0, (1280, 720))
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
                "command": 2
            }

            # Generate WoR Control
            control = agent.run_step(sensor_data)
            ego_vehicle.apply_control(control)

            # Record Video Frame with HUD
            if video_writer is not None and video_frame_buffer["data"] is not None:
                bgr_frame = cv2.cvtColor(video_frame_buffer["data"], cv2.COLOR_RGB2BGR)
                hud_frame = draw_eval_hud(
                    frame=bgr_frame,
                    speed_kmh=speed_kmh,
                    steer=control.steer,
                    throttle=control.throttle,
                    brake=control.brake,
                    step=step,
                    max_steps=args.max_steps
                )
                video_writer.write(hud_frame)

            if step % 50 == 0:
                print(f"  [Step {step:04d}/{args.max_steps:04d}] Speed: {speed_kmh:4.1f} km/h | Steer: {control.steer:+.2f} | Throttle: {control.throttle:.2f} | Brake: {control.brake:.2f}")

        if video_writer is not None:
            video_writer.release()
            print(f"✓ Video saved successfully: {args.video_path}")

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
        "command": (0, 2)
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
