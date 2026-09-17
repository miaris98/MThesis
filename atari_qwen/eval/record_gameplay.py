"""Record high-definition Atari gameplay video with professional HUD overlay."""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from atari_qwen.config.atari_config import AtariConfig, get_config
from atari_qwen.envs.atari_wrappers import make_atari_env
from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic


def draw_hud(
    frame: np.ndarray,
    env_id: str,
    episode: int,
    score: float,
    high_score: float,
    lives: int,
    action_str: str,
    current_sec: float,
    total_sec: float,
    model_name: str = "Qwen-Tiny + ImpalaCNN",
    human_baseline: float = 30.5
) -> np.ndarray:
    """Overlay a professional telemetry HUD on the top and bottom of the frame."""
    h, w, c = frame.shape
    
    # Create canvas with top banner (55px) and bottom banner (40px)
    top_bar_h = 55
    bottom_bar_h = 40
    canvas = np.zeros((h + top_bar_h + bottom_bar_h, w, 3), dtype=np.uint8)
    
    # Fill background banners with sleek dark slate
    canvas[:top_bar_h] = (22, 22, 28)
    canvas[top_bar_h:top_bar_h + h] = frame
    canvas[top_bar_h + h:] = (22, 22, 28)
    
    # Accent lines
    cv2.line(canvas, (0, top_bar_h - 1), (w, top_bar_h - 1), (0, 200, 255), 2)
    cv2.line(canvas, (0, top_bar_h + h), (w, top_bar_h + h), (100, 100, 120), 1)
    
    # Top Banner - Line 1: Game & Model info + Timer
    game_clean = env_id.replace("NoFrameskip-v4", "").replace("-v4", "")
    time_str = f"{int(current_sec // 60):02d}:{int(current_sec % 60):02d} / {int(total_sec // 60):02d}:{int(total_sec % 60):02d}"
    
    cv2.putText(canvas, f"AGENT: {model_name}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"GAME: {game_clean}", (w // 2 - 50, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"TIME: {time_str}", (w - 170, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    
    # Top Banner - Line 2: Score, Lives, High Score
    cv2.putText(canvas, f"SCORE: {score:.0f}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"HIGH: {high_score:.0f}", (140, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"LIVES: {lives}", (w // 2 - 40, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"EPISODE: {episode}", (w - 170, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Bottom Banner: Action + Benchmark context
    action_color = (0, 255, 200) if action_str in ["FIRE", "UP"] else (255, 255, 255)
    cv2.putText(canvas, f"ACTION: {action_str}", (12, top_bar_h + h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, action_color, 1, cv2.LINE_AA)
    
    bench_str = f"Human Benchmark: {human_baseline:.1f} pts"
    cv2.putText(canvas, bench_str, (w - 240, top_bar_h + h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 180), 1, cv2.LINE_AA)
    
    return canvas


def record_gameplay(
    checkpoint_path: Optional[str] = None,
    env_id: str = "BreakoutNoFrameskip-v4",
    duration_seconds: int = 180,
    fps: int = 30,
    output_path: str = "results/atari_qwen/breakout_gameplay_3min.mp4",
    device_str: str = "auto",
    deterministic: bool = True,
    scale_factor: int = 4,
    model_preset: str = "tiny",
    encoder_type: str = "impala_cnn",
    seed: int = 42
) -> str:
    """Record a continuous multi-episode gameplay video for the specified duration."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n========================================================")
    print(f"--> Initializing Atari Gameplay Video Recorder")
    print(f"--> Target Duration: {duration_seconds}s ({duration_seconds/60:.1f} mins) | FPS: {fps}")
    print(f"--> Environment: {env_id} | Device: {device}")
    print(f"--> Output Path: {output_path}")
    print(f"========================================================\n")
    
    # Build evaluation environment
    env_fn = make_atari_env(
        env_id=env_id,
        seed=seed,
        idx=0,
        noop_max=30,
        frame_stack=4,
        clip_reward=False,
        episodic_life=False
    )
    env = env_fn()
    action_meanings = env.unwrapped.get_action_meanings()
    action_dim = env.action_space.n
    
    # Load model
    model = QwenAtariActorCritic(
        action_dim=action_dim,
        in_channels=4,
        preset=model_preset,
        encoder_type=encoder_type
    ).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"--> Loading weights from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            # Try direct load
            try:
                model.load_state_dict(ckpt)
            except Exception as e:
                print(f"--> Warning loading weights: {e}")
        print("--> Checkpoint loaded successfully.")
    else:
        print(f"--> Notice: No checkpoint specified or file not found ({checkpoint_path}). Running with initialized architecture.")
        
    model.eval()
    
    # Setup initial frame to determine dimensions
    obs, info = env.reset(seed=seed)
    raw_screen = env.unwrapped.ale.getScreenRGB()
    h_orig, w_orig, _ = raw_screen.shape
    h_scaled, w_scaled = h_orig * scale_factor, w_orig * scale_factor
    
    # Account for HUD banner heights
    top_bar_h = 55
    bottom_bar_h = 40
    video_h = h_scaled + top_bar_h + bottom_bar_h
    video_w = w_scaled
    
    total_frames_target = duration_seconds * fps
    frames_per_agent_step = max(1, fps // 15)  # 4-skip at 60Hz is 15Hz agent decisions -> 2 frames per step @ 30fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (video_w, video_h))
    
    frames_written = 0
    current_episode = 1
    episode_score = 0.0
    high_score = 0.0
    lives = getattr(env.unwrapped, "ale", None).lives() if hasattr(env.unwrapped, "ale") else 5
    current_action_name = "NOOP"
    
    start_wall_time = time.time()
    
    while frames_written < total_frames_target:
        # 1. Model inference
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
        with torch.no_grad():
            actor_repr, _ = model._forward_transformer(obs_t)
            logits = model.actor_head(actor_repr)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.distributions.Categorical(logits=logits)
                action = probs.sample().item()
                
        current_action_name = action_meanings[action] if action < len(action_meanings) else f"ACT_{action}"
        
        # 2. Step environment
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            
        episode_score += reward
        high_score = max(high_score, episode_score)
        lives = env.unwrapped.ale.lives() if hasattr(env.unwrapped, "ale") else lives
        
        # 3. Render screen & composite HUD
        raw_screen = env.unwrapped.ale.getScreenRGB()
        screen_bgr = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2BGR)
        screen_scaled = cv2.resize(screen_bgr, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST)
        
        current_sec = frames_written / fps
        annotated_frame = draw_hud(
            frame=screen_scaled,
            env_id=env_id,
            episode=current_episode,
            score=episode_score,
            high_score=high_score,
            lives=lives,
            action_str=current_action_name,
            current_sec=current_sec,
            total_sec=duration_seconds,
            model_name=f"Qwen-{model_preset.capitalize()} + {encoder_type.upper()}"
        )
        
        # Write frames (repeating for smooth video pacing matching real Atari speed)
        for _ in range(frames_per_agent_step):
            if frames_written < total_frames_target:
                writer.write(annotated_frame)
                frames_written += 1
                
        # Handle Episode End / Game Over
        if done:
            print(f"--> Episode {current_episode} completed: Score = {episode_score:.0f} | Video Progress: {frames_written}/{total_frames_target} frames ({current_sec:.1f}s)")
            current_episode += 1
            episode_score = 0.0
            obs, info = env.reset(seed=seed + current_episode)
            
        # Logging heartbeat every 30 seconds
        if frames_written % (30 * fps) < frames_per_agent_step:
            pct = (frames_written / total_frames_target) * 100
            print(f"--> Recording Progress: {pct:5.1f}% | Time: {current_sec:5.1f}s / {duration_seconds}s | Current Score: {episode_score:.0f}")
            
    writer.release()
    env.close()
    
    elapsed = time.time() - start_wall_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n========================================================")
    print(f"--> Video Recording Successfully Completed!")
    print(f"--> Output: {output_path} ({file_size_mb:.2f} MB)")
    print(f"--> Total Frames Written: {frames_written} @ {fps} fps ({frames_written/fps:.1f}s)")
    print(f"--> Total Elapsed Real Time: {elapsed:.1f}s")
    print(f"========================================================\n")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Atari Qwen gameplay video with HUD")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4", help="Atari environment ID")
    parser.add_argument("--duration-seconds", type=int, default=180, help="Duration of video in seconds (180 = 3 mins)")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (fps)")
    parser.add_argument("--output-path", type=str, default="results/atari_qwen/breakout_gameplay_3min.mp4", help="Output MP4 file path")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor for Atari resolution (default 4x)")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "small", "100m", "500m"], help="Model preset")
    parser.add_argument("--encoder-type", type=str, default="impala_cnn", choices=["nature_cnn", "impala_cnn", "patch"], help="Visual encoder")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of greedy argmax")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment")
    args = parser.parse_args()
    
    record_gameplay(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        output_path=args.output_path,
        deterministic=not args.stochastic,
        scale_factor=args.scale,
        model_preset=args.preset,
        encoder_type=args.encoder_type,
        seed=args.seed
    )
