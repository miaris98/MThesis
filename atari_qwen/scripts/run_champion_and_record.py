"""Automated Champion Training and 3-Minute Gameplay Video Recorder for Atari Qwen."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna
import torch

from atari_qwen.eval.record_gameplay import record_gameplay


def wait_for_optuna(study_name: str, storage_path: str, poll_interval: int = 15) -> optuna.Study:
    """Wait until the running Optuna study completes all requested trials."""
    print("================================================================")
    print("--> Waiting for Optuna Hyperparameter Study to conclude...")
    print(f"--> Study Name: {study_name}")
    print(f"--> Storage:    {storage_path}")
    print("================================================================")
    
    last_reported_trial = -1
    
    while True:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_path)
            trials = study.trials
            latest = trials[-1] if trials else None
            
            # Check if an optimize_optuna process is still running
            ps = subprocess.run(["pgrep", "-f", "optimize_optuna.py"], capture_output=True, text=True)
            is_running = len(ps.stdout.strip()) > 0
            
            if latest and latest.number != last_reported_trial:
                print(f"--> Current: Trial #{latest.number} [{latest.state.name}] | Intermediates: {latest.intermediate_values}")
                last_reported_trial = latest.number
                
            if not is_running:
                print("--> Optuna training process has concluded!")
                break
        except Exception as e:
            print(f"--> Polling note: {e}")
            
        time.sleep(poll_interval)
        
    print(f"\n========================================================")
    print(f"OPTUNA STUDY COMPLETE! Total trials: {len(study.trials)}")
    print(f"Champion Trial #{study.best_trial.number}: Return = {study.best_value:.2f}")
    print("Champion Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")
    print(f"========================================================\n")
    return study


def train_champion(
    study: optuna.Study,
    env_id: str = "BreakoutNoFrameskip-v4",
    total_timesteps: int = 400_000,
    num_envs: int = 16,
    python_bin: str = sys.executable
) -> str:
    """Train the champion model with winning hyperparameters and return checkpoint path."""
    best = study.best_trial
    params = best.params
    
    lr = params["learning_rate"]
    ent_coef = params["ent_coef"]
    clip_coef = params["clip_coef"]
    num_steps = params["num_steps"]
    gae_lambda = params["gae_lambda"]
    encoder_type = params["encoder_type"]
    model_preset = params["model_preset"]
    
    exp_name = f"champion_trial_{best.number}_{model_preset}_{encoder_type}"
    out_dir = f"results/atari_qwen/{exp_name}"
    
    cmd = [
        python_bin, "atari_qwen/training/train_ppo.py",
        "--env-id", env_id,
        "--preset", model_preset,
        "--encoder-type", encoder_type,
        "--learning-rate", str(lr),
        "--ent-coef", str(ent_coef),
        "--clip-coef", str(clip_coef),
        "--num-steps", str(num_steps),
        "--gae-lambda", str(gae_lambda),
        "--num-envs", str(num_envs),
        "--total-timesteps", str(total_timesteps),
        "--exp-name", exp_name,
        "--log-dir", "results/atari_qwen"
    ]
    
    print(f"--> Launching Champion Model Training...")
    print(f"--> Command: {' '.join(cmd)}\n")
    
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    
    if p.returncode != 0:
        raise RuntimeError(f"Champion training failed with return code {p.returncode}")
        
    # Check for best model
    matches = sorted(Path("results/atari_qwen").glob(f"*{exp_name}*/**/model_best.pt"), key=os.path.getmtime)
    if matches:
        ckpt_path = matches[-1]
    else:
        latest_matches = sorted(Path("results/atari_qwen").glob(f"*{exp_name}*/**/model_latest.pt"), key=os.path.getmtime)
        if latest_matches:
            ckpt_path = latest_matches[-1]
        else:
            raise FileNotFoundError(f"Checkpoint not found for {exp_name} in results/atari_qwen")
            
    print(f"\n--> Champion training finished! Checkpoint: {ckpt_path}\n")
    return str(ckpt_path)


def record_and_sync(
    checkpoint_path: str,
    study: optuna.Study,
    env_id: str = "BreakoutNoFrameskip-v4",
    duration_seconds: int = 180,
    fps: int = 30,
    output_path: str = "results/atari_qwen/breakout_gameplay_3min.mp4",
    repo_id: str = "Miaris/mthesis-atari-qwen"
):
    """Record the 3-minute video and upload to Hugging Face Hub."""
    best = study.best_trial
    preset = best.params.get("model_preset", "tiny")
    encoder_type = best.params.get("encoder_type", "impala_cnn")
    
    print("================================================================")
    print("--> Recording 3-Minute Gameplay Video with HUD Telemetry...")
    print(f"--> Target: {duration_seconds} seconds ({duration_seconds/60:.1f} mins) @ {fps} fps")
    print(f"--> Checkpoint: {checkpoint_path}")
    print(f"--> Destination: {output_path}")
    print("================================================================")
    
    video_file = record_gameplay(
        checkpoint_path=checkpoint_path,
        env_id=env_id,
        duration_seconds=duration_seconds,
        fps=fps,
        output_path=output_path,
        model_preset=preset,
        encoder_type=encoder_type,
        deterministic=True
    )
    
    print(f"\n--> Gameplay video created successfully: {video_file}")
    
    # Upload to Hugging Face Hub if available
    try:
        from huggingface_hub import HfApi
        token = os.getenv("HF_TOKEN")
        api = HfApi(token=token)
        print(f"--> Uploading video to Hugging Face Hub repo '{repo_id}'...")
        api.upload_file(
            path_or_fileobj=video_file,
            path_in_repo="videos/breakout_gameplay_3min.mp4",
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✓ Video successfully uploaded to Hugging Face: https://huggingface.co/{repo_id}/blob/main/videos/breakout_gameplay_3min.mp4")
    except Exception as e:
        print(f"--> HF Hub upload note: {e}")


def main():
    parser = argparse.ArgumentParser(description="Wait for Optuna, train champion, and record 3-min gameplay video")
    parser.add_argument("--study-name", type=str, default="qwen_BreakoutNoFrameskip-v4_tuning")
    parser.add_argument("--storage", type=str, default="sqlite:///results/atari_qwen/optuna.db")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--video-duration", type=int, default=180)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--output-video", type=str, default="results/atari_qwen/breakout_gameplay_3min.mp4")
    parser.add_argument("--repo-id", type=str, default="Miaris/mthesis-atari-qwen")
    parser.add_argument("--skip-train-if-exists", type=str, default=None, help="Use existing checkpoint path")
    args = parser.parse_args()
    
    study = wait_for_optuna(study_name=args.study_name, storage_path=args.storage)
    
    if args.skip_train_if_exists and os.path.exists(args.skip_train_if_exists):
        ckpt_path = args.skip_train_if_exists
        print(f"--> Using existing checkpoint: {ckpt_path}")
    else:
        ckpt_path = train_champion(study=study, env_id=args.env_id)
        
    record_and_sync(
        checkpoint_path=ckpt_path,
        study=study,
        env_id=args.env_id,
        duration_seconds=args.video_duration,
        fps=args.video_fps,
        output_path=args.output_video,
        repo_id=args.repo_id
    )


if __name__ == "__main__":
    main()
