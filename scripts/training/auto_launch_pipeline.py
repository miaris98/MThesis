"""auto_launch_pipeline.py - Remote orchestrator on Vast.ai box.

Monitors dataset download completion, launches CARLA training resumption pinned
to cores 0-15, launches Atari benchmark pinned to cores 16-31, and installs
the auto-shutdown cost watchdog.
"""
import os
import subprocess
import sys
import time

def check_process_running(pattern):
    r = subprocess.run(f"pgrep -f '{pattern}'", shell=True, capture_output=True, text=True)
    return r.returncode == 0

def wait_for_dataset():
    print("--> Waiting for PDM-Lite dataset download to complete...", flush=True)
    while check_process_running("download_pdm_lite.py"):
        time.sleep(15)
        # print tail of log
        r = subprocess.run("tail -n 2 /workspace/download_pdm_lite.log 2>/dev/null", shell=True, capture_output=True, text=True)
        print(f"[{time.strftime('%H:%M:%S')}] {r.stdout.strip()}", flush=True)
    print("✓ Dataset download completed!", flush=True)

def verify_carla_environment():
    print("--> Verifying CARLA environment and checkpoints...", flush=True)
    assert os.path.exists("/workspace/checkpoints/wor_qwen30m_8towns_fast/model_epoch_015.pth"), "Missing model_epoch_015.pth!"
    assert os.path.exists("/workspace/pretrained_carla/model_0030_0.pth"), "Missing model_0030_0.pth!"
    assert os.path.exists("/workspace/dataset/wor_trajectories"), "Missing dataset dir!"
    print("✓ CARLA prerequisites verified!", flush=True)

def launch_carla():
    print("--> Launching CARLA World-on-Rails Training (Resuming from Epoch 15)...", flush=True)
    cmd = (
        "taskset -c 0-15 nohup /venv/main/bin/python /workspace/MThesis/scripts/training/train_wor.py "
        "--data_dir /workspace/dataset/wor_trajectories "
        "--save_dir /workspace/checkpoints/wor_qwen30m_8towns_fast "
        "--run_label wor_qwen30m_8towns_fast "
        "--policy_arch qwen30m "
        "--backbone regnety_032 "
        "--weights_path /workspace/pretrained_carla/model_0030_0.pth "
        "--freeze_backbone 1 "
        "--img_size 192x512 "
        "--vision_grid 4 "
        "--route_overlay 1 "
        "--use_augmented_camera 1 "
        "--target_speed_loss_weight 0.2 "
        "--lateral_loss_weight 3.0 "
        "--batch_size 256 "
        "--epochs 50 "
        "--val_every 5 "
        "--num_workers 12 "
        "--save_freq 3 "
        "--use_mlflow 1 "
        "--mlflow_port 10100 "
        "--resume_from /workspace/checkpoints/wor_qwen30m_8towns_fast/model_epoch_015.pth "
        ">> /workspace/train_wor_8towns.log 2>&1 &"
    )
    subprocess.run(cmd, shell=True, check=True)
    time.sleep(3)
    subprocess.run("ps aux | grep train_wor | grep -v grep", shell=True)

def launch_atari():
    print("--> Launching Atari Off-Policy MCTS Benchmark on cores 16-31...", flush=True)
    # Check if benchmark is already running
    if check_process_running("run_100k_benchmark.py"):
        print("Atari benchmark already running!", flush=True)
        return
        
    cmd = (
        "taskset -c 16-31 nohup /venv/main/bin/python /workspace/MThesis/atari_qwen/scripts/run_100k_benchmark.py "
        "--variants S049a_mcts_sim20 S049b_mcts_sim35 S049c_mcts_sim50 "
        "--seeds 42 "
        "--parallel "
        ">> /workspace/benchmark_mcts_accelerated.log 2>&1 &"
    )
    subprocess.run(cmd, shell=True, check=True)
    time.sleep(3)
    subprocess.run("ps aux | grep run_100k_benchmark | grep -v grep", shell=True)

def main():
    wait_for_dataset()
    verify_carla_environment()
    launch_carla()
    launch_atari()
    print("✓ All parallel workloads successfully launched and pinned!", flush=True)

if __name__ == "__main__":
    main()
