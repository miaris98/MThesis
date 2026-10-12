import os
import subprocess
import sys

def sync_box1():
    print("=== [1/2] SYNCING BOX 1 (CARLA 8-Town Epoch 15) ===")
    target_dir = r"E:\MThesis_EXP\imports\carla_8towns_epoch15_20260922"
    os.makedirs(target_dir, exist_ok=True)
    
    cmd_ssh = [
        "ssh", "-p", "26934", "-o", "StrictHostKeyChecking=no", "root@202.122.49.242",
        "tar -czf - -C /workspace checkpoints/wor_qwen30m_8towns_fast/best_model.pth "
        "checkpoints/wor_qwen30m_8towns_fast/frozen_backbone.pth "
        "checkpoints/wor_qwen30m_8towns_fast/latest_model.pth "
        "checkpoints/wor_qwen30m_8towns_fast/model_epoch_015.pth "
        "checkpoints/wor_qwen30m_8towns_fast/run_config.json "
        "checkpoints/wor_qwen30m_8towns_fast/wor_training_telemetry.csv "
        "train_wor_8towns.log 2>/dev/null"
    ]
    print("Streaming Box 1 files via tar/ssh...")
    p1 = subprocess.Popen(cmd_ssh, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["tar", "-xzf", "-", "-C", target_dir], stdin=p1.stdout)
    p1.stdout.close()
    p2.communicate()
    print(f"Box 1 sync finished with returncode: {p2.returncode}")

def sync_box2():
    print("\n=== [2/2] SYNCING BOX 2 (Atari 100k S049 Step 38k) ===")
    target_dir = r"E:\MThesis_EXP\imports\atari_100k_s049_step38k_20260922"
    os.makedirs(target_dir, exist_ok=True)
    
    cmd_ssh = [
        "ssh", "-p", "43921", "-o", "StrictHostKeyChecking=no", "root@158.181.52.18",
        "tar -czf - -C /workspace "
        "MThesis/results/100k_benchmark/S049a_mcts_sim20_s42 "
        "MThesis/results/100k_benchmark/S049b_mcts_sim35_s42 "
        "MThesis/results/100k_benchmark/S049c_mcts_sim50_s42 "
        "MThesis/results/100k_benchmark/S049_mcts_offpolicy_s42 "
        "MThesis/results/100k_benchmark/_logs "
        "benchmark_mcts_accelerated.log 2>/dev/null"
    ]
    print("Streaming Box 2 files via tar/ssh...")
    p1 = subprocess.Popen(cmd_ssh, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["tar", "-xzf", "-", "-C", target_dir], stdin=p1.stdout)
    p1.stdout.close()
    p2.communicate()
    print(f"Box 2 sync finished with returncode: {p2.returncode}")

if __name__ == "__main__":
    sync_box1()
    sync_box2()
    print("\nAll downloads completed successfully!")
