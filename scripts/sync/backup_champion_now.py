import subprocess
import os
from pathlib import Path

DEST_DIR = Path("E:/MThesis_EXP/carla_wor_8towns_epoch20_champion_20260923")
DEST_DIR.mkdir(parents=True, exist_ok=True)

REMOTE_FILES = [
    "/workspace/checkpoints/wor_qwen30m_8towns_fast/best_model.pth",
    "/workspace/checkpoints/wor_qwen30m_8towns_fast/frozen_backbone.pth",
    "/workspace/checkpoints/wor_qwen30m_8towns_fast/latest_model.pth",
    "/workspace/checkpoints/wor_qwen30m_8towns_fast/run_config.json",
    "/workspace/checkpoints/wor_qwen30m_8towns_fast/wor_training_telemetry.csv",
    "/workspace/train_wor_8towns.log"
]

def backup():
    print(f"--> Backing up Epoch 20 Champion files to {DEST_DIR}...")
    for rf in REMOTE_FILES:
        fname = Path(rf).name
        dest = DEST_DIR / fname
        print(f"Streaming {rf} -> {dest}...")
        cmd = [
            "scp", "-P", "50011", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
            f"root@97.115.163.36:{rf}", str(dest)
        ]
        r = subprocess.run(cmd)
        if r.returncode == 0:
            print(f"✓ Saved {fname} ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"✗ Failed to scp {rf}")

if __name__ == "__main__":
    backup()
