import subprocess
from pathlib import Path
from remote_config import load_config

cfg = load_config()

DEST_DIR = Path(r"E:\MThesis_EXP\carla_wor_8towns_epoch50_final")
DEST_DIR.mkdir(parents=True, exist_ok=True)

REMOTE_DIR = "/workspace/checkpoints/wor_qwen30m_8towns_fast"

FILES = [
    "model_epoch_050.pth",
    "latest_model.pth",
    "best_model.pth",
    "frozen_backbone.pth",
    "wor_training_telemetry.csv",
    "run_config.json",
]

print(f"--> Starting backup of Epoch 50 Final files to {DEST_DIR}...")

for fname in FILES:
    dest_path = DEST_DIR / fname
    print(f"--> Syncing {fname}...")
    r = subprocess.run(
        cfg.scp_download(f"{REMOTE_DIR}/{fname}", str(dest_path)),
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        sz_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"✓ {fname} ({sz_mb:.2f} MB) saved successfully!")
    else:
        print(f"✗ Failed to sync {fname}: {r.stderr}")

# Also copy the full training log
log_dest = DEST_DIR / "train_wor_8towns.log"
subprocess.run(cfg.scp_download("/workspace/train_wor_8towns.log", str(log_dest)))
if log_dest.exists():
    print(f"✓ train_wor_8towns.log ({log_dest.stat().st_size / 1024:.1f} KB) saved!")

print("\n✓ [CARLA Stage 1 Backup Complete] All Epoch 50 final checkpoints archived to E:\\MThesis_EXP!")
