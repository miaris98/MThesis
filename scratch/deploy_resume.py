"""deploy_resume.py — Push and run a CARLA resume script on the remote box."""
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "setup"))
from remote_config import load_config

cfg = load_config()

script = f"""#!/bin/bash
set -e
ulimit -n 1048576

cd {cfg.workspace}
export MLFLOW_ALLOW_FILE_STORE=true

nohup {cfg.python_carla()} scripts/training/train_wor.py \\
    --data_dir /workspace/dataset/wor_trajectories \\
    --save_dir /workspace/checkpoints/wor_qwen30m_8towns_fast \\
    --run_label wor_qwen30m_8towns_fast \\
    --policy_arch qwen30m --backbone regnety_032 \\
    --weights_path /workspace/pretrained_carla/model_0030_0.pth \\
    --freeze_backbone 1 --img_size 192x512 --vision_grid 4 \\
    --route_overlay 1 --use_augmented_camera 1 \\
    --target_speed_loss_weight 0.2 --lateral_loss_weight 3.0 \\
    --batch_size 256 --epochs 50 --val_every 5 \\
    --num_workers 16 --save_freq 3 \\
    --use_mlflow 1 --mlflow_port 10100 \\
    --resume_from /workspace/checkpoints/wor_qwen30m_8towns_fast/latest_model.pth \\
    >> /workspace/train_wor_8towns.log 2>&1 &
echo $!
"""

proc = subprocess.run(
    cfg.ssh_T("cat > /workspace/resume_carla.sh && tr -d '\\r' < /workspace/resume_carla.sh > /workspace/resume_carla_clean.sh && bash /workspace/resume_carla_clean.sh"),
    input=script.encode("utf-8"),
    capture_output=True,
)
print("STDOUT:", proc.stdout.decode("utf-8", errors="replace"))
print("STDERR:", proc.stderr.decode("utf-8", errors="replace"))
