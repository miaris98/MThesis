import os
import subprocess
import time
from remote_config import load_config

cfg = load_config()


def upload_carla():
    src_dir = r"E:\MThesis_EXP\imports\carla_8towns_epoch15_20260922"
    ckpt_rel = r"checkpoints\wor_qwen30m_8towns_fast"
    full_ckpt_dir = os.path.join(src_dir, ckpt_rel)

    if not os.path.exists(full_ckpt_dir):
        print(f"Error: {full_ckpt_dir} does not exist!")
        return False

    remote_ckpt_dir = "/workspace/checkpoints/wor_qwen30m_8towns_fast"
    print(f"--> Creating remote directories...")
    subprocess.run(
        cfg.ssh(f"mkdir -p {remote_ckpt_dir} /workspace/dataset/wor_trajectories /workspace/pretrained_carla"),
        check=True,
    )

    files_to_send = [
        "best_model.pth",
        "frozen_backbone.pth",
        "latest_model.pth",
        "model_epoch_015.pth",
        "run_config.json",
        "wor_training_telemetry.csv",
    ]

    print(f"--> Uploading {len(files_to_send)} CARLA checkpoint files via tar stream...")
    tar_cmd = ["tar", "-czf", "-"] + files_to_send
    ssh_untar_cmd = cfg.ssh(f"tar -xzf - -C {remote_ckpt_dir}")

    t0 = time.time()
    p1 = subprocess.Popen(tar_cmd, cwd=full_ckpt_dir, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(ssh_untar_cmd, stdin=p1.stdout)
    p1.stdout.close()
    p2.communicate()

    if p2.returncode != 0:
        print(f"[ERROR] Tar upload failed with code {p2.returncode}")
        return False

    dt = time.time() - t0
    print(f"✓ Checkpoints uploaded in {dt:.1f}s!")

    # Also upload previous training log
    log_src = os.path.join(src_dir, "train_wor_8towns.log")
    if os.path.exists(log_src):
        print("--> Uploading train_wor_8towns.log...")
        p1 = subprocess.Popen(["tar", "-czf", "-", "train_wor_8towns.log"], cwd=src_dir, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cfg.ssh("tar -xzf - -C /workspace"), stdin=p1.stdout)
        p1.stdout.close()
        p2.communicate()
        print("✓ Previous log uploaded!")

    # Verify
    res = subprocess.run(cfg.ssh(f"ls -lh {remote_ckpt_dir}"), capture_output=True, text=True)
    print("\nRemote checkpoint directory:")
    print(res.stdout)
    return True


if __name__ == "__main__":
    upload_carla()
