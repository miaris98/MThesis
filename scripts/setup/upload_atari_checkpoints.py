import os
import subprocess
import time
from remote_config import load_config

cfg = load_config()


def upload_atari():
    src_dir = r"E:\MThesis_EXP\imports\atari_100k_s049_step38k_20260922\MThesis\results\100k_benchmark"
    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} does not exist!")
        return False

    remote_bench = cfg.remote_path("results/100k_benchmark")
    print(f"--> Creating remote directory {remote_bench}")
    subprocess.run(cfg.ssh(f"mkdir -p {remote_bench}"), check=True)

    items_to_send = [
        "S049a_mcts_sim20_s42",
        "S049b_mcts_sim35_s42",
        "S049c_mcts_sim50_s42",
        "S049_mcts_offpolicy_s42",
        "_logs",
    ]
    existing_items = [item for item in items_to_send if os.path.exists(os.path.join(src_dir, item))]

    print(f"--> Uploading Atari benchmark results and checkpoints: {existing_items}...")
    tar_cmd = ["tar", "-czf", "-"] + existing_items
    ssh_untar_cmd = cfg.ssh(f"tar -xzf - -C {remote_bench}")

    t0 = time.time()
    p1 = subprocess.Popen(tar_cmd, cwd=src_dir, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(ssh_untar_cmd, stdin=p1.stdout)
    p1.stdout.close()
    p2.communicate()

    if p2.returncode != 0:
        print(f"[ERROR] Atari tar upload failed with code {p2.returncode}")
        return False

    dt = time.time() - t0
    print(f"✓ Atari checkpoints uploaded in {dt:.1f}s!")

    res = subprocess.run(cfg.ssh(f"ls -la {remote_bench}"), capture_output=True, text=True)
    print("\nRemote Atari benchmark directory:")
    print(res.stdout)
    return True


if __name__ == "__main__":
    upload_atari()
