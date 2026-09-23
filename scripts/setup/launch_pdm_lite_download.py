import subprocess
import time
from remote_config import load_config

cfg = load_config()


def launch_dataset_download():
    print("--> Launching PDM-Lite dataset download daemon on remote instance...")
    subprocess.run(
        cfg.ssh(
            f"nohup {cfg.python()} {cfg.remote_path('scripts/setup/download_pdm_lite.py')} "
            "--dest /workspace/dataset/wor_trajectories "
            "> /workspace/download_pdm_lite.log 2>&1 &"
        ),
        check=True,
    )
    time.sleep(2)

    res = cfg.run_ssh("head -n 25 /workspace/download_pdm_lite.log 2>/dev/null || echo 'Log not yet created'", check=False)
    print("Initial download log:")
    print(res.stdout)


if __name__ == "__main__":
    launch_dataset_download()
