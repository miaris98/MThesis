import subprocess
from remote_config import load_config

cfg = load_config()


def monitor():
    r = subprocess.run(
        cfg.ssh(
            f"echo '=== [AUTO LAUNCH LOG] ==='; cat /workspace/auto_launch.log 2>/dev/null | tail -n 10; "
            f"echo '=== [DATASET DOWNLOAD] ==='; tail -n 6 /workspace/download_pdm_lite.log 2>/dev/null; "
            "echo '=== [DATASET DISK USAGE] ==='; du -sh /workspace/dataset/wor_trajectories 2>/dev/null; "
            f"echo '=== [BENCHMARK RESULTS] ==='; ls -la {cfg.remote_path('results/100k_benchmark')} 2>/dev/null; "
            "echo '=== [TRAINING PROCESSES] ==='; ps aux | grep -E 'python.*(train_wor|run_100k|auto_launch)' | grep -v grep || true"
        ),
        capture_output=True, text=True,
    )
    print(r.stdout)


if __name__ == "__main__":
    monitor()
