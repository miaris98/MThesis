import subprocess
from remote_config import load_config

cfg = load_config()


def check():
    r = subprocess.run(
        cfg.ssh(
            f"ls -la {cfg.remote_path('results/100k_benchmark')}; echo '---'; "
            f"du -sh {cfg.remote_path('results/100k_benchmark')}/*; echo '---'; "
            "ps aux | grep -E 'tar|run_100k|train_mcts' | grep -v grep || true"
        ),
        capture_output=True, text=True,
    )
    print(r.stdout)


if __name__ == "__main__":
    check()
