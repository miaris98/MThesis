import subprocess
from remote_config import load_config

cfg = load_config()


def check():
    r = subprocess.run(
        cfg.ssh(
            f"tail -n 12 /workspace/download_pdm_lite.log 2>/dev/null; echo '==='; "
            f"du -sh {cfg.remote_path('results/100k_benchmark')} 2>/dev/null; "
            "du -sh /workspace/dataset/wor_trajectories 2>/dev/null"
        ),
        capture_output=True, text=True,
    )
    print(r.stdout)


if __name__ == "__main__":
    check()
