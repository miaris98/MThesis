import subprocess
from remote_config import load_config

cfg = load_config()


def check():
    r = subprocess.run(
        cfg.ssh(
            "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader"
        ),
        capture_output=True, text=True,
    )
    print("GPU:", r.stdout.strip())


if __name__ == "__main__":
    check()
